//! API fetch command implementation
//!
//! This module implements the `api-fetch` command which fetches data from HTTP APIs,
//! processes responses with JSONPath expressions, and stores the results in memory files.

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use clap::ValueEnum;
use jsonpath_lib::select as jsonpath_select;
use memvid_core::{DocMetadata, Memvid, MemvidError, PutOptions};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE, USER_AGENT};
use serde::Deserialize;
use serde_json::{json, Value};
use time::format_description::well_known::Rfc3339;
use time::macros::format_description;
use time::{OffsetDateTime, PrimitiveDateTime};

use crate::config::CliConfig;
use crate::utils::ensure_cli_mutation_allowed;

/// Internal command structure for api-fetch operations
#[derive(Debug)]
pub(crate) struct ApiFetchCommand {
    pub file: PathBuf,
    pub config_path: PathBuf,
    pub dry_run: bool,
    pub mode_override: Option<ApiFetchMode>,
    pub uri_override: Option<String>,
    pub output_json: bool,
    pub lock_timeout_ms: u64,
    pub force_lock: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ApiFetchMode {
    Insert,
    Upsert,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum HttpMethod {
    #[serde(alias = "GET")]
    Get,
    #[serde(alias = "POST")]
    Post,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ResponseContentType {
    #[serde(alias = "JSON")]
    Json,
    #[serde(alias = "RAW")]
    Raw,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RequestBody {
    Json(Value),
    Text(String),
}

#[derive(Debug, Clone, Deserialize)]
struct ExtractConfig {
    pub items: String,
    pub id: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tags: Option<String>,
    #[serde(default)]
    pub metadata: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiFetchConfig {
    pub url: String,
    #[serde(default)]
    pub method: Option<HttpMethod>,
    #[serde(default)]
    pub body: Option<RequestBody>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    pub extract: ExtractConfig,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub mode: Option<ApiFetchMode>,
    #[serde(default)]
    pub tags: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub timestamp: Option<TimestampValue>,
    #[serde(default)]
    pub max_items: Option<usize>,
    #[serde(default)]
    pub content_type: Option<ResponseContentType>,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TimestampValue {
    Number(i64),
    String(String),
}

#[derive(Debug)]
struct PreparedConfig {
    url: String,
    method: HttpMethod,
    body: Option<RequestBody>,
    headers: BTreeMap<String, String>,
    extract: ExtractConfig,
    base_uri: Option<String>,
    mode: ApiFetchMode,
    static_tags: Vec<TagDirective>,
    static_doc_metadata: Option<DocMetadata>,
    static_extra_metadata: BTreeMap<String, Value>,
    static_timestamp: Option<i64>,
    max_items: Option<usize>,
    content_type: ResponseContentType,
    default_title: Option<String>,
}

#[derive(Debug, Clone)]
enum TagDirective {
    Simple(String),
    Pair { key: String, value: String },
}

#[derive(Debug)]
struct FramePlan {
    uri: String,
    title: Option<String>,
    payload: String,
    tags: Vec<TagDirective>,
    doc_metadata: Option<DocMetadata>,
    extra_metadata: BTreeMap<String, Value>,
    timestamp: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameActionKind {
    Insert,
    Update,
    Skip,
}

#[derive(Debug)]
struct FrameResult {
    uri: String,
    action: FrameActionKind,
    sequence: Option<u64>,
    frame_id: Option<u64>,
    reason: Option<String>,
}

pub(crate) fn run_api_fetch(_config: &CliConfig, command: ApiFetchCommand) -> Result<()> {
    let config_data = load_config(&command.config_path)?;
    let prepared = config_data.into_prepared(&command)?;

    let client = crate::http::blocking_client(Duration::from_secs(60))
        .context("failed to build HTTP client")?;

    let headers = build_headers(&prepared.headers)?;
    let response_body = execute_request(&client, &prepared, headers)?;
    let root_value = parse_response(&response_body, prepared.content_type)?;

    let mut warnings = Vec::new();
    let mut plans = build_frame_plans(&prepared, &root_value)?;
    if plans.is_empty() {
        bail!("no items matched extract.items path");
    }

    let mut seen_uris = HashSet::new();
    plans.retain(|plan| {
        if !seen_uris.insert(plan.uri.clone()) {
            warnings.push(format!("duplicate URI planned: {}", plan.uri));
            false
        } else {
            true
        }
    });

    let mut mem = Memvid::open(&command.file)?;
    {
        let settings = mem.lock_settings_mut();
        settings.timeout_ms = command.lock_timeout_ms;
        settings.force_stale = command.force_lock;
    }
    ensure_cli_mutation_allowed(&mem)?;

    let mut results = Vec::new();
    let mut inserted = 0usize;
    let mut updated = 0usize;
    let mut skipped = 0usize;

    for plan in plans {
        let existing = match mem.frame_by_uri(&plan.uri) {
            Ok(frame) => Some(frame),
            Err(MemvidError::FrameNotFoundByUri { .. }) => None,
            Err(err) => return Err(err.into()),
        };

        match (existing, prepared.mode) {
            (Some(frame), ApiFetchMode::Insert) => {
                skipped += 1;
                results.push(FrameResult {
                    uri: plan.uri,
                    action: FrameActionKind::Skip,
                    sequence: None,
                    frame_id: Some(frame.id),
                    reason: Some("frame already exists (mode=insert)".to_string()),
                });
            }
            (Some(frame), ApiFetchMode::Upsert) => {
                if command.dry_run {
                    updated += 1;
                    results.push(FrameResult {
                        uri: plan.uri,
                        action: FrameActionKind::Update,
                        sequence: None,
                        frame_id: Some(frame.id),
                        reason: None,
                    });
                } else {
                    let seq = apply_update(&mut mem, frame.id, &plan)?;
                    results.push(FrameResult {
                        uri: plan.uri,
                        action: FrameActionKind::Update,
                        sequence: Some(seq),
                        frame_id: Some(frame.id),
                        reason: None,
                    });
                    updated += 1;
                }
            }
            (None, _) => {
                if command.dry_run {
                    inserted += 1;
                    results.push(FrameResult {
                        uri: plan.uri,
                        action: FrameActionKind::Insert,
                        sequence: None,
                        frame_id: None,
                        reason: None,
                    });
                } else {
                    let seq = apply_insert(&mut mem, &plan)?;
                    results.push(FrameResult {
                        uri: plan.uri,
                        action: FrameActionKind::Insert,
                        sequence: Some(seq),
                        frame_id: None,
                        reason: None,
                    });
                    inserted += 1;
                }
            }
        }
    }

    if !command.dry_run {
        mem.commit()?;
    }

    if command.output_json {
        print_json_summary(
            &results,
            inserted,
            updated,
            skipped,
            command.dry_run,
            &warnings,
        )?;
    } else {
        print_human_summary(
            &results,
            inserted,
            updated,
            skipped,
            command.dry_run,
            &warnings,
        );
    }

    Ok(())
}

fn load_config(path: &Path) -> Result<ApiFetchConfig> {
    let file = File::open(path)
        .with_context(|| format!("failed to open config file at {}", path.display()))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader)
        .with_context(|| format!("failed to parse config JSON at {}", path.display()))
}

impl ApiFetchConfig {
    fn into_prepared(self, cmd: &ApiFetchCommand) -> Result<PreparedConfig> {
        let mode = cmd
            .mode_override
            .or(self.mode)
            .unwrap_or(ApiFetchMode::Insert);

        let base_uri = cmd
            .uri_override
            .clone()
            .or(self.uri)
            .map(|value| value.trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty());

        let static_tags = parse_tags_value(self.tags)?;
        let (static_doc_metadata, static_extra_metadata) = split_metadata(self.metadata)?;
        let static_timestamp = match self.timestamp {
            Some(value) => Some(parse_timestamp_value(&value)?),
            None => None,
        };

        let content_type = self.content_type.unwrap_or(ResponseContentType::Json);

        Ok(PreparedConfig {
            url: self.url,
            method: self.method.unwrap_or(HttpMethod::Get),
            body: self.body,
            headers: self.headers,
            extract: self.extract,
            base_uri,
            mode,
            static_tags,
            static_doc_metadata,
            static_extra_metadata,
            static_timestamp,
            max_items: self.max_items,
            content_type,
            default_title: self.title,
        })
    }
}

fn parse_tags_value(value: Option<Value>) -> Result<Vec<TagDirective>> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![TagDirective::Simple(s.trim().to_string())]),
        Some(Value::Array(items)) => {
            let mut tags = Vec::new();
            for item in items {
                match item {
                    Value::String(s) => tags.push(TagDirective::Simple(s.trim().to_string())),
                    Value::Object(map) => {
                        if let Some(Value::String(val)) = map.get("value") {
                            if let Some(Value::String(key)) = map.get("key") {
                                tags.push(TagDirective::Pair {
                                    key: key.trim().to_string(),
                                    value: val.trim().to_string(),
                                });
                                continue;
                            }
                        }
                        bail!(
                            "invalid tag entry in array; expected string or {{\"key\",\"value\"}}"
                        );
                    }
                    _ => bail!("tags array must contain strings or key/value objects"),
                }
            }
            Ok(tags)
        }
        Some(Value::Object(object)) => {
            let mut tags = Vec::new();
            for (key, value) in object {
                let key = key.trim().to_string();
                let value_str = value_to_string(&value)?;
                tags.push(TagDirective::Pair {
                    key,
                    value: value_str,
                });
            }
            Ok(tags)
        }
        Some(_) => bail!("unsupported tags format; expected string, array, or object"),
    }
}

fn split_metadata(value: Option<Value>) -> Result<(Option<DocMetadata>, BTreeMap<String, Value>)> {
    match value {
        None | Some(Value::Null) => Ok((None, BTreeMap::new())),
        Some(Value::Object(map)) => {
            let doc_attempt: Result<DocMetadata, _> =
                serde_json::from_value(Value::Object(map.clone()));
            if let Ok(meta) = doc_attempt {
                Ok((Some(meta), BTreeMap::new()))
            } else {
                let mut extras = BTreeMap::new();
                for (key, value) in map {
                    extras.insert(key, value);
                }
                Ok((None, extras))
            }
        }
        Some(other) => {
            let meta: DocMetadata = serde_json::from_value(other.clone())
                .context("metadata must be an object compatible with DocMetadata")?;
            Ok((Some(meta), BTreeMap::new()))
        }
    }
}

fn parse_timestamp_value(value: &TimestampValue) -> Result<i64> {
    match value {
        TimestampValue::Number(num) => Ok(*num),
        TimestampValue::String(text) => parse_timestamp_str(text),
    }
}

fn parse_timestamp_str(raw: &str) -> Result<i64> {
    if let Ok(num) = raw.trim().parse::<i64>() {
        return Ok(num);
    }
    if let Ok(dt) = OffsetDateTime::parse(raw.trim(), &Rfc3339) {
        return Ok(dt.unix_timestamp());
    }
    if let Ok(parsed) = PrimitiveDateTime::parse(
        raw.trim(),
        &format_description!("[year]-[month]-[day] [hour]:[minute]:[second]"),
    ) {
        return Ok(parsed.assume_utc().unix_timestamp());
    }
    if let Ok(parsed) = PrimitiveDateTime::parse(
        raw.trim(),
        &format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]"),
    ) {
        return Ok(parsed.assume_utc().unix_timestamp());
    }
    bail!("unable to parse timestamp: {raw}")
}

fn build_headers(raw: &BTreeMap<String, String>) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    for (key, value) in raw {
        let resolved = resolve_value(value)?;
        let name = HeaderName::try_from(key.as_str())
            .with_context(|| format!("invalid header name: {key}"))?;
        let header_value = HeaderValue::from_str(&resolved)
            .with_context(|| format!("invalid header value for {key}"))?;
        headers.insert(name, header_value);
    }
    Ok(headers)
}

fn resolve_value(raw: &str) -> Result<String> {
    if let Some(env_key) = raw.strip_prefix("env:") {
        let key = env_key.trim();
        let value = env::var(key).with_context(|| {
            format!("environment variable {key} referenced in config is not set")
        })?;
        Ok(value)
    } else {
        Ok(raw.to_string())
    }
}

fn execute_request(
    client: &Client,
    prepared: &PreparedConfig,
    mut headers: HeaderMap,
) -> Result<String> {
    if !headers.contains_key(USER_AGENT) {
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static(concat!("memvid-cli/", env!("CARGO_PKG_VERSION"))),
        );
    }
    if prepared.content_type == ResponseContentType::Json && !headers.contains_key(ACCEPT) {
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    }
    if matches!(prepared.method, HttpMethod::Post)
        && prepared
            .body
            .as_ref()
            .map(|body| matches!(body, RequestBody::Json(_)))
            .unwrap_or(false)
        && !headers.contains_key(CONTENT_TYPE)
    {
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    }

    let mut request = match prepared.method {
        HttpMethod::Get => client.get(&prepared.url),
        HttpMethod::Post => client.post(&prepared.url),
    };
    if let Some(body) = &prepared.body {
        match body {
            RequestBody::Json(value) => {
                request = request.json(value);
            }
            RequestBody::Text(text) => {
                request = request.body(text.clone());
            }
        }
    }
    request = request.headers(headers);

    let response = request
        .send()
        .with_context(|| format!("request to {} failed", prepared.url))?;
    let response = response
        .error_for_status()
        .with_context(|| format!("remote endpoint {} returned an error status", prepared.url))?;
    response.text().context("failed to read response body")
}

fn parse_response(body: &str, kind: ResponseContentType) -> Result<Value> {
    match serde_json::from_str::<Value>(body) {
        Ok(value) => Ok(value),
        Err(err) => {
            if matches!(kind, ResponseContentType::Raw) {
                Ok(Value::String(body.to_string()))
            } else {
                Err(anyhow!("failed to decode response as JSON: {err}"))
            }
        }
    }
}

fn build_frame_plans(prepared: &PreparedConfig, root: &Value) -> Result<Vec<FramePlan>> {
    let mut matches = jsonpath_select(root, &prepared.extract.items)
        .with_context(|| format!("jsonpath '{}' failed", prepared.extract.items))?;
    if matches.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(limit) = prepared.max_items {
        if matches.len() > limit {
            matches.truncate(limit);
        }
    }

    let mut plans = Vec::with_capacity(matches.len());
    for item in matches {
        let plan = build_frame_plan(prepared, item)?;
        plans.push(plan);
    }
    Ok(plans)
}

fn build_frame_plan(prepared: &PreparedConfig, item: &Value) -> Result<FramePlan> {
    let id_matches = jsonpath_select(item, &prepared.extract.id)
        .with_context(|| format!("jsonpath '{}' failed", prepared.extract.id))?;
    let id_value = id_matches.first().ok_or_else(|| {
        anyhow!(
            "extract.id path '{}' returned no results",
            prepared.extract.id
        )
    })?;
    let id = value_to_string(id_value)?.trim().to_string();
    if id.is_empty() {
        bail!("extracted id is empty after trimming");
    }
    let uri = build_uri(prepared.base_uri.as_deref(), &id);

    let derived_title = if let Some(path) = &prepared.extract.title {
        let title_matches =
            jsonpath_select(item, path).with_context(|| format!("jsonpath '{}' failed", path))?;
        title_matches
            .first()
            .map(|value| value_to_string(value))
            .transpose()?
    } else {
        None
    };

    let title = derived_title.or_else(|| prepared.default_title.clone());

    let payload = extract_payload(prepared, item)?;
    let mut tags = prepared.static_tags.clone();
    if let Some(path) = &prepared.extract.tags {
        let tag_values =
            jsonpath_select(item, path).with_context(|| format!("jsonpath '{}' failed", path))?;
        for value in tag_values {
            let tag = value_to_string(value)?;
            if !tag.trim().is_empty() {
                tags.push(TagDirective::Simple(tag.trim().to_string()));
            }
        }
    }

    let mut doc_metadata = prepared.static_doc_metadata.clone();
    let mut extra_metadata = prepared.static_extra_metadata.clone();

    if let Some(path) = &prepared.extract.metadata {
        let meta_matches =
            jsonpath_select(item, path).with_context(|| format!("jsonpath '{}' failed", path))?;
        if let Some(value) = meta_matches.first() {
            match value {
                Value::Null => {}
                Value::Object(map) => {
                    let mut handled = false;
                    if doc_metadata.is_none() {
                        if let Ok(parsed) =
                            serde_json::from_value::<DocMetadata>(Value::Object(map.clone()))
                        {
                            doc_metadata = Some(parsed);
                            handled = true;
                        }
                    }
                    if !handled {
                        for (key, value) in map.iter() {
                            extra_metadata.insert(key.clone(), (*value).clone());
                        }
                    }
                }
                other => {
                    extra_metadata.insert(path.clone(), (*other).clone());
                }
            }
        }
    }

    if doc_metadata
        .as_ref()
        .and_then(|meta| meta.mime.as_ref())
        .is_none()
    {
        let default_mime = match prepared.content_type {
            ResponseContentType::Json => Some("application/json"),
            ResponseContentType::Raw => Some("text/plain"),
        };
        if let Some(mime) = default_mime {
            match doc_metadata.as_mut() {
                Some(existing) => existing.mime = Some(mime.to_string()),
                None => {
                    let mut meta = DocMetadata::default();
                    meta.mime = Some(mime.to_string());
                    doc_metadata = Some(meta);
                }
            }
        }
    }

    let mut timestamp = prepared.static_timestamp;
    if let Some(path) = &prepared.extract.timestamp {
        let ts_matches =
            jsonpath_select(item, path).with_context(|| format!("jsonpath '{}' failed", path))?;
        if let Some(value) = ts_matches.first() {
            timestamp = Some(parse_timestamp_from_value(value)?);
        }
    }

    Ok(FramePlan {
        uri,
        title,
        payload,
        tags,
        doc_metadata,
        extra_metadata,
        timestamp,
    })
}

fn extract_payload(prepared: &PreparedConfig, item: &Value) -> Result<String> {
    if let Some(path) = &prepared.extract.content {
        let payload_matches =
            jsonpath_select(item, path).with_context(|| format!("jsonpath '{}' failed", path))?;
        let value = payload_matches
            .first()
            .ok_or_else(|| anyhow!("extract.content path '{}' returned no results", path))?;
        match prepared.content_type {
            ResponseContentType::Json => value_to_pretty_text(value),
            ResponseContentType::Raw => value_to_plain_string(value),
        }
    } else {
        match prepared.content_type {
            ResponseContentType::Json => value_to_pretty_text(item),
            ResponseContentType::Raw => value_to_plain_string(item),
        }
    }
}

fn value_to_plain_string(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Number(num) => Ok(num.to_string()),
        Value::Bool(flag) => Ok(flag.to_string()),
        other => bail!("expected textual content but found {other}"),
    }
}

fn value_to_pretty_text(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        _ => serde_json::to_string_pretty(value).map_err(|err| anyhow!(err)),
    }
}

fn value_to_string(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Number(num) => Ok(num.to_string()),
        Value::Bool(flag) => Ok(flag.to_string()),
        Value::Null => bail!("value is null"),
        other => serde_json::to_string(other).map_err(|err| anyhow!(err)),
    }
}

fn parse_timestamp_from_value(value: &Value) -> Result<i64> {
    match value {
        Value::Number(num) => num
            .as_i64()
            .ok_or_else(|| anyhow!("timestamp number must fit i64")),
        Value::String(text) => parse_timestamp_str(text),
        other => bail!("unsupported timestamp value: {other}"),
    }
}

fn build_uri(base: Option<&str>, id: &str) -> String {
    match base {
        Some(prefix) => {
            let prefix = prefix.trim_end_matches('/');
            let suffix = id.trim_start_matches('/');
            if prefix.is_empty() {
                suffix.to_string()
            } else if suffix.is_empty() {
                prefix.to_string()
            } else {
                format!("{prefix}/{suffix}")
            }
        }
        None => id.to_string(),
    }
}

fn apply_insert(mem: &mut Memvid, plan: &FramePlan) -> Result<u64> {
    let options = build_put_options(plan);
    let payload = plan.payload.as_bytes();
    mem.put_bytes_with_options(payload, options)
        .context("failed to insert frame")
}

fn apply_update(mem: &mut Memvid, frame_id: u64, plan: &FramePlan) -> Result<u64> {
    let options = build_put_options(plan);
    let payload = plan.payload.as_bytes().to_vec();
    mem.update_frame(frame_id, Some(payload), options, None)
        .context("failed to update frame")
}

fn build_put_options(plan: &FramePlan) -> PutOptions {
    let mut builder = PutOptions::builder()
        .enable_embedding(false)
        .auto_tag(false)
        .extract_dates(false)
        .uri(plan.uri.clone())
        .search_text(plan.payload.clone());

    if let Some(ts) = plan.timestamp {
        builder = builder.timestamp(ts);
    }
    if let Some(title) = &plan.title {
        if !title.trim().is_empty() {
            builder = builder.title(title.clone());
        }
    }
    if let Some(meta) = plan.doc_metadata.clone() {
        builder = builder.metadata(meta);
    }
    for (key, value) in plan.extra_metadata.iter() {
        builder = builder.metadata_entry(key.clone(), value.clone());
    }
    for tag in &plan.tags {
        match tag {
            TagDirective::Simple(tag) => {
                if !tag.trim().is_empty() {
                    builder = builder.push_tag(tag.clone());
                }
            }
            TagDirective::Pair { key, value } => {
                builder = builder.tag(key.clone(), value.clone());
                if !key.trim().is_empty() {
                    builder = builder.push_tag(key.clone());
                }
                if !value.trim().is_empty() && value != key {
                    builder = builder.push_tag(value.clone());
                }
            }
        }
    }

    builder.build()
}

fn print_json_summary(
    results: &[FrameResult],
    inserted: usize,
    updated: usize,
    skipped: usize,
    dry_run: bool,
    warnings: &[String],
) -> Result<()> {
    let items: Vec<Value> = results
        .iter()
        .map(|res| {
            json!({
                "uri": res.uri,
                "action": match res.action {
                    FrameActionKind::Insert => "insert",
                    FrameActionKind::Update => "update",
                    FrameActionKind::Skip => "skip",
                },
                "sequence": res.sequence,
                "frame_id": res.frame_id,
                "reason": res.reason,
            })
        })
        .collect();
    let summary = json!({
        "dry_run": dry_run,
        "counts": {
            "total": results.len(),
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
        },
        "items": items,
        "warnings": warnings,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn print_human_summary(
    results: &[FrameResult],
    inserted: usize,
    updated: usize,
    skipped: usize,
    dry_run: bool,
    warnings: &[String],
) {
    for res in results {
        match res.action {
            FrameActionKind::Insert => {
                if let Some(seq) = res.sequence {
                    println!("+ {} (seq {})", res.uri, seq);
                } else {
                    println!("+ {} (planned)", res.uri);
                }
            }
            FrameActionKind::Update => {
                if let Some(seq) = res.sequence {
                    println!("~ {} (seq {})", res.uri, seq);
                } else {
                    println!("~ {} (planned)", res.uri);
                }
            }
            FrameActionKind::Skip => {
                if let Some(reason) = &res.reason {
                    println!("- {} ({})", res.uri, reason);
                } else {
                    println!("- {} (skipped)", res.uri);
                }
            }
        }
    }
    println!(
        "Summary: inserted {}, updated {}, skipped {}{}",
        inserted,
        updated,
        skipped,
        if dry_run { " (dry run)" } else { "" }
    );
    if !warnings.is_empty() {
        println!("Warnings:");
        for warning in warnings {
            println!("  - {warning}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn resolve_value_reads_env() {
        const VAR: &str = "API_FETCH_TEST_TOKEN";
        let _guard = env_lock();
        unsafe {
            std::env::set_var(VAR, "secret");
        }
        let resolved = resolve_value(&format!("env:{VAR}")).expect("resolve env");
        assert_eq!(resolved, "secret");
        unsafe {
            std::env::remove_var(VAR);
        }
    }

    #[test]
    fn prepared_config_applies_overrides() {
        let raw = json!({
            "url": "https://example.com/api",
            "method": "get",
            "headers": {},
            "extract": {
                "items": "$.items[*]",
                "id": "$.id"
            },
            "mode": "insert",
            "uri": "mem://base",
            "tags": ["static"],
            "metadata": {"mime": "text/plain"},
            "timestamp": 42,
            "content_type": "json"
        });

        let config: ApiFetchConfig = serde_json::from_value(raw).expect("config");
        let command = ApiFetchCommand {
            file: PathBuf::from("memory.mv2"),
            config_path: PathBuf::from("cfg.json"),
            dry_run: false,
            mode_override: Some(ApiFetchMode::Upsert),
            uri_override: Some("mem://override".to_string()),
            output_json: false,
            lock_timeout_ms: 5000,
            force_lock: false,
        };

        let prepared = config.into_prepared(&command).expect("prepared");
        assert_eq!(prepared.mode, ApiFetchMode::Upsert);
        assert_eq!(prepared.base_uri.as_deref(), Some("mem://override"));
        assert_eq!(prepared.static_tags.len(), 1);
        assert!(prepared.static_doc_metadata.is_some());
        assert_eq!(prepared.static_timestamp, Some(42));
        assert_eq!(prepared.content_type, ResponseContentType::Json);
    }

    #[test]
    fn build_frame_plan_extracts_payload_and_tags() {
        let prepared = PreparedConfig {
            url: "https://example.com".to_string(),
            method: HttpMethod::Get,
            body: None,
            headers: BTreeMap::new(),
            extract: ExtractConfig {
                items: "$.items[*]".to_string(),
                id: "$.id".to_string(),
                content: Some("$.text".to_string()),
                tags: Some("$.tags[*]".to_string()),
                metadata: Some("$.meta".to_string()),
                timestamp: Some("$.ts".to_string()),
                title: Some("$.title".to_string()),
            },
            base_uri: Some("mem://base".to_string()),
            mode: ApiFetchMode::Insert,
            static_tags: vec![TagDirective::Simple("static".to_string())],
            static_doc_metadata: None,
            static_extra_metadata: BTreeMap::new(),
            static_timestamp: Some(100),
            max_items: None,
            content_type: ResponseContentType::Raw,
            default_title: Some("fallback".to_string()),
        };

        let root = json!({
            "items": [
                {
                    "id": "doc-1",
                    "text": "hello world",
                    "tags": ["dynamic"],
                    "meta": {"mime": "text/plain"},
                    "ts": 123,
                    "title": "Greeting"
                }
            ]
        });

        let plans = build_frame_plans(&prepared, &root).expect("plans");
        assert_eq!(plans.len(), 1);
        let plan = &plans[0];
        assert_eq!(plan.uri, "mem://base/doc-1");
        assert_eq!(plan.payload, "hello world");
        assert_eq!(plan.title.as_deref(), Some("Greeting"));
        assert_eq!(plan.timestamp, Some(123));
        assert!(plan
            .tags
            .iter()
            .any(|tag| matches!(tag, TagDirective::Simple(value) if value == "dynamic")));
        assert!(plan
            .tags
            .iter()
            .any(|tag| matches!(tag, TagDirective::Simple(value) if value == "static")));
        let meta = plan.doc_metadata.as_ref().expect("doc metadata");
        assert_eq!(meta.mime.as_deref(), Some("text/plain"));
    }

    #[test]
    fn build_frame_plan_defaults_json_mime() {
        let prepared = PreparedConfig {
            url: "https://example.com".to_string(),
            method: HttpMethod::Get,
            body: None,
            headers: BTreeMap::new(),
            extract: ExtractConfig {
                items: "$".to_string(),
                id: "$.id".to_string(),
                content: None,
                tags: None,
                metadata: None,
                timestamp: None,
                title: Some("$.title".to_string()),
            },
            base_uri: Some("memvid://json".to_string()),
            mode: ApiFetchMode::Insert,
            static_tags: Vec::new(),
            static_doc_metadata: None,
            static_extra_metadata: BTreeMap::new(),
            static_timestamp: None,
            max_items: None,
            content_type: ResponseContentType::Json,
            default_title: None,
        };

        let root = json!({
            "id": 1,
            "title": "Example"
        });

        let plans = build_frame_plans(&prepared, &root).expect("plans");
        assert_eq!(plans.len(), 1);
        let plan = &plans[0];
        let meta = plan.doc_metadata.as_ref().expect("doc metadata");
        assert_eq!(meta.mime.as_deref(), Some("application/json"));
    }
}
