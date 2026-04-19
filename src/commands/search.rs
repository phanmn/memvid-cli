//! Search & retrieval command handlers (find, vec-search, ask, timeline, when).
//!
//! Responsibilities:
//! - Parse CLI arguments for search/RAG/timeline.
//! - Call into memvid-core search/ask APIs and present results in JSON or human form.
//! - Keep user-facing errors friendly and deterministic (no panics on malformed flags).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use blake3::hash;
use clap::{ArgAction, Args, ValueEnum};
use colored::Colorize;
use colored_json::ToColoredJson;
#[cfg(feature = "temporal_track")]
use memvid_core::{
    types::SearchHitTemporal, TemporalContext, TemporalFilter, TemporalNormalizer,
    TemporalResolution, TemporalResolutionValue,
};
use memvid_core::{
    types::{
        AdaptiveConfig, AskContextFragment, AskContextFragmentKind, CutoffStrategy,
        SearchHitMetadata,
    },
    AskMode, AskRequest, AskResponse, AskRetriever, FrameId, Memvid, MemvidError, SearchEngineKind,
    SearchHit, SearchRequest, SearchResponse, TimelineEntry, TimelineQueryBuilder, VecEmbedder,
};
#[cfg(feature = "temporal_track")]
use serde::Serialize;
use serde_json::json;
#[cfg(feature = "temporal_track")]
use time::format_description::well_known::Rfc3339;
use time::{Date, PrimitiveDateTime, Time};
#[cfg(feature = "temporal_track")]
use time::{Duration as TimeDuration, Month, OffsetDateTime, UtcOffset};
use tracing::{info, warn};

#[cfg(feature = "local-embeddings")]
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

use memvid_ask_model::{
    run_model_inference, ModelContextFragment, ModelContextFragmentKind, ModelInference,
};

// frame_to_json and print_frame_summary available from commands but not used in this module
use crate::config::{
    load_embedding_runtime, load_embedding_runtime_for_mv2, resolve_llm_context_budget_override,
    try_load_embedding_runtime, try_load_embedding_runtime_for_mv2, CliConfig,
    EmbeddingModelChoice, EmbeddingRuntime,
};
use crate::utils::{
    autodetect_memory_file, format_timestamp, looks_like_memory, open_read_only_mem,
    parse_date_boundary, parse_vector, read_embedding,
};

const OUTPUT_CONTEXT_MAX_LEN: usize = 4_000;
#[cfg(feature = "temporal_track")]
const DEFAULT_TEMPORAL_TZ: &str = "America/Chicago";

fn vec_dimension_mismatch_help(expected: u32, actual: usize) -> String {
    let mut message = format!("Vector dimension mismatch (expected {expected}, got {actual}).");
    message.push_str("\n\nThis usually means the memory was indexed with a different embedding model than the query embedding.");
    if let Some(model) = EmbeddingModelChoice::from_dimension(expected) {
        message.push_str(&format!(
            "\n\nSuggested fix: re-run with `-m {}` (alias: `--embedding-model/--model {}`)",
            model.name(),
            model.name()
        ));
        if model.is_openai() {
            message.push_str(" (and set `OPENAI_API_KEY`).");
        } else {
            message.push('.');
        }
        message.push_str(&format!(
            "\nFor `ask`/`find` only: you can also use `--query-embedding-model {}`.",
            model.name()
        ));
        message.push_str(&format!(
            "\nIf you provided a raw vector (`vec-search --vector/--embedding`), it must have exactly {expected} floats."
        ));
        message.push_str("\nOr use `--mode lex` to disable semantic search.");
    }
    message
}

/// Arguments for the `timeline` subcommand
#[derive(Args)]
pub struct TimelineArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub reverse: bool,
    #[arg(long, value_name = "LIMIT")]
    pub limit: Option<NonZeroU64>,
    #[arg(long, value_name = "TIMESTAMP")]
    pub since: Option<i64>,
    #[arg(long, value_name = "TIMESTAMP")]
    pub until: Option<i64>,
    #[cfg(feature = "temporal_track")]
    #[arg(long = "on", value_name = "PHRASE")]
    pub phrase: Option<String>,
    #[cfg(feature = "temporal_track")]
    #[arg(long = "tz", value_name = "IANA_ZONE")]
    pub tz: Option<String>,
    #[cfg(feature = "temporal_track")]
    #[arg(long = "anchor", value_name = "RFC3339")]
    pub anchor: Option<String>,
    #[cfg(feature = "temporal_track")]
    #[arg(long = "window", value_name = "MINUTES")]
    pub window: Option<u64>,
    /// Replay: Show timeline for frames with ID <= AS_OF_FRAME (time-travel view)
    #[arg(long = "as-of-frame", value_name = "FRAME_ID")]
    pub as_of_frame: Option<u64>,
    /// Replay: Show timeline for frames with timestamp <= AS_OF_TS (time-travel view)
    #[arg(long = "as-of-ts", value_name = "UNIX_TIMESTAMP")]
    pub as_of_ts: Option<i64>,
}

/// Arguments for the `when` subcommand
#[cfg(feature = "temporal_track")]
#[derive(Args)]
pub struct WhenArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long = "on", value_name = "PHRASE")]
    pub phrase: String,
    #[arg(long = "tz", value_name = "IANA_ZONE")]
    pub tz: Option<String>,
    #[arg(long = "anchor", value_name = "RFC3339")]
    pub anchor: Option<String>,
    #[arg(long = "window", value_name = "MINUTES")]
    pub window: Option<u64>,
    #[arg(long, value_name = "LIMIT")]
    pub limit: Option<NonZeroU64>,
    #[arg(long, value_name = "TIMESTAMP")]
    pub since: Option<i64>,
    #[arg(long, value_name = "TIMESTAMP")]
    pub until: Option<i64>,
    #[arg(long)]
    pub reverse: bool,
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the `ask` subcommand
#[derive(Args)]
pub struct AskArgs {
    #[arg(value_name = "TARGET", num_args = 0..)]
    pub targets: Vec<String>,
    #[arg(long = "question", value_name = "TEXT")]
    pub question: Option<String>,
    #[arg(long = "uri", value_name = "URI")]
    pub uri: Option<String>,
    #[arg(long = "scope", value_name = "URI_PREFIX")]
    pub scope: Option<String>,
    #[arg(long = "top-k", value_name = "K", default_value = "8", alias = "limit")]
    pub top_k: usize,
    #[arg(long = "snippet-chars", value_name = "N", default_value = "480")]
    pub snippet_chars: usize,
    #[arg(long = "cursor", value_name = "TOKEN")]
    pub cursor: Option<String>,
    #[arg(long = "mode", value_enum, default_value = "hybrid")]
    pub mode: AskModeArg,
    #[arg(long)]
    pub json: bool,
    #[arg(long = "context-only", action = ArgAction::SetTrue)]
    pub context_only: bool,
    /// Show detailed source information for each citation
    #[arg(long = "sources", action = ArgAction::SetTrue)]
    pub sources: bool,
    /// Mask PII (emails, SSNs, phone numbers, etc.) in context before sending to LLM
    #[arg(long = "mask-pii", action = ArgAction::SetTrue)]
    pub mask_pii: bool,
    /// Include structured memory cards in the context (facts, preferences, etc.)
    #[arg(long = "memories", action = ArgAction::SetTrue)]
    pub memories: bool,
    /// Maximum characters of retrieval context to send to remote LLMs (overrides MEMVID_LLM_CONTEXT_BUDGET)
    #[arg(long = "llm-context-depth", value_name = "CHARS")]
    pub llm_context_depth: Option<usize>,
    #[arg(long = "start", value_name = "DATE")]
    pub start: Option<String>,
    #[arg(long = "end", value_name = "DATE")]
    pub end: Option<String>,
    /// Synthesize an answer with an LLM (defaults to tinyllama when provided without a value).
    ///
    /// Examples:
    /// - `--use-model` (local TinyLlama)
    /// - `--use-model openai` (defaults to gpt-4o-mini; requires OPENAI_API_KEY)
    /// - `--use-model nvidia` (defaults to meta/llama3-8b-instruct; requires NVIDIA_API_KEY)
    /// - `--use-model nvidia:meta/llama3-70b-instruct`
    #[arg(
        long = "use-model",
        value_name = "MODEL",
        num_args = 0..=1,
        default_missing_value = "tinyllama"
    )]
    pub use_model: Option<String>,
    /// Embedding model to use for query (must match the model used during ingestion)
    /// Options: bge-small, bge-base, nomic, gte-large, openai, openai-small, openai-ada
    #[arg(long = "query-embedding-model", value_name = "EMB_MODEL")]
    pub query_embedding_model: Option<String>,
    /// Replay: Filter to frames with ID <= AS_OF_FRAME (time-travel view)
    #[arg(long = "as-of-frame", value_name = "FRAME_ID")]
    pub as_of_frame: Option<u64>,
    /// Replay: Filter to frames with timestamp <= AS_OF_TS (time-travel view)
    #[arg(long = "as-of-ts", value_name = "UNIX_TIMESTAMP")]
    pub as_of_ts: Option<i64>,
    /// Override the default system prompt (useful for providing date context like "Today is March 27, 2023")
    #[arg(long = "system-prompt", value_name = "TEXT")]
    pub system_prompt: Option<String>,
    /// Skip cross-encoder reranking (useful in gated environments where model downloads are blocked)
    #[arg(long = "no-rerank", action = ArgAction::SetTrue)]
    pub no_rerank: bool,

    /// Return verbatim evidence without LLM synthesis.
    /// Shows the most relevant passages with citations, no paraphrasing or summarization.
    #[arg(long = "no-llm", action = ArgAction::SetTrue)]
    pub no_llm: bool,

    // Adaptive retrieval options (enabled by default for best results)
    /// Disable adaptive retrieval and use fixed top-k instead.
    /// By default, adaptive retrieval is enabled with the 'combined' strategy.
    #[arg(long = "no-adaptive", action = ArgAction::SetTrue)]
    pub no_adaptive: bool,
    /// Minimum relevancy ratio vs top score (0.0-1.0). Results below this threshold are excluded.
    /// Example: 0.5 means only include results with score >= 50% of the top result's score.
    #[arg(long = "min-relevancy", value_name = "RATIO", default_value = "0.5")]
    pub min_relevancy: f32,
    /// Maximum results to consider for adaptive retrieval (over-retrieval limit).
    /// Set high enough to capture all potentially relevant results.
    #[arg(long = "max-k", value_name = "K", default_value = "100")]
    pub max_k: usize,
    /// Adaptive cutoff strategy: combined (default), relative, absolute, cliff, or elbow
    #[arg(long = "adaptive-strategy", value_enum, default_value = "combined")]
    pub adaptive_strategy: AdaptiveStrategyArg,
}

/// Ask mode argument
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum AskModeArg {
    Lex,
    Sem,
    Hybrid,
}

impl From<AskModeArg> for AskMode {
    fn from(value: AskModeArg) -> Self {
        match value {
            AskModeArg::Lex => AskMode::Lex,
            AskModeArg::Sem => AskMode::Sem,
            AskModeArg::Hybrid => AskMode::Hybrid,
        }
    }
}

/// Arguments for the `find` subcommand
#[derive(Args)]
pub struct FindArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long = "query", value_name = "TEXT")]
    pub query: String,
    #[arg(long = "uri", value_name = "URI")]
    pub uri: Option<String>,
    #[arg(long = "scope", value_name = "URI_PREFIX")]
    pub scope: Option<String>,
    #[arg(long = "top-k", value_name = "K", default_value = "8", alias = "limit")]
    pub top_k: usize,
    #[arg(long = "snippet-chars", value_name = "N", default_value = "480")]
    pub snippet_chars: usize,
    #[arg(long = "cursor", value_name = "TOKEN")]
    pub cursor: Option<String>,
    #[arg(long)]
    pub json: bool,
    #[arg(long = "json-legacy", conflicts_with = "json")]
    pub json_legacy: bool,
    #[arg(long = "mode", value_enum, default_value = "auto")]
    pub mode: SearchMode,
    /// Replay: Filter to frames with ID <= AS_OF_FRAME (time-travel view)
    #[arg(long = "as-of-frame", value_name = "FRAME_ID")]
    pub as_of_frame: Option<u64>,
    /// Replay: Filter to frames with timestamp <= AS_OF_TS (time-travel view)
    #[arg(long = "as-of-ts", value_name = "UNIX_TIMESTAMP")]
    pub as_of_ts: Option<i64>,
    /// Embedding model to use for query (must match the model used during ingestion)
    /// Options: bge-small, bge-base, nomic, gte-large, openai, openai-small, openai-ada
    #[arg(long = "query-embedding-model", value_name = "EMB_MODEL")]
    pub query_embedding_model: Option<String>,

    // Adaptive retrieval options (enabled by default for best results)
    /// Disable adaptive retrieval and use fixed top-k instead.
    /// By default, adaptive retrieval is enabled with the 'combined' strategy.
    #[arg(long = "no-adaptive", action = ArgAction::SetTrue)]
    pub no_adaptive: bool,
    /// Minimum relevancy ratio vs top score (0.0-1.0). Results below this threshold are excluded.
    /// Example: 0.5 means only include results with score >= 50% of the top result's score.
    #[arg(long = "min-relevancy", value_name = "RATIO", default_value = "0.5")]
    pub min_relevancy: f32,
    /// Maximum results to consider for adaptive retrieval (over-retrieval limit).
    /// Set high enough to capture all potentially relevant results.
    #[arg(long = "max-k", value_name = "K", default_value = "100")]
    pub max_k: usize,
    /// Adaptive cutoff strategy: combined (default), relative, absolute, cliff, or elbow
    #[arg(long = "adaptive-strategy", value_enum, default_value = "combined")]
    pub adaptive_strategy: AdaptiveStrategyArg,

    /// Enable graph-aware search: filter by entity relationships before ranking.
    /// Uses MemoryCards to find entities matching patterns like "who lives in X".
    #[arg(long = "graph", action = ArgAction::SetTrue)]
    pub graph: bool,

    /// Enable hybrid search: combine graph filtering with text search.
    /// Automatically detects relational patterns in the query.
    #[arg(long = "hybrid", action = ArgAction::SetTrue)]
    pub hybrid: bool,

    /// Disable sketch pre-filtering (for benchmarking/debugging).
    /// By default, sketches are used for fast candidate generation if available.
    #[arg(long = "no-sketch", action = ArgAction::SetTrue)]
    pub no_sketch: bool,
}

/// Search mode argument
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum SearchMode {
    Auto,
    Lex,
    Sem,
    /// CLIP visual search using text-to-image embeddings
    #[cfg(feature = "clip")]
    Clip,
}

/// Adaptive retrieval strategy
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum AdaptiveStrategyArg {
    /// Stop when score drops below X% of top score (default)
    Relative,
    /// Stop when score drops below fixed threshold
    Absolute,
    /// Stop when score drops sharply from previous result
    Cliff,
    /// Automatically detect "elbow" in score curve
    Elbow,
    /// Combine relative + cliff + absolute (recommended)
    Combined,
}

/// Arguments for the `vec-search` subcommand
#[derive(Args)]
pub struct VecSearchArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long, conflicts_with = "embedding", value_name = "CSV")]
    pub vector: Option<String>,
    #[arg(long, conflicts_with = "vector", value_name = "PATH", value_parser = clap::value_parser!(PathBuf))]
    pub embedding: Option<PathBuf>,
    #[arg(long, value_name = "K", default_value = "10")]
    pub limit: usize,
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the `audit` subcommand
#[derive(Args)]
pub struct AuditArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    /// The question or topic to audit
    #[arg(value_name = "QUESTION")]
    pub question: String,
    /// Output file path (stdout if not provided)
    #[arg(long = "out", short = 'o', value_name = "PATH", value_parser = clap::value_parser!(PathBuf))]
    pub out: Option<PathBuf>,
    /// Output format
    #[arg(long = "format", value_enum, default_value = "text")]
    pub format: AuditFormat,
    /// Number of sources to retrieve
    #[arg(long = "top-k", value_name = "K", default_value = "10")]
    pub top_k: usize,
    /// Maximum characters per snippet
    #[arg(long = "snippet-chars", value_name = "N", default_value = "500")]
    pub snippet_chars: usize,
    /// Retrieval mode
    #[arg(long = "mode", value_enum, default_value = "hybrid")]
    pub mode: AskModeArg,
    /// Optional scope filter (URI prefix)
    #[arg(long = "scope", value_name = "URI_PREFIX")]
    pub scope: Option<String>,
    /// Start date filter
    #[arg(long = "start", value_name = "DATE")]
    pub start: Option<String>,
    /// End date filter
    #[arg(long = "end", value_name = "DATE")]
    pub end: Option<String>,
    /// Use a model to synthesize the answer (e.g., "ollama:qwen2.5:1.5b")
    #[arg(long = "use-model", value_name = "MODEL")]
    pub use_model: Option<String>,
}

/// Audit output format
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum AuditFormat {
    /// Plain text report
    Text,
    /// Markdown report
    Markdown,
    /// JSON report
    Json,
}

// ============================================================================
// Search & Retrieval command handlers
// ============================================================================

pub fn handle_timeline(_config: &CliConfig, args: TimelineArgs) -> Result<()> {
    let mut mem = open_read_only_mem(&args.file)?;
    let mut builder = TimelineQueryBuilder::default();
    #[cfg(feature = "temporal_track")]
    if args.phrase.is_none()
        && (args.tz.is_some() || args.anchor.is_some() || args.window.is_some())
    {
        bail!("E-TEMP-005 use --on when supplying --tz/--anchor/--window");
    }
    if let Some(limit) = args.limit {
        builder = builder.limit(limit);
    }
    if let Some(since) = args.since {
        builder = builder.since(since);
    }
    if let Some(until) = args.until {
        builder = builder.until(until);
    }
    builder = builder.reverse(args.reverse);
    #[cfg(feature = "temporal_track")]
    let temporal_summary = if let Some(ref phrase) = args.phrase {
        let (filter, summary) = build_temporal_filter(
            phrase,
            args.tz.as_deref(),
            args.anchor.as_deref(),
            args.window,
        )?;
        builder = builder.temporal(filter);
        Some(summary)
    } else {
        None
    };
    let query = builder.build();
    let mut entries = mem.timeline(query)?;

    // Apply Replay filtering if requested
    if args.as_of_frame.is_some() || args.as_of_ts.is_some() {
        entries.retain(|entry| {
            // Check as_of_frame filter
            if let Some(cutoff_frame) = args.as_of_frame {
                if entry.frame_id > cutoff_frame {
                    return false;
                }
            }

            // Check as_of_ts filter
            if let Some(cutoff_ts) = args.as_of_ts {
                if entry.timestamp > cutoff_ts {
                    return false;
                }
            }

            true
        });
    }

    if args.json {
        #[cfg(feature = "temporal_track")]
        if let Some(summary) = temporal_summary.as_ref() {
            println!(
                "{}",
                serde_json::to_string_pretty(&TimelineOutput {
                    temporal: Some(summary_to_output(summary)),
                    entries: &entries,
                })?
            );
        } else {
            println!("{}", serde_json::to_string_pretty(&entries)?);
        }
        #[cfg(not(feature = "temporal_track"))]
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else if entries.is_empty() {
        println!("Timeline is empty");
    } else {
        #[cfg(feature = "temporal_track")]
        if let Some(summary) = temporal_summary.as_ref() {
            print_temporal_summary(summary);
        }
        for entry in entries {
            println!(
                "#{} @ {} — {}",
                entry.frame_id,
                entry.timestamp,
                entry.preview.replace('\n', " ")
            );
            if let Some(uri) = entry.uri.as_deref() {
                println!("  URI: {uri}");
            }
            if !entry.child_frames.is_empty() {
                let child_list = entry
                    .child_frames
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("  Child frames: {child_list}");
            }
            #[cfg(feature = "temporal_track")]
            if let Some(temporal) = entry.temporal.as_ref() {
                print_entry_temporal_details(temporal);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "temporal_track")]
pub fn handle_when(_config: &CliConfig, args: WhenArgs) -> Result<()> {
    let mut mem = open_read_only_mem(&args.file)?;

    let (filter, summary) = build_temporal_filter(
        &args.phrase,
        args.tz.as_deref(),
        args.anchor.as_deref(),
        args.window,
    )?;

    let mut builder = TimelineQueryBuilder::default();
    if let Some(limit) = args.limit {
        builder = builder.limit(limit);
    }
    if let Some(since) = args.since {
        builder = builder.since(since);
    }
    if let Some(until) = args.until {
        builder = builder.until(until);
    }
    builder = builder.reverse(args.reverse).temporal(filter.clone());
    let entries = mem.timeline(builder.build())?;

    if args.json {
        let entry_views: Vec<WhenEntry> = entries.iter().map(entry_to_when_entry).collect();
        let output = WhenOutput {
            summary: summary_to_output(&summary),
            entries: entry_views,
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    print_temporal_summary(&summary);
    if entries.is_empty() {
        println!("No frames matched the resolved window");
        return Ok(());
    }

    for entry in &entries {
        let iso = format_timestamp(entry.timestamp).unwrap_or_default();
        println!(
            "#{} @ {} ({iso}) — {}",
            entry.frame_id,
            entry.timestamp,
            entry.preview.replace('\n', " ")
        );
        if let Some(uri) = entry.uri.as_deref() {
            println!("  URI: {uri}");
        }
        if !entry.child_frames.is_empty() {
            let child_list = entry
                .child_frames
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            println!("  Child frames: {child_list}");
        }
        if let Some(temporal) = entry.temporal.as_ref() {
            print_entry_temporal_details(temporal);
        }
    }

    Ok(())
}

#[cfg(feature = "temporal_track")]
#[derive(Serialize)]
struct TimelineOutput<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    temporal: Option<TemporalSummaryOutput>,
    entries: &'a [TimelineEntry],
}

#[cfg(feature = "temporal_track")]
#[derive(Serialize)]
struct WhenOutput {
    summary: TemporalSummaryOutput,
    entries: Vec<WhenEntry>,
}

#[cfg(feature = "temporal_track")]
#[derive(Serialize)]
struct WhenEntry {
    frame_id: FrameId,
    timestamp: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamp_iso: Option<String>,
    preview: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    uri: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    child_frames: Vec<FrameId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temporal: Option<SearchHitTemporal>,
}

#[cfg(feature = "temporal_track")]
#[derive(Serialize)]
struct TemporalSummaryOutput {
    phrase: String,
    timezone: String,
    anchor_utc: i64,
    anchor_iso: String,
    confidence: u16,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    flags: Vec<&'static str>,
    resolution_kind: &'static str,
    window_start_utc: Option<i64>,
    window_start_iso: Option<String>,
    window_end_utc: Option<i64>,
    window_end_iso: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    window_minutes: Option<u64>,
}

#[cfg(feature = "temporal_track")]
struct TemporalSummary {
    phrase: String,
    tz: String,
    anchor: OffsetDateTime,
    start_utc: Option<i64>,
    end_utc: Option<i64>,
    resolution: TemporalResolution,
    window_minutes: Option<u64>,
}

#[cfg(feature = "temporal_track")]
fn build_temporal_filter(
    phrase: &str,
    tz_override: Option<&str>,
    anchor_override: Option<&str>,
    window_minutes: Option<u64>,
) -> Result<(TemporalFilter, TemporalSummary)> {
    let tz = tz_override
        .unwrap_or(DEFAULT_TEMPORAL_TZ)
        .trim()
        .to_string();
    if tz.is_empty() {
        bail!("E-TEMP-003 timezone must not be empty");
    }

    let anchor = if let Some(raw) = anchor_override {
        OffsetDateTime::parse(raw, &Rfc3339)
            .map_err(|_| anyhow!("E-TEMP-002 anchor must be RFC3339: {raw}"))?
    } else {
        OffsetDateTime::now_utc()
    };

    let context = TemporalContext::new(anchor, tz.clone());
    let normalizer = TemporalNormalizer::new(context);
    let resolution = normalizer
        .resolve(phrase)
        .map_err(|err| anyhow!("E-TEMP-001 {err}"))?;

    let (mut start, mut end) = resolution_bounds(&resolution)?;
    if let Some(minutes) = window_minutes {
        if minutes > 0 {
            let delta = TimeDuration::minutes(minutes as i64);
            if let (Some(s), Some(e)) = (start, end) {
                if s == e {
                    start = Some(s.saturating_sub(delta.whole_seconds()));
                    end = Some(e.saturating_add(delta.whole_seconds()));
                } else {
                    start = Some(s.saturating_sub(delta.whole_seconds()));
                    end = Some(e.saturating_add(delta.whole_seconds()));
                }
            }
        }
    }

    let filter = TemporalFilter {
        start_utc: start,
        end_utc: end,
        phrase: None,
        tz: None,
    };

    let summary = TemporalSummary {
        phrase: phrase.to_owned(),
        tz,
        anchor,
        start_utc: start,
        end_utc: end,
        resolution,
        window_minutes,
    };

    Ok((filter, summary))
}

#[cfg(feature = "temporal_track")]
fn summary_to_output(summary: &TemporalSummary) -> TemporalSummaryOutput {
    TemporalSummaryOutput {
        phrase: summary.phrase.clone(),
        timezone: summary.tz.clone(),
        anchor_utc: summary.anchor.unix_timestamp(),
        anchor_iso: summary
            .anchor
            .format(&Rfc3339)
            .unwrap_or_else(|_| summary.anchor.unix_timestamp().to_string()),
        confidence: summary.resolution.confidence,
        flags: summary
            .resolution
            .flags
            .iter()
            .map(|flag| flag.as_str())
            .collect(),
        resolution_kind: resolution_kind(&summary.resolution),
        window_start_utc: summary.start_utc,
        window_start_iso: summary.start_utc.and_then(format_timestamp),
        window_end_utc: summary.end_utc,
        window_end_iso: summary.end_utc.and_then(format_timestamp),
        window_minutes: summary.window_minutes,
    }
}

#[cfg(feature = "temporal_track")]
fn entry_to_when_entry(entry: &TimelineEntry) -> WhenEntry {
    WhenEntry {
        frame_id: entry.frame_id,
        timestamp: entry.timestamp,
        timestamp_iso: format_timestamp(entry.timestamp),
        preview: entry.preview.clone(),
        uri: entry.uri.clone(),
        child_frames: entry.child_frames.clone(),
        temporal: entry.temporal.clone(),
    }
}

#[cfg(feature = "temporal_track")]
fn print_temporal_summary(summary: &TemporalSummary) {
    println!("Phrase: \"{}\"", summary.phrase);
    println!("Timezone: {}", summary.tz);
    println!(
        "Anchor: {}",
        summary
            .anchor
            .format(&Rfc3339)
            .unwrap_or_else(|_| summary.anchor.unix_timestamp().to_string())
    );
    let start_iso = summary.start_utc.and_then(format_timestamp);
    let end_iso = summary.end_utc.and_then(format_timestamp);
    match (start_iso, end_iso) {
        (Some(start), Some(end)) if start == end => println!("Resolved to: {start}"),
        (Some(start), Some(end)) => println!("Window: {start} → {end}"),
        (Some(start), None) => println!("Window start: {start}"),
        (None, Some(end)) => println!("Window end: {end}"),
        _ => println!("Window: (not resolved)"),
    }
    println!("Confidence: {}", summary.resolution.confidence);
    let flags: Vec<&'static str> = summary
        .resolution
        .flags
        .iter()
        .map(|flag| flag.as_str())
        .collect();
    if !flags.is_empty() {
        println!("Flags: {}", flags.join(", "));
    }
    if let Some(window) = summary.window_minutes {
        if window > 0 {
            println!("Window padding: {window} minute(s)");
        }
    }
    println!();
}

#[cfg(feature = "temporal_track")]
fn print_entry_temporal_details(temporal: &SearchHitTemporal) {
    if let Some(anchor) = temporal.anchor.as_ref() {
        let iso = anchor
            .iso_8601
            .clone()
            .or_else(|| format_timestamp(anchor.ts_utc));
        println!(
            "  Anchor: {} (source: {:?})",
            iso.unwrap_or_else(|| anchor.ts_utc.to_string()),
            anchor.source
        );
    }
    if !temporal.mentions.is_empty() {
        println!("  Mentions:");
        for mention in &temporal.mentions {
            let iso = mention
                .iso_8601
                .clone()
                .or_else(|| format_timestamp(mention.ts_utc))
                .unwrap_or_else(|| mention.ts_utc.to_string());
            let mut details = format!(
                "    - {} ({:?}, confidence {})",
                iso, mention.kind, mention.confidence
            );
            if let Some(text) = mention.text.as_deref() {
                details.push_str(&format!(" — \"{}\"", text));
            }
            println!("{details}");
        }
    }
}

#[cfg(feature = "temporal_track")]
fn resolution_bounds(resolution: &TemporalResolution) -> Result<(Option<i64>, Option<i64>)> {
    match &resolution.value {
        TemporalResolutionValue::Date(date) => {
            let ts = date_to_timestamp(*date);
            Ok((Some(ts), Some(ts)))
        }
        TemporalResolutionValue::DateTime(dt) => {
            let ts = dt.unix_timestamp();
            Ok((Some(ts), Some(ts)))
        }
        TemporalResolutionValue::DateRange { start, end } => Ok((
            Some(date_to_timestamp(*start)),
            Some(date_to_timestamp(*end)),
        )),
        TemporalResolutionValue::DateTimeRange { start, end } => {
            Ok((Some(start.unix_timestamp()), Some(end.unix_timestamp())))
        }
        TemporalResolutionValue::Month { year, month } => {
            let start_date = Date::from_calendar_date(*year, *month, 1)
                .map_err(|_| anyhow!("invalid month resolution"))?;
            let end_date = last_day_in_month(*year, *month)
                .map_err(|_| anyhow!("invalid month resolution"))?;
            Ok((
                Some(date_to_timestamp(start_date)),
                Some(date_to_timestamp(end_date)),
            ))
        }
    }
}

#[cfg(feature = "temporal_track")]
fn resolution_kind(resolution: &TemporalResolution) -> &'static str {
    match resolution.value {
        TemporalResolutionValue::Date(_) => "date",
        TemporalResolutionValue::DateTime(_) => "datetime",
        TemporalResolutionValue::DateRange { .. } => "date_range",
        TemporalResolutionValue::DateTimeRange { .. } => "datetime_range",
        TemporalResolutionValue::Month { .. } => "month",
    }
}

#[cfg(feature = "temporal_track")]
fn date_to_timestamp(date: Date) -> i64 {
    PrimitiveDateTime::new(date, Time::MIDNIGHT)
        .assume_offset(UtcOffset::UTC)
        .unix_timestamp()
}

#[cfg(feature = "temporal_track")]
fn last_day_in_month(year: i32, month: Month) -> Result<Date> {
    let mut date = Date::from_calendar_date(year, month, 1)
        .map_err(|_| anyhow!("invalid month resolution"))?;
    while let Some(next) = date.next_day() {
        if next.month() == month {
            date = next;
        } else {
            break;
        }
    }
    Ok(date)
}

#[cfg(feature = "temporal_track")]

fn apply_model_context_fragments(response: &mut AskResponse, fragments: Vec<ModelContextFragment>) {
    if fragments.is_empty() {
        return;
    }

    response.context_fragments = fragments
        .into_iter()
        .map(|fragment| AskContextFragment {
            rank: fragment.rank,
            frame_id: fragment.frame_id,
            uri: fragment.uri,
            title: fragment.title,
            score: fragment.score,
            matches: fragment.matches,
            range: Some(fragment.range),
            chunk_range: fragment.chunk_range,
            text: fragment.text,
            kind: Some(match fragment.kind {
                ModelContextFragmentKind::Full => AskContextFragmentKind::Full,
                ModelContextFragmentKind::Summary => AskContextFragmentKind::Summary,
            }),
            #[cfg(feature = "temporal_track")]
            temporal: None,
        })
        .collect();
}

pub fn handle_ask(config: &CliConfig, args: AskArgs) -> Result<()> {
    // Check if plan allows query operations (blocks expired subscriptions)
    crate::utils::require_active_plan(config, "ask")?;

    // Track query usage against plan quota
    crate::api::track_query_usage(config, 1)?;

    if args.uri.is_some() && args.scope.is_some() {
        warn!("--scope ignored because --uri is provided");
    }

    let mut question_tokens = Vec::new();
    let mut file_path: Option<PathBuf> = None;
    for token in &args.targets {
        if file_path.is_none() && looks_like_memory(token) {
            file_path = Some(PathBuf::from(token));
        } else {
            question_tokens.push(token.clone());
        }
    }

    let positional_question = if question_tokens.is_empty() {
        None
    } else {
        Some(question_tokens.join(" "))
    };

    let question = args
        .question
        .or(positional_question)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let question = question
        .ok_or_else(|| anyhow!("provide a question via positional arguments or --question"))?;

    // Expand query for better retrieval using LLM (expands abbreviations, adds synonyms)
    // This happens when --use-model is set or we have an API key
    let (original_question, search_query) = {
        // For query expansion, we use the fastest available model
        // Priority: OpenAI > Groq > Anthropic > XAI > Mistral
        let (model_for_expansion, api_key_for_expansion): (Option<&str>, Option<String>) =
            if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                // OpenAI available - use gpt-4o-mini (fastest, cheapest)
                (Some("gpt-4o-mini"), Some(key))
            } else if let Ok(key) = std::env::var("GROQ_API_KEY") {
                // Groq available - use llama-3.1-8b-instant (very fast)
                (Some("llama-3.1-8b-instant"), Some(key))
            } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
                // Anthropic available - use haiku
                (Some("claude-haiku-4-5"), Some(key))
            } else if let Ok(key) = std::env::var("XAI_API_KEY") {
                // XAI available - use grok-4-fast
                (Some("grok-4-fast"), Some(key))
            } else if let Ok(key) = std::env::var("MISTRAL_API_KEY") {
                // Mistral available - use mistral-small
                (Some("mistral-small-latest"), Some(key))
            } else {
                // No fast model available for expansion
                (None, None)
            };

        // DISABLED: Query expansion for ask command
        // The ask command has sophisticated retrieval with fallbacks, aggregation detection,
        // temporal boosting, and diverse retrieval strategies. Query expansion often strips
        // out important semantic context (temporal markers, aggregation signals, analytical
        // keywords) that these strategies depend on. The original question is preserved
        // to ensure all downstream detection and ranking works correctly.
        //
        // Query expansion may be appropriate for simple keyword searches, but for complex
        // natural language questions it causes more problems than it solves.
        let _ = (model_for_expansion, api_key_for_expansion); // suppress unused warnings
        (question.clone(), question.clone())
    };

    let memory_path = match file_path {
        Some(path) => path,
        None => autodetect_memory_file()?,
    };

    let start = parse_date_boundary(args.start.as_ref(), false)?;
    let end = parse_date_boundary(args.end.as_ref(), true)?;
    if let (Some(start_ts), Some(end_ts)) = (start, end) {
        if end_ts < start_ts {
            anyhow::bail!("--end must not be earlier than --start");
        }
    }

    // Open MV2 file first to get vector dimension for auto-detection
    let mut mem = Memvid::open(&memory_path)?;

    // Load active replay session if one exists
    #[cfg(feature = "replay")]
    let _ = mem.load_active_session();

    // Get the vector dimension from the MV2 file for auto-detection
    let mv2_dimension = mem.effective_vec_index_dimension()?;

    // Check if memory has any vectors - if not, force lexical mode
    let stats = mem.stats()?;
    let has_vectors = stats.vector_count > 0;
    let effective_mode = if !has_vectors
        && matches!(args.mode, AskModeArg::Sem | AskModeArg::Hybrid)
    {
        tracing::info!("Memory has no embeddings (vector_count=0); falling back to lexical mode");
        AskModeArg::Lex
    } else {
        args.mode.clone()
    };

    let ask_mode: AskMode = effective_mode.clone().into();
    let inferred_model_override = match effective_mode {
        AskModeArg::Lex => None,
        AskModeArg::Sem | AskModeArg::Hybrid => match mem.embedding_identity_summary(10_000) {
            memvid_core::EmbeddingIdentitySummary::Single(identity) => {
                identity.model.map(String::from)
            }
            memvid_core::EmbeddingIdentitySummary::Mixed(identities) => {
                let models: Vec<_> = identities
                    .iter()
                    .filter_map(|entry| entry.identity.model.as_deref())
                    .collect();
                anyhow::bail!(
                    "memory contains mixed embedding models; semantic queries are unsafe.\n\n\
                    Detected models: {:?}\n\n\
                    Suggested fix: split into separate memories per embedding model.",
                    models
                );
            }
            memvid_core::EmbeddingIdentitySummary::Unknown => None,
        },
    };
    let emb_model_override = args
        .query_embedding_model
        .as_deref()
        .or(inferred_model_override.as_deref());
    let runtime = match effective_mode {
        AskModeArg::Lex => None,
        AskModeArg::Sem => Some(load_embedding_runtime_for_mv2(
            config,
            emb_model_override,
            mv2_dimension,
        )?),
        AskModeArg::Hybrid => {
            // For hybrid, use auto-detection from MV2 dimension
            try_load_embedding_runtime_for_mv2(config, emb_model_override, mv2_dimension).or_else(
                || {
                    // Force a load; if it fails we error below.
                    load_embedding_runtime_for_mv2(config, emb_model_override, mv2_dimension)
                        .ok()
                        .map(|rt| {
                            tracing::debug!("hybrid ask: loaded embedding runtime after fallback");
                            rt
                        })
                },
            )
        }
    };
    if runtime.is_none() && matches!(effective_mode, AskModeArg::Sem | AskModeArg::Hybrid) {
        anyhow::bail!(
            "semantic embeddings unavailable; install/cached model required for {:?} mode",
            effective_mode
        );
    }

    let embedder = runtime.as_ref().map(|inner| inner as &dyn VecEmbedder);

    // Build adaptive config (enabled by default, use --no-adaptive to disable)
    let adaptive = if !args.no_adaptive {
        Some(AdaptiveConfig {
            enabled: true,
            max_results: args.max_k,
            min_results: 1,
            normalize_scores: true,
            strategy: match args.adaptive_strategy {
                AdaptiveStrategyArg::Relative => CutoffStrategy::RelativeThreshold {
                    min_ratio: args.min_relevancy,
                },
                AdaptiveStrategyArg::Absolute => CutoffStrategy::AbsoluteThreshold {
                    min_score: args.min_relevancy,
                },
                AdaptiveStrategyArg::Cliff => CutoffStrategy::ScoreCliff {
                    max_drop_ratio: 0.3,
                },
                AdaptiveStrategyArg::Elbow => CutoffStrategy::Elbow { sensitivity: 1.0 },
                AdaptiveStrategyArg::Combined => CutoffStrategy::Combined {
                    relative_threshold: args.min_relevancy,
                    max_drop_ratio: 0.3,
                    absolute_min: 0.3,
                },
            },
        })
    } else {
        None
    };

    let request = AskRequest {
        question: search_query, // Use expanded query for retrieval
        top_k: args.top_k,
        snippet_chars: args.snippet_chars,
        uri: args.uri.clone(),
        scope: args.scope.clone(),
        cursor: args.cursor.clone(),
        start,
        end,
        #[cfg(feature = "temporal_track")]
        temporal: None,
        context_only: args.context_only,
        mode: ask_mode,
        as_of_frame: args.as_of_frame,
        as_of_ts: args.as_of_ts,
        adaptive,
        acl_context: None,
        acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
    };
    let mut response = mem.ask(request, embedder).map_err(|err| match err {
        MemvidError::VecDimensionMismatch { expected, actual } => {
            anyhow!(vec_dimension_mismatch_help(expected, actual))
        }
        other => anyhow!(other),
    })?;

    // Restore original question for display and LLM synthesis
    // (search_query was used for retrieval but original_question is shown to user)
    response.question = original_question;

    // Apply cross-encoder reranking for better precision on preference/personalization queries
    // This is especially important for questions like "What should I..." where semantic
    // similarity doesn't capture personal relevance well.
    // Skip if --no-rerank is set (useful in gated environments where model downloads are blocked)
    // Skip for temporal/recency queries - cross-encoder doesn't understand temporal context
    // and would override the recency boost from lexical search
    let is_temporal_query = {
        let q_lower = response.question.to_lowercase();
        q_lower.contains("current")
            || q_lower.contains("latest")
            || q_lower.contains("recent")
            || q_lower.contains("now")
            || q_lower.contains("today")
            || q_lower.contains("updated")
            || q_lower.contains("new ")
            || q_lower.contains("newest")
    };
    if !args.no_rerank
        && !response.retrieval.hits.is_empty()
        && matches!(effective_mode, AskModeArg::Sem | AskModeArg::Hybrid)
        && !is_temporal_query
    {
        // Create a temporary SearchResponse for reranking
        let mut search_response = SearchResponse {
            query: response.question.clone(),
            hits: response.retrieval.hits.clone(),
            total_hits: response.retrieval.hits.len(),
            params: memvid_core::SearchParams {
                top_k: args.top_k,
                snippet_chars: args.snippet_chars,
                cursor: None,
            },
            elapsed_ms: 0,
            engine: memvid_core::SearchEngineKind::Hybrid,
            next_cursor: None,
            context: String::new(),
            stale_index_skips: 0,
        };

        if let Err(e) = apply_cross_encoder_rerank(&mut search_response) {
            warn!("Cross-encoder reranking failed: {e}");
        } else {
            // Update the response hits with reranked order
            response.retrieval.hits = search_response.hits;
            // Rebuild context from reranked hits
            response.retrieval.context = response
                .retrieval
                .hits
                .iter()
                .take(10) // Use top-10 for context
                .map(|hit| hit.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n---\n\n");
        }
    }

    // Inject memory cards into context if --memories flag is set
    if args.memories {
        let memory_context = build_memory_context(&mem);
        if !memory_context.is_empty() {
            // Prepend memory context to retrieval context
            response.retrieval.context = format!(
                "=== KNOWN FACTS ===\n{}\n\n=== RETRIEVED CONTEXT ===\n{}",
                memory_context, response.retrieval.context
            );
        }
    }

    // Inject entity context from Logic-Mesh if entities were found in search hits
    let entity_context = build_entity_context_from_hits(&response.retrieval.hits);
    if !entity_context.is_empty() {
        // Prepend entity context to retrieval context
        response.retrieval.context = format!(
            "=== ENTITIES MENTIONED ===\n{}\n\n{}",
            entity_context, response.retrieval.context
        );
    }

    // Apply PII masking if requested
    if args.mask_pii {
        use memvid_core::pii::mask_pii;

        // Mask the aggregated context
        response.retrieval.context = mask_pii(&response.retrieval.context);

        // Mask text in each hit
        for hit in &mut response.retrieval.hits {
            hit.text = mask_pii(&hit.text);
            if let Some(chunk_text) = &hit.chunk_text {
                hit.chunk_text = Some(mask_pii(chunk_text));
            }
        }
    }

    let llm_context_override = resolve_llm_context_budget_override(args.llm_context_depth)?;

    let mut model_result: Option<ModelInference> = None;
    if args.no_llm {
        // --no-llm: return verbatim evidence without LLM synthesis
        if args.use_model.is_some() {
            warn!("--use-model ignored because --no-llm disables LLM synthesis");
        }
        if args.json {
            emit_verbatim_evidence_json(&response, args.sources, &mut mem)?;
        } else {
            emit_verbatim_evidence_pretty(&response, args.sources, &mut mem);
        }

        // Save active replay session if one exists
        #[cfg(feature = "replay")]
        let _ = mem.save_active_session();

        return Ok(());
    } else if response.context_only {
        if args.use_model.is_some() {
            warn!("--use-model ignored because --context-only disables synthesis");
        }
    } else if let Some(model_name) = args.use_model.as_deref() {
        match run_model_inference(
            model_name,
            &response.question,
            &response.retrieval.context,
            response.retrieval.hits.as_slice(),
            llm_context_override,
            None,
            args.system_prompt.as_deref(),
        ) {
            Ok(inference) => {
                response.answer = Some(inference.answer.answer.clone());
                response.retrieval.context = inference.context_body.clone();
                apply_model_context_fragments(&mut response, inference.context_fragments.clone());
                model_result = Some(inference);
            }
            Err(err) => {
                warn!(
                    "model inference unavailable for '{}': {err}. Falling back to default summary.",
                    model_name
                );
            }
        }
    }

    // Record the ask action if a replay session is active
    #[cfg(feature = "replay")]
    if let Some(ref inference) = model_result {
        if let Some(model_name) = args.use_model.as_deref() {
            // Extract frame IDs from retrieval hits for replay audit
            let retrieved_frames: Vec<u64> = response
                .retrieval
                .hits
                .iter()
                .map(|hit| hit.frame_id)
                .collect();

            mem.record_ask_action(
                &response.question,
                model_name, // provider
                model_name, // model
                inference.answer.answer.as_bytes(),
                0, // duration_ms not tracked at this level
                retrieved_frames,
            );
        }
    }

    if args.json {
        if let Some(model_name) = args.use_model.as_deref() {
            emit_model_json(
                &response,
                model_name,
                model_result.as_ref(),
                args.sources,
                &mut mem,
            )?;
        } else {
            emit_ask_json(
                &response,
                effective_mode.clone(),
                model_result.as_ref(),
                args.sources,
                &mut mem,
            )?;
        }
    } else {
        emit_ask_pretty(
            &response,
            effective_mode.clone(),
            model_result.as_ref(),
            args.sources,
            &mut mem,
        );
    }

    // Save active replay session if one exists
    #[cfg(feature = "replay")]
    let _ = mem.save_active_session();

    Ok(())
}

/// Handle graph-aware find with --graph or --hybrid flags
fn handle_graph_find(mem: &mut Memvid, args: &FindArgs) -> Result<()> {
    use memvid_core::graph_search::{hybrid_search, QueryPlanner};
    use memvid_core::types::QueryPlan;

    let planner = QueryPlanner::new();

    // Create query plan based on mode
    let plan = if args.graph {
        // Pure graph mode - let planner detect patterns
        let plan = planner.plan(&args.query, args.top_k);
        // If it's a hybrid plan from auto-detection, convert to graph-only
        match plan {
            QueryPlan::Hybrid { graph_filter, .. } if !graph_filter.is_empty() => {
                QueryPlan::graph_only(graph_filter, args.top_k)
            }
            _ => plan,
        }
    } else {
        // Hybrid mode - use the auto-detected plan
        planner.plan(&args.query, args.top_k)
    };

    // Execute the search
    let hits = hybrid_search(mem, &plan)?;

    if args.json {
        // JSON output
        let output = serde_json::json!({
            "query": args.query,
            "mode": if args.graph { "graph" } else { "hybrid" },
            "plan": format!("{:?}", plan),
            "hits": hits.iter().map(|h| {
                serde_json::json!({
                    "frame_id": h.frame_id,
                    "score": h.score,
                    "graph_score": h.graph_score,
                    "vector_score": h.vector_score,
                    "matched_entity": h.matched_entity,
                    "preview": h.preview,
                })
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        // Human-readable output
        let mode_str = if args.graph { "Graph" } else { "Hybrid" };
        println!("{} search for: \"{}\"", mode_str, args.query);
        println!("Plan: {:?}", plan);
        println!();

        if hits.is_empty() {
            println!("No results found.");
        } else {
            println!("Results ({} hits):", hits.len());
            for (i, hit) in hits.iter().enumerate() {
                println!();
                println!(
                    "{}. Frame {} (score: {:.3}, graph: {:.2}, text: {:.2})",
                    i + 1,
                    hit.frame_id,
                    hit.score,
                    hit.graph_score,
                    hit.vector_score
                );
                if let Some(entity) = &hit.matched_entity {
                    println!("   Matched entity: {}", entity);
                }
                if let Some(preview) = &hit.preview {
                    let truncated = if preview.len() > 200 {
                        format!("{}...", &preview[..200])
                    } else {
                        preview.clone()
                    };
                    println!("   {}", truncated.replace('\n', " "));
                }
            }
        }
    }

    Ok(())
}

pub fn handle_find(config: &CliConfig, args: FindArgs) -> Result<()> {
    // Check if plan allows query operations (blocks expired subscriptions)
    crate::utils::require_active_plan(config, "find")?;

    // Track query usage against plan quota
    crate::api::track_query_usage(config, 1)?;

    let mut mem = open_read_only_mem(&args.file)?;

    // Load active replay session if one exists
    #[cfg(feature = "replay")]
    let _ = mem.load_active_session();

    // Handle graph-aware and hybrid search modes
    if args.graph || args.hybrid {
        return handle_graph_find(&mut mem, &args);
    }

    if args.uri.is_some() && args.scope.is_some() {
        warn!("--scope ignored because --uri is provided");
    }

    // Get vector dimension from MV2 for auto-detection
    let mv2_dimension = mem.effective_vec_index_dimension()?;
    let identity_summary = match args.mode {
        SearchMode::Sem | SearchMode::Auto => Some(mem.embedding_identity_summary(10_000)),
        #[cfg(feature = "clip")]
        SearchMode::Clip => None,
        SearchMode::Lex => None,
    };

    let mut semantic_allowed = true;
    let inferred_model_override = match identity_summary.as_ref() {
        Some(memvid_core::EmbeddingIdentitySummary::Single(identity)) => {
            identity.model.as_deref().map(|value| value.to_string())
        }
        Some(memvid_core::EmbeddingIdentitySummary::Mixed(identities)) => {
            let models: Vec<_> = identities
                .iter()
                .filter_map(|entry| entry.identity.model.as_deref())
                .collect();
            if args.mode == SearchMode::Sem {
                anyhow::bail!(
                    "memory contains mixed embedding models; semantic queries are unsafe.\n\n\
                    Detected models: {:?}\n\n\
                    Suggested fix: split into separate memories per embedding model.",
                    models
                );
            }
            warn!(
                "semantic search disabled: mixed embedding models detected: {:?}",
                models
            );
            semantic_allowed = false;
            None
        }
        _ => None,
    };

    let emb_model_override = args
        .query_embedding_model
        .as_deref()
        .or(inferred_model_override.as_deref());

    let (mode_label, runtime_option) = match args.mode {
        SearchMode::Lex => ("Lexical (forced)".to_string(), None),
        SearchMode::Sem => {
            let runtime =
                load_embedding_runtime_for_mv2(config, emb_model_override, mv2_dimension)?;
            ("Semantic (vector search)".to_string(), Some(runtime))
        }
        SearchMode::Auto => {
            if !semantic_allowed {
                ("Lexical (semantic unsafe)".to_string(), None)
            } else if let Some(runtime) =
                try_load_embedding_runtime_for_mv2(config, emb_model_override, mv2_dimension)
            {
                ("Hybrid (lexical + semantic)".to_string(), Some(runtime))
            } else {
                ("Lexical (semantic unavailable)".to_string(), None)
            }
        }
        #[cfg(feature = "clip")]
        SearchMode::Clip => ("CLIP (visual search)".to_string(), None),
    };

    let mode_key = match args.mode {
        SearchMode::Sem => "semantic",
        SearchMode::Lex => "text",
        SearchMode::Auto => {
            if runtime_option.is_some() {
                "hybrid"
            } else {
                "text"
            }
        }
        #[cfg(feature = "clip")]
        SearchMode::Clip => "clip",
    };

    // For CLIP mode, use CLIP visual search
    #[cfg(feature = "clip")]
    if args.mode == SearchMode::Clip {
        use memvid_core::clip::{ClipConfig, ClipModel};

        // Initialize CLIP model
        let config = ClipConfig::default();
        let clip = ClipModel::new(config).map_err(|e| {
            anyhow!("Failed to initialize CLIP model: {}. Make sure the MobileCLIP-S2 ONNX models are installed.", e)
        })?;

        // Encode query text
        let query_embedding = clip
            .encode_text(&args.query)
            .map_err(|e| anyhow!("Failed to encode query text: {}", e))?;

        // Search CLIP index
        let hits = mem.search_clip(&query_embedding, args.top_k)?;

        // Debug distances before filtering
        for hit in &hits {
            if let Ok(frame) = mem.frame_by_id(hit.frame_id) {
                tracing::debug!(
                    frame_id = hit.frame_id,
                    title = %frame.title.unwrap_or_default(),
                    page = hit.page,
                    distance = hit.distance,
                    cosine = 1.0 - (hit.distance * hit.distance / 2.0),
                    "CLIP raw hit"
                );
            } else {
                tracing::debug!(
                    frame_id = hit.frame_id,
                    page = hit.page,
                    distance = hit.distance,
                    cosine = 1.0 - (hit.distance * hit.distance / 2.0),
                    "CLIP raw hit (missing frame)"
                );
            }
        }

        // CLIP distance threshold for filtering poor matches
        // CLIP uses L2 distance on normalized embeddings:
        //   - distance² = 2(1 - cosine_similarity)
        //   - distance = 0 → identical (cosine_sim = 1)
        //   - distance = 1.0 → cosine_sim = 0.5 (50% match)
        //   - distance = 1.26 → cosine_sim = 0.20 (20% match - our threshold)
        //   - distance = √2 ≈ 1.41 → orthogonal (cosine_sim = 0)
        //   - distance = 2.0 → opposite (cosine_sim = -1)
        //
        // MobileCLIP text-to-image matching typically produces lower scores than expected.
        // Good matches are usually in the 0.20-0.35 cosine similarity range.
        // We filter at distance > 1.26 (cosine_sim < 0.20) to remove clearly irrelevant results.
        const CLIP_MAX_DISTANCE: f32 = 1.26;

        // Convert CLIP hits to SearchResponse format, filtering by threshold
        let search_hits: Vec<SearchHit> = hits
            .into_iter()
            .filter(|hit| hit.distance < CLIP_MAX_DISTANCE)
            .enumerate()
            .filter_map(|(rank, hit)| {
                // Convert L2 distance to cosine similarity for display
                // cos_sim = 1 - (distance² / 2)
                let cosine_similarity = 1.0 - (hit.distance * hit.distance / 2.0);

                // Get frame preview for snippet
                let preview = mem.frame_preview_by_id(hit.frame_id).ok()?;
                let uri = mem.frame_by_id(hit.frame_id).ok().and_then(|f| f.uri);
                let base_title = mem.frame_by_id(hit.frame_id).ok().and_then(|f| f.title);
                let title = match (base_title, hit.page) {
                    (Some(t), Some(p)) => Some(format!("{t} (page {p})")),
                    (Some(t), None) => Some(t),
                    (None, Some(p)) => Some(format!("Page {p}")),
                    _ => None,
                };
                Some(SearchHit {
                    rank: rank + 1,
                    frame_id: hit.frame_id,
                    uri: uri.unwrap_or_else(|| format!("mv2://frame/{}", hit.frame_id)),
                    title,
                    text: preview.clone(),
                    chunk_text: Some(preview),
                    range: (0, 0),
                    chunk_range: None,
                    matches: 0,
                    score: Some(cosine_similarity),
                    metadata: None,
                })
            })
            .collect();

        let response = SearchResponse {
            query: args.query.clone(),
            hits: search_hits.clone(),
            total_hits: search_hits.len(),
            params: memvid_core::SearchParams {
                top_k: args.top_k,
                snippet_chars: args.snippet_chars,
                cursor: args.cursor.clone(),
            },
            elapsed_ms: 0,
            engine: SearchEngineKind::Hybrid, // Use Hybrid as placeholder
            next_cursor: None,
            context: String::new(),
            stale_index_skips: 0,
        };

        if args.json_legacy {
            warn!("--json-legacy is deprecated; use --json for mv2.search.v1 output");
            emit_legacy_search_json(&response)?;
        } else if args.json {
            emit_search_json(&response, mode_key)?;
        } else {
            println!(
                "mode: {}   k={}   time: {} ms",
                mode_label, response.params.top_k, response.elapsed_ms
            );
            println!("engine: clip (MobileCLIP-S2)");
            println!(
                "hits: {} (showing {})",
                response.total_hits,
                response.hits.len()
            );
            emit_search_table(&response);
        }
        return Ok(());
    }

    // For semantic mode, use pure vector search.
    let (response, engine_label, adaptive_stats) = if args.mode == SearchMode::Sem {
        let runtime = runtime_option
            .as_ref()
            .ok_or_else(|| anyhow!("Semantic search requires an embedding runtime"))?;

        // Embed the query
        let query_embedding = runtime.embed_query(&args.query)?;

        // Use pure vector search (adaptive by default, use --no-adaptive to disable)
        let scope = args.scope.as_deref().or(args.uri.as_deref());

        if !args.no_adaptive {
            // Build adaptive config from CLI args
            let strategy = match args.adaptive_strategy {
                AdaptiveStrategyArg::Relative => CutoffStrategy::RelativeThreshold {
                    min_ratio: args.min_relevancy,
                },
                AdaptiveStrategyArg::Absolute => CutoffStrategy::AbsoluteThreshold {
                    min_score: args.min_relevancy,
                },
                AdaptiveStrategyArg::Cliff => CutoffStrategy::ScoreCliff {
                    max_drop_ratio: 0.35, // 35% drop triggers cutoff
                },
                AdaptiveStrategyArg::Elbow => CutoffStrategy::Elbow { sensitivity: 1.0 },
                AdaptiveStrategyArg::Combined => CutoffStrategy::Combined {
                    relative_threshold: args.min_relevancy,
                    max_drop_ratio: 0.35,
                    absolute_min: 0.3,
                },
            };

            let config = AdaptiveConfig {
                enabled: true,
                max_results: args.max_k,
                min_results: 1,
                strategy,
                normalize_scores: true,
            };

            match mem.search_adaptive(
                &args.query,
                &query_embedding,
                config,
                args.snippet_chars,
                scope,
            ) {
                Ok(result) => {
                    let mut resp = SearchResponse {
                        query: args.query.clone(),
                        hits: result.results,
                        total_hits: result.stats.returned,
                        params: memvid_core::SearchParams {
                            top_k: result.stats.returned,
                            snippet_chars: args.snippet_chars,
                            cursor: args.cursor.clone(),
                        },
                        elapsed_ms: 0,
                        engine: SearchEngineKind::Hybrid,
                        next_cursor: None,
                        context: String::new(),
                        stale_index_skips: 0,
                    };
                    apply_preference_rerank(&mut resp);
                    (
                        resp,
                        "semantic (adaptive vector search)".to_string(),
                        Some(result.stats),
                    )
                }
                Err(e) => {
                    if let MemvidError::VecDimensionMismatch { expected, actual } = e {
                        return Err(anyhow!(vec_dimension_mismatch_help(expected, actual)));
                    }

                    warn!("Adaptive search failed ({e}), falling back to fixed-k");
                    match mem.vec_search_with_embedding(
                        &args.query,
                        &query_embedding,
                        args.top_k,
                        args.snippet_chars,
                        scope,
                    ) {
                        Ok(mut resp) => {
                            apply_preference_rerank(&mut resp);
                            (resp, "semantic (vector search fallback)".to_string(), None)
                        }
                        Err(e2) => {
                            if let MemvidError::VecDimensionMismatch { expected, actual } = e2 {
                                return Err(anyhow!(vec_dimension_mismatch_help(expected, actual)));
                            }
                            return Err(anyhow!(
                                "Both adaptive and fixed-k search failed: {e}, {e2}"
                            ));
                        }
                    }
                }
            }
        } else {
            // Standard fixed-k vector search
            match mem.vec_search_with_embedding(
                &args.query,
                &query_embedding,
                args.top_k,
                args.snippet_chars,
                scope,
            ) {
                Ok(mut resp) => {
                    // Apply preference boost to rerank results for preference-seeking queries
                    apply_preference_rerank(&mut resp);
                    (resp, "semantic (vector search)".to_string(), None)
                }
                Err(e) => {
                    if let MemvidError::VecDimensionMismatch { expected, actual } = e {
                        return Err(anyhow!(vec_dimension_mismatch_help(expected, actual)));
                    }

                    // Fall back to lexical search + rerank if vector search fails
                    warn!("Vector search failed ({e}), falling back to lexical + rerank");
                    let request = SearchRequest {
                        query: args.query.clone(),
                        top_k: args.top_k,
                        snippet_chars: args.snippet_chars,
                        uri: args.uri.clone(),
                        scope: args.scope.clone(),
                        cursor: args.cursor.clone(),
                        #[cfg(feature = "temporal_track")]
                        temporal: None,
                        as_of_frame: args.as_of_frame,
                        as_of_ts: args.as_of_ts,
                        no_sketch: args.no_sketch,
                        acl_context: None,
                        acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
                    };
                    let mut resp = mem.search(request)?;
                    apply_semantic_rerank(runtime, &mut mem, &mut resp)?;
                    (resp, "semantic (fallback rerank)".to_string(), None)
                }
            }
        }
    } else {
        // For lexical and auto modes, use existing behavior
        let request = SearchRequest {
            query: args.query.clone(),
            top_k: args.top_k,
            snippet_chars: args.snippet_chars,
            uri: args.uri.clone(),
            scope: args.scope.clone(),
            cursor: args.cursor.clone(),
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: args.as_of_frame,
            as_of_ts: args.as_of_ts,
            no_sketch: args.no_sketch,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        };

        let mut resp = mem.search(request)?;

        if matches!(resp.engine, SearchEngineKind::LexFallback) && args.mode != SearchMode::Lex {
            warn!("Search index unavailable; returning basic text results");
        }

        let mut engine_label = match resp.engine {
            SearchEngineKind::Tantivy => "text (tantivy)".to_string(),
            SearchEngineKind::LexFallback => "text (fallback)".to_string(),
            SearchEngineKind::Hybrid => "hybrid".to_string(),
        };

        if runtime_option.is_some() {
            engine_label = format!("hybrid ({engine_label} + semantic)");
        }

        if let Some(ref runtime) = runtime_option {
            apply_semantic_rerank(runtime, &mut mem, &mut resp)?;
        }

        (resp, engine_label, None)
    };

    if args.json_legacy {
        warn!("--json-legacy is deprecated; use --json for mv2.search.v1 output");
        emit_legacy_search_json(&response)?;
    } else if args.json {
        emit_search_json(&response, mode_key)?;
    } else {
        println!(
            "mode: {}   k={}   time: {} ms",
            mode_label, response.params.top_k, response.elapsed_ms
        );
        println!("engine: {}", engine_label);

        // Show adaptive retrieval stats if enabled
        if let Some(ref stats) = adaptive_stats {
            println!(
                "adaptive: {} -> {} results (cutoff: {}, top: {:.3}, ratio: {:.1}%)",
                stats.total_considered,
                stats.returned,
                stats.triggered_by,
                stats.top_score.unwrap_or(0.0),
                stats.cutoff_ratio.unwrap_or(0.0) * 100.0
            );
        }

        println!(
            "hits: {} (showing {})",
            response.total_hits,
            response.hits.len()
        );
        emit_search_table(&response);
    }

    // Save active replay session if one exists
    #[cfg(feature = "replay")]
    let _ = mem.save_active_session();

    Ok(())
}

pub fn handle_vec_search(config: &CliConfig, args: VecSearchArgs) -> Result<()> {
    // Track query usage against plan quota
    crate::api::track_query_usage(config, 1)?;

    let mut mem = open_read_only_mem(&args.file)?;
    let vector = if let Some(path) = args.embedding.as_deref() {
        read_embedding(path)?
    } else if let Some(vector_string) = &args.vector {
        parse_vector(vector_string)?
    } else {
        anyhow::bail!("provide --vector or --embedding for search input");
    };

    let hits = mem
        .search_vec(&vector, args.limit)
        .map_err(|err| match err {
            MemvidError::VecDimensionMismatch { expected, actual } => {
                anyhow!(vec_dimension_mismatch_help(expected, actual))
            }
            other => anyhow!(other),
        })?;
    let mut enriched = Vec::with_capacity(hits.len());
    for hit in hits {
        let preview = mem.frame_preview_by_id(hit.frame_id)?;
        enriched.push((hit.frame_id, hit.distance, preview));
    }

    if args.json {
        let json_hits: Vec<_> = enriched
            .iter()
            .map(|(frame_id, distance, preview)| {
                json!({
                    "frame_id": frame_id,
                    "distance": distance,
                    "preview": preview,
                })
            })
            .collect();
        let json_str = serde_json::to_string_pretty(&json_hits)?;
        println!("{}", json_str.to_colored_json_auto()?);
    } else if enriched.is_empty() {
        println!("No vector matches found");
    } else {
        for (frame_id, distance, preview) in enriched {
            println!("frame {frame_id} (distance {distance:.6}): {preview}");
        }
    }
    Ok(())
}

pub fn handle_audit(config: &CliConfig, args: AuditArgs) -> Result<()> {
    use memvid_core::AuditOptions;
    use std::fs::File;
    use std::io::Write;

    let mut mem = Memvid::open(&args.file)?;

    // Parse date boundaries
    let start = parse_date_boundary(args.start.as_ref(), false)?;
    let end = parse_date_boundary(args.end.as_ref(), true)?;
    if let (Some(start_ts), Some(end_ts)) = (start, end) {
        if end_ts < start_ts {
            anyhow::bail!("--end must not be earlier than --start");
        }
    }

    // Set up embedding runtime if needed
    let ask_mode: AskMode = args.mode.into();
    let runtime = match args.mode {
        AskModeArg::Lex => None,
        AskModeArg::Sem => Some(load_embedding_runtime(config)?),
        AskModeArg::Hybrid => try_load_embedding_runtime(config),
    };
    let embedder = runtime.as_ref().map(|inner| inner as &dyn VecEmbedder);

    // Build audit options
    let options = AuditOptions {
        top_k: Some(args.top_k),
        snippet_chars: Some(args.snippet_chars),
        mode: Some(ask_mode),
        scope: args.scope,
        start,
        end,
        include_snippets: true,
    };

    // Run the audit
    let mut report = mem.audit(&args.question, Some(options), embedder)?;

    // If --use-model is provided, run model inference to synthesize the answer
    if let Some(model_name) = args.use_model.as_deref() {
        // Build context from sources for model inference
        let context = report
            .sources
            .iter()
            .filter_map(|s| s.snippet.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        match run_model_inference(
            model_name,
            &report.question,
            &context,
            &[], // No hits needed for audit
            None,
            None,
            None, // No system prompt override for audit
        ) {
            Ok(inference) => {
                report.answer = Some(inference.answer.answer);
                report.notes.push(format!(
                    "Answer synthesized by model: {}",
                    inference.answer.model
                ));
            }
            Err(err) => {
                warn!(
                    "model inference unavailable for '{}': {err}. Using default answer.",
                    model_name
                );
            }
        }
    }

    // Format the output
    let output = match args.format {
        AuditFormat::Text => report.to_text(),
        AuditFormat::Markdown => report.to_markdown(),
        AuditFormat::Json => serde_json::to_string_pretty(&report)?,
    };

    // Write output
    if let Some(out_path) = args.out {
        let mut file = File::create(&out_path)?;
        file.write_all(output.as_bytes())?;
        println!("Audit report written to: {}", out_path.display());
    } else {
        println!("{}", output);
    }

    Ok(())
}

fn emit_search_json(response: &SearchResponse, mode: &str) -> Result<()> {
    let hits: Vec<_> = response.hits.iter().map(search_hit_to_json).collect();

    let mut additional_params = serde_json::Map::new();
    if let Some(cursor) = &response.params.cursor {
        additional_params.insert("cursor".into(), json!(cursor));
    }

    let mut params = serde_json::Map::new();
    params.insert("top_k".into(), json!(response.params.top_k));
    params.insert("snippet_chars".into(), json!(response.params.snippet_chars));
    params.insert("mode".into(), json!(mode));
    params.insert(
        "additional_params".into(),
        serde_json::Value::Object(additional_params),
    );

    let mut metadata_json = serde_json::Map::new();
    metadata_json.insert("elapsed_ms".into(), json!(response.elapsed_ms));
    metadata_json.insert("total_hits".into(), json!(response.total_hits));
    metadata_json.insert(
        "next_cursor".into(),
        match &response.next_cursor {
            Some(cursor) => json!(cursor),
            None => serde_json::Value::Null,
        },
    );
    metadata_json.insert("engine".into(), json!(response.engine));
    metadata_json.insert("params".into(), serde_json::Value::Object(params));

    let body = json!({
        "version": "mv2.result.v2",
        "query": response.query,
        "metadata": metadata_json,
        "hits": hits,
        "context": response.context,
    });
    let json_str = serde_json::to_string_pretty(&body)?;
    println!("{}", json_str.to_colored_json_auto()?);
    Ok(())
}

fn emit_ask_json(
    response: &AskResponse,
    requested_mode: AskModeArg,
    inference: Option<&ModelInference>,
    include_sources: bool,
    mem: &mut Memvid,
) -> Result<()> {
    let hits: Vec<_> = response
        .retrieval
        .hits
        .iter()
        .map(search_hit_to_json)
        .collect();

    let citations: Vec<_> = response
        .citations
        .iter()
        .map(|citation| {
            let mut map = serde_json::Map::new();
            map.insert("index".into(), json!(citation.index));
            map.insert("frame_id".into(), json!(citation.frame_id));
            map.insert("uri".into(), json!(citation.uri));
            if let Some(range) = citation.chunk_range {
                map.insert("chunk_range".into(), json!([range.0, range.1]));
            }
            if let Some(score) = citation.score {
                map.insert("score".into(), json!(score));
            }
            serde_json::Value::Object(map)
        })
        .collect();

    let mut body = json!({
        "version": "mv2.ask.v1",
        "question": response.question,
        "answer": response.answer,
        "context_only": response.context_only,
        "mode": ask_mode_display(requested_mode),
        "retriever": ask_retriever_display(response.retriever),
        "top_k": response.retrieval.params.top_k,
        "results": hits,
        "citations": citations,
        "stats": {
            "retrieval_ms": response.stats.retrieval_ms,
            "synthesis_ms": response.stats.synthesis_ms,
            "latency_ms": response.stats.latency_ms,
        },
        "engine": search_engine_label(&response.retrieval.engine),
        "total_hits": response.retrieval.total_hits,
        "next_cursor": response.retrieval.next_cursor,
        "context": truncate_with_ellipsis(&response.retrieval.context, OUTPUT_CONTEXT_MAX_LEN),
    });

    if let Some(inf) = inference {
        let model = &inf.answer;
        if let serde_json::Value::Object(ref mut map) = body {
            map.insert("model".into(), json!(model.requested));
            if model.model != model.requested {
                map.insert("model_used".into(), json!(model.model));
            }
            map.insert("cached".into(), json!(inf.cached));
            // Add usage and cost if available
            if let Some(usage) = &inf.usage {
                map.insert(
                    "usage".into(),
                    json!({
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "cost_usd": if inf.cached { 0.0 } else { usage.cost_usd },
                        "saved_cost_usd": if inf.cached { usage.cost_usd } else { 0.0 },
                    }),
                );
            }
            // Add grounding/hallucination score if available
            if let Some(grounding) = &inf.grounding {
                map.insert(
                    "grounding".into(),
                    json!({
                        "score": grounding.score,
                        "label": grounding.label(),
                        "sentence_count": grounding.sentence_count,
                        "grounded_sentences": grounding.grounded_sentences,
                        "has_warning": grounding.has_warning,
                        "warning_reason": grounding.warning_reason,
                    }),
                );
            }
        }
    }

    // Add detailed sources if requested
    if include_sources {
        if let serde_json::Value::Object(ref mut map) = body {
            let sources = build_sources_json(response, mem);
            map.insert("sources".into(), json!(sources));
        }
    }

    // Add follow-up suggestions if confidence is low
    if let Some(follow_up) = build_follow_up_suggestions(response, inference, mem) {
        if let serde_json::Value::Object(ref mut map) = body {
            map.insert("follow_up".into(), follow_up);
        }
    }

    println!("{}", serde_json::to_string_pretty(&body)?);
    Ok(())
}

fn build_sources_json(response: &AskResponse, mem: &mut Memvid) -> Vec<serde_json::Value> {
    response
        .citations
        .iter()
        .enumerate()
        .map(|(idx, citation)| {
            let mut source = serde_json::Map::new();
            source.insert("index".into(), json!(idx + 1));
            source.insert("frame_id".into(), json!(citation.frame_id));
            source.insert("uri".into(), json!(citation.uri));

            if let Some(range) = citation.chunk_range {
                source.insert("chunk_range".into(), json!([range.0, range.1]));
            }
            if let Some(score) = citation.score {
                source.insert("score".into(), json!(score));
            }

            // Get frame metadata for rich source information
            if let Ok(frame) = mem.frame_by_id(citation.frame_id) {
                if let Some(title) = frame.title {
                    source.insert("title".into(), json!(title));
                }
                if !frame.tags.is_empty() {
                    source.insert("tags".into(), json!(frame.tags));
                }
                if !frame.labels.is_empty() {
                    source.insert("labels".into(), json!(frame.labels));
                }
                source.insert("frame_timestamp".into(), json!(frame.timestamp));
                if !frame.content_dates.is_empty() {
                    source.insert("content_dates".into(), json!(frame.content_dates));
                }
            }

            // Get snippet from hit
            if let Some(hit) = response
                .retrieval
                .hits
                .iter()
                .find(|h| h.frame_id == citation.frame_id)
            {
                let snippet = hit.chunk_text.clone().unwrap_or_else(|| hit.text.clone());
                source.insert("snippet".into(), json!(snippet));
            }

            serde_json::Value::Object(source)
        })
        .collect()
}

/// Build follow-up suggestions when the answer has low grounding/confidence.
/// Helps users understand what the memory contains and suggests relevant questions.
fn build_follow_up_suggestions(
    response: &AskResponse,
    inference: Option<&ModelInference>,
    mem: &mut Memvid,
) -> Option<serde_json::Value> {
    // Check if we need follow-up suggestions
    let needs_followup = inference
        .and_then(|inf| inf.grounding.as_ref())
        .map(|g| g.score < 0.3 || g.has_warning)
        .unwrap_or(false);

    // Also trigger if retrieval hits have very low scores or no hits
    let low_retrieval = response
        .retrieval
        .hits
        .first()
        .and_then(|h| h.score)
        .map(|score| score < -2.0)
        .unwrap_or(true);

    if !needs_followup && !low_retrieval {
        return None;
    }

    // Get available topics from the memory by sampling timeline entries
    let limit = std::num::NonZeroU64::new(20).unwrap();
    let timeline_query = TimelineQueryBuilder::default().limit(limit).build();

    let available_topics: Vec<String> = mem
        .timeline(timeline_query)
        .ok()
        .map(|entries| {
            entries
                .iter()
                .filter_map(|e| {
                    // Extract meaningful preview/title
                    let preview = e.preview.trim();
                    if preview.is_empty() || preview.len() < 5 {
                        return None;
                    }
                    // Get first line or truncate
                    let first_line = preview.lines().next().unwrap_or(preview);
                    if first_line.len() > 60 {
                        Some(format!("{}...", &first_line[..57]))
                    } else {
                        Some(first_line.to_string())
                    }
                })
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .take(5)
                .collect()
        })
        .unwrap_or_default();

    // Determine the reason for low confidence
    let reason = if response.retrieval.hits.is_empty() || low_retrieval {
        "No relevant information found in memory"
    } else if inference
        .and_then(|i| i.grounding.as_ref())
        .map(|g| g.has_warning)
        .unwrap_or(false)
    {
        "Answer may not be well-supported by the available context"
    } else {
        "Low confidence in the answer"
    };

    // Generate suggestion questions based on available topics
    let suggestions: Vec<String> = if available_topics.is_empty() {
        vec![
            "What information is stored in this memory?".to_string(),
            "Can you list the main topics covered?".to_string(),
        ]
    } else {
        available_topics
            .iter()
            .take(3)
            .map(|topic| format!("Tell me about {}", topic))
            .chain(std::iter::once(
                "What topics are in this memory?".to_string(),
            ))
            .collect()
    };

    Some(json!({
        "needed": true,
        "reason": reason,
        "hint": if available_topics.is_empty() {
            "This memory may not contain information about your query."
        } else {
            "This memory contains information about different topics. Try asking about those instead."
        },
        "available_topics": available_topics,
        "suggestions": suggestions
    }))
}

fn emit_model_json(
    response: &AskResponse,
    requested_model: &str,
    inference: Option<&ModelInference>,
    include_sources: bool,
    mem: &mut Memvid,
) -> Result<()> {
    let answer = response.answer.clone().unwrap_or_default();
    let requested_label = inference
        .map(|m| m.answer.requested.clone())
        .unwrap_or_else(|| requested_model.to_string());
    let used_label = inference
        .map(|m| m.answer.model.clone())
        .unwrap_or_else(|| requested_model.to_string());

    let mut body = json!({
        "question": response.question,
        "model": requested_label,
        "model_used": used_label,
        "answer": answer,
        "context": truncate_with_ellipsis(&response.retrieval.context, OUTPUT_CONTEXT_MAX_LEN),
    });

    // Add usage and cost if available
    if let Some(inf) = inference {
        if let serde_json::Value::Object(ref mut map) = body {
            map.insert("cached".into(), json!(inf.cached));
            if let Some(usage) = &inf.usage {
                map.insert(
                    "usage".into(),
                    json!({
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "cost_usd": if inf.cached { 0.0 } else { usage.cost_usd },
                        "saved_cost_usd": if inf.cached { usage.cost_usd } else { 0.0 },
                    }),
                );
            }
            if let Some(grounding) = &inf.grounding {
                map.insert(
                    "grounding".into(),
                    json!({
                        "score": grounding.score,
                        "label": grounding.label(),
                        "sentence_count": grounding.sentence_count,
                        "grounded_sentences": grounding.grounded_sentences,
                        "has_warning": grounding.has_warning,
                        "warning_reason": grounding.warning_reason,
                    }),
                );
            }
        }
    }

    // Add detailed sources if requested
    if include_sources {
        if let serde_json::Value::Object(ref mut map) = body {
            let sources = build_sources_json(response, mem);
            map.insert("sources".into(), json!(sources));
        }
    }

    // Add follow-up suggestions if confidence is low
    if let Some(follow_up) = build_follow_up_suggestions(response, inference, mem) {
        if let serde_json::Value::Object(ref mut map) = body {
            map.insert("follow_up".into(), follow_up);
        }
    }

    // Use colored JSON output
    let json_str = serde_json::to_string_pretty(&body)?;
    println!("{}", json_str.to_colored_json_auto()?);
    Ok(())
}

fn emit_ask_pretty(
    response: &AskResponse,
    requested_mode: AskModeArg,
    inference: Option<&ModelInference>,
    include_sources: bool,
    mem: &mut Memvid,
) {
    println!(
        "mode: {}   retriever: {}   k={}   latency: {} ms (retrieval {} ms)",
        ask_mode_pretty(requested_mode),
        ask_retriever_pretty(response.retriever),
        response.retrieval.params.top_k,
        response.stats.latency_ms,
        response.stats.retrieval_ms
    );
    if let Some(inference) = inference {
        let model = &inference.answer;
        let cached_label = if inference.cached { " [CACHED]" } else { "" };
        if model.requested.trim() == model.model {
            println!("model: {}{}", model.model, cached_label);
        } else {
            println!(
                "model requested: {}   model used: {}{}",
                model.requested, model.model, cached_label
            );
        }
        // Display usage and cost if available
        if let Some(usage) = &inference.usage {
            let cost_label = if inference.cached {
                format!("$0.00 (saved ${:.6})", usage.cost_usd)
            } else {
                format!("${:.6}", usage.cost_usd)
            };
            println!(
                "tokens: {} input + {} output = {}   cost: {}",
                usage.input_tokens, usage.output_tokens, usage.total_tokens, cost_label
            );
        }
        // Display grounding/hallucination score
        if let Some(grounding) = &inference.grounding {
            let warning = if grounding.has_warning {
                format!(
                    " [WARNING: {}]",
                    grounding
                        .warning_reason
                        .as_deref()
                        .unwrap_or("potential hallucination")
                )
            } else {
                String::new()
            };
            println!(
                "grounding: {:.0}% ({}) - {}/{} sentences grounded{}",
                grounding.score * 100.0,
                grounding.label(),
                grounding.grounded_sentences,
                grounding.sentence_count,
                warning
            );
        }
    }
    println!(
        "engine: {}",
        search_engine_label(&response.retrieval.engine)
    );
    println!(
        "hits: {} (showing {})",
        response.retrieval.total_hits,
        response.retrieval.hits.len()
    );

    if response.context_only {
        println!();
        println!("Context-only mode: synthesis disabled.");
        println!();
    } else if let Some(answer) = &response.answer {
        println!();
        println!("Answer:\n{answer}");
        println!();
    }

    if !response.citations.is_empty() {
        println!("Citations:");
        for citation in &response.citations {
            match citation.score {
                Some(score) => println!(
                    "[{}] {} (frame {}, score {:.3})",
                    citation.index, citation.uri, citation.frame_id, score
                ),
                None => println!(
                    "[{}] {} (frame {})",
                    citation.index, citation.uri, citation.frame_id
                ),
            }
        }
        println!();
    }

    // Print detailed sources if requested
    if include_sources && !response.citations.is_empty() {
        println!("=== SOURCES ===");
        println!();
        for citation in &response.citations {
            println!("[{}] {}", citation.index, citation.uri);

            // Get frame metadata
            if let Ok(frame) = mem.frame_by_id(citation.frame_id) {
                if let Some(title) = &frame.title {
                    println!("    Title: {}", title);
                }
                println!("    Frame ID: {}", citation.frame_id);
                if let Some(score) = citation.score {
                    println!("    Score: {:.4}", score);
                }
                if let Some((start, end)) = citation.chunk_range {
                    println!("    Range: [{}..{})", start, end);
                }
                if !frame.tags.is_empty() {
                    println!("    Tags: {}", frame.tags.join(", "));
                }
                if !frame.labels.is_empty() {
                    println!("    Labels: {}", frame.labels.join(", "));
                }
                println!("    Timestamp: {}", frame.timestamp);
                if !frame.content_dates.is_empty() {
                    println!("    Content Dates: {}", frame.content_dates.join(", "));
                }
            }

            // Get snippet from hit
            if let Some(hit) = response
                .retrieval
                .hits
                .iter()
                .find(|h| h.frame_id == citation.frame_id)
            {
                let snippet = hit.chunk_text.as_ref().unwrap_or(&hit.text);
                let truncated = if snippet.len() > 200 {
                    format!("{}...", &snippet[..200])
                } else {
                    snippet.clone()
                };
                println!("    Snippet: {}", truncated.replace('\n', " "));
            }
            println!();
        }
    }

    if !include_sources {
        println!();
        emit_search_table(&response.retrieval);
    }

    // Display follow-up suggestions if confidence is low
    if let Some(follow_up) = build_follow_up_suggestions(response, inference, mem) {
        if let Some(needed) = follow_up.get("needed").and_then(|v| v.as_bool()) {
            if needed {
                println!();
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("💡 FOLLOW-UP SUGGESTIONS");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                if let Some(reason) = follow_up.get("reason").and_then(|v| v.as_str()) {
                    println!("Reason: {}", reason);
                }

                if let Some(hint) = follow_up.get("hint").and_then(|v| v.as_str()) {
                    println!("Hint: {}", hint);
                }

                if let Some(topics) = follow_up.get("available_topics").and_then(|v| v.as_array()) {
                    if !topics.is_empty() {
                        println!();
                        println!("Available topics in this memory:");
                        for topic in topics.iter().filter_map(|t| t.as_str()) {
                            println!("  • {}", topic);
                        }
                    }
                }

                if let Some(suggestions) = follow_up.get("suggestions").and_then(|v| v.as_array()) {
                    if !suggestions.is_empty() {
                        println!();
                        println!("Try asking:");
                        for (i, suggestion) in
                            suggestions.iter().filter_map(|s| s.as_str()).enumerate()
                        {
                            println!("  {}. \"{}\"", i + 1, suggestion);
                        }
                    }
                }
                println!();
            }
        }
    }
}

/// Emit verbatim evidence as JSON without LLM synthesis.
/// Format: {evidence: [{source, text, score}], question, hits, stats}
fn emit_verbatim_evidence_json(
    response: &AskResponse,
    include_sources: bool,
    mem: &mut Memvid,
) -> Result<()> {
    // Build evidence array from hits - verbatim excerpts with citations
    let evidence: Vec<_> = response
        .retrieval
        .hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| {
            let mut entry = serde_json::Map::new();
            entry.insert("index".into(), json!(idx + 1));
            entry.insert("frame_id".into(), json!(hit.frame_id));
            entry.insert("uri".into(), json!(&hit.uri));
            if let Some(title) = &hit.title {
                entry.insert("title".into(), json!(title));
            }
            // Use chunk_text if available (more specific), otherwise full text
            let verbatim = hit.chunk_text.as_ref().unwrap_or(&hit.text);
            entry.insert("text".into(), json!(verbatim));
            if let Some(score) = hit.score {
                entry.insert("score".into(), json!(score));
            }
            serde_json::Value::Object(entry)
        })
        .collect();

    // Build sources array if requested
    let sources: Option<Vec<_>> = if include_sources {
        Some(
            response
                .retrieval
                .hits
                .iter()
                .filter_map(|hit| {
                    mem.frame_by_id(hit.frame_id).ok().map(|frame| {
                        let mut source = serde_json::Map::new();
                        source.insert("frame_id".into(), json!(frame.id));
                        source.insert(
                            "uri".into(),
                            json!(frame.uri.as_deref().unwrap_or("(unknown)")),
                        );
                        if let Some(title) = &frame.title {
                            source.insert("title".into(), json!(title));
                        }
                        source.insert("timestamp".into(), json!(frame.timestamp.to_string()));
                        if !frame.tags.is_empty() {
                            source.insert("tags".into(), json!(frame.tags));
                        }
                        if !frame.labels.is_empty() {
                            source.insert("labels".into(), json!(frame.labels));
                        }
                        serde_json::Value::Object(source)
                    })
                })
                .collect(),
        )
    } else {
        None
    };

    let mut body = json!({
        "version": "mv2.evidence.v1",
        "mode": "verbatim",
        "question": response.question,
        "evidence": evidence,
        "evidence_count": evidence.len(),
        "total_hits": response.retrieval.total_hits,
        "stats": {
            "retrieval_ms": response.stats.retrieval_ms,
            "latency_ms": response.stats.latency_ms,
        },
        "engine": search_engine_label(&response.retrieval.engine),
    });

    if let (Some(sources), serde_json::Value::Object(ref mut map)) = (sources, &mut body) {
        map.insert("sources".into(), json!(sources));
    }

    let json_str = serde_json::to_string_pretty(&body)?;
    println!("{}", json_str.to_colored_json_auto()?);
    Ok(())
}

/// Emit verbatim evidence in human-readable format without LLM synthesis.
fn emit_verbatim_evidence_pretty(response: &AskResponse, include_sources: bool, mem: &mut Memvid) {
    println!(
        "mode: {}   latency: {} ms (retrieval {} ms)",
        "verbatim evidence".cyan(),
        response.stats.latency_ms,
        response.stats.retrieval_ms
    );
    println!(
        "engine: {}",
        search_engine_label(&response.retrieval.engine)
    );
    println!(
        "hits: {} (showing {})",
        response.retrieval.total_hits,
        response.retrieval.hits.len()
    );
    println!();

    // Header
    println!("{}", "━".repeat(60));
    println!(
        "{}",
        format!(
            "VERBATIM EVIDENCE for: \"{}\"",
            truncate_with_ellipsis(&response.question, 40)
        )
        .bold()
    );
    println!("{}", "━".repeat(60));
    println!();

    if response.retrieval.hits.is_empty() {
        println!("No evidence found.");
        return;
    }

    // Calculate score range for normalization (BM25 scores can be negative)
    let scores: Vec<Option<f32>> = response.retrieval.hits.iter().map(|h| h.score).collect();
    let (min_score, max_score) = score_range(&scores);

    // Display each piece of evidence with citation
    for (idx, hit) in response.retrieval.hits.iter().enumerate() {
        let uri = &hit.uri;
        let title = hit.title.as_deref().unwrap_or("Untitled");
        let score_str = hit
            .score
            .map(|s| {
                let normalized = normalize_bm25_for_display(s, min_score, max_score);
                format!(" (relevance: {:.0}%)", normalized)
            })
            .unwrap_or_default();

        println!(
            "{}",
            format!("[{}] {}{}", idx + 1, title, score_str)
                .green()
                .bold()
        );
        println!("    Source: {} (frame {})", uri, hit.frame_id);
        println!();

        // Show verbatim text - prefer chunk_text if available
        let verbatim = hit.chunk_text.as_ref().unwrap_or(&hit.text);
        // Indent each line for readability
        for line in verbatim.lines() {
            if !line.trim().is_empty() {
                println!("    │ {}", line);
            }
        }
        println!();
    }

    // Print detailed sources if requested
    if include_sources {
        println!("{}", "━".repeat(60));
        println!("{}", "SOURCE DETAILS".bold());
        println!("{}", "━".repeat(60));
        println!();

        for (idx, hit) in response.retrieval.hits.iter().enumerate() {
            if let Ok(frame) = mem.frame_by_id(hit.frame_id) {
                println!(
                    "{}",
                    format!(
                        "[{}] {}",
                        idx + 1,
                        frame.uri.as_deref().unwrap_or("(unknown)")
                    )
                    .cyan()
                );
                if let Some(title) = &frame.title {
                    println!("    Title: {}", title);
                }
                println!("    Frame ID: {}", frame.id);
                println!("    Timestamp: {}", frame.timestamp);
                if !frame.tags.is_empty() {
                    println!("    Tags: {}", frame.tags.join(", "));
                }
                if !frame.labels.is_empty() {
                    println!("    Labels: {}", frame.labels.join(", "));
                }
                if !frame.content_dates.is_empty() {
                    println!("    Content Dates: {}", frame.content_dates.join(", "));
                }
                println!();
            }
        }
    }

    // Note about no LLM synthesis
    println!("{}", "─".repeat(60));
    println!(
        "{}",
        "Note: Showing verbatim evidence without LLM synthesis.".dimmed()
    );
    println!(
        "{}",
        "Use --use-model to get an AI-synthesized answer.".dimmed()
    );
}

fn emit_legacy_search_json(response: &SearchResponse) -> Result<()> {
    let hits: Vec<_> = response
        .hits
        .iter()
        .map(|hit| {
            json!({
                "frame_id": hit.frame_id,
                "matches": hit.matches,
                "snippets": [hit.text.clone()],
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&hits)?);
    Ok(())
}

fn emit_search_table(response: &SearchResponse) {
    if response.hits.is_empty() {
        println!("No results for '{}'.", response.query);
        return;
    }

    // Calculate score range for normalization (BM25 scores can be negative)
    let scores: Vec<Option<f32>> = response.hits.iter().map(|h| h.score).collect();
    let (min_score, max_score) = score_range(&scores);

    for hit in &response.hits {
        println!("#{} {} (matches {})", hit.rank, hit.uri, hit.matches);
        if let Some(title) = &hit.title {
            println!("  Title: {title}");
        }
        if let Some(score) = hit.score {
            let normalized = normalize_bm25_for_display(score, min_score, max_score);
            println!("  Relevance: {:.0}%", normalized);
        }
        println!("  Range: [{}..{})", hit.range.0, hit.range.1);
        if let Some((chunk_start, chunk_end)) = hit.chunk_range {
            println!("  Chunk: [{}..{})", chunk_start, chunk_end);
        }
        if let Some(chunk_text) = &hit.chunk_text {
            println!("  Chunk Text: {}", chunk_text.trim());
        }
        if let Some(metadata) = &hit.metadata {
            if let Some(track) = &metadata.track {
                println!("  Track: {track}");
            }
            if !metadata.tags.is_empty() {
                println!("  Tags: {}", metadata.tags.join(", "));
            }
            if !metadata.labels.is_empty() {
                println!("  Labels: {}", metadata.labels.join(", "));
            }
            if let Some(created_at) = &metadata.created_at {
                println!("  Created: {created_at}");
            }
            if !metadata.content_dates.is_empty() {
                println!("  Content Dates: {}", metadata.content_dates.join(", "));
            }
            if !metadata.entities.is_empty() {
                let entity_strs: Vec<String> = metadata
                    .entities
                    .iter()
                    .map(|e| format!("{} ({})", e.name, e.kind))
                    .collect();
                println!("  Entities: {}", entity_strs.join(", "));
            }
        }
        println!("  Snippet: {}", hit.text.trim());
        println!();
    }
    if let Some(cursor) = &response.next_cursor {
        println!("Next cursor: {cursor}");
    }
}

fn ask_mode_display(mode: AskModeArg) -> &'static str {
    match mode {
        AskModeArg::Lex => "lex",
        AskModeArg::Sem => "sem",
        AskModeArg::Hybrid => "hybrid",
    }
}

fn ask_mode_pretty(mode: AskModeArg) -> &'static str {
    match mode {
        AskModeArg::Lex => "Lexical",
        AskModeArg::Sem => "Semantic",
        AskModeArg::Hybrid => "Hybrid",
    }
}

fn ask_retriever_display(retriever: AskRetriever) -> &'static str {
    match retriever {
        AskRetriever::Lex => "lex",
        AskRetriever::Semantic => "semantic",
        AskRetriever::Hybrid => "hybrid",
        AskRetriever::LexFallback => "lex_fallback",
        AskRetriever::TimelineFallback => "timeline_fallback",
    }
}

fn ask_retriever_pretty(retriever: AskRetriever) -> &'static str {
    match retriever {
        AskRetriever::Lex => "Lexical",
        AskRetriever::Semantic => "Semantic",
        AskRetriever::Hybrid => "Hybrid",
        AskRetriever::LexFallback => "Lexical (fallback)",
        AskRetriever::TimelineFallback => "Timeline (fallback)",
    }
}

fn search_engine_label(engine: &SearchEngineKind) -> &'static str {
    match engine {
        SearchEngineKind::Tantivy => "text (tantivy)",
        SearchEngineKind::LexFallback => "text (fallback)",
        SearchEngineKind::Hybrid => "hybrid",
    }
}

fn build_hit_id(uri: &str, frame_id: u64, start: usize) -> String {
    let digest = hash(uri.as_bytes()).to_hex().to_string();
    let prefix_len = digest.len().min(12);
    let prefix = &digest[..prefix_len];
    format!("mv2-hit-{prefix}-{frame_id}-{start}")
}

fn truncate_with_ellipsis(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }

    let truncated: String = text.chars().take(limit).collect();
    format!("{truncated}...")
}

/// Normalize a BM25 score to 0-100 range for user-friendly display.
///
/// BM25 scores can be negative (Tantivy uses log-based TF which can go negative
/// for very common terms). This function normalizes scores relative to the
/// min/max in the result set so users see intuitive 0-100 values.
///
/// - Returns 100.0 if min == max (all scores equal)
/// - Returns normalized 0-100 value based on position in [min, max] range
fn normalize_bm25_for_display(score: f32, min_score: f32, max_score: f32) -> f32 {
    if (max_score - min_score).abs() < f32::EPSILON {
        // All scores are the same, show 100%
        return 100.0;
    }
    // Normalize to 0-100 range
    ((score - min_score) / (max_score - min_score) * 100.0).clamp(0.0, 100.0)
}

/// Extract min and max scores from a slice of optional scores.
fn score_range(scores: &[Option<f32>]) -> (f32, f32) {
    let valid_scores: Vec<f32> = scores.iter().filter_map(|s| *s).collect();
    if valid_scores.is_empty() {
        return (0.0, 0.0);
    }
    let min = valid_scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = valid_scores
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

fn search_hit_to_json(hit: &SearchHit) -> serde_json::Value {
    let mut hit_json = serde_json::Map::new();
    hit_json.insert("rank".into(), json!(hit.rank));
    if let Some(score) = hit.score {
        hit_json.insert("score".into(), json!(score));
    }
    hit_json.insert(
        "id".into(),
        json!(build_hit_id(&hit.uri, hit.frame_id, hit.range.0)),
    );
    hit_json.insert("frame_id".into(), json!(hit.frame_id));
    hit_json.insert("uri".into(), json!(hit.uri));
    if let Some(title) = &hit.title {
        hit_json.insert("title".into(), json!(title));
    }
    let chunk_range = hit.chunk_range.unwrap_or(hit.range);
    hit_json.insert("chunk_range".into(), json!([chunk_range.0, chunk_range.1]));
    hit_json.insert("range".into(), json!([hit.range.0, hit.range.1]));
    hit_json.insert("text".into(), json!(hit.text));

    let metadata = hit.metadata.clone().unwrap_or_else(|| SearchHitMetadata {
        matches: hit.matches,
        ..SearchHitMetadata::default()
    });
    let mut meta_json = serde_json::Map::new();
    meta_json.insert("matches".into(), json!(metadata.matches));
    if !metadata.tags.is_empty() {
        meta_json.insert("tags".into(), json!(metadata.tags));
    }
    if !metadata.labels.is_empty() {
        meta_json.insert("labels".into(), json!(metadata.labels));
    }
    if let Some(track) = metadata.track {
        meta_json.insert("track".into(), json!(track));
    }
    if let Some(created_at) = metadata.created_at {
        meta_json.insert("created_at".into(), json!(created_at));
    }
    if !metadata.content_dates.is_empty() {
        meta_json.insert("content_dates".into(), json!(metadata.content_dates));
    }
    if !metadata.entities.is_empty() {
        let entities_json: Vec<serde_json::Value> = metadata
            .entities
            .iter()
            .map(|e| {
                let mut ent = serde_json::Map::new();
                ent.insert("name".into(), json!(e.name));
                ent.insert("kind".into(), json!(e.kind));
                if let Some(conf) = e.confidence {
                    ent.insert("confidence".into(), json!(conf));
                }
                serde_json::Value::Object(ent)
            })
            .collect();
        meta_json.insert("entities".into(), json!(entities_json));
    }
    hit_json.insert("metadata".into(), serde_json::Value::Object(meta_json));
    serde_json::Value::Object(hit_json)
}
/// Apply Reciprocal Rank Fusion (RRF) to combine lexical and semantic rankings.
///
/// RRF is mathematically superior to raw score combination because:
/// - BM25 scores are unbounded (0 to infinity)
/// - Cosine similarity is bounded (-1 to 1)
/// - RRF normalizes by using only RANKS, not raw scores
///
/// Formula: Score(d) = sum(1 / (k + rank(d))) where k=60 is standard
fn apply_semantic_rerank(
    runtime: &EmbeddingRuntime,
    mem: &mut Memvid,
    response: &mut SearchResponse,
) -> Result<()> {
    if response.hits.is_empty() {
        return Ok(());
    }

    let query_embedding = runtime.embed_query(&response.query)?;
    let mut semantic_scores: HashMap<u64, f32> = HashMap::new();
    for hit in &response.hits {
        if let Some(embedding) = mem.frame_embedding(hit.frame_id)? {
            if embedding.len() == runtime.dimension() {
                let score = cosine_similarity(&query_embedding, &embedding);
                semantic_scores.insert(hit.frame_id, score);
            }
        }
    }

    if semantic_scores.is_empty() {
        return Ok(());
    }

    // Sort by semantic score to get semantic ranks
    let mut sorted_semantic: Vec<(u64, f32)> = semantic_scores
        .iter()
        .map(|(frame_id, score)| (*frame_id, *score))
        .collect();
    sorted_semantic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut semantic_rank: HashMap<u64, usize> = HashMap::new();
    for (idx, (frame_id, _)) in sorted_semantic.iter().enumerate() {
        semantic_rank.insert(*frame_id, idx + 1);
    }

    // Check if query is preference-seeking (suggests, recommend, should I, etc.)
    let query_lower = response.query.to_lowercase();
    let is_preference_query = query_lower.contains("suggest")
        || query_lower.contains("recommend")
        || query_lower.contains("should i")
        || query_lower.contains("what should")
        || query_lower.contains("prefer")
        || query_lower.contains("favorite")
        || query_lower.contains("best for me");

    // Pure RRF: Use ONLY ranks, NOT raw scores
    // This prevents a "confidently wrong" high-scoring vector from burying
    // a "precisely correct" keyword match
    const RRF_K: f32 = 60.0;

    let mut ordering: Vec<(usize, f32, usize)> = response
        .hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| {
            let lexical_rank = hit.rank;

            // RRF score for lexical rank
            let lexical_rrf = 1.0 / (RRF_K + lexical_rank as f32);

            // RRF score for semantic rank
            let semantic_rrf = semantic_rank
                .get(&hit.frame_id)
                .map(|rank| 1.0 / (RRF_K + *rank as f32))
                .unwrap_or(0.0);

            // Apply preference boost for hits containing user preference signals
            // This is a small bonus for content with first-person preference indicators
            let preference_boost = if is_preference_query {
                compute_preference_boost(&hit.text) * 0.01 // Scale down to RRF magnitude
            } else {
                0.0
            };

            // Pure RRF: Only rank-based scores, no raw similarity scores
            let combined = lexical_rrf + semantic_rrf + preference_boost;
            (idx, combined, lexical_rank)
        })
        .collect();

    ordering.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });

    let mut reordered = Vec::with_capacity(response.hits.len());
    for (rank_idx, (idx, _, _)) in ordering.into_iter().enumerate() {
        let mut hit = response.hits[idx].clone();
        hit.rank = rank_idx + 1;
        reordered.push(hit);
    }

    response.hits = reordered;
    Ok(())
}

/// Rerank search results by boosting hits that contain user preference signals.
/// Only applies when the query appears to be seeking recommendations or preferences.
fn apply_preference_rerank(response: &mut SearchResponse) {
    if response.hits.is_empty() {
        return;
    }

    // Check if query is preference-seeking
    let query_lower = response.query.to_lowercase();
    let is_preference_query = query_lower.contains("suggest")
        || query_lower.contains("recommend")
        || query_lower.contains("should i")
        || query_lower.contains("what should")
        || query_lower.contains("prefer")
        || query_lower.contains("favorite")
        || query_lower.contains("best for me");

    if !is_preference_query {
        return;
    }

    // Compute boost scores for each hit
    let mut scored: Vec<(usize, f32, f32)> = response
        .hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| {
            let original_score = hit.score.unwrap_or(0.0);
            let preference_boost = compute_preference_boost(&hit.text);
            let boosted_score = original_score + preference_boost;
            (idx, boosted_score, original_score)
        })
        .collect();

    // Sort by boosted score (descending)
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal))
    });

    // Reorder hits
    let mut reordered = Vec::with_capacity(response.hits.len());
    for (rank_idx, (idx, _, _)) in scored.into_iter().enumerate() {
        let mut hit = response.hits[idx].clone();
        hit.rank = rank_idx + 1;
        reordered.push(hit);
    }

    response.hits = reordered;
}

/// Compute a boost score for hits that contain user preference signals.
/// This helps surface context where users express their preferences,
/// habits, or personal information that's relevant to recommendation queries.
///
/// Key insight: We want to distinguish content where the user describes
/// their ESTABLISHED situation/preferences (high boost) from content where
/// the user is making a REQUEST (low boost). Both use first-person language,
/// but they serve different purposes for personalization.
fn compute_preference_boost(text: &str) -> f32 {
    let text_lower = text.to_lowercase();
    let mut boost = 0.0f32;

    // Strong signals: Past/present user experiences and possessions
    // These describe what the user HAS DONE, HAS, or DOES REGULARLY
    let established_context = [
        // Past tense - indicates actual experience
        "i've been",
        "i've had",
        "i've used",
        "i've tried",
        "i recently",
        "i just",
        "lately",
        "i started",
        "i bought",
        "i harvested",
        "i grew",
        // Current possessions/ownership (indicates established context)
        "my garden",
        "my home",
        "my house",
        "my setup",
        "my equipment",
        "my camera",
        "my car",
        "my phone",
        "i have a",
        "i own",
        "i got a",
        // Established habits/preferences
        "i prefer",
        "i like to",
        "i love to",
        "i enjoy",
        "i usually",
        "i always",
        "i typically",
        "my favorite",
        "i tend to",
        "i often",
        // Regular activities (indicates ongoing behavior)
        "i use",
        "i grow",
        "i cook",
        "i make",
        "i work on",
        "i'm into",
        "i collect",
    ];
    for pattern in established_context {
        if text_lower.contains(pattern) {
            boost += 0.15;
        }
    }

    // Moderate signals: General first-person statements
    let first_person = [" i ", " my ", " me "];
    for pattern in first_person {
        if text_lower.contains(pattern) {
            boost += 0.02;
        }
    }

    // Weak signals: Requests/intentions (not yet established preferences)
    // These indicate the user wants something, but don't describe established context
    let request_patterns = [
        "i'm trying to",
        "i want to",
        "i need to",
        "looking for",
        "can you suggest",
        "can you help",
    ];
    for pattern in request_patterns {
        if text_lower.contains(pattern) {
            boost += 0.02;
        }
    }

    // Cap the boost to avoid over-weighting
    boost.min(0.5)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        sum_a += x * x;
        sum_b += y * y;
    }

    if sum_a <= f32::EPSILON || sum_b <= f32::EPSILON {
        0.0
    } else {
        dot / (sum_a.sqrt() * sum_b.sqrt())
    }
}

/// Apply cross-encoder reranking to search results.
///
/// Cross-encoders directly score query-document pairs and can understand
/// more nuanced relevance than bi-encoders (embeddings). This is especially
/// useful for personalization queries where semantic similarity != relevance.
///
/// Uses JINA-reranker-v1-turbo-en (~86MB model) for fast, high-quality reranking.
#[cfg(feature = "local-embeddings")]
fn apply_cross_encoder_rerank(response: &mut SearchResponse) -> Result<()> {
    if response.hits.is_empty() || response.hits.len() < 2 {
        return Ok(());
    }

    // Only rerank if we have enough candidates
    let candidates_to_rerank = response.hits.len().min(50);

    // Initialize the reranker (model will be downloaded on first use, ~86MB)
    // Using JINA Turbo - faster than BGE while maintaining good accuracy
    let options = RerankInitOptions::new(RerankerModel::JINARerankerV1TurboEn)
        .with_show_download_progress(true);

    let mut reranker = match TextRerank::try_new(options) {
        Ok(r) => r,
        Err(e) => {
            warn!("Failed to initialize cross-encoder reranker: {e}");
            return Ok(());
        }
    };

    // Prepare documents for reranking (owned Strings to avoid lifetime issues)
    let documents: Vec<String> = response.hits[..candidates_to_rerank]
        .iter()
        .map(|hit| hit.text.clone())
        .collect();

    // Rerank using cross-encoder
    info!("Cross-encoder reranking {} candidates", documents.len());
    let rerank_results = match reranker.rerank(response.query.clone(), documents, false, None) {
        Ok(results) => results,
        Err(e) => {
            warn!("Cross-encoder reranking failed: {e}");
            return Ok(());
        }
    };

    // Blend cross-encoder scores with original scores to preserve temporal boosting.
    // The original score includes recency boost; purely replacing it loses temporal relevance.
    // We collect (blended_score, original_idx) pairs and sort by blended score.
    let mut scored_hits: Vec<(f32, usize)> = Vec::with_capacity(rerank_results.len());

    // Find score range for normalization (original scores can be negative for BM25)
    let original_scores: Vec<f32> = response.hits[..candidates_to_rerank]
        .iter()
        .filter_map(|h| h.score)
        .collect();
    let orig_min = original_scores
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let orig_max = original_scores
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let orig_range = (orig_max - orig_min).max(0.001); // Avoid division by zero

    for result in rerank_results.iter() {
        let original_idx = result.index;
        let cross_encoder_score = result.score; // Already normalized 0-1

        // Normalize original score to 0-1 range
        let original_score = response.hits[original_idx].score.unwrap_or(0.0);
        let normalized_original = (original_score - orig_min) / orig_range;

        // Blend: 20% cross-encoder (relevance) + 80% original (includes temporal boost)
        // Very heavy weight on original score to preserve temporal ranking
        // The original score already incorporates BM25 + recency boost
        let blended = cross_encoder_score * 0.2 + normalized_original * 0.8;

        scored_hits.push((blended, original_idx));
    }

    // Sort by blended score (descending)
    scored_hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build reordered hits with new ranks
    let mut reordered = Vec::with_capacity(response.hits.len());
    for (new_rank, (blended_score, original_idx)) in scored_hits.into_iter().enumerate() {
        let mut hit = response.hits[original_idx].clone();
        hit.rank = new_rank + 1;
        // Store blended score for reference
        hit.score = Some(blended_score);
        reordered.push(hit);
    }

    // Add any remaining hits that weren't reranked (beyond top-50)
    for hit in response.hits.iter().skip(candidates_to_rerank) {
        let mut h = hit.clone();
        h.rank = reordered.len() + 1;
        reordered.push(h);
    }

    response.hits = reordered;
    info!("Cross-encoder reranking complete");
    Ok(())
}

/// Stub for cross-encoder reranking when local-embeddings is disabled.
/// Does nothing - reranking is skipped silently.
#[cfg(not(feature = "local-embeddings"))]
fn apply_cross_encoder_rerank(_response: &mut SearchResponse) -> Result<()> {
    Ok(())
}

/// Build a context string from memory cards stored in the MV2 file.
/// Groups facts by entity for better LLM comprehension.
fn build_memory_context(mem: &Memvid) -> String {
    let entities = mem.memory_entities();
    if entities.is_empty() {
        return String::new();
    }

    let mut sections = Vec::new();
    for entity in entities {
        let cards = mem.get_entity_memories(&entity);
        if cards.is_empty() {
            continue;
        }

        let mut entity_lines = Vec::new();
        for card in cards {
            // Format: "slot: value" with optional polarity indicator
            let polarity_marker = card
                .polarity
                .as_ref()
                .map(|p| match p.to_string().as_str() {
                    "Positive" => " (+)",
                    "Negative" => " (-)",
                    _ => "",
                })
                .unwrap_or("");
            entity_lines.push(format!(
                "  - {}: {}{}",
                card.slot, card.value, polarity_marker
            ));
        }

        sections.push(format!("{}:\n{}", entity, entity_lines.join("\n")));
    }

    sections.join("\n\n")
}

/// Build a context string from entities found in search hits.
/// Groups entities by type for better LLM comprehension.
fn build_entity_context_from_hits(hits: &[SearchHit]) -> String {
    use std::collections::HashMap;

    // Collect unique entities by kind
    let mut entities_by_kind: HashMap<String, Vec<String>> = HashMap::new();

    for hit in hits {
        if let Some(metadata) = &hit.metadata {
            for entity in &metadata.entities {
                entities_by_kind
                    .entry(entity.kind.clone())
                    .or_default()
                    .push(entity.name.clone());
            }
        }
    }

    if entities_by_kind.is_empty() {
        return String::new();
    }

    // Deduplicate and format
    let mut sections = Vec::new();
    let mut sorted_kinds: Vec<_> = entities_by_kind.keys().collect();
    sorted_kinds.sort();

    for kind in sorted_kinds {
        let names = entities_by_kind.get(kind).unwrap();
        let mut unique_names: Vec<_> = names.iter().collect();
        unique_names.sort();
        unique_names.dedup();

        let names_str = unique_names
            .iter()
            .take(10) // Limit to 10 entities per kind
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        sections.push(format!("{}: {}", kind, names_str));
    }

    sections.join("\n")
}
