//! NVIDIA Embeddings Provider
//!
//! Implements high-quality embeddings via NVIDIA Integrate API (`/v1/embeddings`).
//!
//! ## Environment Variables
//! - `NVIDIA_API_KEY`: Required
//! - `NVIDIA_BASE_URL`: Optional (default: https://integrate.api.nvidia.com)
//! - `NVIDIA_EMBEDDING_MODEL`: Optional (default: nvidia/nv-embed-v1)
//! - `NVIDIA_EMBEDDING_BATCH_SIZE`: Optional (default: 64)
//!
//! The NVIDIA API supports different `input_type` values. Memvid uses:
//! - `passage` for document/chunk embeddings
//! - `query` for query embeddings

use anyhow::{anyhow, bail, Result};
use reqwest::blocking::Client;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::warn;

const DEFAULT_NVIDIA_BASE_URL: &str = "https://integrate.api.nvidia.com";
const DEFAULT_NVIDIA_EMBEDDING_MODEL: &str = "nvidia/nv-embed-v1";

const DEFAULT_BATCH_SIZE: usize = 64;
const MAX_BATCH_SIZE: usize = 256;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

fn truncate_to_chars(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }

    let truncated = &text[..max_chars];
    let end = truncated
        .char_indices()
        .rev()
        .next()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(max_chars);
    text[..end].to_string()
}

fn extract_error_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if let Some(message) = value.get("error").and_then(|v| v.as_str()) {
        return Some(message.to_string());
    }
    if let Some(message) = value.get("message").and_then(|v| v.as_str()) {
        return Some(message.to_string());
    }
    None
}

fn parse_token_limit_error(message: &str) -> Option<(usize, usize)> {
    if !message
        .to_ascii_lowercase()
        .contains("exceeds maximum allowed token size")
    {
        return None;
    }

    let mut numbers = Vec::new();
    let mut current = String::new();
    for ch in message.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            if let Ok(value) = current.parse::<usize>() {
                numbers.push(value);
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        if let Ok(value) = current.parse::<usize>() {
            numbers.push(value);
        }
    }

    if numbers.len() >= 2 {
        Some((numbers[0], numbers[1]))
    } else {
        None
    }
}

#[derive(Debug, Serialize)]
struct NvidiaEmbeddingRequest<'a> {
    input: Vec<&'a str>,
    model: &'a str,
    #[serde(rename = "input_type")]
    input_type: &'a str,
    #[serde(rename = "encoding_format")]
    encoding_format: &'a str,
    truncate: &'a str,
}

#[derive(Debug, Deserialize)]
struct NvidiaEmbeddingResponse {
    data: Vec<NvidiaEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct NvidiaEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Clone, Debug)]
pub struct NvidiaEmbeddingProvider {
    api_key: String,
    base_url: String,
    model: String,
    batch_size: usize,
    document_input_type: String,
    query_input_type: String,
    encoding_format: String,
    truncate: String,
    client: Client,
}

impl NvidiaEmbeddingProvider {
    pub fn from_env(explicit_model_override: Option<&str>) -> Result<Self> {
        let api_key = std::env::var("NVIDIA_API_KEY").map_err(|_| {
            anyhow!("NVIDIA_API_KEY environment variable is required for NVIDIA embeddings")
        })?;
        if api_key.trim().is_empty() {
            bail!("NVIDIA_API_KEY cannot be empty");
        }

        let base_url = std::env::var("NVIDIA_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_NVIDIA_BASE_URL.to_string());
        let base_url = base_url.trim().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            bail!("NVIDIA_BASE_URL cannot be empty");
        }

        let model = explicit_model_override
            .and_then(|value| {
                let trimmed = value.trim();
                (!trimmed.is_empty()).then_some(trimmed.to_string())
            })
            .or_else(|| {
                std::env::var("NVIDIA_EMBEDDING_MODEL")
                    .ok()
                    .map(|s| s.trim().to_string())
            })
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_NVIDIA_EMBEDDING_MODEL.to_string());

        let batch_size = std::env::var("NVIDIA_EMBEDDING_BATCH_SIZE")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_BATCH_SIZE)
            .clamp(1, MAX_BATCH_SIZE);

        let client = crate::http::blocking_client(REQUEST_TIMEOUT)
            .map_err(|err| anyhow!("Failed to create HTTP client: {err}"))?;

        let truncate = std::env::var("NVIDIA_EMBEDDING_TRUNCATE")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "NONE".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            batch_size,
            document_input_type: "passage".to_string(),
            query_input_type: "query".to_string(),
            encoding_format: "float".to_string(),
            truncate,
            client,
        })
    }

    pub fn kind(&self) -> &'static str {
        "nvidia"
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_retry(&self.document_input_type, texts, 3)
    }

    pub fn embed_queries(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_retry(&self.query_input_type, texts, 3)
    }

    pub fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_passages(&[text])?;
        out.pop()
            .ok_or_else(|| anyhow!("NVIDIA embeddings API returned no embedding output"))
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_queries(&[text])?;
        out.pop()
            .ok_or_else(|| anyhow!("NVIDIA embeddings API returned no embedding output"))
    }

    fn embeddings_url(&self) -> String {
        format!("{}/v1/embeddings", self.base_url)
    }

    fn embed_batch_with_retry(
        &self,
        input_type: &str,
        texts: &[&str],
        max_retries: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let embeddings = self.call_nvidia_with_retry(input_type, chunk, max_retries)?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    fn call_nvidia_with_retry(
        &self,
        input_type: &str,
        texts: &[&str],
        max_retries: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let url = self.embeddings_url();
        let request = NvidiaEmbeddingRequest {
            input: texts.to_vec(),
            model: &self.model,
            input_type,
            encoding_format: &self.encoding_format,
            truncate: &self.truncate,
        };

        let mut attempt = 0usize;
        let mut backoff = Duration::from_millis(500);
        let max_backoff = Duration::from_secs(8);

        loop {
            attempt += 1;
            let response = self
                .client
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&request)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().unwrap_or_default();

                    if status.is_success() {
                        let mut decoded: NvidiaEmbeddingResponse = serde_json::from_str(&body)
                            .map_err(|err| {
                                anyhow!("failed to decode NVIDIA embeddings response: {err}")
                            })?;
                        decoded.data.sort_by_key(|item| item.index);

                        if decoded.data.len() != texts.len() {
                            bail!(
                                "NVIDIA embeddings API returned {} embeddings for {} inputs",
                                decoded.data.len(),
                                texts.len()
                            );
                        }

                        let embeddings: Vec<Vec<f32>> = decoded
                            .data
                            .into_iter()
                            .map(|item| item.embedding)
                            .collect();

                        if embeddings.iter().any(|emb| emb.is_empty()) {
                            bail!("NVIDIA embeddings API returned an empty embedding vector");
                        }

                        return Ok(embeddings);
                    }

                    let retryable =
                        status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                    if retryable && attempt <= max_retries {
                        warn!(
                            "NVIDIA embeddings API returned {status} (attempt {attempt}/{max_attempts}); retrying in {backoff:?}: {body}",
                            max_attempts = max_retries + 1
                        );
                        std::thread::sleep(backoff);
                        backoff = (backoff * 2).min(max_backoff);
                        continue;
                    }

                    if status == StatusCode::BAD_REQUEST {
                        if let Some(message) = extract_error_message(&body) {
                            if let Some((actual, max)) = parse_token_limit_error(&message) {
                                let mut factor =
                                    (max as f64 / actual.max(1) as f64).clamp(0.05, 0.95) * 0.95;
                                warn!(
                                    "NVIDIA embeddings input exceeds token limit ({actual} > {max}); retrying with automatic truncation"
                                );

                                for _ in 0..3 {
                                    let owned: Vec<String> = texts
                                        .iter()
                                        .map(|text| {
                                            let target =
                                                ((text.len() as f64) * factor).floor() as usize;
                                            truncate_to_chars(text, target.max(256))
                                        })
                                        .collect();
                                    let refs: Vec<&str> =
                                        owned.iter().map(|text| text.as_str()).collect();
                                    let request = NvidiaEmbeddingRequest {
                                        input: refs,
                                        model: &self.model,
                                        input_type,
                                        encoding_format: &self.encoding_format,
                                        truncate: &self.truncate,
                                    };

                                    let resp = self
                                        .client
                                        .post(&url)
                                        .bearer_auth(&self.api_key)
                                        .json(&request)
                                        .send()
                                        .map_err(|err| {
                                            anyhow!("NVIDIA embeddings request failed: {err}")
                                        })?;

                                    let status = resp.status();
                                    let body = resp.text().unwrap_or_default();
                                    if status.is_success() {
                                        let mut decoded: NvidiaEmbeddingResponse =
                                            serde_json::from_str(&body).map_err(|err| {
                                                anyhow!(
                                                    "failed to decode NVIDIA embeddings response: {err}"
                                                )
                                            })?;
                                        decoded.data.sort_by_key(|item| item.index);

                                        if decoded.data.len() != texts.len() {
                                            bail!(
                                                "NVIDIA embeddings API returned {} embeddings for {} inputs",
                                                decoded.data.len(),
                                                texts.len()
                                            );
                                        }

                                        let embeddings: Vec<Vec<f32>> = decoded
                                            .data
                                            .into_iter()
                                            .map(|item| item.embedding)
                                            .collect();

                                        if embeddings.iter().any(|emb| emb.is_empty()) {
                                            bail!(
                                                "NVIDIA embeddings API returned an empty embedding vector"
                                            );
                                        }

                                        return Ok(embeddings);
                                    }

                                    if status == StatusCode::BAD_REQUEST {
                                        if let Some(message) = extract_error_message(&body) {
                                            if parse_token_limit_error(&message).is_some() {
                                                factor = (factor * 0.85).clamp(0.02, 0.8);
                                                continue;
                                            }
                                        }
                                    }

                                    bail!(
                                        "NVIDIA embeddings API returned error status {status}: {body}"
                                    );
                                }

                                bail!(
                                    "NVIDIA embeddings input exceeds token limit and could not be truncated automatically.\n\
                                     Try enabling smaller chunks (or disable contextual prefixes) and retry. You can also set NVIDIA_EMBEDDING_TRUNCATE=END if your model supports server-side truncation."
                                );
                            }
                        }
                    }

                    bail!("NVIDIA embeddings API returned error status {status}: {body}");
                }
                Err(err) => {
                    let retryable = err.is_timeout() || err.is_connect();
                    if retryable && attempt <= max_retries {
                        warn!(
                            "NVIDIA embeddings request failed (attempt {attempt}/{max_attempts}); retrying in {backoff:?}: {err}",
                            max_attempts = max_retries + 1
                        );
                        std::thread::sleep(backoff);
                        backoff = (backoff * 2).min(max_backoff);
                        continue;
                    }

                    bail!("NVIDIA embeddings request failed: {err}");
                }
            }
        }
    }
}
