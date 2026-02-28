//! Gemini (Google AI) Embeddings Provider
//!
//! This module provides an `EmbeddingProvider` implementation that uses
//! Google's Gemini API for generating high-quality embeddings.
//!
//! ## Environment Variables
//! - `GOOGLE_API_KEY` or `GEMINI_API_KEY`: Required API key for Google AI
//! - `GEMINI_EMBEDDING_MODEL`: Optional model override (default: text-embedding-004)
//!
//! ## Features
//! - Supports all Gemini embedding models
//! - Efficient batch processing
//! - Thread-safe for concurrent use

use anyhow::{anyhow, bail, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Gemini embeddings API base URL
const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Default embedding model
const DEFAULT_MODEL: &str = "text-embedding-004";

/// Maximum texts per batch
const MAX_BATCH_SIZE: usize = 100;

/// Request timeout
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum characters for embedding text to avoid exceeding token limits.
const MAX_EMBEDDING_TEXT_LEN: usize = 20_000;

/// Truncate text to MAX_EMBEDDING_TEXT_LEN to avoid token limit errors.
fn truncate_for_embedding(text: &str) -> std::borrow::Cow<'_, str> {
    if text.len() <= MAX_EMBEDDING_TEXT_LEN {
        std::borrow::Cow::Borrowed(text)
    } else {
        let end = text[..MAX_EMBEDDING_TEXT_LEN]
            .char_indices()
            .rev()
            .next()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(MAX_EMBEDDING_TEXT_LEN);
        warn!(
            "Truncating embedding text from {} to {} chars to avoid token limit",
            text.len(),
            end
        );
        std::borrow::Cow::Owned(text[..end].to_string())
    }
}

/// Gemini embed content request
#[derive(Debug, Serialize)]
struct GeminiEmbedRequest {
    content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
}

/// Gemini batch embed request
#[derive(Debug, Serialize)]
struct GeminiBatchEmbedRequest {
    requests: Vec<GeminiEmbedRequestItem>,
}

#[derive(Debug, Serialize)]
struct GeminiEmbedRequestItem {
    model: String,
    content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

/// Gemini embed response
#[derive(Debug, Deserialize)]
struct GeminiEmbedResponse {
    embedding: GeminiEmbedding,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}

/// Gemini batch embed response
#[derive(Debug, Deserialize)]
struct GeminiBatchEmbedResponse {
    embeddings: Vec<GeminiEmbedding>,
}

/// Gemini error response
#[derive(Debug, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    message: String,
    code: i32,
}

/// Gemini Embedding Provider
#[derive(Clone)]
pub struct GeminiEmbeddingProvider {
    api_key: String,
    model: String,
    client: Client,
    dimension: usize,
}

impl std::fmt::Debug for GeminiEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiEmbeddingProvider")
            .field("model", &self.model)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl GeminiEmbeddingProvider {
    /// Create a new Gemini embedding provider
    pub fn new(api_key: String, model: Option<&str>) -> Result<Self> {
        if api_key.is_empty() {
            bail!("Gemini API key cannot be empty");
        }

        let client = crate::http::blocking_client(REQUEST_TIMEOUT)
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let model = model.unwrap_or(DEFAULT_MODEL).to_string();

        // Dimension depends on model:
        // text-embedding-004: 768 (default, can be reduced)
        // gemini-embedding-001: 3072 (can be truncated)
        let dimension = if model.contains("gemini-embedding") {
            3072
        } else {
            768 // text-embedding-004 default
        };

        Ok(Self {
            api_key,
            model,
            client,
            dimension,
        })
    }

    /// Create provider from environment variables
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| {
                anyhow!("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
            })?;

        let model = std::env::var("GEMINI_EMBEDDING_MODEL").ok();
        Self::new(api_key, model.as_deref())
    }

    /// Get model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get provider kind
    pub fn kind(&self) -> &'static str {
        "gemini"
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Embed a single text
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let text = truncate_for_embedding(text);
        self.embed_with_retry(&text, 3)
    }

    /// Embed multiple texts in batch
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Truncate all texts first
        let truncated: Vec<std::borrow::Cow<'_, str>> =
            texts.iter().map(|t| truncate_for_embedding(t)).collect();

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in batches
        for chunk in truncated.chunks(MAX_BATCH_SIZE) {
            let embeddings = self.embed_batch_with_retry(chunk, 3)?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embed single text with retry logic
    fn embed_with_retry(&self, text: &str, max_retries: usize) -> Result<Vec<f32>> {
        let url = format!(
            "{}/{}:embedContent?key={}",
            GEMINI_API_BASE, self.model, self.api_key
        );

        let request = GeminiEmbedRequest {
            content: GeminiContent {
                parts: vec![GeminiPart {
                    text: text.to_string(),
                }],
            },
            task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
        };

        let mut last_error = None;

        for attempt in 0..max_retries {
            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().unwrap_or_default();

                    if status.is_success() {
                        let embed_response: GeminiEmbedResponse = serde_json::from_str(&body)
                            .map_err(|e| anyhow!("Failed to parse Gemini response: {}", e))?;

                        debug!(
                            "Gemini embedding: {} values, model={}",
                            embed_response.embedding.values.len(),
                            self.model
                        );

                        return Ok(embed_response.embedding.values);
                    }

                    // Check for rate limiting
                    if status.as_u16() == 429 {
                        let backoff = Duration::from_millis(500 * (1 << attempt));
                        warn!(
                            "Rate limited by Gemini, retrying in {:?} (attempt {}/{})",
                            backoff,
                            attempt + 1,
                            max_retries
                        );
                        std::thread::sleep(backoff);
                        last_error = Some(anyhow!("Rate limited"));
                        continue;
                    }

                    // Try to parse error response
                    if let Ok(error_response) = serde_json::from_str::<GeminiErrorResponse>(&body) {
                        return Err(anyhow!(
                            "Gemini API error ({}): {}",
                            error_response.error.code,
                            error_response.error.message
                        ));
                    }

                    return Err(anyhow!(
                        "Gemini API request failed with status {}: {}",
                        status,
                        body
                    ));
                }
                Err(e) => {
                    if attempt < max_retries - 1 {
                        let backoff = Duration::from_millis(500 * (1 << attempt));
                        warn!(
                            "Gemini request failed, retrying in {:?} (attempt {}/{}): {}",
                            backoff,
                            attempt + 1,
                            max_retries,
                            e
                        );
                        std::thread::sleep(backoff);
                        last_error = Some(anyhow!("Request failed: {}", e));
                        continue;
                    }
                    return Err(anyhow!("Gemini API request failed: {}", e));
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("Failed to embed after {} retries", max_retries)))
    }

    /// Embed batch with retry logic
    fn embed_batch_with_retry(
        &self,
        texts: &[std::borrow::Cow<'_, str>],
        max_retries: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "{}/{}:batchEmbedContents?key={}",
            GEMINI_API_BASE, self.model, self.api_key
        );

        let requests: Vec<GeminiEmbedRequestItem> = texts
            .iter()
            .map(|text| GeminiEmbedRequestItem {
                model: format!("models/{}", self.model),
                content: GeminiContent {
                    parts: vec![GeminiPart {
                        text: text.to_string(),
                    }],
                },
                task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
            })
            .collect();

        let batch_request = GeminiBatchEmbedRequest { requests };

        let mut last_error = None;

        for attempt in 0..max_retries {
            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&batch_request)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().unwrap_or_default();

                    if status.is_success() {
                        let batch_response: GeminiBatchEmbedResponse = serde_json::from_str(&body)
                            .map_err(|e| anyhow!("Failed to parse Gemini batch response: {}", e))?;

                        debug!(
                            "Gemini batch embeddings: {} texts, model={}",
                            batch_response.embeddings.len(),
                            self.model
                        );

                        return Ok(batch_response
                            .embeddings
                            .into_iter()
                            .map(|e| e.values)
                            .collect());
                    }

                    // Check for rate limiting
                    if status.as_u16() == 429 {
                        let backoff = Duration::from_millis(500 * (1 << attempt));
                        warn!(
                            "Rate limited by Gemini, retrying in {:?} (attempt {}/{})",
                            backoff,
                            attempt + 1,
                            max_retries
                        );
                        std::thread::sleep(backoff);
                        last_error = Some(anyhow!("Rate limited"));
                        continue;
                    }

                    // Try to parse error response
                    if let Ok(error_response) = serde_json::from_str::<GeminiErrorResponse>(&body) {
                        return Err(anyhow!(
                            "Gemini API error ({}): {}",
                            error_response.error.code,
                            error_response.error.message
                        ));
                    }

                    return Err(anyhow!(
                        "Gemini API request failed with status {}: {}",
                        status,
                        body
                    ));
                }
                Err(e) => {
                    if attempt < max_retries - 1 {
                        let backoff = Duration::from_millis(500 * (1 << attempt));
                        warn!(
                            "Gemini batch request failed, retrying in {:?} (attempt {}/{}): {}",
                            backoff,
                            attempt + 1,
                            max_retries,
                            e
                        );
                        std::thread::sleep(backoff);
                        last_error = Some(anyhow!("Request failed: {}", e));
                        continue;
                    }
                    return Err(anyhow!("Gemini API batch request failed: {}", e));
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| anyhow!("Failed to embed batch after {} retries", max_retries)))
    }
}

/// Helper to create a Gemini provider or return error
pub fn try_gemini_provider() -> Option<GeminiEmbeddingProvider> {
    match GeminiEmbeddingProvider::from_env() {
        Ok(provider) => {
            info!("Gemini embedding provider available");
            Some(provider)
        }
        Err(e) => {
            debug!("Gemini provider not available: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_api_key() {
        let result = GeminiEmbeddingProvider::new(String::new(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_dimensions() {
        let provider = GeminiEmbeddingProvider::new("test-key".to_string(), None).unwrap();
        assert_eq!(provider.dimension(), 768);

        let provider =
            GeminiEmbeddingProvider::new("test-key".to_string(), Some("gemini-embedding-001"))
                .unwrap();
        assert_eq!(provider.dimension(), 3072);
    }

    #[test]
    #[ignore] // Requires valid API key
    fn test_real_embedding() {
        let provider = GeminiEmbeddingProvider::from_env().expect("GOOGLE_API_KEY must be set");
        let embedding = provider.embed_text("Hello, world!").expect("embed");
        assert!(!embedding.is_empty());
    }
}
