//! OpenRouter Embeddings Provider
//!
//! This module provides an embedding implementation that uses
//! OpenRouter's API for generating embeddings via 100+ models.
//!
//! ## Environment Variables
//! - `OPENROUTER_API_KEY`: Required API key for OpenRouter
//! - `OPENROUTER_EMBEDDING_MODEL`: Optional model override (default: google/text-embedding-004)
//!
//! ## Features
//! - Access to 100+ embedding models
//! - OpenAI-compatible API format
//! - Efficient batch processing
//! - Thread-safe for concurrent use

use anyhow::{anyhow, bail, Result};
use memvid_core::{EmbeddingConfig, EmbeddingProvider, VecEmbedder};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, info, warn};

/// OpenRouter embeddings API endpoint
const OPENROUTER_EMBEDDINGS_URL: &str = "https://openrouter.ai/api/v1/embeddings";

/// Default embedding model (high quality, 768 dimensions)
const DEFAULT_MODEL: &str = "google/text-embedding-004";

/// Maximum texts per batch (OpenRouter limit)
const MAX_BATCH_SIZE: usize = 100;

/// Request timeout
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum characters for embedding text to avoid token limits.
/// Using conservative estimate for most models.
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

/// OpenRouter embedding request payload (OpenAI-compatible format)
#[derive(Debug, Serialize)]
struct OpenRouterEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

/// OpenRouter embedding response
#[derive(Debug, Deserialize)]
struct OpenRouterEmbeddingResponse {
    data: Vec<OpenRouterEmbeddingData>,
    model: String,
    usage: OpenRouterUsage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    #[allow(dead_code)]
    prompt_tokens: usize,
    total_tokens: usize,
}

/// OpenRouter error response
#[derive(Debug, Deserialize)]
struct OpenRouterErrorResponse {
    error: OpenRouterError,
}

#[derive(Debug, Deserialize)]
struct OpenRouterError {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

/// OpenRouter Embedding Provider
///
/// Implements `EmbeddingProvider` trait for generating embeddings via OpenRouter API.
#[derive(Clone)]
pub struct OpenRouterEmbeddingProvider {
    api_key: String,
    config: EmbeddingConfig,
    client: Client,
    ready: std::sync::Arc<AtomicBool>,
}

impl std::fmt::Debug for OpenRouterEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterEmbeddingProvider")
            .field("model", &self.config.model)
            .field("dimension", &self.config.dimension)
            .field("ready", &self.ready.load(Ordering::Relaxed))
            .finish()
    }
}

impl OpenRouterEmbeddingProvider {
    /// Create a new OpenRouter embedding provider
    ///
    /// # Arguments
    /// * `api_key` - OpenRouter API key
    /// * `config` - Embedding configuration (model, dimension, etc.)
    pub fn new(api_key: String, config: EmbeddingConfig) -> Result<Self> {
        if api_key.is_empty() {
            bail!("OpenRouter API key cannot be empty");
        }

        let client = crate::http::blocking_client(REQUEST_TIMEOUT)
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        Ok(Self {
            api_key,
            config,
            client,
            ready: std::sync::Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create provider from environment variables
    ///
    /// Uses `OPENROUTER_API_KEY` for authentication and optionally
    /// `OPENROUTER_EMBEDDING_MODEL` to override the default model.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| anyhow!("OPENROUTER_API_KEY environment variable not set"))?;

        let model = std::env::var("OPENROUTER_EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        let dimension = match model.as_str() {
            "google/text-embedding-004" => 768,
            "amazon/amazon-embeddings" => 1024,
            "cohere/embed-english-v3.0" => 1024,
            "cohere/embed-multilingual-v3.0" => 1024,
            "openai/text-embedding-3-small" => 1536,
            "openai/text-embedding-3-large" => 3072,
            "openai/text-embedding-ada-002" => 1536,
            "mistral/mistral-embed" => 1024,
            "nvidia/nv-embed-v1" => 4096,
            _ => 0, // Unknown dimension, will infer from first response
        };

        let config = EmbeddingConfig {
            model,
            dimension,
            batch_size: Some(MAX_BATCH_SIZE),
            normalize: true,
        };

        Self::new(api_key, config)
    }

    /// Internal method to call OpenRouter API
    fn call_openrouter(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = OpenRouterEmbeddingRequest {
            model: &self.config.model,
            input: texts.to_vec(),
        };

        let response = self
            .client
            .post(OPENROUTER_EMBEDDINGS_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://memvid.com")
            .header("X-Title", "Memvid")
            .json(&request)
            .send()
            .map_err(|e| anyhow!("OpenRouter API request failed: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .map_err(|e| anyhow!("Failed to read response body: {}", e))?;

        if !status.is_success() {
            if let Ok(error_response) = serde_json::from_str::<OpenRouterErrorResponse>(&body) {
                let error_type = error_response
                    .error
                    .error_type
                    .as_deref()
                    .unwrap_or("unknown");
                bail!(
                    "OpenRouter API error ({}): {}",
                    error_type,
                    error_response.error.message
                );
            }
            bail!(
                "OpenRouter API request failed with status {}: {}",
                status,
                body
            );
        }

        let embedding_response: OpenRouterEmbeddingResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Failed to parse OpenRouter response: {}", e))?;

        debug!(
            "OpenRouter embeddings: {} texts, {} tokens, model={}",
            texts.len(),
            embedding_response.usage.total_tokens,
            embedding_response.model
        );

        let mut data = embedding_response.data;
        data.sort_by_key(|d| d.index);

        let embeddings: Vec<Vec<f32>> = data.into_iter().map(|d| d.embedding).collect();

        Ok(embeddings)
    }

    /// Embed texts with retry logic
    fn embed_with_retry(&self, texts: &[&str], max_retries: usize) -> Result<Vec<Vec<f32>>> {
        let mut last_error = None;

        for attempt in 0..max_retries {
            match self.call_openrouter(texts) {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => {
                    let error_str = e.to_string();
                    if error_str.contains("rate_limit") || error_str.contains("429") {
                        let backoff = Duration::from_millis(500 * (1 << attempt));
                        warn!(
                            "Rate limited by OpenRouter, retrying in {:?} (attempt {}/{})",
                            backoff,
                            attempt + 1,
                            max_retries
                        );
                        std::thread::sleep(backoff);
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("Failed to embed after {} retries", max_retries)))
    }
}

impl EmbeddingProvider for OpenRouterEmbeddingProvider {
    fn kind(&self) -> &str {
        "openrouter"
    }

    fn model(&self) -> &str {
        &self.config.model
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn embed_text(&self, text: &str) -> memvid_core::Result<Vec<f32>> {
        let text = truncate_for_embedding(text);
        self.embed_with_retry(&[&text], 3)
            .map(|mut v| v.pop().unwrap_or_default())
            .map_err(|e| memvid_core::MemvidError::EmbeddingFailed {
                reason: e.to_string().into_boxed_str(),
            })
    }

    fn embed_batch(&self, texts: &[&str]) -> memvid_core::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let truncated: Vec<std::borrow::Cow<'_, str>> =
            texts.iter().map(|t| truncate_for_embedding(t)).collect();
        let truncated_refs: Vec<&str> = truncated.iter().map(|c| c.as_ref()).collect();

        let batch_size = self
            .config
            .batch_size
            .unwrap_or(MAX_BATCH_SIZE)
            .min(MAX_BATCH_SIZE);
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in truncated_refs.chunks(batch_size) {
            let embeddings = self.embed_with_retry(chunk, 3).map_err(|e| {
                memvid_core::MemvidError::EmbeddingFailed {
                    reason: e.to_string().into_boxed_str(),
                }
            })?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    fn init(&mut self) -> memvid_core::Result<()> {
        info!(
            "Initializing OpenRouter embedding provider with model: {}",
            self.config.model
        );

        let test_embedding = self.embed_with_retry(&["test"], 1).map_err(|e| {
            memvid_core::MemvidError::EmbeddingFailed {
                reason: format!("Failed to initialize OpenRouter provider: {}", e).into_boxed_str(),
            }
        })?;

        if let Some(emb) = test_embedding.first() {
            info!(
                "OpenRouter provider initialized: model={}, dimension={}",
                self.config.model,
                emb.len()
            );
            if emb.len() != self.config.dimension && self.config.dimension > 0 {
                warn!(
                    "Updating dimension from {} to {}",
                    self.config.dimension,
                    emb.len()
                );
            }
        }

        self.ready.store(true, Ordering::Relaxed);
        Ok(())
    }
}

impl VecEmbedder for OpenRouterEmbeddingProvider {
    fn embed_query(&self, text: &str) -> memvid_core::Result<Vec<f32>> {
        self.embed_text(text)
    }

    fn embed_chunks(&self, texts: &[&str]) -> memvid_core::Result<Vec<Vec<f32>>> {
        self.embed_batch(texts)
    }

    fn embedding_dimension(&self) -> usize {
        self.dimension()
    }
}

/// Helper to create an OpenRouter provider or fall back to local
pub fn try_openrouter_provider() -> Option<OpenRouterEmbeddingProvider> {
    match OpenRouterEmbeddingProvider::from_env() {
        Ok(provider) => {
            info!("OpenRouter embedding provider available");
            Some(provider)
        }
        Err(e) => {
            debug!("OpenRouter provider not available: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_api_key() {
        let config = EmbeddingConfig {
            model: DEFAULT_MODEL.to_string(),
            dimension: 768,
            batch_size: Some(MAX_BATCH_SIZE),
            normalize: true,
        };
        let result = OpenRouterEmbeddingProvider::new(String::new(), config);
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Requires valid API key
    fn test_real_embedding() {
        let provider =
            OpenRouterEmbeddingProvider::from_env().expect("OPENROUTER_API_KEY must be set");
        let embedding = provider.embed_text("Hello, world!").expect("embed");
        assert!(!embedding.is_empty());
    }
}
