//! OpenAI Embeddings Provider
//!
//! This module provides an `EmbeddingProvider` implementation that uses
//! OpenAI's text-embedding API for generating high-quality embeddings.
//!
//! ## Environment Variables
//! - `OPENAI_API_KEY`: Required API key for OpenAI
//! - `OPENAI_EMBEDDING_MODEL`: Optional model override (default: text-embedding-3-large)
//!
//! ## Features
//! - Supports all OpenAI embedding models
//! - Efficient batch processing (up to 100 texts per request)
//! - Automatic rate limiting with exponential backoff
//! - Thread-safe for concurrent use

use anyhow::{anyhow, bail, Result};
use memvid_core::{EmbeddingConfig, EmbeddingProvider, VecEmbedder};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, info, warn};

/// OpenAI embeddings API endpoint
const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";

/// Maximum texts per batch (OpenAI limit)
const MAX_BATCH_SIZE: usize = 100;

/// Request timeout
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum characters for embedding text to avoid exceeding OpenAI's 8192 token limit.
/// Using ~3 chars/token estimate (conservative for dense content), 20K chars ≈ 6.6K tokens.
const MAX_EMBEDDING_TEXT_LEN: usize = 20_000;

/// Truncate text to MAX_EMBEDDING_TEXT_LEN to avoid token limit errors.
fn truncate_for_embedding(text: &str) -> std::borrow::Cow<'_, str> {
    if text.len() <= MAX_EMBEDDING_TEXT_LEN {
        std::borrow::Cow::Borrowed(text)
    } else {
        // Find a safe char boundary
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

/// OpenAI embedding request payload
#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

/// OpenAI embedding response
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    #[allow(dead_code)]
    prompt_tokens: usize,
    total_tokens: usize,
}

/// OpenAI error response
#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

#[derive(Debug, Deserialize)]
struct OpenAIError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

/// OpenAI Embedding Provider
///
/// Implements `EmbeddingProvider` trait for generating embeddings via OpenAI API.
#[derive(Clone)]
pub struct OpenAIEmbeddingProvider {
    api_key: String,
    config: EmbeddingConfig,
    client: Client,
    ready: std::sync::Arc<AtomicBool>,
}

impl std::fmt::Debug for OpenAIEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbeddingProvider")
            .field("model", &self.config.model)
            .field("dimension", &self.config.dimension)
            .field("ready", &self.ready.load(Ordering::Relaxed))
            .finish()
    }
}

impl OpenAIEmbeddingProvider {
    /// Create a new OpenAI embedding provider
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key
    /// * `config` - Embedding configuration (model, dimension, etc.)
    ///
    /// # Example
    /// ```ignore
    /// let provider = OpenAIEmbeddingProvider::new(
    ///     "sk-...".to_string(),
    ///     EmbeddingConfig::openai_large(),
    /// )?;
    /// ```
    pub fn new(api_key: String, config: EmbeddingConfig) -> Result<Self> {
        if api_key.is_empty() {
            bail!("OpenAI API key cannot be empty");
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
    /// Uses `OPENAI_API_KEY` for authentication and optionally
    /// `OPENAI_EMBEDDING_MODEL` to override the default model.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow!("OPENAI_API_KEY environment variable not set"))?;

        let config = match std::env::var("OPENAI_EMBEDDING_MODEL") {
            Ok(model) => match model.as_str() {
                "text-embedding-3-small" => EmbeddingConfig::openai_small(),
                "text-embedding-ada-002" => EmbeddingConfig::openai_ada(),
                "text-embedding-3-large" | _ => EmbeddingConfig::openai_large(),
            },
            Err(_) => EmbeddingConfig::openai_large(),
        };

        Self::new(api_key, config)
    }

    /// Create provider with text-embedding-3-large (default, highest quality)
    pub fn large(api_key: String) -> Result<Self> {
        Self::new(api_key, EmbeddingConfig::openai_large())
    }

    /// Create provider with text-embedding-3-small (faster, lower cost)
    pub fn small(api_key: String) -> Result<Self> {
        Self::new(api_key, EmbeddingConfig::openai_small())
    }

    /// Create provider with text-embedding-ada-002 (legacy)
    pub fn ada(api_key: String) -> Result<Self> {
        Self::new(api_key, EmbeddingConfig::openai_ada())
    }

    /// Internal method to call OpenAI API
    fn call_openai(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = OpenAIEmbeddingRequest {
            model: &self.config.model,
            input: texts.to_vec(),
            dimensions: None, // Use model's native dimension
        };

        let response = self
            .client
            .post(OPENAI_EMBEDDINGS_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| anyhow!("OpenAI API request failed: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .map_err(|e| anyhow!("Failed to read response body: {}", e))?;

        if !status.is_success() {
            // Try to parse error response
            if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&body) {
                bail!(
                    "OpenAI API error ({}): {}",
                    error_response.error.error_type,
                    error_response.error.message
                );
            }
            bail!("OpenAI API request failed with status {}: {}", status, body);
        }

        let embedding_response: OpenAIEmbeddingResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Failed to parse OpenAI response: {}", e))?;

        debug!(
            "OpenAI embeddings: {} texts, {} tokens, model={}",
            texts.len(),
            embedding_response.usage.total_tokens,
            embedding_response.model
        );

        // Sort by index and extract embeddings
        let mut data = embedding_response.data;
        data.sort_by_key(|d| d.index);

        let embeddings: Vec<Vec<f32>> = data.into_iter().map(|d| d.embedding).collect();

        // Validate dimensions
        if let Some(first) = embeddings.first() {
            if first.len() != self.config.dimension {
                warn!(
                    "OpenAI returned dimension {} but expected {}",
                    first.len(),
                    self.config.dimension
                );
            }
        }

        Ok(embeddings)
    }

    /// Embed texts with retry logic
    fn embed_with_retry(&self, texts: &[&str], max_retries: usize) -> Result<Vec<Vec<f32>>> {
        let mut last_error = None;

        for attempt in 0..max_retries {
            match self.call_openai(texts) {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => {
                    let error_str = e.to_string();
                    if error_str.contains("rate_limit") || error_str.contains("429") {
                        let backoff = Duration::from_millis(500 * (1 << attempt));
                        warn!(
                            "Rate limited by OpenAI, retrying in {:?} (attempt {}/{})",
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

impl EmbeddingProvider for OpenAIEmbeddingProvider {
    fn kind(&self) -> &str {
        "openai"
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

        // Truncate all texts first to avoid token limit errors
        let truncated: Vec<std::borrow::Cow<'_, str>> =
            texts.iter().map(|t| truncate_for_embedding(t)).collect();
        let truncated_refs: Vec<&str> = truncated.iter().map(|c| c.as_ref()).collect();

        // Process in batches of MAX_BATCH_SIZE
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
        // Validate API key with a small test request
        info!(
            "Initializing OpenAI embedding provider with model: {}",
            self.config.model
        );

        let test_embedding = self.embed_with_retry(&["test"], 1).map_err(|e| {
            memvid_core::MemvidError::EmbeddingFailed {
                reason: format!("Failed to initialize OpenAI provider: {}", e).into_boxed_str(),
            }
        })?;

        if let Some(emb) = test_embedding.first() {
            info!(
                "OpenAI provider initialized: model={}, dimension={}",
                self.config.model,
                emb.len()
            );
            // Update dimension if different from expected
            if emb.len() != self.config.dimension {
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

/// Implement VecEmbedder for compatibility with existing memvid code
impl VecEmbedder for OpenAIEmbeddingProvider {
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

/// Helper to create an OpenAI provider or fall back to local
pub fn try_openai_provider() -> Option<OpenAIEmbeddingProvider> {
    match OpenAIEmbeddingProvider::from_env() {
        Ok(provider) => {
            info!("OpenAI embedding provider available");
            Some(provider)
        }
        Err(e) => {
            debug!("OpenAI provider not available: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_dimensions() {
        assert_eq!(EmbeddingConfig::openai_large().dimension, 3072);
        assert_eq!(EmbeddingConfig::openai_small().dimension, 1536);
        assert_eq!(EmbeddingConfig::openai_ada().dimension, 1536);
    }

    #[test]
    fn test_empty_api_key() {
        let result = OpenAIEmbeddingProvider::new(String::new(), EmbeddingConfig::openai_large());
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Requires valid API key
    fn test_real_embedding() {
        let provider = OpenAIEmbeddingProvider::from_env().expect("OPENAI_API_KEY must be set");
        let embedding = provider.embed_text("Hello, world!").expect("embed");
        assert!(!embedding.is_empty());
        assert_eq!(embedding.len(), 3072); // text-embedding-3-large
    }
}
