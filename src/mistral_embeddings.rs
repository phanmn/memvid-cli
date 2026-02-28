//! Mistral AI Embeddings Provider
//!
//! This module provides an embedding implementation that uses
//! Mistral AI's embeddings API for generating high-quality embeddings.
//!
//! ## Environment Variables
//! - `MISTRAL_API_KEY`: Required API key for Mistral AI
//! - `MISTRAL_EMBEDDING_MODEL`: Optional model override (default: mistral-embed)
//!
//! ## Features
//! - Uses mistral-embed model (1024 dimensions)
//! - Efficient batch processing
//! - Thread-safe for concurrent use

use anyhow::{anyhow, bail, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Mistral embeddings API endpoint
const MISTRAL_EMBEDDINGS_URL: &str = "https://api.mistral.ai/v1/embeddings";

/// Default embedding model
const DEFAULT_MODEL: &str = "mistral-embed";

/// Maximum texts per batch (Mistral supports batching)
const MAX_BATCH_SIZE: usize = 100;

/// Request timeout
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum characters for embedding text to avoid token limits.
/// mistral-embed has an 8192 token limit, using conservative estimate.
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

/// Mistral embedding request payload
#[derive(Debug, Serialize)]
struct MistralEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

/// Mistral embedding response
#[derive(Debug, Deserialize)]
struct MistralEmbeddingResponse {
    data: Vec<MistralEmbeddingData>,
    model: String,
    usage: MistralUsage,
}

#[derive(Debug, Deserialize)]
struct MistralEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MistralUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// Mistral error response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MistralErrorResponse {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

/// Mistral Embedding Provider
#[derive(Clone)]
pub struct MistralEmbeddingProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl std::fmt::Debug for MistralEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralEmbeddingProvider")
            .field("model", &self.model)
            .finish()
    }
}

impl MistralEmbeddingProvider {
    /// Embedding dimension for mistral-embed
    pub const DIMENSION: usize = 1024;

    /// Create a new Mistral embedding provider
    pub fn new(api_key: String, model: Option<&str>) -> Result<Self> {
        if api_key.is_empty() {
            bail!("Mistral API key cannot be empty");
        }

        let client = crate::http::blocking_client(REQUEST_TIMEOUT)
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let model = model.unwrap_or(DEFAULT_MODEL).to_string();

        Ok(Self {
            api_key,
            model,
            client,
        })
    }

    /// Create provider from environment variables
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| anyhow!("MISTRAL_API_KEY environment variable not set"))?;

        let model = std::env::var("MISTRAL_EMBEDDING_MODEL").ok();
        Self::new(api_key, model.as_deref())
    }

    /// Get model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get provider kind
    pub fn kind(&self) -> &'static str {
        "mistral"
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        Self::DIMENSION
    }

    /// Embed a single text
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let text = truncate_for_embedding(text);
        self.embed_with_retry(&[&text], 3)
            .map(|mut v| v.pop().unwrap_or_default())
    }

    /// Embed multiple texts in batch
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Truncate all texts first
        let truncated: Vec<std::borrow::Cow<'_, str>> =
            texts.iter().map(|t| truncate_for_embedding(t)).collect();
        let truncated_refs: Vec<&str> = truncated.iter().map(|c| c.as_ref()).collect();

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in batches
        for chunk in truncated_refs.chunks(MAX_BATCH_SIZE) {
            let embeddings = self.embed_with_retry(chunk, 3)?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embed texts with retry logic
    fn embed_with_retry(&self, texts: &[&str], max_retries: usize) -> Result<Vec<Vec<f32>>> {
        let request = MistralEmbeddingRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let mut last_error = None;

        for attempt in 0..max_retries {
            let response = self
                .client
                .post(MISTRAL_EMBEDDINGS_URL)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().unwrap_or_default();

                    if status.is_success() {
                        let embed_response: MistralEmbeddingResponse = serde_json::from_str(&body)
                            .map_err(|e| anyhow!("Failed to parse Mistral response: {}", e))?;

                        debug!(
                            "Mistral embeddings: {} texts, {} tokens, model={}",
                            texts.len(),
                            embed_response.usage.total_tokens,
                            embed_response.model
                        );

                        // Sort by index and extract embeddings
                        let mut data = embed_response.data;
                        data.sort_by_key(|d| d.index);

                        let embeddings: Vec<Vec<f32>> =
                            data.into_iter().map(|d| d.embedding).collect();

                        // Validate dimensions
                        if let Some(first) = embeddings.first() {
                            if first.len() != Self::DIMENSION {
                                warn!(
                                    "Mistral returned dimension {} but expected {}",
                                    first.len(),
                                    Self::DIMENSION
                                );
                            }
                        }

                        return Ok(embeddings);
                    }

                    // Check for rate limiting
                    if status.as_u16() == 429 {
                        let backoff = Duration::from_millis(500 * (1 << attempt));
                        warn!(
                            "Rate limited by Mistral, retrying in {:?} (attempt {}/{})",
                            backoff,
                            attempt + 1,
                            max_retries
                        );
                        std::thread::sleep(backoff);
                        last_error = Some(anyhow!("Rate limited"));
                        continue;
                    }

                    // Try to parse error response
                    if let Ok(error_response) = serde_json::from_str::<MistralErrorResponse>(&body)
                    {
                        return Err(anyhow!("Mistral API error: {}", error_response.message));
                    }

                    return Err(anyhow!(
                        "Mistral API request failed with status {}: {}",
                        status,
                        body
                    ));
                }
                Err(e) => {
                    if attempt < max_retries - 1 {
                        let backoff = Duration::from_millis(500 * (1 << attempt));
                        warn!(
                            "Mistral request failed, retrying in {:?} (attempt {}/{}): {}",
                            backoff,
                            attempt + 1,
                            max_retries,
                            e
                        );
                        std::thread::sleep(backoff);
                        last_error = Some(anyhow!("Request failed: {}", e));
                        continue;
                    }
                    return Err(anyhow!("Mistral API request failed: {}", e));
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("Failed to embed after {} retries", max_retries)))
    }
}

/// Helper to create a Mistral provider or return None
pub fn try_mistral_provider() -> Option<MistralEmbeddingProvider> {
    match MistralEmbeddingProvider::from_env() {
        Ok(provider) => {
            info!("Mistral embedding provider available");
            Some(provider)
        }
        Err(e) => {
            debug!("Mistral provider not available: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_api_key() {
        let result = MistralEmbeddingProvider::new(String::new(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension() {
        let provider = MistralEmbeddingProvider::new("test-key".to_string(), None).unwrap();
        assert_eq!(provider.dimension(), 1024);
    }

    #[test]
    #[ignore] // Requires valid API key
    fn test_real_embedding() {
        let provider = MistralEmbeddingProvider::from_env().expect("MISTRAL_API_KEY must be set");
        let embedding = provider.embed_text("Hello, world!").expect("embed");
        assert!(!embedding.is_empty());
        assert_eq!(embedding.len(), 1024);
    }
}
