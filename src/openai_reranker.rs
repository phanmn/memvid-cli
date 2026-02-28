//! OpenAI Reranker Provider
//!
//! This module provides a `Reranker` implementation that uses OpenAI's
//! GPT models to score and rerank search results for improved relevance.
//!
//! ## Environment Variables
//! - `OPENAI_API_KEY`: Required API key for OpenAI
//! - `OPENAI_RERANK_MODEL`: Optional model override (default: gpt-4o-mini)
//!
//! ## Features
//! - Uses structured prompting to score relevance
//! - Efficient batch processing with configurable concurrency
//! - Automatic retry with exponential backoff
//! - Thread-safe for concurrent use

use anyhow::{anyhow, bail, Result};
use memvid_core::{Reranker, RerankerConfig, RerankerDocument, RerankerResult};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, info, warn};

/// OpenAI chat completions API endpoint
const OPENAI_CHAT_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Default model for reranking (fast and cost-effective)
const DEFAULT_RERANK_MODEL: &str = "gpt-4o-mini";

/// Request timeout for reranking
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum documents to rerank in a single prompt
const MAX_DOCS_PER_PROMPT: usize = 20;

/// OpenAI chat request payload
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// OpenAI chat response
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    #[allow(dead_code)]
    prompt_tokens: usize,
    #[allow(dead_code)]
    completion_tokens: usize,
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

/// Parsed relevance score from LLM response
#[derive(Debug, Deserialize)]
struct RelevanceScore {
    id: u64,
    score: f32,
}

/// OpenAI Reranker Provider
///
/// Uses GPT models to evaluate query-document relevance for improved ranking.
#[derive(Clone)]
pub struct OpenAIReranker {
    api_key: String,
    model: String,
    config: RerankerConfig,
    client: Client,
    ready: std::sync::Arc<AtomicBool>,
}

impl std::fmt::Debug for OpenAIReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIReranker")
            .field("model", &self.model)
            .field("max_candidates", &self.config.max_candidates)
            .field("ready", &self.ready.load(Ordering::Relaxed))
            .finish()
    }
}

impl OpenAIReranker {
    /// Create a new OpenAI reranker
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key
    /// * `model` - Model to use (e.g., "gpt-4o-mini", "gpt-4o")
    /// * `config` - Reranker configuration
    pub fn new(api_key: String, model: Option<String>, config: RerankerConfig) -> Result<Self> {
        if api_key.is_empty() {
            bail!("OpenAI API key cannot be empty");
        }

        let client = crate::http::blocking_client(REQUEST_TIMEOUT)
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        Ok(Self {
            api_key,
            model: model.unwrap_or_else(|| DEFAULT_RERANK_MODEL.to_string()),
            config,
            client,
            ready: std::sync::Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create reranker from environment variables
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow!("OPENAI_API_KEY environment variable not set"))?;

        let model = std::env::var("OPENAI_RERANK_MODEL").ok();

        Self::new(api_key, model, RerankerConfig::default())
    }

    /// Create reranker with high precision config
    pub fn high_precision(api_key: String) -> Result<Self> {
        Self::new(
            api_key,
            Some("gpt-4o".to_string()),
            RerankerConfig::high_precision(),
        )
    }

    /// Create reranker with high recall config
    pub fn high_recall(api_key: String) -> Result<Self> {
        Self::new(api_key, None, RerankerConfig::high_recall())
    }

    /// Build the reranking prompt
    fn build_prompt(&self, query: &str, documents: &[&RerankerDocument]) -> String {
        let mut prompt = format!(
            r#"You are a relevance scoring assistant. Given a query and a list of documents, score each document's relevance to the query on a scale of 0.0 to 1.0.

Query: "{}"

Documents:
"#,
            query
        );

        for (idx, doc) in documents.iter().enumerate() {
            let preview = if doc.text.len() > 500 {
                format!("{}...", &doc.text[..500])
            } else {
                doc.text.clone()
            };
            prompt.push_str(&format!("\n[{}] ID={}: {}\n", idx + 1, doc.id, preview));
        }

        prompt.push_str(
            r#"
Return a JSON array of objects with "id" and "score" fields for each document.
Score based on semantic relevance, not just keyword matching.
Consider:
- Direct answers to the query
- Related context that helps answer the query
- Factual relevance even if wording differs

Output format (JSON only, no explanation):
[{"id": 123, "score": 0.95}, {"id": 456, "score": 0.72}, ...]
"#,
        );

        prompt
    }

    /// Parse relevance scores from LLM response
    fn parse_scores(&self, response: &str) -> Result<Vec<RelevanceScore>> {
        // Try to find JSON array in response
        let json_start = response
            .find('[')
            .ok_or_else(|| anyhow!("No JSON array found"))?;
        let json_end = response
            .rfind(']')
            .ok_or_else(|| anyhow!("No JSON array end found"))?;

        let json_str = &response[json_start..=json_end];
        let scores: Vec<RelevanceScore> = serde_json::from_str(json_str)
            .map_err(|e| anyhow!("Failed to parse scores: {} from: {}", e, json_str))?;

        Ok(scores)
    }

    /// Call OpenAI chat API
    fn call_openai(&self, prompt: &str) -> Result<String> {
        let messages = vec![
            ChatMessage {
                role: "system",
                content: "You are a document relevance scoring assistant. Output only valid JSON.",
            },
            ChatMessage {
                role: "user",
                content: prompt,
            },
        ];

        let request = ChatRequest {
            model: &self.model,
            messages,
            temperature: 0.0,
            max_tokens: 1024,
        };

        let response = self
            .client
            .post(OPENAI_CHAT_URL)
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
            if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&body) {
                bail!(
                    "OpenAI API error ({}): {}",
                    error_response.error.error_type,
                    error_response.error.message
                );
            }
            bail!("OpenAI API request failed with status {}: {}", status, body);
        }

        let chat_response: ChatResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Failed to parse OpenAI response: {}", e))?;

        let content = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow!("No response content"))?;

        debug!(
            "OpenAI rerank: {} tokens used, model={}",
            chat_response.usage.total_tokens, self.model
        );

        Ok(content)
    }

    /// Rerank with retry logic
    fn rerank_with_retry(
        &self,
        query: &str,
        documents: &[&RerankerDocument],
        max_retries: usize,
    ) -> Result<Vec<RelevanceScore>> {
        let prompt = self.build_prompt(query, documents);
        let mut last_error = None;

        for attempt in 0..max_retries {
            match self.call_openai(&prompt) {
                Ok(response) => match self.parse_scores(&response) {
                    Ok(scores) => return Ok(scores),
                    Err(e) => {
                        warn!("Failed to parse scores (attempt {}): {}", attempt + 1, e);
                        last_error = Some(e);
                    }
                },
                Err(e) => {
                    let error_str = e.to_string();
                    if error_str.contains("rate_limit") || error_str.contains("429") {
                        let backoff = Duration::from_millis(1000 * (1 << attempt));
                        warn!(
                            "Rate limited, retrying in {:?} (attempt {}/{})",
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

        Err(last_error.unwrap_or_else(|| anyhow!("Failed after {} retries", max_retries)))
    }
}

impl Reranker for OpenAIReranker {
    fn kind(&self) -> &'static str {
        "openai"
    }

    fn rerank(
        &self,
        query: &str,
        documents: &[RerankerDocument],
        top_k: usize,
    ) -> memvid_core::Result<Vec<RerankerResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Limit candidates
        let max_candidates = self.config.max_candidates.min(documents.len());
        let candidates: Vec<&RerankerDocument> = documents.iter().take(max_candidates).collect();

        // Process in batches if needed
        let mut all_scores: Vec<RelevanceScore> = Vec::new();

        for chunk in candidates.chunks(MAX_DOCS_PER_PROMPT) {
            let scores = self.rerank_with_retry(query, chunk, 3).map_err(|e| {
                memvid_core::MemvidError::RerankFailed {
                    reason: e.to_string().into_boxed_str(),
                }
            })?;
            all_scores.extend(scores);
        }

        // Build results with original ranks
        let mut results: Vec<RerankerResult> = all_scores
            .into_iter()
            .filter_map(|score| {
                let original_rank = documents.iter().position(|d| d.id == score.id)?;
                if score.score < self.config.min_score {
                    return None;
                }
                Some(RerankerResult {
                    id: score.id,
                    score: score.score,
                    original_rank: original_rank + 1,
                    new_rank: 0, // Will be set after sorting
                })
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign new ranks and limit to top_k
        let top_k = top_k.min(self.config.top_k);
        for (idx, result) in results.iter_mut().enumerate() {
            result.new_rank = idx + 1;
        }

        Ok(results.into_iter().take(top_k).collect())
    }

    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    fn init(&mut self) -> memvid_core::Result<()> {
        info!("Initializing OpenAI reranker with model: {}", self.model);

        // Test with a simple rerank to validate API key
        let test_docs = vec![RerankerDocument::new(0, "Test document")];
        let _ = self
            .rerank_with_retry("test query", &[&test_docs[0]], 1)
            .map_err(|e| memvid_core::MemvidError::RerankFailed {
                reason: format!("Failed to initialize reranker: {}", e).into_boxed_str(),
            })?;

        info!("OpenAI reranker initialized successfully");
        self.ready.store(true, Ordering::Relaxed);
        Ok(())
    }
}

/// Helper to create an OpenAI reranker or return None
pub fn try_openai_reranker() -> Option<OpenAIReranker> {
    match OpenAIReranker::from_env() {
        Ok(reranker) => {
            info!("OpenAI reranker available");
            Some(reranker)
        }
        Err(e) => {
            debug!("OpenAI reranker not available: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_api_key() {
        let result = OpenAIReranker::new(String::new(), None, RerankerConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_build_prompt() {
        let reranker =
            OpenAIReranker::new("test-key".to_string(), None, RerankerConfig::default()).unwrap();

        let docs = vec![
            RerankerDocument::new(1, "First document about Rust"),
            RerankerDocument::new(2, "Second document about Python"),
        ];

        let doc_refs: Vec<&RerankerDocument> = docs.iter().collect();
        let prompt = reranker.build_prompt("What is Rust?", &doc_refs);

        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("ID=1"));
        assert!(prompt.contains("ID=2"));
        assert!(prompt.contains("First document"));
        assert!(prompt.contains("Second document"));
    }

    #[test]
    fn test_parse_scores() {
        let reranker =
            OpenAIReranker::new("test-key".to_string(), None, RerankerConfig::default()).unwrap();

        let response = r#"Here are the scores:
[{"id": 1, "score": 0.95}, {"id": 2, "score": 0.42}]"#;

        let scores = reranker.parse_scores(response).unwrap();
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].id, 1);
        assert!((scores[0].score - 0.95).abs() < 0.01);
        assert_eq!(scores[1].id, 2);
        assert!((scores[1].score - 0.42).abs() < 0.01);
    }

    #[test]
    #[ignore] // Requires valid API key
    fn test_real_rerank() {
        let reranker = OpenAIReranker::from_env().expect("OPENAI_API_KEY must be set");

        let docs = vec![
            RerankerDocument::new(
                1,
                "Rust is a systems programming language focused on safety.",
            ),
            RerankerDocument::new(2, "Python is great for data science and machine learning."),
            RerankerDocument::new(3, "Rust provides memory safety without garbage collection."),
        ];

        let results = reranker.rerank("What makes Rust safe?", &docs, 2).unwrap();
        assert!(!results.is_empty());
        // Document about Rust safety should rank higher
        assert!(results[0].id == 1 || results[0].id == 3);
    }
}
