//! Claude (Anthropic) enrichment engine using Claude Haiku 4.5.
//!
//! This engine uses the Anthropic API to extract structured memory cards
//! from text content. Supports parallel batch processing for speed.
//! Uses Claude Haiku 4.5 - the fastest Claude model.

use anyhow::{anyhow, Result};
use memvid_core::enrich::{EnrichmentContext, EnrichmentEngine, EnrichmentResult};
use memvid_core::types::{MemoryCard, MemoryCardBuilder, MemoryKind, Polarity};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// The extraction prompt for Claude
const EXTRACTION_PROMPT: &str = r#"You are a memory extraction assistant. Extract structured facts from the text.

For each distinct fact, preference, event, or relationship mentioned, output a memory card in this exact format:
MEMORY_START
kind: <Fact|Preference|Event|Profile|Relationship|Other>
entity: <the main entity this memory is about, use "user" for the human in the conversation>
slot: <a short key describing what aspect of the entity>
value: <the actual information>
polarity: <Positive|Negative|Neutral>
MEMORY_END

Only extract information that is explicitly stated. Do not infer or guess.
If there are no clear facts to extract, output MEMORY_NONE.

Extract memories from this text:
"#;

/// Claude API request message
#[derive(Debug, Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

/// Claude API request
#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
}

/// Claude API response
#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

/// Claude enrichment engine using Claude Haiku 4.5 (fastest) with parallel processing.
pub struct ClaudeEngine {
    /// API key
    api_key: String,
    /// Model to use
    model: String,
    /// Whether the engine is initialized
    ready: bool,
    /// Number of parallel workers (default: 20)
    parallelism: usize,
    /// Shared HTTP client (built in `init`)
    client: Option<Client>,
}

impl ClaudeEngine {
    /// Create a new Claude engine.
    pub fn new() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        Self {
            api_key,
            model: "claude-haiku-4-5-20251001".to_string(),
            ready: false,
            parallelism: 20,
            client: None,
        }
    }

    /// Create with a specific model.
    pub fn with_model(model: &str) -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        Self {
            api_key,
            model: model.to_string(),
            ready: false,
            parallelism: 20,
            client: None,
        }
    }

    /// Set parallelism level.
    pub fn with_parallelism(mut self, n: usize) -> Self {
        self.parallelism = n;
        self
    }

    /// Run inference via Claude API (blocking, thread-safe).
    fn run_inference_blocking(
        client: &Client,
        api_key: &str,
        model: &str,
        text: &str,
    ) -> Result<String> {
        let prompt = format!("{}\n\n{}", EXTRACTION_PROMPT, text);

        let request = ClaudeRequest {
            model: model.to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt,
            }],
        };

        let response = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| anyhow!("Claude API request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow!("Claude API error {}: {}", status, body));
        }

        let claude_response: ClaudeResponse = response
            .json()
            .map_err(|e| anyhow!("Failed to parse Claude response: {}", e))?;

        claude_response
            .content
            .iter()
            .find(|c| c.content_type == "text")
            .and_then(|c| c.text.clone())
            .ok_or_else(|| anyhow!("No text response from Claude"))
    }

    /// Parse the LLM output into memory cards.
    fn parse_output(output: &str, frame_id: u64, uri: &str, timestamp: i64) -> Vec<MemoryCard> {
        let mut cards = Vec::new();

        if output.contains("MEMORY_NONE") {
            return cards;
        }

        for block in output.split("MEMORY_START") {
            let block = block.trim();
            if block.is_empty() || !block.contains("MEMORY_END") {
                continue;
            }

            let block = block.split("MEMORY_END").next().unwrap_or("").trim();

            let mut kind = None;
            let mut entity = None;
            let mut slot = None;
            let mut value = None;
            let mut polarity = Polarity::Neutral;

            for line in block.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("kind:") {
                    kind = parse_memory_kind(rest.trim());
                } else if let Some(rest) = line.strip_prefix("entity:") {
                    entity = Some(rest.trim().to_string());
                } else if let Some(rest) = line.strip_prefix("slot:") {
                    slot = Some(rest.trim().to_string());
                } else if let Some(rest) = line.strip_prefix("value:") {
                    value = Some(rest.trim().to_string());
                } else if let Some(rest) = line.strip_prefix("polarity:") {
                    polarity = parse_polarity(rest.trim());
                }
            }

            if let (Some(k), Some(e), Some(s), Some(v)) = (kind, entity, slot, value) {
                if !e.is_empty() && !s.is_empty() && !v.is_empty() {
                    match MemoryCardBuilder::new()
                        .kind(k)
                        .entity(&e)
                        .slot(&s)
                        .value(&v)
                        .polarity(polarity)
                        .source(frame_id, Some(uri.to_string()))
                        .document_date(timestamp)
                        .engine("claude:claude-haiku-4-5", "1.0.0")
                        .build(0)
                    {
                        Ok(card) => cards.push(card),
                        Err(err) => {
                            warn!("Failed to build memory card: {}", err);
                        }
                    }
                }
            }
        }

        cards
    }

    /// Process multiple frames in parallel and return all cards.
    pub fn enrich_batch(
        &self,
        contexts: Vec<EnrichmentContext>,
    ) -> Result<Vec<(u64, Vec<MemoryCard>)>> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow!("Claude engine not initialized (init() not called)"))?
            .clone();
        let client = Arc::new(client);
        let api_key = Arc::new(self.api_key.clone());
        let model = Arc::new(self.model.clone());
        let total = contexts.len();

        info!(
            "Starting parallel enrichment of {} frames with {} workers",
            total, self.parallelism
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.parallelism)
            .build()
            .map_err(|err| anyhow!("failed to build enrichment thread pool: {err}"))?;

        let results: Vec<(u64, Vec<MemoryCard>)> = pool.install(|| {
            contexts
                .into_par_iter()
                .enumerate()
                .map(|(i, ctx)| {
                    if ctx.text.is_empty() {
                        return (ctx.frame_id, vec![]);
                    }

                    if i > 0 && i % 50 == 0 {
                        info!("Enrichment progress: {}/{} frames", i, total);
                    }

                    match Self::run_inference_blocking(&client, &api_key, &model, &ctx.text) {
                        Ok(output) => {
                            debug!(
                                "Claude output for frame {}: {}",
                                ctx.frame_id,
                                &output[..output.len().min(100)]
                            );
                            let cards =
                                Self::parse_output(&output, ctx.frame_id, &ctx.uri, ctx.timestamp);
                            (ctx.frame_id, cards)
                        }
                        Err(err) => {
                            warn!(
                                "Claude inference failed for frame {}: {}",
                                ctx.frame_id, err
                            );
                            (ctx.frame_id, vec![])
                        }
                    }
                })
                .collect()
        });

        info!(
            "Parallel enrichment complete: {} frames processed",
            results.len()
        );
        Ok(results)
    }
}

fn parse_memory_kind(s: &str) -> Option<MemoryKind> {
    match s.to_lowercase().as_str() {
        "fact" => Some(MemoryKind::Fact),
        "preference" => Some(MemoryKind::Preference),
        "event" => Some(MemoryKind::Event),
        "profile" => Some(MemoryKind::Profile),
        "relationship" => Some(MemoryKind::Relationship),
        "other" => Some(MemoryKind::Other),
        _ => None,
    }
}

fn parse_polarity(s: &str) -> Polarity {
    match s.to_lowercase().as_str() {
        "positive" => Polarity::Positive,
        "negative" => Polarity::Negative,
        _ => Polarity::Neutral,
    }
}

impl EnrichmentEngine for ClaudeEngine {
    fn kind(&self) -> &str {
        "claude:claude-haiku-4-5"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> memvid_core::Result<()> {
        if self.api_key.is_empty() {
            return Err(memvid_core::MemvidError::EmbeddingFailed {
                reason: "ANTHROPIC_API_KEY environment variable not set".into(),
            });
        }
        let client = crate::http::blocking_client(Duration::from_secs(60)).map_err(|err| {
            memvid_core::MemvidError::EmbeddingFailed {
                reason: format!("Failed to create Claude HTTP client: {err}").into(),
            }
        })?;
        self.client = Some(client);
        self.ready = true;
        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.ready
    }

    fn enrich(&self, ctx: &EnrichmentContext) -> EnrichmentResult {
        if ctx.text.is_empty() {
            return EnrichmentResult::empty();
        }

        let client = match self.client.as_ref() {
            Some(client) => client,
            None => {
                return EnrichmentResult::failed(
                    "Claude engine not initialized (init() not called)".to_string(),
                )
            }
        };

        match Self::run_inference_blocking(client, &self.api_key, &self.model, &ctx.text) {
            Ok(output) => {
                debug!("Claude output for frame {}: {}", ctx.frame_id, output);
                let cards = Self::parse_output(&output, ctx.frame_id, &ctx.uri, ctx.timestamp);
                EnrichmentResult::success(cards)
            }
            Err(err) => EnrichmentResult::failed(format!("Claude inference failed: {}", err)),
        }
    }
}

impl Default for ClaudeEngine {
    fn default() -> Self {
        Self::new()
    }
}
