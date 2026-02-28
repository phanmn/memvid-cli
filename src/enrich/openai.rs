//! OpenAI-based enrichment engine using GPT-4o-mini.
//!
//! This engine uses the OpenAI API to extract structured memory cards
//! from text content. Supports parallel batch processing for speed.

use anyhow::{anyhow, Result};
use memvid_core::enrich::{EnrichmentContext, EnrichmentEngine, EnrichmentResult};
use memvid_core::types::{MemoryCard, MemoryCardBuilder, MemoryKind, Polarity};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// The extraction prompt for GPT-4o-mini (single frame)
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

/// The extraction prompt for batched frames (multiple frames per API call)
const BATCH_EXTRACTION_PROMPT: &str = r#"You are a memory extraction assistant. Extract structured facts from multiple text blocks.

Each text block is labeled with a FRAME_ID. For each distinct fact in each block, output a memory card with the frame_id field:

MEMORY_START
frame_id: <the FRAME_ID of the source text>
kind: <Fact|Preference|Event|Profile|Relationship|Other>
entity: <the main entity this memory is about, use "user" for the human in the conversation>
slot: <a short key describing what aspect of the entity>
value: <the actual information>
polarity: <Positive|Negative|Neutral>
MEMORY_END

Only extract information that is explicitly stated. Do not infer or guess.
If a text block has no facts, output MEMORY_NONE with its frame_id.

Process these text blocks:
"#;

/// OpenAI API request message
#[derive(Debug, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

/// OpenAI API request
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
}

/// OpenAI API response
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

/// OpenAI enrichment engine using GPT-4o-mini with parallel processing.
pub struct OpenAiEngine {
    /// API key
    api_key: String,
    /// Model to use
    model: String,
    /// Whether the engine is initialized
    ready: bool,
    /// Number of parallel workers (default: 100)
    parallelism: usize,
    /// Number of frames to batch per API call (default: 10)
    batch_size: usize,
    /// Shared HTTP client (built in `init`)
    client: Option<Client>,
}

impl OpenAiEngine {
    /// Create a new OpenAI engine.
    pub fn new() -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        Self {
            api_key,
            model: "gpt-4o-mini".to_string(),
            ready: false,
            parallelism: 20, // 20 concurrent requests (balanced for API rate limits)
            batch_size: 10,  // 10 frames per API call
            client: None,
        }
    }

    /// Create with a specific model.
    pub fn with_model(model: &str) -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        Self {
            api_key,
            model: model.to_string(),
            ready: false,
            parallelism: 20,
            batch_size: 10,
            client: None,
        }
    }

    /// Set parallelism level.
    pub fn with_parallelism(mut self, n: usize) -> Self {
        self.parallelism = n;
        self
    }

    /// Set batch size (number of frames per API call).
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1); // At least 1
        self
    }

    /// Run inference via OpenAI API (blocking, thread-safe).
    fn run_inference_blocking(
        client: &Client,
        api_key: &str,
        model: &str,
        text: &str,
    ) -> Result<String> {
        let prompt = format!("{}\n\n{}", EXTRACTION_PROMPT, text);

        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            max_tokens: 1024,
            temperature: 0.0,
        };

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| anyhow!("OpenAI API request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow!("OpenAI API error {}: {}", status, body));
        }

        let chat_response: ChatResponse = response
            .json()
            .map_err(|e| anyhow!("Failed to parse OpenAI response: {}", e))?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow!("No response from OpenAI"))
    }

    /// Parse the LLM output into memory cards.
    fn parse_output(output: &str, frame_id: u64, uri: &str, timestamp: i64) -> Vec<MemoryCard> {
        let mut cards = Vec::new();

        // Check for "no memories" signal
        if output.contains("MEMORY_NONE") {
            return cards;
        }

        // Parse MEMORY_START...MEMORY_END blocks
        for block in output.split("MEMORY_START") {
            let block = block.trim();
            if block.is_empty() || !block.contains("MEMORY_END") {
                continue;
            }

            let block = block.split("MEMORY_END").next().unwrap_or("").trim();

            // Parse fields
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

            // Build memory card if we have required fields
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
                        .engine("openai:gpt-4o-mini", "1.0.0")
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

    /// Run batched inference for multiple frames in a single API call.
    fn run_batched_inference_blocking(
        client: &Client,
        api_key: &str,
        model: &str,
        contexts: &[&EnrichmentContext],
    ) -> Result<String> {
        // Build the batched prompt with frame markers
        let mut prompt = BATCH_EXTRACTION_PROMPT.to_string();
        for ctx in contexts {
            prompt.push_str(&format!(
                "\n\n=== FRAME_ID: {} ===\n{}",
                ctx.frame_id, ctx.text
            ));
        }

        // Use larger max_tokens for batched requests
        let max_tokens = 1024 + (contexts.len() as u32 * 512);

        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            max_tokens: max_tokens.min(4096), // Cap at 4096
            temperature: 0.0,
        };

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| anyhow!("OpenAI API request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow!("OpenAI API error {}: {}", status, body));
        }

        let chat_response: ChatResponse = response
            .json()
            .map_err(|e| anyhow!("Failed to parse OpenAI response: {}", e))?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow!("No response from OpenAI"))
    }

    /// Parse batched LLM output into memory cards grouped by frame_id.
    fn parse_batched_output(
        output: &str,
        contexts: &[&EnrichmentContext],
    ) -> std::collections::HashMap<u64, Vec<MemoryCard>> {
        let mut results: std::collections::HashMap<u64, Vec<MemoryCard>> =
            std::collections::HashMap::new();

        // Initialize empty results for all frames
        for ctx in contexts {
            results.insert(ctx.frame_id, Vec::new());
        }

        // Build a lookup for context metadata
        let ctx_lookup: std::collections::HashMap<u64, &EnrichmentContext> =
            contexts.iter().map(|c| (c.frame_id, *c)).collect();

        // Parse MEMORY_START...MEMORY_END blocks
        for block in output.split("MEMORY_START") {
            let block = block.trim();
            if block.is_empty() || !block.contains("MEMORY_END") {
                continue;
            }

            let block = block.split("MEMORY_END").next().unwrap_or("").trim();

            // Parse fields including frame_id
            let mut frame_id: Option<u64> = None;
            let mut kind = None;
            let mut entity = None;
            let mut slot = None;
            let mut value = None;
            let mut polarity = Polarity::Neutral;

            for line in block.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("frame_id:") {
                    frame_id = rest.trim().parse().ok();
                } else if let Some(rest) = line.strip_prefix("kind:") {
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

            // Build memory card if we have required fields
            if let (Some(fid), Some(k), Some(e), Some(s), Some(v)) =
                (frame_id, kind, entity, slot, value)
            {
                if let Some(ctx) = ctx_lookup.get(&fid) {
                    let uri = &ctx.uri;
                    let timestamp = ctx.timestamp;

                    if !e.is_empty() && !s.is_empty() && !v.is_empty() {
                        match MemoryCardBuilder::new()
                            .kind(k)
                            .entity(&e)
                            .slot(&s)
                            .value(&v)
                            .polarity(polarity)
                            .source(fid, Some(uri.to_string()))
                            .document_date(timestamp)
                            .engine("openai:gpt-4o-mini", "1.0.0")
                            .build(0)
                        {
                            Ok(card) => {
                                results.entry(fid).or_default().push(card);
                            }
                            Err(err) => {
                                warn!("Failed to build memory card: {}", err);
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Process multiple frames in parallel and return all cards.
    /// This is the key method for fast enrichment.
    /// Uses batching to reduce API calls: batch_size frames per API call.
    pub fn enrich_batch(
        &self,
        contexts: Vec<EnrichmentContext>,
    ) -> Result<Vec<(u64, Vec<MemoryCard>)>> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow!("OpenAI engine not initialized (init() not called)"))?
            .clone();
        let client = Arc::new(client);
        let api_key = Arc::new(self.api_key.clone());
        let model = Arc::new(self.model.clone());
        let total = contexts.len();
        let batch_size = self.batch_size;

        // Calculate number of batches
        let num_batches = (total + batch_size - 1) / batch_size;

        info!(
            "Starting parallel enrichment of {} frames with {} workers, {} frames per batch ({} batches)",
            total, self.parallelism, batch_size, num_batches
        );

        // Create batches of contexts
        let batches: Vec<Vec<EnrichmentContext>> = contexts
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Use rayon for parallel processing of batches
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.parallelism)
            .build()
            .map_err(|err| anyhow!("failed to build enrichment thread pool: {err}"))?;

        let batch_results: Vec<std::collections::HashMap<u64, Vec<MemoryCard>>> =
            pool.install(|| {
                batches
                    .into_par_iter()
                    .enumerate()
                    .map(|(batch_idx, batch)| {
                        // Filter out empty texts
                        let non_empty: Vec<&EnrichmentContext> =
                            batch.iter().filter(|ctx| !ctx.text.is_empty()).collect();

                        if non_empty.is_empty() {
                            // Return empty results for all frames in this batch
                            return batch.iter().map(|ctx| (ctx.frame_id, Vec::new())).collect();
                        }

                        // Progress logging every 10 batches
                        if batch_idx > 0 && batch_idx % 10 == 0 {
                            info!("Enrichment progress: {} batches processed", batch_idx);
                        }

                        match Self::run_batched_inference_blocking(
                            &client, &api_key, &model, &non_empty,
                        ) {
                            Ok(output) => {
                                debug!(
                                    "OpenAI batch output (batch {}): {}...",
                                    batch_idx,
                                    &output[..output.len().min(100)]
                                );
                                Self::parse_batched_output(&output, &non_empty)
                            }
                            Err(err) => {
                                warn!(
                                    "OpenAI batch inference failed (batch {}): {}",
                                    batch_idx, err
                                );
                                // Return empty results for all frames in this batch
                                batch.iter().map(|ctx| (ctx.frame_id, Vec::new())).collect()
                            }
                        }
                    })
                    .collect()
            });

        // Flatten batch results into a single vec
        let mut results: Vec<(u64, Vec<MemoryCard>)> = Vec::with_capacity(total);
        for batch_map in batch_results {
            for (frame_id, cards) in batch_map {
                results.push((frame_id, cards));
            }
        }

        info!(
            "Parallel enrichment complete: {} frames processed in {} batches",
            results.len(),
            num_batches
        );
        Ok(results)
    }
}

/// Parse a memory kind string into the enum.
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

/// Parse a polarity string into the enum.
fn parse_polarity(s: &str) -> Polarity {
    match s.to_lowercase().as_str() {
        "positive" => Polarity::Positive,
        "negative" => Polarity::Negative,
        _ => Polarity::Neutral,
    }
}

impl EnrichmentEngine for OpenAiEngine {
    fn kind(&self) -> &str {
        "openai:gpt-4o-mini"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> memvid_core::Result<()> {
        if self.api_key.is_empty() {
            return Err(memvid_core::MemvidError::EmbeddingFailed {
                reason: "OPENAI_API_KEY environment variable not set".into(),
            });
        }
        // Use longer timeout for batched requests (multiple frames per call)
        let client = crate::http::blocking_client(Duration::from_secs(120)).map_err(|err| {
            memvid_core::MemvidError::EmbeddingFailed {
                reason: format!("Failed to create OpenAI HTTP client: {err}").into(),
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
                    "OpenAI engine not initialized (init() not called)".to_string(),
                )
            }
        };

        match Self::run_inference_blocking(client, &self.api_key, &self.model, &ctx.text) {
            Ok(output) => {
                debug!("OpenAI output for frame {}: {}", ctx.frame_id, output);
                let cards = Self::parse_output(&output, ctx.frame_id, &ctx.uri, ctx.timestamp);
                EnrichmentResult::success(cards)
            }
            Err(err) => EnrichmentResult::failed(format!("OpenAI inference failed: {}", err)),
        }
    }
}

impl Default for OpenAiEngine {
    fn default() -> Self {
        Self::new()
    }
}
