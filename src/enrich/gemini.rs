//! Gemini (Google) enrichment engine using Gemini 2.5 Flash.
//!
//! This engine uses the Google AI Studio API to extract structured memory cards
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

/// The extraction prompt for Gemini
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

/// Gemini API request
#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

/// Gemini API response
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: CandidateContent,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    parts: Vec<CandidatePart>,
}

#[derive(Debug, Deserialize)]
struct CandidatePart {
    text: Option<String>,
}

/// Gemini enrichment engine using Gemini 2.5 Flash with parallel processing.
pub struct GeminiEngine {
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

impl GeminiEngine {
    /// Create a new Gemini engine.
    pub fn new() -> Self {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .unwrap_or_default();
        Self {
            api_key,
            model: "gemini-2.5-flash".to_string(),
            ready: false,
            parallelism: 20,
            client: None,
        }
    }

    /// Create with a specific model.
    pub fn with_model(model: &str) -> Self {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .unwrap_or_default();
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

    /// Run inference via Gemini API (blocking, thread-safe).
    fn run_inference_blocking(
        client: &Client,
        api_key: &str,
        model: &str,
        text: &str,
    ) -> Result<String> {
        let prompt = format!("{}\n\n{}", EXTRACTION_PROMPT, text);

        let request = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart { text: prompt }],
            }],
            generation_config: GenerationConfig {
                temperature: 0.0,
                max_output_tokens: 1024,
            },
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, api_key
        );

        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| anyhow!("Gemini API request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow!("Gemini API error {}: {}", status, body));
        }

        let gemini_response: GeminiResponse = response
            .json()
            .map_err(|e| anyhow!("Failed to parse Gemini response: {}", e))?;

        gemini_response
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content.parts.into_iter().next())
            .and_then(|p| p.text)
            .ok_or_else(|| anyhow!("No text response from Gemini"))
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
                        .engine("gemini:gemini-2.5-flash", "1.0.0")
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
            .ok_or_else(|| anyhow!("Gemini engine not initialized (init() not called)"))?
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
                                "Gemini output for frame {}: {}",
                                ctx.frame_id,
                                &output[..output.len().min(100)]
                            );
                            let cards =
                                Self::parse_output(&output, ctx.frame_id, &ctx.uri, ctx.timestamp);
                            (ctx.frame_id, cards)
                        }
                        Err(err) => {
                            warn!(
                                "Gemini inference failed for frame {}: {}",
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

impl EnrichmentEngine for GeminiEngine {
    fn kind(&self) -> &str {
        "gemini:gemini-2.5-flash"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> memvid_core::Result<()> {
        if self.api_key.is_empty() {
            return Err(memvid_core::MemvidError::EmbeddingFailed {
                reason: "GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set".into(),
            });
        }
        let client = crate::http::blocking_client(Duration::from_secs(60)).map_err(|err| {
            memvid_core::MemvidError::EmbeddingFailed {
                reason: format!("Failed to create Gemini HTTP client: {err}").into(),
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
                    "Gemini engine not initialized (init() not called)".to_string(),
                )
            }
        };

        match Self::run_inference_blocking(client, &self.api_key, &self.model, &ctx.text) {
            Ok(output) => {
                debug!("Gemini output for frame {}: {}", ctx.frame_id, output);
                let cards = Self::parse_output(&output, ctx.frame_id, &ctx.uri, ctx.timestamp);
                EnrichmentResult::success(cards)
            }
            Err(err) => EnrichmentResult::failed(format!("Gemini inference failed: {}", err)),
        }
    }
}

impl Default for GeminiEngine {
    fn default() -> Self {
        Self::new()
    }
}
