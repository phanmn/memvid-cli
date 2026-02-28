//! LLM-based enrichment engine using Phi-3.5 Mini.
//!
//! This engine uses a local Phi-3.5 model to extract structured memory cards
//! from text content through prompted inference.

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use memvid_core::enrich::{EnrichmentContext, EnrichmentEngine, EnrichmentResult};
use memvid_core::types::{MemoryCard, MemoryCardBuilder, MemoryKind, Polarity};
use tokio::runtime::Runtime;
use tracing::{debug, warn};

/// The prompt template for Phi-3.5 memory extraction.
const EXTRACTION_PROMPT: &str = r#"<|system|>
You are a memory extraction assistant. Your task is to extract structured facts from text.

For each distinct fact, preference, event, or relationship mentioned, output a memory card in this exact format:
MEMORY_START
kind: <Fact|Preference|Event|Profile|Relationship|Other>
entity: <the main entity this memory is about>
slot: <a short key describing what aspect of the entity>
value: <the actual information>
polarity: <Positive|Negative|Neutral>
MEMORY_END

Only extract information that is explicitly stated. Do not infer or guess.
If there are no clear facts to extract, output MEMORY_NONE.
<|end|>
<|user|>
Extract memories from this text:

{text}
<|end|>
<|assistant|>
"#;

/// Maximum context window size for Phi-3.5
const MAX_CONTEXT_TOKENS: u32 = 4096;

/// Maximum tokens to generate per extraction
const MAX_OUTPUT_TOKENS: usize = 1024;

/// Maximum input text length (characters) to process
const MAX_INPUT_CHARS: usize = 8192;

/// LLM enrichment engine using Phi-3.5 Mini.
pub struct LlmEngine {
    /// Path to the GGUF model file.
    model_path: PathBuf,
    /// Loaded model (lazy initialization).
    model: Mutex<Option<LlamaModel>>,
    /// Whether the engine is initialized.
    ready: bool,
    /// Engine version.
    version: String,
}

impl LlmEngine {
    /// Create a new LLM engine with the specified model path.
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            model: Mutex::new(None),
            ready: false,
            version: "1.0.0".to_string(),
        }
    }

    /// Load the model from disk.
    fn load_model(&self) -> Result<LlamaModel> {
        if !self.model_path.exists() {
            return Err(anyhow!(
                "Model file not found: {}",
                self.model_path.display()
            ));
        }

        // Suppress llama.cpp logging
        unsafe {
            std::env::set_var("GGML_LOG_LEVEL", "ERROR");
            std::env::set_var("LLAMA_LOG_LEVEL", "ERROR");
        }

        debug!("Loading model from {}", self.model_path.display());

        LlamaModel::load_from_file(&self.model_path, LlamaParams::default())
            .map_err(|err| anyhow!("Failed to load Phi-3.5 model: {}", err))
    }

    /// Run inference on the given text and return raw output.
    fn run_inference(&self, text: &str) -> Result<String> {
        let model_guard = self
            .model
            .lock()
            .map_err(|_| anyhow!("Model lock poisoned"))?;

        let model = model_guard
            .as_ref()
            .ok_or_else(|| anyhow!("LLM engine not initialized. Call init() first."))?;

        // Truncate input if too long
        let truncated_text = if text.len() > MAX_INPUT_CHARS {
            &text[..MAX_INPUT_CHARS]
        } else {
            text
        };

        // Build the prompt
        let prompt = EXTRACTION_PROMPT.replace("{text}", truncated_text);

        // Create session
        let mut session_params = SessionParams::default();
        session_params.n_ctx = MAX_CONTEXT_TOKENS;
        session_params.n_batch = 512.min(MAX_CONTEXT_TOKENS);
        if session_params.n_ubatch == 0 {
            session_params.n_ubatch = 512;
        }

        let mut session = model
            .create_session(session_params)
            .map_err(|err| anyhow!("Failed to create LLM session: {}", err))?;

        // Tokenize the prompt
        let mut tokens = model
            .tokenize_bytes(prompt.as_bytes(), true, true)
            .map_err(|err| anyhow!("Failed to tokenize prompt: {}", err))?;

        // Ensure we have room for output
        let max_tokens = MAX_CONTEXT_TOKENS as usize;
        let reserved = MAX_OUTPUT_TOKENS + 64;
        if tokens.len() >= max_tokens.saturating_sub(reserved) {
            let target = max_tokens.saturating_sub(reserved).max(1);
            let tail_start = tokens.len().saturating_sub(target);
            tokens = tokens.split_off(tail_start);
        }

        // Prime the context
        session
            .advance_context_with_tokens(&tokens)
            .map_err(|err| anyhow!("Failed to prime LLM context: {}", err))?;

        // Generate completion
        let handle = session
            .start_completing_with(StandardSampler::default(), MAX_OUTPUT_TOKENS)
            .map_err(|err| anyhow!("Failed to start LLM completion: {}", err))?;

        // Run async generation synchronously
        let runtime =
            Runtime::new().map_err(|err| anyhow!("Failed to create tokio runtime: {}", err))?;

        let generated = runtime.block_on(async { handle.into_string_async().await });

        Ok(generated.trim().to_string())
    }

    /// Parse the LLM output into memory cards.
    fn parse_output(&self, output: &str, ctx: &EnrichmentContext) -> Vec<MemoryCard> {
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
                    // ID 0 will be reassigned when added to MemoriesTrack
                    match MemoryCardBuilder::new()
                        .kind(k)
                        .entity(&e)
                        .slot(&s)
                        .value(&v)
                        .polarity(polarity)
                        .source(ctx.frame_id, Some(ctx.uri.clone()))
                        .document_date(ctx.timestamp)
                        .engine("llm:phi-3.5-mini", "1.0.0")
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

impl EnrichmentEngine for LlmEngine {
    fn kind(&self) -> &str {
        "llm:phi-3.5-mini"
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn init(&mut self) -> memvid_core::Result<()> {
        let model = self
            .load_model()
            .map_err(|err| memvid_core::MemvidError::EmbeddingFailed {
                reason: format!("{}", err).into_boxed_str(),
            })?;
        *self
            .model
            .lock()
            .map_err(|_| memvid_core::MemvidError::EmbeddingFailed {
                reason: "Model lock poisoned".into(),
            })? = Some(model);
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

        match self.run_inference(&ctx.text) {
            Ok(output) => {
                debug!("LLM output for frame {}: {}", ctx.frame_id, output);
                let cards = self.parse_output(&output, ctx);
                EnrichmentResult::success(cards)
            }
            Err(err) => EnrichmentResult::failed(format!("LLM inference failed: {}", err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_memory_kind() {
        assert_eq!(parse_memory_kind("Fact"), Some(MemoryKind::Fact));
        assert_eq!(
            parse_memory_kind("PREFERENCE"),
            Some(MemoryKind::Preference)
        );
        assert_eq!(parse_memory_kind("event"), Some(MemoryKind::Event));
        assert_eq!(parse_memory_kind("invalid"), None);
    }

    #[test]
    fn test_parse_polarity() {
        assert_eq!(parse_polarity("Positive"), Polarity::Positive);
        assert_eq!(parse_polarity("NEGATIVE"), Polarity::Negative);
        assert_eq!(parse_polarity("Neutral"), Polarity::Neutral);
        assert_eq!(parse_polarity("unknown"), Polarity::Neutral);
    }

    #[test]
    fn test_parse_output() {
        let engine = LlmEngine::new(PathBuf::from("/tmp/test.gguf"));
        let ctx = EnrichmentContext::new(
            1,
            "mv2://test/1".to_string(),
            "Test text".to_string(),
            None,
            1700000000,
            None,
        );

        // Test parsing a valid memory block
        let output = r#"
MEMORY_START
kind: Fact
entity: John
slot: employer
value: Anthropic
polarity: Positive
MEMORY_END
"#;
        let cards = engine.parse_output(output, &ctx);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].entity, "John");
        assert_eq!(cards[0].slot, "employer");
        assert_eq!(cards[0].value, "Anthropic");
        assert_eq!(cards[0].kind, MemoryKind::Fact);
    }

    #[test]
    fn test_parse_output_none() {
        let engine = LlmEngine::new(PathBuf::from("/tmp/test.gguf"));
        let ctx = EnrichmentContext::new(
            1,
            "mv2://test/1".to_string(),
            "Test text".to_string(),
            None,
            1700000000,
            None,
        );

        let output = "MEMORY_NONE";
        let cards = engine.parse_output(output, &ctx);
        assert!(cards.is_empty());
    }

    #[test]
    fn test_parse_output_multiple() {
        let engine = LlmEngine::new(PathBuf::from("/tmp/test.gguf"));
        let ctx = EnrichmentContext::new(
            1,
            "mv2://test/1".to_string(),
            "Test text".to_string(),
            None,
            1700000000,
            None,
        );

        let output = r#"
MEMORY_START
kind: Fact
entity: Alice
slot: role
value: Engineer
polarity: Neutral
MEMORY_END

MEMORY_START
kind: Preference
entity: Bob
slot: drink
value: Coffee
polarity: Positive
MEMORY_END
"#;
        let cards = engine.parse_output(output, &ctx);
        assert_eq!(cards.len(), 2);
        assert_eq!(cards[0].entity, "Alice");
        assert_eq!(cards[1].entity, "Bob");
    }
}
