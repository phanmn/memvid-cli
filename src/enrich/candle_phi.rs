//! Candle-based Phi-3 enrichment engine using quantized GGUF models.
//!
//! This engine uses Hugging Face Candle to run quantized Phi-3 models locally
//! for extracting structured memory cards from text content.

#![cfg(feature = "candle-llm")]

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use candle_core::Device;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;
use hf_hub::api::sync::Api;
use memvid_core::enrich::{EnrichmentContext, EnrichmentEngine, EnrichmentResult};
use memvid_core::types::{MemoryCard, MemoryCardBuilder, MemoryKind, Polarity};
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

/// The prompt template for Phi-3 memory extraction.
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

/// Maximum tokens to generate per extraction
const MAX_OUTPUT_TOKENS: usize = 1024;

/// Maximum input text length (characters) to process
const MAX_INPUT_CHARS: usize = 8192;

/// Hugging Face model repository (GGUF quantized version - much smaller)
const PHI3_MINI_REPO: &str = "microsoft/Phi-3-mini-4k-instruct-gguf";
/// GGUF file name (Q4 quantized, ~2.4GB)
const PHI3_GGUF_FILE: &str = "Phi-3-mini-4k-instruct-q4.gguf";

/// Loaded Phi-3 model state
struct LoadedModel {
    model: Phi3,
    tokenizer: Tokenizer,
    device: Device,
}

/// Candle-based Phi-3 enrichment engine using quantized GGUF.
pub struct CandlePhiEngine {
    /// Hugging Face model repo or local path
    model_source: ModelSource,
    /// Loaded model (lazy initialization)
    loaded: Mutex<Option<LoadedModel>>,
    /// Whether the engine is initialized
    ready: bool,
    /// Engine version
    version: String,
}

/// Source for the model
enum ModelSource {
    /// Load from Hugging Face Hub (downloads to HF cache)
    HuggingFace { repo: String, file: String },
    /// Load from local GGUF file
    Local { path: PathBuf },
    /// Load from memvid models directory (downloads there if needed)
    MemvidModels { models_dir: PathBuf },
}

impl CandlePhiEngine {
    /// Create a new Candle Phi engine that loads from Hugging Face Hub.
    pub fn from_hub(repo: Option<&str>) -> Self {
        Self {
            model_source: ModelSource::HuggingFace {
                repo: repo.unwrap_or(PHI3_MINI_REPO).to_string(),
                file: PHI3_GGUF_FILE.to_string(),
            },
            loaded: Mutex::new(None),
            ready: false,
            version: "1.0.0".to_string(),
        }
    }

    /// Create a new Candle Phi engine that loads from a local GGUF file.
    pub fn from_local(path: PathBuf) -> Self {
        Self {
            model_source: ModelSource::Local { path },
            loaded: Mutex::new(None),
            ready: false,
            version: "1.0.0".to_string(),
        }
    }

    /// Create a new Candle Phi engine that uses the memvid models directory.
    /// Downloads the model to ~/.memvid/models/llm/phi-3-mini-q4/ if not present.
    pub fn from_memvid_models(models_dir: PathBuf) -> Self {
        Self {
            model_source: ModelSource::MemvidModels { models_dir },
            loaded: Mutex::new(None),
            ready: false,
            version: "1.0.0".to_string(),
        }
    }

    /// Load the model from the configured source.
    fn load_model(&self) -> Result<LoadedModel> {
        let device = Device::Cpu;
        info!("Loading quantized Phi-3 model on device: {:?}", device);

        let (gguf_path, tokenizer_path) = match &self.model_source {
            ModelSource::HuggingFace { repo, file } => {
                info!(
                    "Downloading GGUF model from Hugging Face: {}/{}",
                    repo, file
                );
                let api = Api::new()?;
                let model_repo = api.model(repo.clone());

                // Download the GGUF file
                let gguf_path = model_repo.get(file)?;
                info!("Downloaded GGUF to: {:?}", gguf_path);

                // Get tokenizer from the non-GGUF repo (GGUF repo doesn't have tokenizer.json)
                let tokenizer_repo = api.model("microsoft/Phi-3-mini-4k-instruct".to_string());
                let tokenizer_path = tokenizer_repo.get("tokenizer.json")?;

                (gguf_path, tokenizer_path)
            }
            ModelSource::Local { path } => {
                if !path.exists() {
                    return Err(anyhow!("GGUF file not found: {}", path.display()));
                }

                // For local, assume tokenizer is in the same directory
                let tokenizer_path = path
                    .parent()
                    .map(|p| p.join("tokenizer.json"))
                    .ok_or_else(|| anyhow!("Cannot determine tokenizer path"))?;

                (path.clone(), tokenizer_path)
            }
            ModelSource::MemvidModels { models_dir } => {
                // Use ~/.memvid/models/llm/phi-3-mini-q4/ directory
                let model_dir = models_dir.join("llm").join("phi-3-mini-q4");
                let gguf_path = model_dir.join(PHI3_GGUF_FILE);
                let tokenizer_path = model_dir.join("tokenizer.json");

                if gguf_path.exists() && tokenizer_path.exists() {
                    info!("Using existing model from: {:?}", model_dir);
                } else {
                    info!(
                        "Downloading model to memvid models directory: {:?}",
                        model_dir
                    );
                    std::fs::create_dir_all(&model_dir)?;

                    // Download from HuggingFace and copy to our directory
                    let api = Api::new()?;

                    // Download GGUF
                    let gguf_repo = api.model(PHI3_MINI_REPO.to_string());
                    let hf_gguf = gguf_repo.get(PHI3_GGUF_FILE)?;
                    if !gguf_path.exists() {
                        info!("Copying GGUF to: {:?}", gguf_path);
                        std::fs::copy(&hf_gguf, &gguf_path)?;
                    }

                    // Download tokenizer
                    let tokenizer_repo = api.model("microsoft/Phi-3-mini-4k-instruct".to_string());
                    let hf_tokenizer = tokenizer_repo.get("tokenizer.json")?;
                    if !tokenizer_path.exists() {
                        info!("Copying tokenizer to: {:?}", tokenizer_path);
                        std::fs::copy(&hf_tokenizer, &tokenizer_path)?;
                    }

                    info!("Model installed to: {:?}", model_dir);
                }

                (gguf_path, tokenizer_path)
            }
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Load GGUF model
        info!("Loading GGUF file: {:?}", gguf_path);
        let mut file = std::fs::File::open(&gguf_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("Failed to read GGUF: {}", e))?;

        let model = Phi3::from_gguf(false, content, &mut file, &device)?;
        info!("Phi-3 quantized model loaded successfully");

        Ok(LoadedModel {
            model,
            tokenizer,
            device,
        })
    }

    /// Run inference on the given text and return raw output.
    fn run_inference(&self, text: &str) -> Result<String> {
        let mut loaded_guard = self
            .loaded
            .lock()
            .map_err(|_| anyhow!("Model lock poisoned"))?;

        let loaded = loaded_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Candle Phi engine not initialized. Call init() first."))?;

        // Truncate input if too long
        let truncated_text = if text.len() > MAX_INPUT_CHARS {
            &text[..MAX_INPUT_CHARS]
        } else {
            text
        };

        // Build the prompt
        let prompt = EXTRACTION_PROMPT.replace("{text}", truncated_text);

        // Tokenize
        let encoding = loaded
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        debug!("Input tokens: {}", tokens.len());

        // Run forward pass and generate tokens
        let mut logits_processor = LogitsProcessor::new(42, None, None);
        let mut generated_tokens = Vec::new();
        let eos_token = loaded
            .tokenizer
            .token_to_id("<|end|>")
            .or_else(|| loaded.tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(0);

        // Process the initial prompt
        let input = candle_core::Tensor::new(&tokens[..], &loaded.device)?.unsqueeze(0)?;
        let logits = loaded.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let logits = logits.to_dtype(candle_core::DType::F32)?;

        let next_token = logits_processor.sample(&logits)?;
        generated_tokens.push(next_token);
        tokens.push(next_token);

        // Generate tokens one at a time
        for i in 0..MAX_OUTPUT_TOKENS {
            if next_token == eos_token {
                break;
            }

            let input = candle_core::Tensor::new(&[tokens[tokens.len() - 1]], &loaded.device)?
                .unsqueeze(0)?;

            let logits = loaded.model.forward(&input, tokens.len() - 1)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;
            let logits = logits.to_dtype(candle_core::DType::F32)?;

            let next_token = logits_processor.sample(&logits)?;
            generated_tokens.push(next_token);
            tokens.push(next_token);

            if next_token == eos_token || i >= MAX_OUTPUT_TOKENS - 1 {
                break;
            }
        }

        // Decode generated tokens
        let output = loaded
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;

        Ok(output.trim().to_string())
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
                    match MemoryCardBuilder::new()
                        .kind(k)
                        .entity(&e)
                        .slot(&s)
                        .value(&v)
                        .polarity(polarity)
                        .source(ctx.frame_id, Some(ctx.uri.clone()))
                        .document_date(ctx.timestamp)
                        .engine("candle:phi-3-mini-q4", "1.0.0")
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

impl EnrichmentEngine for CandlePhiEngine {
    fn kind(&self) -> &str {
        "candle:phi-3-mini-q4"
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
            .loaded
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
                debug!("Candle Phi-3 output for frame {}: {}", ctx.frame_id, output);
                let cards = self.parse_output(&output, ctx);
                EnrichmentResult::success(cards)
            }
            Err(err) => EnrichmentResult::failed(format!("Candle inference failed: {}", err)),
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
}
