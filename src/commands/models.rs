//! Model management commands for all model types: embeddings, rerankers, and LLMs.
//!
//! Supports listing, installing, and removing models for semantic search, reranking,
//! and local LLM inference.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use clap::{Args, Subcommand, ValueEnum};

use crate::config::CliConfig;

// ============================================================================
// Embedding Models
// ============================================================================

/// Local embedding models available for installation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// BGE-small-en-v1.5: Fast, 384-dim, ~33 MB
    BgeSmall,
    /// BGE-base-en-v1.5: Balanced, 768-dim, ~110 MB
    BgeBase,
    /// Nomic-embed-text-v1.5: High accuracy, 768-dim, ~137 MB
    Nomic,
    /// GTE-large-en-v1.5: Best semantic depth, 1024-dim, ~327 MB
    GteLarge,
}

impl EmbeddingModel {
    fn all() -> impl Iterator<Item = EmbeddingModel> {
        [
            EmbeddingModel::BgeSmall,
            EmbeddingModel::BgeBase,
            EmbeddingModel::Nomic,
            EmbeddingModel::GteLarge,
        ]
        .into_iter()
    }

    fn display_name(&self) -> &'static str {
        match self {
            EmbeddingModel::BgeSmall => "BGE-small-en-v1.5",
            EmbeddingModel::BgeBase => "BGE-base-en-v1.5",
            EmbeddingModel::Nomic => "Nomic-embed-text-v1.5",
            EmbeddingModel::GteLarge => "GTE-large-en-v1.5",
        }
    }

    fn cli_name(&self) -> &'static str {
        match self {
            EmbeddingModel::BgeSmall => "bge-small",
            EmbeddingModel::BgeBase => "bge-base",
            EmbeddingModel::Nomic => "nomic",
            EmbeddingModel::GteLarge => "gte-large",
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            EmbeddingModel::BgeSmall => 384,
            EmbeddingModel::BgeBase => 768,
            EmbeddingModel::Nomic => 768,
            EmbeddingModel::GteLarge => 1024,
        }
    }

    fn size_mb(&self) -> usize {
        match self {
            EmbeddingModel::BgeSmall => 33,
            EmbeddingModel::BgeBase => 110,
            EmbeddingModel::Nomic => 137,
            EmbeddingModel::GteLarge => 327,
        }
    }

    fn hf_repo(&self) -> &'static str {
        match self {
            EmbeddingModel::BgeSmall => "BAAI/bge-small-en-v1.5",
            EmbeddingModel::BgeBase => "BAAI/bge-base-en-v1.5",
            EmbeddingModel::Nomic => "nomic-ai/nomic-embed-text-v1.5",
            EmbeddingModel::GteLarge => "thenlper/gte-large",
        }
    }

    fn is_default(&self) -> bool {
        matches!(self, EmbeddingModel::BgeSmall)
    }
}

// ============================================================================
// Reranker Models
// ============================================================================

/// Local reranker models available for installation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerModel {
    /// Jina-reranker-v1-turbo-en: Fast English reranking, ~86 MB
    JinaTurbo,
    /// Jina-reranker-v2-base-multilingual: Multilingual, ~200 MB
    JinaMultilingual,
    /// BGE-reranker-base: English/Chinese, ~200 MB
    BgeRerankerBase,
    /// BGE-reranker-v2-m3: Best multilingual, ~400 MB
    BgeRerankerV2M3,
}

impl RerankerModel {
    fn all() -> impl Iterator<Item = RerankerModel> {
        [
            RerankerModel::JinaTurbo,
            RerankerModel::JinaMultilingual,
            RerankerModel::BgeRerankerBase,
            RerankerModel::BgeRerankerV2M3,
        ]
        .into_iter()
    }

    fn display_name(&self) -> &'static str {
        match self {
            RerankerModel::JinaTurbo => "Jina-reranker-v1-turbo-en",
            RerankerModel::JinaMultilingual => "Jina-reranker-v2-base-multilingual",
            RerankerModel::BgeRerankerBase => "BGE-reranker-base",
            RerankerModel::BgeRerankerV2M3 => "BGE-reranker-v2-m3",
        }
    }

    fn cli_name(&self) -> &'static str {
        match self {
            RerankerModel::JinaTurbo => "jina-turbo",
            RerankerModel::JinaMultilingual => "jina-multilingual",
            RerankerModel::BgeRerankerBase => "bge-reranker-base",
            RerankerModel::BgeRerankerV2M3 => "bge-reranker-v2-m3",
        }
    }

    fn size_mb(&self) -> usize {
        match self {
            RerankerModel::JinaTurbo => 86,
            RerankerModel::JinaMultilingual => 200,
            RerankerModel::BgeRerankerBase => 200,
            RerankerModel::BgeRerankerV2M3 => 400,
        }
    }

    fn hf_repo(&self) -> &'static str {
        match self {
            RerankerModel::JinaTurbo => "jinaai/jina-reranker-v1-turbo-en",
            RerankerModel::JinaMultilingual => "jinaai/jina-reranker-v2-base-multilingual",
            RerankerModel::BgeRerankerBase => "BAAI/bge-reranker-base",
            RerankerModel::BgeRerankerV2M3 => "rozgo/bge-reranker-v2-m3",
        }
    }

    fn language(&self) -> &'static str {
        match self {
            RerankerModel::JinaTurbo => "English",
            RerankerModel::JinaMultilingual => "Multilingual",
            RerankerModel::BgeRerankerBase => "English/Chinese",
            RerankerModel::BgeRerankerV2M3 => "Multilingual",
        }
    }

    fn is_default(&self) -> bool {
        matches!(self, RerankerModel::JinaTurbo)
    }
}

// ============================================================================
// LLM Models
// ============================================================================

/// Known LLM models available for installation
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LlmModel {
    /// Phi-3.5 Mini Instruct (Q4_K_M quantization, ~2.4 GB)
    #[value(name = "phi-3.5-mini")]
    Phi35Mini,
    /// Phi-3.5 Mini Instruct (Q8_0 quantization, ~3.9 GB)
    #[value(name = "phi-3.5-mini-q8")]
    Phi35MiniQ8,
}

impl LlmModel {
    /// Returns the Hugging Face model ID
    fn hf_repo(&self) -> &'static str {
        match self {
            LlmModel::Phi35Mini | LlmModel::Phi35MiniQ8 => "bartowski/Phi-3.5-mini-instruct-GGUF",
        }
    }

    /// Returns the filename within the Hugging Face repo
    fn hf_filename(&self) -> &'static str {
        match self {
            LlmModel::Phi35Mini => "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            LlmModel::Phi35MiniQ8 => "Phi-3.5-mini-instruct-Q8_0.gguf",
        }
    }

    /// Returns the expected file size in bytes (approximate)
    fn expected_size_bytes(&self) -> u64 {
        match self {
            LlmModel::Phi35Mini => 2_360_000_000,   // ~2.4 GB
            LlmModel::Phi35MiniQ8 => 3_860_000_000, // ~3.9 GB
        }
    }

    /// Returns the local directory name for this model
    fn local_dir_name(&self) -> &'static str {
        match self {
            LlmModel::Phi35Mini => "phi-3.5-mini-q4",
            LlmModel::Phi35MiniQ8 => "phi-3.5-mini-q8",
        }
    }

    /// Returns a human-readable display name
    fn display_name(&self) -> &'static str {
        match self {
            LlmModel::Phi35Mini => "Phi-3.5 Mini Instruct (Q4_K_M)",
            LlmModel::Phi35MiniQ8 => "Phi-3.5 Mini Instruct (Q8_0)",
        }
    }

    fn cli_name(&self) -> &'static str {
        match self {
            LlmModel::Phi35Mini => "phi-3.5-mini",
            LlmModel::Phi35MiniQ8 => "phi-3.5-mini-q8",
        }
    }

    /// Returns an iterator over all known models
    fn all() -> impl Iterator<Item = LlmModel> {
        [LlmModel::Phi35Mini, LlmModel::Phi35MiniQ8].into_iter()
    }

    fn is_default(&self) -> bool {
        matches!(self, LlmModel::Phi35Mini)
    }
}

// ============================================================================
// CLIP Models (Visual Search)
// ============================================================================

/// CLIP models available for installation
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClipModel {
    /// MobileCLIP-S2 int8 quantized: Fast, 512-dim, ~101 MB total (default)
    #[value(name = "mobileclip-s2")]
    MobileClipS2,
    /// MobileCLIP-S2 fp16: Better accuracy, 512-dim, ~199 MB total
    #[value(name = "mobileclip-s2-fp16")]
    MobileClipS2Fp16,
    /// SigLIP-base quantized: Higher quality, 768-dim, ~211 MB total
    #[value(name = "siglip-base")]
    SigLipBase,
}

impl ClipModel {
    fn all() -> impl Iterator<Item = ClipModel> {
        [
            ClipModel::MobileClipS2,
            ClipModel::MobileClipS2Fp16,
            ClipModel::SigLipBase,
        ]
        .into_iter()
    }

    fn display_name(&self) -> &'static str {
        match self {
            ClipModel::MobileClipS2 => "MobileCLIP-S2 (int8 quantized)",
            ClipModel::MobileClipS2Fp16 => "MobileCLIP-S2 (fp16)",
            ClipModel::SigLipBase => "SigLIP-base (quantized)",
        }
    }

    fn cli_name(&self) -> &'static str {
        match self {
            ClipModel::MobileClipS2 => "mobileclip-s2",
            ClipModel::MobileClipS2Fp16 => "mobileclip-s2-fp16",
            ClipModel::SigLipBase => "siglip-base",
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            ClipModel::MobileClipS2 | ClipModel::MobileClipS2Fp16 => 512,
            ClipModel::SigLipBase => 768,
        }
    }

    fn total_size_mb(&self) -> f32 {
        match self {
            ClipModel::MobileClipS2 => 100.8,     // 36.7 + 64.1
            ClipModel::MobileClipS2Fp16 => 198.7, // 71.7 + 127.0
            ClipModel::SigLipBase => 210.5,       // 99.5 + 111.0
        }
    }

    fn vision_url(&self) -> &'static str {
        match self {
            ClipModel::MobileClipS2 => "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/onnx/vision_model_int8.onnx",
            ClipModel::MobileClipS2Fp16 => "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/onnx/vision_model_fp16.onnx",
            ClipModel::SigLipBase => "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/onnx/vision_model_quantized.onnx",
        }
    }

    fn text_url(&self) -> &'static str {
        match self {
            ClipModel::MobileClipS2 => "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/onnx/text_model_int8.onnx",
            ClipModel::MobileClipS2Fp16 => "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/onnx/text_model_fp16.onnx",
            ClipModel::SigLipBase => "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/onnx/text_model_quantized.onnx",
        }
    }

    fn vision_filename(&self) -> &'static str {
        match self {
            ClipModel::MobileClipS2 => "mobileclip-s2_vision.onnx",
            ClipModel::MobileClipS2Fp16 => "mobileclip-s2-fp16_vision.onnx",
            ClipModel::SigLipBase => "siglip-base_vision.onnx",
        }
    }

    fn text_filename(&self) -> &'static str {
        match self {
            ClipModel::MobileClipS2 => "mobileclip-s2_text.onnx",
            ClipModel::MobileClipS2Fp16 => "mobileclip-s2-fp16_text.onnx",
            ClipModel::SigLipBase => "siglip-base_text.onnx",
        }
    }

    fn is_default(&self) -> bool {
        matches!(self, ClipModel::MobileClipS2)
    }
}

// ============================================================================
// NER Models (Named Entity Recognition for Logic-Mesh)
// ============================================================================

/// NER models available for installation
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum NerModel {
    /// DistilBERT-NER: Fast & accurate NER, ~261 MB, 92% F1
    #[value(name = "distilbert-ner")]
    DistilbertNer,
}

impl NerModel {
    fn all() -> impl Iterator<Item = NerModel> {
        [NerModel::DistilbertNer].into_iter()
    }

    fn display_name(&self) -> &'static str {
        match self {
            NerModel::DistilbertNer => "DistilBERT-NER (dslim)",
        }
    }

    fn cli_name(&self) -> &'static str {
        match self {
            NerModel::DistilbertNer => "distilbert-ner",
        }
    }

    fn size_mb(&self) -> f32 {
        match self {
            NerModel::DistilbertNer => 261.0,
        }
    }

    fn model_url(&self) -> &'static str {
        match self {
            NerModel::DistilbertNer => {
                "https://huggingface.co/dslim/distilbert-NER/resolve/main/onnx/model.onnx"
            }
        }
    }

    fn tokenizer_url(&self) -> &'static str {
        match self {
            NerModel::DistilbertNer => {
                "https://huggingface.co/dslim/distilbert-NER/resolve/main/tokenizer.json"
            }
        }
    }

    fn local_dir_name(&self) -> &'static str {
        match self {
            NerModel::DistilbertNer => "distilbert-ner",
        }
    }

    fn is_default(&self) -> bool {
        matches!(self, NerModel::DistilbertNer)
    }
}

// ============================================================================
// Whisper Models (Audio Transcription)
// ============================================================================

/// Whisper models - now using Candle with auto-download from HuggingFace
/// No manual installation needed, kept for backwards compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum WhisperModel {
    /// whisper-small-en: Good quality English transcription (auto-downloads via Candle)
    #[value(name = "whisper-small-en")]
    WhisperSmallEn,
}

// WhisperModel is kept for CLI argument compatibility but models auto-download via Candle

// ============================================================================
// External Models (API-based, no download required)
// ============================================================================

/// External embedding providers (API-based)
struct ExternalEmbeddingProvider {
    name: &'static str,
    models: &'static [(&'static str, usize, &'static str)], // (model_name, dimensions, description)
    env_var: &'static str,
}

const EXTERNAL_EMBEDDING_PROVIDERS: &[ExternalEmbeddingProvider] = &[
    ExternalEmbeddingProvider {
        name: "OpenAI",
        models: &[
            ("text-embedding-3-large", 3072, "Highest quality"),
            ("text-embedding-3-small", 1536, "Good balance"),
            ("text-embedding-ada-002", 1536, "Legacy"),
        ],
        env_var: "OPENAI_API_KEY",
    },
    ExternalEmbeddingProvider {
        name: "Cohere",
        models: &[
            ("embed-english-v3.0", 1024, "English"),
            ("embed-multilingual-v3.0", 1024, "Multilingual"),
        ],
        env_var: "COHERE_API_KEY",
    },
    ExternalEmbeddingProvider {
        name: "Voyage",
        models: &[
            ("voyage-3", 1024, "Code & technical docs"),
            ("voyage-3-lite", 512, "Lightweight"),
        ],
        env_var: "VOYAGE_API_KEY",
    },
];

// ============================================================================
// CLI Commands
// ============================================================================

/// Model management commands
#[derive(Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Subcommand)]
pub enum ModelsCommand {
    /// Install an LLM model for enrichment
    Install(ModelsInstallArgs),
    /// List all available and installed models
    List(ModelsListArgs),
    /// Remove an installed model
    Remove(ModelsRemoveArgs),
    /// Verify model integrity
    Verify(ModelsVerifyArgs),
}

#[derive(Args)]
pub struct ModelsInstallArgs {
    /// LLM model to install (phi-3.5-mini, phi-3.5-mini-q8)
    #[arg(value_enum, group = "model_choice")]
    pub model: Option<LlmModel>,

    /// CLIP model to install for visual search (mobileclip-s2, mobileclip-s2-fp16, siglip-base)
    #[arg(long, value_enum, group = "model_choice")]
    pub clip: Option<ClipModel>,

    /// NER model to install for Logic-Mesh entity extraction (gliner-int8)
    #[arg(long, value_enum, group = "model_choice")]
    pub ner: Option<NerModel>,

    /// Whisper model (auto-downloads via Candle, no install needed)
    #[arg(long, value_enum, group = "model_choice", hide = true)]
    pub whisper: Option<WhisperModel>,

    /// Force re-download even if already installed
    #[arg(long, short)]
    pub force: bool,
}

#[derive(Args)]
pub struct ModelsListArgs {
    /// Output in JSON format
    #[arg(long)]
    pub json: bool,

    /// Show only a specific model type
    #[arg(long, value_enum)]
    pub model_type: Option<ModelType>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ModelType {
    /// Embedding models for semantic search
    Embedding,
    /// Reranker models for result reranking
    Reranker,
    /// LLM models for local inference
    Llm,
    /// CLIP models for visual search
    Clip,
    /// NER models for Logic-Mesh entity extraction
    Ner,
    /// Whisper models for audio transcription
    Whisper,
    /// External API-based models
    External,
}

#[derive(Args)]
pub struct ModelsRemoveArgs {
    /// Model to remove
    #[arg(value_enum)]
    pub model: LlmModel,

    /// Skip confirmation prompt
    #[arg(long, short)]
    pub yes: bool,
}

#[derive(Args)]
pub struct ModelsVerifyArgs {
    /// Model to verify (verifies all if not specified)
    #[arg(value_enum)]
    pub model: Option<LlmModel>,
}

/// Information about an installed model
#[derive(Debug, Clone, serde::Serialize)]
pub struct InstalledModel {
    pub name: String,
    pub model_type: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub verified: bool,
}

// ============================================================================
// Directory Helpers
// ============================================================================

/// Get the LLM models directory
fn llm_models_dir(config: &CliConfig) -> PathBuf {
    config.models_dir.join("llm")
}

/// Get the fastembed cache directory (for embeddings and rerankers)
#[cfg(feature = "local-embeddings")]
fn fastembed_cache_dir(config: &CliConfig) -> PathBuf {
    config.models_dir.clone()
}

/// Get the path where a specific LLM model should be stored
fn llm_model_path(config: &CliConfig, model: LlmModel) -> PathBuf {
    llm_models_dir(config)
        .join(model.local_dir_name())
        .join(model.hf_filename())
}

/// Check if an LLM model is installed
fn is_llm_model_installed(config: &CliConfig, model: LlmModel) -> bool {
    let path = llm_model_path(config, model);
    path.exists() && path.is_file()
}

/// Get the CLIP models directory
fn clip_models_dir(config: &CliConfig) -> PathBuf {
    config.models_dir.clone()
}

/// Get the path where a specific CLIP model vision encoder should be stored
fn clip_vision_path(config: &CliConfig, model: ClipModel) -> PathBuf {
    clip_models_dir(config).join(model.vision_filename())
}

/// Get the path where a specific CLIP model text encoder should be stored
fn clip_text_path(config: &CliConfig, model: ClipModel) -> PathBuf {
    clip_models_dir(config).join(model.text_filename())
}

/// Check if a CLIP model is fully installed (both vision and text encoders)
fn is_clip_model_installed(config: &CliConfig, model: ClipModel) -> bool {
    let vision_path = clip_vision_path(config, model);
    let text_path = clip_text_path(config, model);
    vision_path.exists() && vision_path.is_file() && text_path.exists() && text_path.is_file()
}

/// Check if a CLIP model is partially installed
fn clip_model_status(config: &CliConfig, model: ClipModel) -> (&'static str, bool, bool) {
    let has_vision = clip_vision_path(config, model).exists();
    let has_text = clip_text_path(config, model).exists();
    let status = match (has_vision, has_text) {
        (true, true) => "✓ installed",
        (true, false) => "⚠ partial (missing text)",
        (false, true) => "⚠ partial (missing vision)",
        (false, false) => "○ available",
    };
    (status, has_vision, has_text)
}

/// Get the NER models directory
fn ner_models_dir(config: &CliConfig) -> PathBuf {
    config.models_dir.clone()
}

/// Get the path where a specific NER model should be stored
fn ner_model_path(config: &CliConfig, model: NerModel) -> PathBuf {
    ner_models_dir(config)
        .join(model.local_dir_name())
        .join("model.onnx")
}

/// Get the path where a specific NER tokenizer should be stored
fn ner_tokenizer_path(config: &CliConfig, model: NerModel) -> PathBuf {
    ner_models_dir(config)
        .join(model.local_dir_name())
        .join("tokenizer.json")
}

/// Check if a NER model is fully installed (both model and tokenizer)
fn is_ner_model_installed(config: &CliConfig, model: NerModel) -> bool {
    let model_path = ner_model_path(config, model);
    let tokenizer_path = ner_tokenizer_path(config, model);
    model_path.exists()
        && model_path.is_file()
        && tokenizer_path.exists()
        && tokenizer_path.is_file()
}

/// Check NER model install status
fn ner_model_status(config: &CliConfig, model: NerModel) -> (&'static str, bool, bool) {
    let has_model = ner_model_path(config, model).exists();
    let has_tokenizer = ner_tokenizer_path(config, model).exists();
    let status = match (has_model, has_tokenizer) {
        (true, true) => "✓ installed",
        (true, false) => "⚠ partial (missing tokenizer)",
        (false, true) => "⚠ partial (missing model)",
        (false, false) => "○ available",
    };
    (status, has_model, has_tokenizer)
}

// Note: Whisper models now auto-download from HuggingFace via Candle.
// No manual installation helpers are needed - the models are cached in ~/.cache/huggingface/hub/

/// Scan fastembed cache for installed embedding/reranker models
#[cfg(feature = "local-embeddings")]
fn scan_fastembed_cache(config: &CliConfig) -> Vec<(String, PathBuf, u64)> {
    let cache_dir = fastembed_cache_dir(config);
    let mut installed = Vec::new();

    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                // fastembed caches models in directories like "models--BAAI--bge-small-en-v1.5"
                if name.starts_with("models--") {
                    let size = dir_size(&path).unwrap_or(0);
                    let model_name = name.replace("models--", "").replace("--", "/");
                    installed.push((model_name, path, size));
                }
            }
        }
    }

    installed
}

/// Stub for scan_fastembed_cache when local-embeddings is disabled
#[cfg(not(feature = "local-embeddings"))]
fn scan_fastembed_cache(_config: &CliConfig) -> Vec<(String, PathBuf, u64)> {
    Vec::new()
}

/// Calculate directory size recursively
fn dir_size(path: &Path) -> io::Result<u64> {
    let mut size = 0;
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                size += dir_size(&path)?;
            } else {
                size += entry.metadata()?.len();
            }
        }
    }
    Ok(size)
}

// ============================================================================
// Command Handlers
// ============================================================================

/// Handle the models command
pub fn handle_models(config: &CliConfig, args: ModelsArgs) -> Result<()> {
    match args.command {
        ModelsCommand::Install(install_args) => handle_models_install(config, install_args),
        ModelsCommand::List(list_args) => handle_models_list(config, list_args),
        ModelsCommand::Remove(remove_args) => handle_models_remove(config, remove_args),
        ModelsCommand::Verify(verify_args) => handle_models_verify(config, verify_args),
    }
}

/// Handle model installation
pub fn handle_models_install(config: &CliConfig, args: ModelsInstallArgs) -> Result<()> {
    // Check if CLIP model is being installed
    if let Some(clip_model) = args.clip {
        return handle_clip_install(config, clip_model, args.force);
    }

    // Check if NER model is being installed
    if let Some(ner_model) = args.ner {
        return handle_ner_install(config, ner_model, args.force);
    }

    // Whisper models auto-download via Candle, no manual install needed
    if args.whisper.is_some() {
        println!("ℹ️  Whisper models now auto-download from HuggingFace on first use.");
        println!("   No manual installation required!");
        println!();
        println!("   Just use: memvid put file.mv2 --input audio.mp3 --transcribe");
        println!();
        println!("   The model will download automatically (~244 MB for whisper-small-en).");
        return Ok(());
    }

    // Check if LLM model is being installed
    if let Some(llm_model) = args.model {
        return handle_llm_install(config, llm_model, args.force);
    }

    // Neither specified - show help
    bail!(
        "Please specify a model to install:\n\
         \n\
         LLM models (for local inference):\n\
         \x20 memvid models install phi-3.5-mini\n\
         \x20 memvid models install phi-3.5-mini-q8\n\
         \n\
         CLIP models (for visual search):\n\
         \x20 memvid models install --clip mobileclip-s2\n\
         \x20 memvid models install --clip mobileclip-s2-fp16\n\
         \x20 memvid models install --clip siglip-base\n\
         \n\
         NER models (for Logic-Mesh entity extraction):\n\
         \x20 memvid models install --ner distilbert-ner\n\
         \n\
         Note: Whisper models auto-download on first use (no install needed)"
    );
}

/// Handle CLIP model installation
fn handle_clip_install(config: &CliConfig, model: ClipModel, force: bool) -> Result<()> {
    let vision_path = clip_vision_path(config, model);
    let text_path = clip_text_path(config, model);

    if is_clip_model_installed(config, model) && !force {
        println!(
            "{} is already installed at {}",
            model.display_name(),
            clip_models_dir(config).display()
        );
        println!("Use --force to re-download.");
        return Ok(());
    }

    if config.offline {
        bail!(
            "Cannot install models while offline (MEMVID_OFFLINE=1). \
             Run without MEMVID_OFFLINE to download the model."
        );
    }

    // Create the directory structure
    fs::create_dir_all(clip_models_dir(config))?;

    println!("Installing {}...", model.display_name());
    println!("Dimensions: {}", model.dimensions());
    println!("Total size: {:.1} MB", model.total_size_mb());
    println!();

    // Download vision encoder
    println!("Downloading vision encoder...");
    download_file(model.vision_url(), &vision_path)?;

    // Download text encoder
    println!();
    println!("Downloading text encoder...");
    download_file(model.text_url(), &text_path)?;

    // Calculate total size
    let vision_size = fs::metadata(&vision_path).map(|m| m.len()).unwrap_or(0);
    let text_size = fs::metadata(&text_path).map(|m| m.len()).unwrap_or(0);
    let total_size = vision_size + text_size;

    println!();
    println!(
        "Successfully installed {} ({:.1} MB)",
        model.display_name(),
        total_size as f64 / 1_000_000.0
    );
    println!("Vision encoder: {}", vision_path.display());
    println!("Text encoder: {}", text_path.display());
    println!();
    println!("Usage:");
    println!("  memvid put photos.mv2 --input ./images/ --clip");
    println!("  memvid find photos.mv2 --query \"sunset over ocean\" --mode clip");

    Ok(())
}

/// Handle NER model installation
fn handle_ner_install(config: &CliConfig, model: NerModel, force: bool) -> Result<()> {
    let model_path = ner_model_path(config, model);
    let tokenizer_path = ner_tokenizer_path(config, model);

    if is_ner_model_installed(config, model) && !force {
        println!(
            "{} is already installed at {}",
            model.display_name(),
            model_path.parent().unwrap_or(&model_path).display()
        );
        println!("Use --force to re-download.");
        return Ok(());
    }

    if config.offline {
        bail!(
            "Cannot install models while offline (MEMVID_OFFLINE=1). \
             Run without MEMVID_OFFLINE to download the model."
        );
    }

    // Create the directory structure
    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Installing {}...", model.display_name());
    println!("Size: {:.1} MB", model.size_mb());
    println!();

    // Download model ONNX
    println!("Downloading model...");
    download_file(model.model_url(), &model_path)?;

    // Download tokenizer
    println!();
    println!("Downloading tokenizer...");
    download_file(model.tokenizer_url(), &tokenizer_path)?;

    // Calculate total size
    let model_size = fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
    let tokenizer_size = fs::metadata(&tokenizer_path).map(|m| m.len()).unwrap_or(0);
    let total_size = model_size + tokenizer_size;

    println!();
    println!(
        "Successfully installed {} ({:.1} MB)",
        model.display_name(),
        total_size as f64 / 1_000_000.0
    );
    println!("Model: {}", model_path.display());
    println!("Tokenizer: {}", tokenizer_path.display());
    println!();
    println!("Usage:");
    println!("  memvid enrich file.mv2 --logic-mesh");
    println!("  memvid follow traverse file.mv2 --start \"John\" --link manager");

    Ok(())
}

// Note: Whisper model installation is no longer needed - Candle auto-downloads from HuggingFace.
// The --whisper install flag now just shows an informational message (see handle_models_install).

/// Handle LLM model installation
fn handle_llm_install(config: &CliConfig, model: LlmModel, force: bool) -> Result<()> {
    let target_path = llm_model_path(config, model);

    if is_llm_model_installed(config, model) && !force {
        println!(
            "{} is already installed at {}",
            model.display_name(),
            target_path.display()
        );
        println!("Use --force to re-download.");
        return Ok(());
    }

    if config.offline {
        bail!(
            "Cannot install models while offline (MEMVID_OFFLINE=1). \
             Run without MEMVID_OFFLINE to download the model."
        );
    }

    // Create the directory structure
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Installing {}...", model.display_name());
    println!("Repository: {}", model.hf_repo());
    println!("File: {}", model.hf_filename());
    println!(
        "Expected size: {:.1} GB",
        model.expected_size_bytes() as f64 / 1_000_000_000.0
    );
    println!();

    // Download using curl
    download_llm_model(model, &target_path)?;

    // Verify the download
    let metadata = fs::metadata(&target_path)?;
    let size = metadata.len();

    // Allow 10% variance in file size
    let min_size = (model.expected_size_bytes() as f64 * 0.9) as u64;
    let max_size = (model.expected_size_bytes() as f64 * 1.1) as u64;

    if size < min_size || size > max_size {
        eprintln!(
            "Warning: Downloaded file size ({:.2} GB) differs significantly from expected ({:.2} GB)",
            size as f64 / 1_000_000_000.0,
            model.expected_size_bytes() as f64 / 1_000_000_000.0
        );
    }

    println!();
    println!(
        "Successfully installed {} ({:.2} GB)",
        model.display_name(),
        size as f64 / 1_000_000_000.0
    );
    println!("Location: {}", target_path.display());

    Ok(())
}

/// Download a file from a URL using curl
fn download_file(url: &str, target_path: &Path) -> Result<()> {
    println!("URL: {}", url);

    let status = std::process::Command::new("curl")
        .args([
            "-L",             // Follow redirects
            "--progress-bar", // Show progress bar
            "-o",
            target_path
                .to_str()
                .ok_or_else(|| anyhow!("Invalid target path"))?,
            url,
        ])
        .status()?;

    if !status.success() {
        // Clean up partial download
        let _ = fs::remove_file(target_path);
        bail!("Download failed. Please check your internet connection and try again.");
    }

    Ok(())
}

/// Download an LLM model from Hugging Face
fn download_llm_model(model: LlmModel, target_path: &Path) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model.hf_repo(),
        model.hf_filename()
    );

    println!("Downloading from Hugging Face...");
    download_file(&url, target_path)
}

/// Handle listing models
pub fn handle_models_list(config: &CliConfig, args: ModelsListArgs) -> Result<()> {
    let fastembed_installed = scan_fastembed_cache(config);

    if args.json {
        return handle_models_list_json(config, &fastembed_installed);
    }

    // Check which sections to show
    let show_all = args.model_type.is_none();
    let show_embedding = show_all || matches!(args.model_type, Some(ModelType::Embedding));
    let show_reranker = show_all || matches!(args.model_type, Some(ModelType::Reranker));
    let show_llm = show_all || matches!(args.model_type, Some(ModelType::Llm));
    let show_clip = show_all || matches!(args.model_type, Some(ModelType::Clip));
    let show_ner = show_all || matches!(args.model_type, Some(ModelType::Ner));
    let show_whisper = show_all || matches!(args.model_type, Some(ModelType::Whisper));
    let show_external = show_all || matches!(args.model_type, Some(ModelType::External));

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                       MEMVID MODEL CATALOG                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Show models directory
    println!("Models Directory: {}", config.models_dir.display());
    println!();

    // =========================================================================
    // Embedding Models
    // =========================================================================
    if show_embedding {
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ 📊 EMBEDDING MODELS (Semantic Search)                            │");
        println!("├──────────────────────────────────────────────────────────────────┤");

        for model in EmbeddingModel::all() {
            let is_installed = fastembed_installed.iter().any(|(name, _, _)| {
                name.contains(&model.hf_repo().replace("/", "--").replace("--", "/"))
            });

            let status = if is_installed {
                "✓ installed"
            } else {
                "○ available"
            };
            let default_marker = if model.is_default() { " (default)" } else { "" };

            println!(
                "│ {:20} {:4}D  {:>4} MB  {:15}{}",
                model.cli_name(),
                model.dimensions(),
                model.size_mb(),
                status,
                default_marker
            );
        }

        println!("│                                                                  │");
        println!("│ Usage: memvid put mem.mv2 --input doc.pdf --embedding            │");
        println!("│        --embedding-model nomic                                   │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // =========================================================================
    // Reranker Models
    // =========================================================================
    if show_reranker {
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ 🔄 RERANKER MODELS (Result Reranking)                            │");
        println!("├──────────────────────────────────────────────────────────────────┤");

        for model in RerankerModel::all() {
            let is_installed = fastembed_installed.iter().any(|(name, _, _)| {
                let repo = model.hf_repo();
                name.to_lowercase()
                    .contains(&repo.to_lowercase().replace("/", "--").replace("--", "/"))
                    || name
                        .to_lowercase()
                        .contains(&repo.split('/').last().unwrap_or("").to_lowercase())
            });

            let status = if is_installed {
                "✓ installed"
            } else {
                "○ available"
            };
            let default_marker = if model.is_default() { " (default)" } else { "" };

            println!(
                "│ {:25} {:>4} MB  {:12}  {:12}{}",
                model.cli_name(),
                model.size_mb(),
                model.language(),
                status,
                default_marker
            );
        }

        println!("│                                                                  │");
        println!("│ Reranking is automatic in hybrid search mode (--mode auto)       │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // =========================================================================
    // LLM Models
    // =========================================================================
    if show_llm {
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ 🤖 LLM MODELS (Local Inference)                                  │");
        println!("├──────────────────────────────────────────────────────────────────┤");

        for model in LlmModel::all() {
            let is_installed = is_llm_model_installed(config, model);
            let status = if is_installed {
                "✓ installed"
            } else {
                "○ available"
            };
            let default_marker = if model.is_default() { " (default)" } else { "" };

            println!(
                "│ {:20} {:>5.1} GB  {:15}{}",
                model.cli_name(),
                model.expected_size_bytes() as f64 / 1_000_000_000.0,
                status,
                default_marker
            );

            if is_installed {
                println!("│   Path: {}", llm_model_path(config, model).display());
            }
        }

        println!("│                                                                  │");
        println!("│ Install: memvid models install phi-3.5-mini                      │");
        println!("│ Usage:   memvid ask file.mv2 --question \"...\" --model candle:phi │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // =========================================================================
    // CLIP Models (Visual Search)
    // =========================================================================
    if show_clip {
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ 🖼️  CLIP MODELS (Visual Search)                                   │");
        println!("├──────────────────────────────────────────────────────────────────┤");

        for model in ClipModel::all() {
            let (status, _, _) = clip_model_status(config, model);
            let default_marker = if model.is_default() { " (default)" } else { "" };

            println!(
                "│ {:20} {:4}D  {:>6.1} MB  {:15}{}",
                model.cli_name(),
                model.dimensions(),
                model.total_size_mb(),
                status,
                default_marker
            );
        }

        println!("│                                                                  │");
        println!("│ Install: memvid models install --clip mobileclip-s2              │");
        println!("│ Usage:   memvid put photos.mv2 --input ./images/ --clip          │");
        println!("│          memvid find photos.mv2 --query \"sunset\" --mode clip     │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // =========================================================================
    // NER Models (Logic-Mesh Entity Extraction)
    // =========================================================================
    if show_ner {
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ 🔗 NER MODELS (Logic-Mesh Entity Extraction)                      │");
        println!("├──────────────────────────────────────────────────────────────────┤");

        for model in NerModel::all() {
            let (status, _, _) = ner_model_status(config, model);
            let default_marker = if model.is_default() { " (default)" } else { "" };

            println!(
                "│ {:20} {:>6.1} MB  {:15}{}",
                model.cli_name(),
                model.size_mb(),
                status,
                default_marker
            );
        }

        println!("│                                                                  │");
        println!("│ Install: memvid models install --ner distilbert-ner              │");
        println!("│ Usage:   memvid put file.mv2 --input doc.txt --logic-mesh        │");
        println!("│          memvid follow traverse file.mv2 --start \"John\"          │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // =========================================================================
    // Whisper Models (Audio Transcription) - Using Candle (auto-download)
    // =========================================================================
    if show_whisper {
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ 🎙️  WHISPER MODELS (Audio Transcription via Candle)               │");
        println!("├──────────────────────────────────────────────────────────────────┤");
        println!("│ whisper-small-en          244 MB  Auto-download    (default)     │");
        println!("│ whisper-small             244 MB  Auto-download    multilingual  │");
        println!("│ whisper-tiny-en            75 MB  Auto-download    fastest       │");
        println!("│ whisper-base-en           145 MB  Auto-download                  │");
        println!("│                                                                  │");
        println!("│ Models download automatically from HuggingFace on first use.     │");
        println!("│ GPU acceleration: --features metal (Mac) or --features cuda      │");
        println!("│                                                                  │");
        println!("│ Usage: memvid put file.mv2 --input audio.mp3 --transcribe        │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // =========================================================================
    // External Models (API-based)
    // =========================================================================
    if show_external {
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ ☁️  EXTERNAL MODELS (API-based, no download required)             │");
        println!("├──────────────────────────────────────────────────────────────────┤");

        for provider in EXTERNAL_EMBEDDING_PROVIDERS {
            let api_key_set = std::env::var(provider.env_var).is_ok();
            let key_status = if api_key_set {
                format!("{} ✓", provider.env_var)
            } else {
                format!("{} ○", provider.env_var)
            };

            println!("│ {} ({}):", provider.name, key_status);

            for (model_name, dim, desc) in provider.models.iter() {
                println!("│   {:30} {:4}D  {}", model_name, dim, desc);
            }
            println!("│");
        }

        println!("│ Usage: export OPENAI_API_KEY=sk-...                              │");
        println!("│        memvid put mem.mv2 --input doc.pdf --embedding            │");
        println!("│        --embedding-model openai-small                            │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // =========================================================================
    // Installed Models Summary
    // =========================================================================
    if !fastembed_installed.is_empty() {
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!(
            "│ 📦 INSTALLED MODELS (cached in {})     │",
            config.models_dir.display()
        );
        println!("├──────────────────────────────────────────────────────────────────┤");

        let mut total_size: u64 = 0;

        for (name, _path, size) in &fastembed_installed {
            total_size += size;
            println!(
                "│ {:40} {:>8.1} MB",
                if name.len() > 40 {
                    format!("{}...", &name[..37])
                } else {
                    name.clone()
                },
                *size as f64 / 1_000_000.0
            );
        }

        // Add LLM models
        for model in LlmModel::all() {
            if is_llm_model_installed(config, model) {
                let path = llm_model_path(config, model);
                if let Ok(meta) = fs::metadata(&path) {
                    total_size += meta.len();
                    println!(
                        "│ {:40} {:>8.1} MB",
                        model.display_name(),
                        meta.len() as f64 / 1_000_000.0
                    );
                }
            }
        }

        println!("├──────────────────────────────────────────────────────────────────┤");
        println!("│ Total: {:>55.1} MB │", total_size as f64 / 1_000_000.0);
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // =========================================================================
    // Quick Help
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║ QUICK REFERENCE                                                  ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║ memvid models list                    List all models            ║");
    println!("║ memvid models list --model-type llm   List only LLM models       ║");
    println!("║ memvid models install phi-3.5-mini    Install LLM model          ║");
    println!("║ memvid models remove phi-3.5-mini     Remove LLM model           ║");
    println!("║ memvid models verify                  Verify installed models    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Handle JSON output for models list
fn handle_models_list_json(
    config: &CliConfig,
    fastembed_installed: &[(String, PathBuf, u64)],
) -> Result<()> {
    let output = serde_json::json!({
        "models_dir": config.models_dir,
        "embedding_models": EmbeddingModel::all().map(|m| {
            let is_installed = fastembed_installed
                .iter()
                .any(|(name, _, _)| name.contains(m.hf_repo()));
            serde_json::json!({
                "name": m.cli_name(),
                "display_name": m.display_name(),
                "dimensions": m.dimensions(),
                "size_mb": m.size_mb(),
                "hf_repo": m.hf_repo(),
                "installed": is_installed,
                "is_default": m.is_default(),
            })
        }).collect::<Vec<_>>(),
        "reranker_models": RerankerModel::all().map(|m| {
            serde_json::json!({
                "name": m.cli_name(),
                "display_name": m.display_name(),
                "size_mb": m.size_mb(),
                "hf_repo": m.hf_repo(),
                "language": m.language(),
                "is_default": m.is_default(),
            })
        }).collect::<Vec<_>>(),
        "llm_models": LlmModel::all().map(|m| {
            serde_json::json!({
                "name": m.cli_name(),
                "display_name": m.display_name(),
                "size_gb": m.expected_size_bytes() as f64 / 1_000_000_000.0,
                "hf_repo": m.hf_repo(),
                "installed": is_llm_model_installed(config, m),
                "path": if is_llm_model_installed(config, m) {
                    Some(llm_model_path(config, m))
                } else {
                    None
                },
                "is_default": m.is_default(),
            })
        }).collect::<Vec<_>>(),
        "external_providers": EXTERNAL_EMBEDDING_PROVIDERS.iter().map(|p| {
            serde_json::json!({
                "name": p.name,
                "env_var": p.env_var,
                "configured": std::env::var(p.env_var).is_ok(),
                "models": p.models.iter().map(|(name, dim, desc)| {
                    serde_json::json!({
                        "name": name,
                        "dimensions": dim,
                        "description": desc,
                    })
                }).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
        "installed_cache": fastembed_installed.iter().map(|(name, path, size)| {
            serde_json::json!({
                "name": name,
                "path": path,
                "size_bytes": size,
            })
        }).collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Handle model removal
pub fn handle_models_remove(config: &CliConfig, args: ModelsRemoveArgs) -> Result<()> {
    let model = args.model;
    let path = llm_model_path(config, model);

    if !path.exists() {
        println!("{} is not installed.", model.display_name());
        return Ok(());
    }

    if !args.yes {
        print!(
            "Remove {} ({})? [y/N] ",
            model.display_name(),
            path.display()
        );
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    fs::remove_file(&path)?;

    // Try to remove parent directory if empty
    if let Some(parent) = path.parent() {
        let _ = fs::remove_dir(parent);
    }

    println!("Removed {}.", model.display_name());
    Ok(())
}

/// Handle model verification
pub fn handle_models_verify(config: &CliConfig, args: ModelsVerifyArgs) -> Result<()> {
    let models_to_verify: Vec<LlmModel> = match args.model {
        Some(m) => vec![m],
        None => LlmModel::all()
            .filter(|m| is_llm_model_installed(config, *m))
            .collect(),
    };

    if models_to_verify.is_empty() {
        println!("No LLM models installed to verify.");
        return Ok(());
    }

    let mut all_ok = true;

    for model in models_to_verify {
        let path = llm_model_path(config, model);
        print!("Verifying {}... ", model.display_name());
        io::stdout().flush()?;

        match verify_model_file(&path, model) {
            Ok(()) => println!("OK"),
            Err(err) => {
                println!("FAILED");
                eprintln!("  Error: {}", err);
                all_ok = false;
            }
        }
    }

    if !all_ok {
        bail!("Some models failed verification.");
    }

    Ok(())
}

/// Verify a model file exists and has reasonable size
fn verify_model_file(path: &Path, model: LlmModel) -> Result<()> {
    if !path.exists() {
        bail!("Model file does not exist");
    }

    let metadata = fs::metadata(path)?;
    let size = metadata.len();

    // Check minimum size (at least 50% of expected)
    let min_size = model.expected_size_bytes() / 2;
    if size < min_size {
        bail!(
            "Model file too small ({:.2} GB, expected at least {:.2} GB)",
            size as f64 / 1_000_000_000.0,
            min_size as f64 / 1_000_000_000.0
        );
    }

    // Check GGUF magic bytes
    let mut file = fs::File::open(path)?;
    let mut magic = [0u8; 4];
    io::Read::read_exact(&mut file, &mut magic)?;

    // GGUF magic is "GGUF" (0x46554747)
    if &magic != b"GGUF" {
        bail!("Invalid GGUF file (bad magic bytes)");
    }

    Ok(())
}

/// Get the path to an installed model, or None if not installed
pub fn get_installed_model_path(config: &CliConfig, model: LlmModel) -> Option<PathBuf> {
    let path = llm_model_path(config, model);
    if path.exists() && path.is_file() {
        Some(path)
    } else {
        None
    }
}

/// Get the default LLM model for enrichment
pub fn default_enrichment_model() -> LlmModel {
    LlmModel::Phi35Mini
}
