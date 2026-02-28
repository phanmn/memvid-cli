//! Data operation command handlers (put, update, delete, api-fetch).
//!
//! Focused on ingesting files/bytes into `.mv2` memories, updating/deleting frames,
//! and fetching remote content. Keeps ingestion flags and capacity checks consistent
//! with the core engine while presenting concise CLI output.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::io::{self, Cursor, Write};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use blake3::hash;
use clap::{ArgAction, Args};
use color_thief::{get_palette, Color, ColorFormat};
use exif::{Reader as ExifReader, Tag, Value as ExifValue};
use glob::glob;
use image::ImageReader;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use infer;
#[cfg(feature = "clip")]
use memvid_core::clip::render_pdf_pages_for_clip;
#[cfg(feature = "clip")]
use memvid_core::get_image_info;
use memvid_core::table::{extract_tables, store_table, TableExtractionOptions};
#[cfg(feature = "clip")]
use memvid_core::FrameRole;
use memvid_core::{
    normalize_text, AudioSegmentMetadata, DocAudioMetadata, DocExifMetadata, DocGpsMetadata,
    DocMetadata, DocumentFormat, EnrichmentEngine, MediaManifest, Memvid, MemvidError, PutOptions,
    ReaderHint, ReaderRegistry, RulesEngine, Stats, Tier, TimelineQueryBuilder,
    MEMVID_EMBEDDING_DIMENSION_KEY, MEMVID_EMBEDDING_MODEL_KEY, MEMVID_EMBEDDING_PROVIDER_KEY,
};
#[cfg(feature = "parallel_segments")]
use memvid_core::{BuildOpts, ParallelInput, ParallelPayload};
use pdf_extract::OutputError as PdfExtractError;
use serde_json::json;
use tracing::{info, warn};
use tree_magic_mini::from_u8 as magic_from_u8;
use uuid::Uuid;

const MAX_SEARCH_TEXT_LEN: usize = 32_768;
/// Maximum characters for embedding text to avoid exceeding OpenAI's 8192 token limit.
/// Using ~4 chars/token estimate, 30K chars ≈ 7.5K tokens (safe margin).
const MAX_EMBEDDING_TEXT_LEN: usize = 20_000;

/// Parse a timestamp from either epoch seconds or human-readable date formats.
/// Accepts:
/// - Epoch seconds: "1673740800"
/// - ISO format: "2023-01-15"
/// - US format: "Jan 15, 2023" or "January 15, 2023"
/// - Slash format: "01/15/2023" or "1/15/2023"
fn parse_timestamp(s: &str) -> Result<i64, String> {
    let s = s.trim();

    // Try parsing as epoch timestamp first (pure digits, optionally negative)
    if s.chars().all(|c| c.is_ascii_digit())
        || (s.starts_with('-') && s[1..].chars().all(|c| c.is_ascii_digit()))
    {
        return s
            .parse::<i64>()
            .map_err(|e| format!("Invalid epoch timestamp: {e}"));
    }

    // Try various date formats
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};

    // Parse date and return midnight UTC
    fn date_to_epoch(date: NaiveDate) -> i64 {
        let datetime = NaiveDateTime::new(date, NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        Utc.from_utc_datetime(&datetime).timestamp()
    }

    // Try ISO format: "2023-01-15"
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(date_to_epoch(date));
    }

    // Try US formats: "Jan 15, 2023" or "January 15, 2023"
    if let Ok(date) = NaiveDate::parse_from_str(s, "%b %d, %Y") {
        return Ok(date_to_epoch(date));
    }
    if let Ok(date) = NaiveDate::parse_from_str(s, "%B %d, %Y") {
        return Ok(date_to_epoch(date));
    }

    // Try slash format: "01/15/2023" or "1/15/2023"
    if let Ok(date) = NaiveDate::parse_from_str(s, "%m/%d/%Y") {
        return Ok(date_to_epoch(date));
    }

    // Try European format: "15-01-2023" or "15/01/2023"
    if let Ok(date) = NaiveDate::parse_from_str(s, "%d-%m-%Y") {
        return Ok(date_to_epoch(date));
    }
    if let Ok(date) = NaiveDate::parse_from_str(s, "%d/%m/%Y") {
        return Ok(date_to_epoch(date));
    }

    Err(format!(
        "Invalid timestamp format: '{}'. Use epoch seconds (1673740800) or date format (Jan 15, 2023 / 2023-01-15 / 01/15/2023)",
        s
    ))
}
const COLOR_PALETTE_SIZE: u8 = 5;
const COLOR_PALETTE_QUALITY: u8 = 8;
const MAGIC_SNIFF_BYTES: usize = 16;
/// Maximum number of images to extract from a PDF for CLIP visual search.
/// Set high to capture all images; processing stops when no more images found.
#[cfg(feature = "clip")]
const CLIP_PDF_MAX_IMAGES: usize = 100;
#[cfg(feature = "clip")]
const CLIP_PDF_TARGET_PX: u32 = 768;

/// Truncate text for embedding to fit within OpenAI token limits.
/// Returns a truncated string that fits within MAX_EMBEDDING_TEXT_LEN.
fn truncate_for_embedding(text: &str) -> String {
    if text.len() <= MAX_EMBEDDING_TEXT_LEN {
        text.to_string()
    } else {
        // Find a char boundary to avoid splitting UTF-8
        let truncated = &text[..MAX_EMBEDDING_TEXT_LEN];
        // Try to find the last complete character
        let end = truncated
            .char_indices()
            .rev()
            .next()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(MAX_EMBEDDING_TEXT_LEN);
        text[..end].to_string()
    }
}

fn apply_embedding_identity_metadata(options: &mut PutOptions, runtime: &EmbeddingRuntime) {
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_PROVIDER_KEY.to_string(),
        runtime.provider_kind().to_string(),
    );
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_MODEL_KEY.to_string(),
        runtime.provider_model_id(),
    );
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
        runtime.dimension().to_string(),
    );
}

fn apply_embedding_identity_metadata_from_choice(
    options: &mut PutOptions,
    model: EmbeddingModelChoice,
    dimension: usize,
    model_id_override: Option<&str>,
) {
    let provider = match model {
        EmbeddingModelChoice::OpenAILarge
        | EmbeddingModelChoice::OpenAISmall
        | EmbeddingModelChoice::OpenAIAda => "openai",
        EmbeddingModelChoice::Nvidia => "nvidia",
        _ => "fastembed",
    };

    let model_id = match model {
        EmbeddingModelChoice::Nvidia => model_id_override
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                value
                    .trim_start_matches("nvidia:")
                    .trim_start_matches("nv:")
            })
            .filter(|value| !value.is_empty())
            .map(|value| {
                if value.contains('/') {
                    value.to_string()
                } else {
                    format!("nvidia/{value}")
                }
            })
            .unwrap_or_else(|| model.canonical_model_id().to_string()),
        _ => model.canonical_model_id().to_string(),
    };

    options.extra_metadata.insert(
        MEMVID_EMBEDDING_PROVIDER_KEY.to_string(),
        provider.to_string(),
    );
    options
        .extra_metadata
        .insert(MEMVID_EMBEDDING_MODEL_KEY.to_string(), model_id);
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
        dimension.to_string(),
    );
}

/// Get the default local model path for contextual retrieval (phi-3.5-mini).
/// Uses MEMVID_MODELS_DIR or defaults to ~/.memvid/models/llm/phi-3-mini-q4/Phi-3.5-mini-instruct-Q4_K_M.gguf
#[cfg(feature = "llama-cpp")]
fn get_local_contextual_model_path() -> Result<PathBuf> {
    let models_dir = if let Ok(custom) = env::var("MEMVID_MODELS_DIR") {
        // Handle ~ expansion manually
        let expanded = if custom.starts_with("~/") {
            if let Ok(home) = env::var("HOME") {
                PathBuf::from(home).join(&custom[2..])
            } else {
                PathBuf::from(&custom)
            }
        } else {
            PathBuf::from(&custom)
        };
        expanded
    } else {
        // Default to ~/.memvid/models
        let home = env::var("HOME").map_err(|_| anyhow!("HOME environment variable not set"))?;
        PathBuf::from(home).join(".memvid").join("models")
    };

    let model_path = models_dir
        .join("llm")
        .join("phi-3-mini-q4")
        .join("Phi-3.5-mini-instruct-Q4_K_M.gguf");

    if model_path.exists() {
        Ok(model_path)
    } else {
        Err(anyhow!(
            "Local model not found at {}. Run 'memvid models install phi-3.5-mini' first.",
            model_path.display()
        ))
    }
}

use pathdiff::diff_paths;

use crate::api_fetch::{run_api_fetch, ApiFetchCommand, ApiFetchMode};
use crate::commands::{extension_from_mime, frame_to_json, print_frame_summary};
use crate::config::{
    load_embedding_runtime_for_mv2, CliConfig, EmbeddingModelChoice, EmbeddingRuntime,
};
use crate::contextual::{apply_contextual_prefixes, ContextualEngine};
use crate::error::{CapacityExceededMessage, DuplicateUriError};
use crate::utils::{
    apply_lock_cli, ensure_cli_mutation_allowed, format_bytes, frame_status_str, read_payload,
    select_frame,
};
#[cfg(feature = "temporal_enrich")]
use memvid_core::enrich_chunks as temporal_enrich_chunks;

/// Helper function to parse boolean flags from environment variables
fn parse_bool_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Helper function to check for parallel segments environment override
#[cfg(feature = "parallel_segments")]
fn parallel_env_override() -> Option<bool> {
    std::env::var("MEMVID_PARALLEL_SEGMENTS")
        .ok()
        .and_then(|value| parse_bool_flag(&value))
}

/// Check if CLIP model files are installed
#[cfg(feature = "clip")]
#[allow(dead_code)]
fn is_clip_model_installed() -> bool {
    let models_dir = if let Ok(custom) = env::var("MEMVID_MODELS_DIR") {
        PathBuf::from(custom)
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".memvid/models")
    } else {
        PathBuf::from(".memvid/models")
    };

    let model_name = env::var("MEMVID_CLIP_MODEL").unwrap_or_else(|_| "mobileclip-s2".to_string());

    let vision_model = models_dir.join(format!("{}_vision.onnx", model_name));
    let text_model = models_dir.join(format!("{}_text.onnx", model_name));

    vision_model.exists() && text_model.exists()
}

/// Minimum confidence threshold for NER entities (0.0-1.0)
/// Note: Lower threshold captures more entities but also more noise
#[cfg(feature = "logic_mesh")]
const NER_MIN_CONFIDENCE: f32 = 0.50;

/// Minimum length for entity names (characters)
#[cfg(feature = "logic_mesh")]
const NER_MIN_ENTITY_LEN: usize = 2;

/// Maximum gap (in characters) between adjacent entities to merge
/// Allows merging "S" + "&" + "P Global" when they're close together
#[cfg(feature = "logic_mesh")]
const MAX_MERGE_GAP: usize = 3;

/// Merge adjacent NER entities that appear to be tokenization artifacts.
///
/// The BERT tokenizer splits names at special characters like `&`, producing
/// separate entities like "S", "&", "P Global" instead of "S&P Global".
/// This function merges adjacent entities of the same type when they're
/// contiguous or nearly contiguous in the original text.
#[cfg(feature = "logic_mesh")]
fn merge_adjacent_entities(
    entities: Vec<memvid_core::ExtractedEntity>,
    original_text: &str,
) -> Vec<memvid_core::ExtractedEntity> {
    use memvid_core::ExtractedEntity;

    if entities.is_empty() {
        return entities;
    }

    // Sort by byte position
    let mut sorted: Vec<ExtractedEntity> = entities;
    sorted.sort_by_key(|e| e.byte_start);

    let mut merged: Vec<ExtractedEntity> = Vec::new();

    for entity in sorted {
        if let Some(last) = merged.last_mut() {
            // Check if this entity is adjacent to the previous one and same type
            let gap = entity.byte_start.saturating_sub(last.byte_end);
            let same_type = last.entity_type == entity.entity_type;

            // Check what's in the gap (if any)
            let gap_is_mergeable = if gap == 0 {
                true
            } else if gap <= MAX_MERGE_GAP
                && entity.byte_start <= original_text.len()
                && last.byte_end <= original_text.len()
            {
                // Gap content should be whitespace (but not newlines), punctuation, or connectors
                let gap_text = &original_text[last.byte_end..entity.byte_start];
                // Don't merge across newlines - that indicates separate entities
                if gap_text.contains('\n') || gap_text.contains('\r') {
                    false
                } else {
                    gap_text.chars().all(|c| {
                        c == ' ' || c == '\t' || c == '&' || c == '-' || c == '\'' || c == '.'
                    })
                }
            } else {
                false
            };

            if same_type && gap_is_mergeable {
                // Merge: extend the previous entity to include this one
                let merged_text = if entity.byte_end <= original_text.len() {
                    original_text[last.byte_start..entity.byte_end].to_string()
                } else {
                    format!("{}{}", last.text, entity.text)
                };

                tracing::debug!(
                    "Merged entities: '{}' + '{}' -> '{}' (gap: {} chars)",
                    last.text,
                    entity.text,
                    merged_text,
                    gap
                );

                last.text = merged_text;
                last.byte_end = entity.byte_end;
                // Average the confidence
                last.confidence = (last.confidence + entity.confidence) / 2.0;
                continue;
            }
        }

        merged.push(entity);
    }

    merged
}

/// Filter and clean extracted NER entities to remove noise.
/// Returns true if the entity should be kept.
#[cfg(feature = "logic_mesh")]
fn is_valid_entity(text: &str, confidence: f32) -> bool {
    // Filter by confidence
    if confidence < NER_MIN_CONFIDENCE {
        return false;
    }

    let trimmed = text.trim();

    // Filter by minimum length
    if trimmed.len() < NER_MIN_ENTITY_LEN {
        return false;
    }

    // Filter out entities that are just whitespace or punctuation
    if trimmed
        .chars()
        .all(|c| c.is_whitespace() || c.is_ascii_punctuation())
    {
        return false;
    }

    // Filter out entities containing newlines (malformed extractions)
    if trimmed.contains('\n') || trimmed.contains('\r') {
        return false;
    }

    // Filter out entities that look like partial words (start/end with ##)
    if trimmed.starts_with("##") || trimmed.ends_with("##") {
        return false;
    }

    // Filter out entities ending with period (likely sentence fragments)
    if trimmed.ends_with('.') {
        return false;
    }

    let lower = trimmed.to_lowercase();

    // Filter out common single-letter false positives
    if trimmed.len() == 1 {
        return false;
    }

    // Filter out 2-letter tokens that are common noise
    if trimmed.len() == 2 {
        // Allow legitimate 2-letter entities like "EU", "UN", "UK", "US"
        if matches!(lower.as_str(), "eu" | "un" | "uk" | "us" | "ai" | "it") {
            return true;
        }
        // Filter other 2-letter tokens
        return false;
    }

    // Filter out common noise words
    if matches!(
        lower.as_str(),
        "the" | "api" | "url" | "pdf" | "inc" | "ltd" | "llc"
    ) {
        return false;
    }

    // Filter out partial tokenization artifacts
    // e.g., "S&P Global" becomes "P Global", "IHS Markit" becomes "HS Markit"
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() >= 2 {
        let first_word = words[0];
        // If first word is 1-2 chars and looks like a fragment, filter it
        if first_word.len() <= 2 && first_word.chars().all(|c| c.is_uppercase()) {
            // But allow "US Bank", "UK Office", etc.
            if !matches!(
                first_word.to_lowercase().as_str(),
                "us" | "uk" | "eu" | "un"
            ) {
                return false;
            }
        }
    }

    // Filter entities that start with lowercase (likely partial words)
    if let Some(first_char) = trimmed.chars().next() {
        if first_char.is_lowercase() {
            return false;
        }
    }

    true
}

/// Parse key=value pairs for tags
pub fn parse_key_val(s: &str) -> Result<(String, String)> {
    let (key, value) = s
        .split_once('=')
        .ok_or_else(|| anyhow!("expected KEY=VALUE, got `{s}`"))?;
    Ok((key.to_string(), value.to_string()))
}

/// Lock-related CLI arguments (shared across commands)
#[derive(Args, Clone, Copy, Debug)]
pub struct LockCliArgs {
    /// Maximum time to wait for an active writer before failing (ms)
    #[arg(long = "lock-timeout", value_name = "MS", default_value_t = 250)]
    pub lock_timeout: u64,
    /// Attempt a stale takeover if the recorded writer heartbeat has expired
    #[arg(long = "force", action = ArgAction::SetTrue)]
    pub force: bool,
}

/// Arguments for the `put` subcommand
#[derive(Args)]
pub struct PutArgs {
    /// Path to the memory file to append to
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    /// Emit JSON instead of human-readable output
    #[arg(long)]
    pub json: bool,
    /// Read payload bytes from a file instead of stdin
    #[arg(long, value_name = "PATH")]
    pub input: Option<String>,
    /// Override the derived URI for the document
    #[arg(long, value_name = "URI")]
    pub uri: Option<String>,
    /// Override the derived title for the document
    #[arg(long, value_name = "TITLE")]
    pub title: Option<String>,
    /// Optional timestamp to store on the frame.
    /// Accepts epoch seconds (1673740800) or human-readable dates:
    /// "Jan 15, 2023", "2023-01-15", "01/15/2023"
    #[arg(long, value_parser = parse_timestamp, value_name = "DATE")]
    pub timestamp: Option<i64>,
    /// Track name for the frame
    #[arg(long)]
    pub track: Option<String>,
    /// Kind metadata for the frame
    #[arg(long)]
    pub kind: Option<String>,
    /// Store the payload as a binary video without transcoding
    #[arg(long, action = ArgAction::SetTrue)]
    pub video: bool,
    /// Attempt to attach a transcript (Dev/Enterprise tiers only)
    #[arg(long, action = ArgAction::SetTrue)]
    pub transcript: bool,
    /// Tags to attach in key=value form
    #[arg(long = "tag", value_parser = parse_key_val, value_name = "KEY=VALUE")]
    pub tags: Vec<(String, String)>,
    /// Free-form labels to attach to the frame
    #[arg(long = "label", value_name = "LABEL")]
    pub labels: Vec<String>,
    /// Additional metadata payloads expressed as JSON
    #[arg(long = "metadata", value_name = "JSON")]
    pub metadata: Vec<String>,
    /// Disable automatic tag generation from extracted text
    #[arg(long = "no-auto-tag", action = ArgAction::SetTrue)]
    pub no_auto_tag: bool,
    /// Disable automatic date extraction from content
    #[arg(long = "no-extract-dates", action = ArgAction::SetTrue)]
    pub no_extract_dates: bool,
    /// Disable automatic triplet extraction (SPO facts as MemoryCards)
    #[arg(long = "no-extract-triplets", action = ArgAction::SetTrue)]
    pub no_extract_triplets: bool,
    /// Force audio analysis and tagging when ingesting binary data
    #[arg(long, action = ArgAction::SetTrue)]
    pub audio: bool,
    /// Transcribe audio using Whisper and use the transcript for text embedding
    /// Requires the 'whisper' feature to be enabled
    #[arg(long, action = ArgAction::SetTrue)]
    pub transcribe: bool,
    /// Override the default segment duration (in seconds) when generating audio snippets
    #[arg(long = "audio-segment-seconds", value_name = "SECS")]
    pub audio_segment_seconds: Option<u32>,
    /// Compute semantic embeddings using the installed model
    /// Increases storage overhead from ~5KB to ~270KB per document
    #[arg(long, aliases = ["embeddings"], action = ArgAction::SetTrue, conflicts_with = "no_embedding")]
    pub embedding: bool,
    /// Explicitly disable embeddings (default, but kept for clarity)
    #[arg(long = "no-embedding", action = ArgAction::SetTrue)]
    pub no_embedding: bool,
    /// Path to a JSON file containing a pre-computed embedding vector (e.g., from OpenAI)
    /// The JSON file should contain a flat array of floats like [0.1, 0.2, ...]
    #[arg(long = "embedding-vec", value_name = "JSON_PATH")]
    pub embedding_vec: Option<PathBuf>,
    /// Embedding model identity for the provided `--embedding-vec` vector.
    ///
    /// This is written into per-frame `extra_metadata` (`memvid.embedding.*`) so that
    /// semantic queries can auto-select the correct runtime across CLI/SDKs.
    ///
    /// Options: bge-small, bge-base, nomic, gte-large, openai, openai-small, openai-ada, nvidia
    /// (also accepts canonical IDs like `text-embedding-3-small` and `BAAI/bge-small-en-v1.5`).
    #[arg(
        long = "embedding-vec-model",
        value_name = "EMB_MODEL",
        requires = "embedding_vec"
    )]
    pub embedding_vec_model: Option<String>,
    /// Enable Product Quantization compression for vectors (16x compression)
    /// Reduces vector storage from ~270KB to ~20KB per document with ~95% accuracy
    /// Only applies when --embedding is enabled
    #[arg(long, action = ArgAction::SetTrue)]
    pub vector_compression: bool,
    /// Enable contextual retrieval: prepend LLM-generated context to each chunk before embedding
    /// This improves retrieval accuracy for preference/personalization queries
    /// Requires OPENAI_API_KEY or a local model (phi-3.5-mini)
    #[arg(long, action = ArgAction::SetTrue)]
    pub contextual: bool,
    /// Model to use for contextual retrieval (default: gpt-4o-mini, or "local" for phi-3.5-mini)
    #[arg(long = "contextual-model", value_name = "MODEL")]
    pub contextual_model: Option<String>,
    /// Automatically extract tables from PDF documents
    /// Uses aggressive mode with fallbacks for best results
    #[arg(long, action = ArgAction::SetTrue)]
    pub tables: bool,
    /// Disable automatic table extraction (when --tables is the default)
    #[arg(long = "no-tables", action = ArgAction::SetTrue)]
    pub no_tables: bool,
    /// Time budget for text extraction per document (milliseconds)
    /// When set, extraction stops early and queues the frame for background enrichment
    /// Use `memvid process-queue` to complete extraction later
    #[arg(long = "extraction-budget-ms", value_name = "MS")]
    pub extraction_budget_ms: Option<u64>,
    /// Enable CLIP visual embeddings for images and PDF pages
    /// Allows text-to-image search using natural language queries
    /// Disabled by default; use --clip to enable
    #[cfg(feature = "clip")]
    #[arg(long, action = ArgAction::SetTrue)]
    pub clip: bool,

    /// Disable CLIP visual embeddings (no-op, CLIP is disabled by default)
    #[cfg(feature = "clip")]
    #[arg(long, action = ArgAction::SetTrue, hide = true)]
    pub no_clip: bool,

    /// Replace any existing frame with the same URI instead of inserting a duplicate
    #[arg(long, conflicts_with = "allow_duplicate")]
    pub update_existing: bool,
    /// Allow inserting a new frame even if a frame with the same URI already exists
    #[arg(long)]
    pub allow_duplicate: bool,

    /// Enable temporal enrichment: resolve relative time phrases ("last year") to absolute dates
    /// Prepends resolved temporal context to chunks for improved temporal question accuracy
    /// Requires the temporal_enrich feature in memvid-core
    #[arg(long, action = ArgAction::SetTrue)]
    pub temporal_enrich: bool,

    /// Bind to a dashboard memory ID (UUID or 24-char ObjectId) for capacity and tracking
    /// If the file is not already bound, this will sync the capacity ticket from the dashboard
    #[arg(long = "memory-id", value_name = "ID")]
    pub memory_id: Option<crate::commands::tickets::MemoryId>,

    /// Build Logic-Mesh entity graph during ingestion (opt-in)
    /// Extracts entities (people, organizations, locations, etc.) and their relationships
    /// Enables graph traversal with `memvid follow`
    /// Requires NER model: `memvid models install --ner distilbert-ner`
    #[arg(long, action = ArgAction::SetTrue)]
    pub logic_mesh: bool,

    /// Use the experimental parallel segment builder (requires --features parallel_segments)
    #[cfg(feature = "parallel_segments")]
    #[arg(long, action = ArgAction::SetTrue)]
    pub parallel_segments: bool,
    /// Force the legacy ingestion path even when the parallel feature is available
    #[cfg(feature = "parallel_segments")]
    #[arg(long, action = ArgAction::SetTrue)]
    pub no_parallel_segments: bool,
    /// Target tokens per segment when using the parallel builder
    #[cfg(feature = "parallel_segments")]
    #[arg(long = "parallel-seg-tokens", value_name = "TOKENS")]
    pub parallel_segment_tokens: Option<usize>,
    /// Target pages per segment when using the parallel builder
    #[cfg(feature = "parallel_segments")]
    #[arg(long = "parallel-seg-pages", value_name = "PAGES")]
    pub parallel_segment_pages: Option<usize>,
    /// Worker threads used by the parallel builder
    #[cfg(feature = "parallel_segments")]
    #[arg(long = "parallel-threads", value_name = "N")]
    pub parallel_threads: Option<usize>,
    /// Worker queue depth used by the parallel builder
    #[cfg(feature = "parallel_segments")]
    #[arg(long = "parallel-queue-depth", value_name = "N")]
    pub parallel_queue_depth: Option<usize>,

    /// Store raw binary content along with extracted text.
    /// By default, only extracted text + BLAKE3 hash is stored (--no-raw behavior).
    /// Use --raw when you need to retrieve the original file later.
    #[arg(long = "raw", action = ArgAction::SetTrue)]
    pub raw: bool,

    /// Don't store raw binary content (DEFAULT behavior).
    /// Only extracted text + BLAKE3 hash is stored.
    /// This flag is kept for backwards compatibility but is now the default.
    #[arg(long = "no-raw", action = ArgAction::SetTrue, hide = true)]
    pub no_raw: bool,

    /// Skip ingestion if identical content (by BLAKE3 hash) already exists in the memory.
    /// Returns the existing frame's ID instead of creating a duplicate.
    #[arg(long, action = ArgAction::SetTrue)]
    pub dedup: bool,

    /// Run rules-based memory extraction after ingestion (default: enabled)
    /// Extracts facts, preferences, events, relationships from ingested content
    #[arg(long, action = ArgAction::SetTrue, default_value_t = true)]
    pub enrich: bool,

    /// Disable automatic rules-based enrichment
    #[arg(long = "no-enrich", action = ArgAction::SetTrue)]
    pub no_enrich: bool,

    #[command(flatten)]
    pub lock: LockCliArgs,
}

#[cfg(feature = "parallel_segments")]
impl PutArgs {
    pub fn wants_parallel(&self) -> bool {
        if self.parallel_segments {
            return true;
        }
        if self.no_parallel_segments {
            return false;
        }
        if let Some(env_flag) = parallel_env_override() {
            return env_flag;
        }
        false
    }

    pub fn sanitized_parallel_opts(&self) -> memvid_core::BuildOpts {
        let mut opts = memvid_core::BuildOpts::default();
        if let Some(tokens) = self.parallel_segment_tokens {
            opts.segment_tokens = tokens;
        }
        if let Some(pages) = self.parallel_segment_pages {
            opts.segment_pages = pages;
        }
        if let Some(threads) = self.parallel_threads {
            opts.threads = threads;
        }
        if let Some(depth) = self.parallel_queue_depth {
            opts.queue_depth = depth;
        }
        opts.sanitize();
        opts
    }
}

#[cfg(not(feature = "parallel_segments"))]
impl PutArgs {
    pub fn wants_parallel(&self) -> bool {
        false
    }
}

/// Arguments for the `api-fetch` subcommand
#[derive(Args)]
pub struct ApiFetchArgs {
    /// Path to the memory file to ingest into
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    /// Path to the fetch configuration JSON file
    #[arg(value_name = "CONFIG", value_parser = clap::value_parser!(PathBuf))]
    pub config: PathBuf,
    /// Perform a dry run without writing to the memory
    #[arg(long, action = ArgAction::SetTrue)]
    pub dry_run: bool,
    /// Override the configured ingest mode
    #[arg(long, value_name = "MODE")]
    pub mode: Option<ApiFetchMode>,
    /// Override the base URI used when constructing frame URIs
    #[arg(long, value_name = "URI")]
    pub uri: Option<String>,
    /// Emit JSON instead of human-readable output
    #[arg(long, action = ArgAction::SetTrue)]
    pub json: bool,

    #[command(flatten)]
    pub lock: LockCliArgs,
}

/// Arguments for the `update` subcommand
#[derive(Args)]
pub struct UpdateArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long = "frame-id", value_name = "ID", conflicts_with = "uri")]
    pub frame_id: Option<u64>,
    #[arg(long, value_name = "URI", conflicts_with = "frame_id")]
    pub uri: Option<String>,
    #[arg(long = "input", value_name = "PATH", value_parser = clap::value_parser!(PathBuf))]
    pub input: Option<PathBuf>,
    #[arg(long = "set-uri", value_name = "URI")]
    pub set_uri: Option<String>,
    #[arg(long, value_name = "TITLE")]
    pub title: Option<String>,
    /// Timestamp for the frame. Accepts epoch seconds or human-readable dates.
    #[arg(long, value_parser = parse_timestamp, value_name = "DATE")]
    pub timestamp: Option<i64>,
    #[arg(long, value_name = "TRACK")]
    pub track: Option<String>,
    #[arg(long, value_name = "KIND")]
    pub kind: Option<String>,
    #[arg(long = "tag", value_name = "KEY=VALUE", value_parser = parse_key_val)]
    pub tags: Vec<(String, String)>,
    #[arg(long = "label", value_name = "LABEL")]
    pub labels: Vec<String>,
    #[arg(long = "metadata", value_name = "JSON")]
    pub metadata: Vec<String>,
    #[arg(long, aliases = ["embeddings"], action = ArgAction::SetTrue)]
    pub embeddings: bool,
    #[arg(long)]
    pub json: bool,

    #[command(flatten)]
    pub lock: LockCliArgs,
}

/// Arguments for the `delete` subcommand
#[derive(Args)]
pub struct DeleteArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long = "frame-id", value_name = "ID", conflicts_with = "uri")]
    pub frame_id: Option<u64>,
    #[arg(long, value_name = "URI", conflicts_with = "frame_id")]
    pub uri: Option<String>,
    #[arg(long, action = ArgAction::SetTrue)]
    pub yes: bool,
    #[arg(long)]
    pub json: bool,

    #[command(flatten)]
    pub lock: LockCliArgs,
}

/// Handler for `memvid api-fetch`
pub fn handle_api_fetch(config: &CliConfig, args: ApiFetchArgs) -> Result<()> {
    let command = ApiFetchCommand {
        file: args.file,
        config_path: args.config,
        dry_run: args.dry_run,
        mode_override: args.mode,
        uri_override: args.uri,
        output_json: args.json,
        lock_timeout_ms: args.lock.lock_timeout,
        force_lock: args.lock.force,
    };
    run_api_fetch(config, command)
}

/// Handler for `memvid delete`
pub fn handle_delete(_config: &CliConfig, args: DeleteArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;
    ensure_cli_mutation_allowed(&mem)?;
    apply_lock_cli(&mut mem, &args.lock);
    let frame = select_frame(&mut mem, args.frame_id, args.uri.as_deref())?;

    if !args.yes {
        if let Some(uri) = &frame.uri {
            println!("About to delete frame {} ({uri})", frame.id);
        } else {
            println!("About to delete frame {}", frame.id);
        }
        print!("Confirm? [y/N]: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let confirmed = matches!(input.trim().to_ascii_lowercase().as_str(), "y" | "yes");
        if !confirmed {
            println!("Aborted");
            return Ok(());
        }
    }

    let seq = mem.delete_frame(frame.id)?;
    mem.commit()?;
    let deleted = mem.frame_by_id(frame.id)?;

    if args.json {
        let json = json!({
            "frame_id": frame.id,
            "sequence": seq,
            "status": frame_status_str(deleted.status),
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!("Deleted frame {} (seq {})", frame.id, seq);
    }

    Ok(())
}
pub fn handle_put(config: &CliConfig, args: PutArgs) -> Result<()> {
    // Check if plan allows write operations (blocks expired subscriptions)
    crate::utils::require_active_plan(config, "put")?;

    let mut mem = Memvid::open(&args.file)?;
    ensure_cli_mutation_allowed(&mem)?;
    apply_lock_cli(&mut mem, &args.lock);

    // If memory-id is provided and file isn't already bound, bind it now
    if let Some(memory_id) = &args.memory_id {
        let needs_binding = match mem.get_memory_binding() {
            Some(binding) => binding.memory_id != **memory_id,
            None => true,
        };
        if needs_binding {
            match crate::commands::creation::bind_to_dashboard_memory(
                config, &mut mem, &args.file, memory_id,
            ) {
                Ok((bound_id, capacity)) => {
                    eprintln!(
                        "✓ Bound to dashboard memory: {} (capacity: {})",
                        bound_id,
                        crate::utils::format_bytes(capacity)
                    );
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to bind to dashboard memory: {}", e);
                    eprintln!("   Continuing with existing capacity. Bind manually with:");
                    eprintln!(
                        "   memvid tickets sync {} --memory-id {}",
                        args.file.display(),
                        memory_id
                    );
                }
            }
        }
    }

    // Load active replay session if one exists
    #[cfg(feature = "replay")]
    let _ = mem.load_active_session();
    let mut stats = mem.stats()?;
    if stats.frame_count == 0 && !stats.has_lex_index {
        mem.enable_lex()?;
        stats = mem.stats()?;
    }

    // Check if current memory size exceeds free tier limit
    // If so, require API key to continue
    crate::utils::ensure_capacity_with_api_key(stats.size_bytes, 0, config)?;

    // Set vector compression mode if requested
    if args.vector_compression {
        mem.set_vector_compression(memvid_core::VectorCompression::Pq96);
    }

    let mut capacity_guard = CapacityGuard::from_stats(&stats);
    let use_parallel = args.wants_parallel();
    #[cfg(feature = "parallel_segments")]
    let mut parallel_opts = if use_parallel {
        Some(args.sanitized_parallel_opts())
    } else {
        None
    };
    #[cfg(feature = "parallel_segments")]
    let mut pending_parallel_inputs: Vec<ParallelInput> = Vec::new();
    #[cfg(feature = "parallel_segments")]
    let mut pending_parallel_indices: Vec<usize> = Vec::new();
    let input_set = resolve_inputs(args.input.as_deref())?;
    if args.embedding && args.embedding_vec.is_some() {
        anyhow::bail!("--embedding-vec conflicts with --embedding; choose one");
    }
    if args.embedding_vec.is_some() {
        match &input_set {
            InputSet::Files(paths) if paths.len() != 1 => {
                anyhow::bail!("--embedding-vec requires exactly one input file (or stdin)")
            }
            _ => {}
        }
    }

    if args.video {
        if let Some(kind) = &args.kind {
            if !kind.eq_ignore_ascii_case("video") {
                anyhow::bail!("--video conflicts with explicit --kind={kind}");
            }
        }
        if matches!(input_set, InputSet::Stdin) {
            anyhow::bail!("--video requires --input <file>");
        }
    }

    #[allow(dead_code)]
    enum EmbeddingMode {
        None,
        Auto,
        Explicit,
    }

    // PHASE 1: Make embeddings opt-in, not opt-out
    // Removed auto-embedding mode to reduce storage bloat (270 KB/doc → 5 KB/doc without vectors)
    // Auto-detect embedding model from existing MV2 dimension to ensure compatibility
    let mv2_dimension = mem.effective_vec_index_dimension()?;
    let inferred_embedding_model = if args.embedding {
        match mem.embedding_identity_summary(10_000) {
            memvid_core::EmbeddingIdentitySummary::Single(identity) => {
                identity.model.map(String::from)
            }
            memvid_core::EmbeddingIdentitySummary::Mixed(identities) => {
                let models: Vec<_> = identities
                    .iter()
                    .filter_map(|entry| entry.identity.model.as_deref())
                    .collect();
                anyhow::bail!(
                    "memory contains mixed embedding models; refusing to add more embeddings.\n\n\
                    Detected models: {:?}\n\n\
                    Suggested fix: separate memories per embedding model, or re-ingest into a fresh .mv2.",
                    models
                );
            }
            memvid_core::EmbeddingIdentitySummary::Unknown => None,
        }
    } else {
        None
    };
    let (embedding_mode, embedding_runtime) = if args.embedding {
        (
            EmbeddingMode::Explicit,
            Some(load_embedding_runtime_for_mv2(
                config,
                inferred_embedding_model.as_deref(),
                mv2_dimension,
            )?),
        )
    } else if args.embedding_vec.is_some() {
        (EmbeddingMode::Explicit, None)
    } else {
        (EmbeddingMode::None, None)
    };
    let embedding_enabled = args.embedding || args.embedding_vec.is_some();
    let runtime_ref = embedding_runtime.as_ref();

    // Enable CLIP index only if --clip flag is set explicitly
    // CLIP is disabled by default to avoid unexpected overhead
    #[cfg(feature = "clip")]
    let clip_enabled = args.clip;
    #[cfg(feature = "clip")]
    let clip_model = if clip_enabled {
        mem.enable_clip()?;
        if !args.json {
            eprintln!("ℹ️  CLIP visual embeddings enabled");
            eprintln!("   Model: MobileCLIP-S2 (512 dimensions)");
            eprintln!("   Use 'memvid find --mode clip' to search with natural language");
            eprintln!();
        }
        // Initialize CLIP model for generating image embeddings
        match memvid_core::ClipModel::default_model() {
            Ok(model) => Some(model),
            Err(e) => {
                warn!("Failed to load CLIP model: {e}. CLIP embeddings will be skipped.");
                None
            }
        }
    } else {
        None
    };

    // Warn user about vector storage overhead (PHASE 2)
    if embedding_enabled && !args.json {
        let capacity = stats.capacity_bytes;
        if args.vector_compression {
            // PHASE 3: PQ compression reduces storage 16x
            let max_docs = capacity / 20_000; // ~20 KB/doc with PQ compression
            eprintln!("ℹ️  Vector embeddings enabled with Product Quantization compression");
            eprintln!("   Storage: ~20 KB per document (16x compressed)");
            eprintln!("   Accuracy: ~95% recall@10 maintained");
            eprintln!(
                "   Capacity: ~{} documents max for {:.1} GB",
                max_docs,
                capacity as f64 / 1e9
            );
            eprintln!();
        } else {
            let max_docs = capacity / 270_000; // Conservative estimate: 270 KB/doc
            eprintln!("⚠️  WARNING: Vector embeddings enabled (uncompressed)");
            eprintln!("   Storage: ~270 KB per document");
            eprintln!(
                "   Capacity: ~{} documents max for {:.1} GB",
                max_docs,
                capacity as f64 / 1e9
            );
            eprintln!("   Use --vector-compression for 16x storage savings (~20 KB/doc)");
            eprintln!("   Use --no-embedding for lexical search only (~5 KB/doc)");
            eprintln!();
        }
    }

    let embed_progress = if embedding_enabled {
        let total = match &input_set {
            InputSet::Files(paths) => Some(paths.len() as u64),
            InputSet::Stdin => None,
        };
        Some(create_task_progress_bar(total, "embed", false))
    } else {
        None
    };
    let mut embedded_docs = 0usize;

    if let InputSet::Files(paths) = &input_set {
        if paths.len() > 1 {
            if args.uri.is_some() || args.title.is_some() {
                anyhow::bail!(
                    "--uri/--title apply to a single file; omit them when using a directory or glob"
                );
            }
        }
    }

    let mut reports = Vec::new();
    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut bytes_added = 0u64;
    let mut bytes_input = 0u64;
    let mut no_raw_count = 0usize;
    let mut summary_warnings: Vec<String> = Vec::new();
    #[cfg(feature = "clip")]
    let mut clip_embeddings_added = false;

    let mut capacity_reached = false;
    match input_set {
        InputSet::Stdin => {
            processed += 1;
            match read_payload(None) {
                Ok(payload) => {
                    let mut analyze_spinner = if args.json {
                        None
                    } else {
                        Some(create_spinner("Analyzing stdin payload..."))
                    };
                    match ingest_payload(
                        &mut mem,
                        &args,
                        config,
                        payload,
                        None,
                        capacity_guard.as_mut(),
                        embedding_enabled,
                        runtime_ref,
                        use_parallel,
                    ) {
                        Ok(outcome) => {
                            if let Some(pb) = analyze_spinner.take() {
                                pb.finish_and_clear();
                            }
                            bytes_added += outcome.report.stored_bytes as u64;
                            bytes_input += outcome.report.original_bytes as u64;
                            if outcome.report.no_raw {
                                no_raw_count += 1;
                            }
                            if args.json {
                                summary_warnings
                                    .extend(outcome.report.extraction.warnings.iter().cloned());
                            }
                            reports.push(outcome.report);
                            let report_index = reports.len() - 1;
                            if !args.json && !use_parallel {
                                if let Some(report) = reports.get(report_index) {
                                    print_report(report);
                                }
                            }
                            if args.transcript {
                                let notice = transcript_notice_message(&mut mem)?;
                                if args.json {
                                    summary_warnings.push(notice);
                                } else {
                                    println!("{notice}");
                                }
                            }
                            if outcome.embedded {
                                if let Some(pb) = embed_progress.as_ref() {
                                    pb.inc(1);
                                }
                                embedded_docs += 1;
                            }
                            if use_parallel {
                                #[cfg(feature = "parallel_segments")]
                                if let Some(input) = outcome.parallel_input {
                                    pending_parallel_indices.push(report_index);
                                    pending_parallel_inputs.push(input);
                                }
                            } else if let Some(guard) = capacity_guard.as_mut() {
                                mem.commit()?;
                                let stats = mem.stats()?;
                                guard.update_after_commit(&stats);
                                guard.check_and_warn_capacity(); // PHASE 2: Capacity warnings
                            }
                        }
                        Err(err) => {
                            if let Some(pb) = analyze_spinner.take() {
                                pb.finish_and_clear();
                            }
                            if err.downcast_ref::<DuplicateUriError>().is_some() {
                                return Err(err);
                            }
                            warn!("skipped stdin payload: {err}");
                            skipped += 1;
                            if err.downcast_ref::<CapacityExceededMessage>().is_some() {
                                capacity_reached = true;
                            }
                        }
                    }
                }
                Err(err) => {
                    warn!("failed to read stdin: {err}");
                    skipped += 1;
                }
            }
        }
        InputSet::Files(ref paths) => {
            let mut transcript_notice_printed = false;
            for path in paths {
                processed += 1;
                match read_payload(Some(&path)) {
                    Ok(payload) => {
                        let label = path.display().to_string();
                        let mut analyze_spinner = if args.json {
                            None
                        } else {
                            Some(create_spinner(&format!("Analyzing {label}...")))
                        };
                        match ingest_payload(
                            &mut mem,
                            &args,
                            config,
                            payload,
                            Some(&path),
                            capacity_guard.as_mut(),
                            embedding_enabled,
                            runtime_ref,
                            use_parallel,
                        ) {
                            Ok(outcome) => {
                                if let Some(pb) = analyze_spinner.take() {
                                    pb.finish_and_clear();
                                }
                                bytes_added += outcome.report.stored_bytes as u64;
                                bytes_input += outcome.report.original_bytes as u64;
                                if outcome.report.no_raw {
                                    no_raw_count += 1;
                                }

                                // Generate CLIP embedding for image files
                                #[cfg(feature = "clip")]
                                if let Some(ref model) = clip_model {
                                    let mime = outcome.report.mime.as_deref();
                                    let clip_frame_id = outcome.report.frame_id;

                                    if is_image_file(&path) {
                                        match model.encode_image_file(&path) {
                                            Ok(embedding) => {
                                                if let Err(e) = mem.add_clip_embedding_with_page(
                                                    clip_frame_id,
                                                    None,
                                                    embedding,
                                                ) {
                                                    warn!(
                                                        "Failed to add CLIP embedding for {}: {e}",
                                                        path.display()
                                                    );
                                                } else {
                                                    info!(
                                                        "Added CLIP embedding for frame {}",
                                                        clip_frame_id
                                                    );
                                                    clip_embeddings_added = true;
                                                }
                                            }
                                            Err(e) => {
                                                warn!(
                                                    "Failed to encode CLIP embedding for {}: {e}",
                                                    path.display()
                                                );
                                            }
                                        }
                                    } else if matches!(mime, Some(m) if m == "application/pdf") {
                                        // Commit before adding child frames to ensure WAL has space
                                        if let Err(e) = mem.commit() {
                                            warn!("Failed to commit before CLIP extraction: {e}");
                                        }

                                        match render_pdf_pages_for_clip(
                                            &path,
                                            CLIP_PDF_MAX_IMAGES,
                                            CLIP_PDF_TARGET_PX,
                                        ) {
                                            Ok(rendered_pages) => {
                                                use rayon::prelude::*;

                                                // Pre-encode all images to PNG in parallel for better performance
                                                // Filter out junk images (solid colors, black, too small)
                                                let path_for_closure = path.clone();

                                                // Temporarily suppress panic output during image encoding
                                                // (some PDFs have malformed images that cause panics in the image crate)
                                                let prev_hook = std::panic::take_hook();
                                                std::panic::set_hook(Box::new(|_| {
                                                    // Silently ignore panics - we handle them with catch_unwind
                                                }));

                                                let encoded_images: Vec<_> = rendered_pages
                                                    .into_par_iter()
                                                    .filter_map(|(page_num, image)| {
                                                        // Check if image is worth embedding (not junk)
                                                        let image_info = get_image_info(&image);
                                                        if !image_info.should_embed() {
                                                            info!(
                                                                "Skipping junk image for {} page {} (variance={:.4}, {}x{})",
                                                                path_for_closure.display(),
                                                                page_num,
                                                                image_info.color_variance,
                                                                image_info.width,
                                                                image_info.height
                                                            );
                                                            return None;
                                                        }

                                                        // Convert to RGB8 first to ensure consistent buffer size
                                                        // (some PDFs have images with alpha or other formats that cause panics)
                                                        // Use catch_unwind to handle any panics from corrupted image data
                                                        let encode_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                                            let rgb_image = image.to_rgb8();
                                                            let mut png_bytes = Vec::new();
                                                            image::DynamicImage::ImageRgb8(rgb_image).write_to(
                                                                &mut Cursor::new(&mut png_bytes),
                                                                image::ImageFormat::Png,
                                                            ).map(|()| png_bytes)
                                                        }));

                                                        match encode_result {
                                                            Ok(Ok(png_bytes)) => Some((page_num, image, png_bytes)),
                                                            Ok(Err(e)) => {
                                                                info!(
                                                                    "Skipping image for {} page {}: {e}",
                                                                    path_for_closure.display(),
                                                                    page_num
                                                                );
                                                                None
                                                            }
                                                            Err(_) => {
                                                                // Panic occurred - corrupted image data, silently skip
                                                                info!(
                                                                    "Skipping malformed image for {} page {}",
                                                                    path_for_closure.display(),
                                                                    page_num
                                                                );
                                                                None
                                                            }
                                                        }
                                                    })
                                                    .collect();

                                                // Restore the original panic hook
                                                std::panic::set_hook(prev_hook);

                                                // Process encoded images sequentially (storage + CLIP encoding)
                                                // Commit after each image to prevent WAL overflow (PNG images can be large)
                                                let mut child_frame_offset = 1u64;
                                                let parent_uri = outcome.report.uri.clone();
                                                let parent_title = outcome
                                                    .report
                                                    .title
                                                    .clone()
                                                    .unwrap_or_else(|| "Image".to_string());
                                                let total_images = encoded_images.len();

                                                // Show progress bar for CLIP extraction
                                                let clip_pb = if !args.json && total_images > 0 {
                                                    let pb = ProgressBar::new(total_images as u64);
                                                    pb.set_style(
                                                        ProgressStyle::with_template(
                                                            "  CLIP {bar:40.cyan/blue} {pos}/{len} images ({eta})"
                                                        )
                                                        .unwrap_or_else(|_| ProgressStyle::default_bar())
                                                        .progress_chars("█▓░"),
                                                    );
                                                    Some(pb)
                                                } else {
                                                    None
                                                };

                                                for (page_num, image, png_bytes) in encoded_images {
                                                    let child_uri = format!(
                                                        "{}/image/{}",
                                                        parent_uri, child_frame_offset
                                                    );
                                                    let child_title = format!(
                                                        "{} - Image {} (page {})",
                                                        parent_title, child_frame_offset, page_num
                                                    );

                                                    let child_options = PutOptions::builder()
                                                        .uri(&child_uri)
                                                        .title(&child_title)
                                                        .parent_id(clip_frame_id)
                                                        .role(FrameRole::ExtractedImage)
                                                        .auto_tag(false)
                                                        .extract_dates(false)
                                                        .build();

                                                    match mem.put_bytes_with_options(&png_bytes, child_options) {
                                                        Ok(_child_seq) => {
                                                            let child_frame_id = clip_frame_id + child_frame_offset;
                                                            child_frame_offset += 1;

                                                            match model.encode_image(&image) {
                                                                Ok(embedding) => {
                                                                    if let Err(e) = mem
                                                                        .add_clip_embedding_with_page(
                                                                            child_frame_id,
                                                                            Some(page_num),
                                                                            embedding,
                                                                        )
                                                                    {
                                                                        warn!(
                                                                            "Failed to add CLIP embedding for {} page {}: {e}",
                                                                            path.display(),
                                                                            page_num
                                                                        );
                                                                    } else {
                                                                        info!(
                                                                            "Added CLIP image frame {} for {} page {}",
                                                                            child_frame_id,
                                                                            clip_frame_id,
                                                                            page_num
                                                                        );
                                                                        clip_embeddings_added = true;
                                                                    }
                                                                }
                                                                Err(e) => warn!(
                                                                    "Failed to encode CLIP embedding for {} page {}: {e}",
                                                                    path.display(),
                                                                    page_num
                                                                ),
                                                            }

                                                            // Commit after each image to checkpoint WAL
                                                            if let Err(e) = mem.commit() {
                                                                warn!("Failed to commit after image {}: {e}", child_frame_offset);
                                                            }
                                                        }
                                                        Err(e) => warn!(
                                                            "Failed to store image frame for {} page {}: {e}",
                                                            path.display(),
                                                            page_num
                                                        ),
                                                    }

                                                    if let Some(ref pb) = clip_pb {
                                                        pb.inc(1);
                                                    }
                                                }

                                                if let Some(pb) = clip_pb {
                                                    pb.finish_and_clear();
                                                }
                                            }
                                            Err(err) => warn!(
                                                "Failed to render PDF pages for CLIP {}: {err}",
                                                path.display()
                                            ),
                                        }
                                    }
                                }

                                if args.json {
                                    summary_warnings
                                        .extend(outcome.report.extraction.warnings.iter().cloned());
                                }
                                reports.push(outcome.report);
                                let report_index = reports.len() - 1;
                                if !args.json && !use_parallel {
                                    if let Some(report) = reports.get(report_index) {
                                        print_report(report);
                                    }
                                }
                                if args.transcript && !transcript_notice_printed {
                                    let notice = transcript_notice_message(&mut mem)?;
                                    if args.json {
                                        summary_warnings.push(notice);
                                    } else {
                                        println!("{notice}");
                                    }
                                    transcript_notice_printed = true;
                                }
                                if outcome.embedded {
                                    if let Some(pb) = embed_progress.as_ref() {
                                        pb.inc(1);
                                    }
                                    embedded_docs += 1;
                                }
                                if use_parallel {
                                    #[cfg(feature = "parallel_segments")]
                                    if let Some(input) = outcome.parallel_input {
                                        pending_parallel_indices.push(report_index);
                                        pending_parallel_inputs.push(input);
                                    }
                                } else if let Some(guard) = capacity_guard.as_mut() {
                                    mem.commit()?;
                                    let stats = mem.stats()?;
                                    guard.update_after_commit(&stats);
                                    guard.check_and_warn_capacity(); // PHASE 2: Capacity warnings
                                }
                            }
                            Err(err) => {
                                if let Some(pb) = analyze_spinner.take() {
                                    pb.finish_and_clear();
                                }
                                if err.downcast_ref::<DuplicateUriError>().is_some() {
                                    return Err(err);
                                }
                                warn!("skipped {}: {err}", path.display());
                                skipped += 1;
                                if err.downcast_ref::<CapacityExceededMessage>().is_some() {
                                    capacity_reached = true;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        warn!("failed to read {}: {err}", path.display());
                        skipped += 1;
                    }
                }
                if capacity_reached {
                    break;
                }
            }
        }
    }

    if capacity_reached {
        warn!("capacity limit reached; remaining inputs were not processed");
    }

    if let Some(pb) = embed_progress.as_ref() {
        pb.finish_with_message("embed");
    }

    if let Some(runtime) = embedding_runtime.as_ref() {
        if embedded_docs > 0 {
            match embedding_mode {
                EmbeddingMode::Explicit => println!(
                    "\u{2713} vectors={}  dim={}  hnsw(M=16, efC=200)",
                    embedded_docs,
                    runtime.dimension()
                ),
                EmbeddingMode::Auto => println!(
                    "\u{2713} vectors={}  dim={}  hnsw(M=16, efC=200)  (auto)",
                    embedded_docs,
                    runtime.dimension()
                ),
                EmbeddingMode::None => {}
            }
        } else {
            match embedding_mode {
                EmbeddingMode::Explicit => {
                    println!("No embeddings were generated (no textual content available)");
                }
                EmbeddingMode::Auto => {
                    println!(
                        "Semantic runtime available, but no textual content produced embeddings"
                    );
                }
                EmbeddingMode::None => {}
            }
        }
    }

    if reports.is_empty() {
        if args.json {
            if capacity_reached {
                summary_warnings.push("capacity_limit_reached".to_string());
            }
            let summary_json = json!({
                "processed": processed,
                "ingested": 0,
                "skipped": skipped,
                "bytes_added": bytes_added,
                "bytes_added_human": format_bytes(bytes_added),
                "embedded_documents": embedded_docs,
                "capacity_reached": capacity_reached,
                "warnings": summary_warnings,
                "reports": Vec::<serde_json::Value>::new(),
            });
            println!("{}", serde_json::to_string_pretty(&summary_json)?);
        } else {
            println!(
                "Summary: processed {}, ingested 0, skipped {}, bytes added 0 B",
                processed, skipped
            );
        }
        return Ok(());
    }

    #[cfg(feature = "parallel_segments")]
    let mut parallel_flush_spinner = if use_parallel && !args.json {
        Some(create_spinner("Flushing parallel segments..."))
    } else {
        None
    };

    if use_parallel {
        #[cfg(feature = "parallel_segments")]
        {
            let mut opts = parallel_opts.take().unwrap_or_else(|| {
                let mut opts = BuildOpts::default();
                opts.sanitize();
                opts
            });
            // Set vector compression mode from mem state
            opts.vec_compression = mem.vector_compression().clone();
            let seqs = if pending_parallel_inputs.is_empty() {
                mem.commit_parallel(opts)?;
                Vec::new()
            } else {
                mem.put_parallel_inputs(&pending_parallel_inputs, opts)?
            };
            if let Some(pb) = parallel_flush_spinner.take() {
                pb.finish_with_message("Flushed parallel segments");
            }
            for (idx, seq) in pending_parallel_indices.into_iter().zip(seqs.into_iter()) {
                if let Some(report) = reports.get_mut(idx) {
                    report.seq = seq;
                }
            }
        }
        #[cfg(not(feature = "parallel_segments"))]
        {
            mem.commit()?;
        }
    } else {
        mem.commit()?;
    }

    // Persist CLIP embeddings added after the primary commit (avoids rebuild ordering issues).
    #[cfg(feature = "clip")]
    if clip_embeddings_added {
        mem.commit()?;
    }

    if !args.json && use_parallel {
        for report in &reports {
            print_report(report);
        }
    }

    // Table extraction for PDF files
    let mut tables_extracted = 0usize;
    let mut table_summaries: Vec<serde_json::Value> = Vec::new();
    if args.tables && !args.no_tables {
        if let InputSet::Files(ref paths) = input_set {
            let pdf_paths: Vec<_> = paths
                .iter()
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case("pdf"))
                        .unwrap_or(false)
                })
                .collect();

            if !pdf_paths.is_empty() {
                let table_spinner = if args.json {
                    None
                } else {
                    Some(create_spinner(&format!(
                        "Extracting tables from {} PDF(s)...",
                        pdf_paths.len()
                    )))
                };

                // Use aggressive mode with auto fallback for best results
                let table_options = TableExtractionOptions::builder()
                    .mode(memvid_core::table::ExtractionMode::Aggressive)
                    .min_rows(2)
                    .min_cols(2)
                    .min_quality(memvid_core::table::TableQuality::Low)
                    .merge_multi_page(true)
                    .build();

                for pdf_path in pdf_paths {
                    if let Ok(pdf_bytes) = std::fs::read(pdf_path) {
                        let filename = pdf_path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown.pdf");

                        if let Ok(result) = extract_tables(&pdf_bytes, filename, &table_options) {
                            for table in &result.tables {
                                if let Ok((meta_id, _row_ids)) = store_table(&mut mem, table, true)
                                {
                                    tables_extracted += 1;
                                    table_summaries.push(json!({
                                        "table_id": table.table_id,
                                        "source_file": table.source_file,
                                        "meta_frame_id": meta_id,
                                        "rows": table.n_rows,
                                        "cols": table.n_cols,
                                        "quality": format!("{:?}", table.quality),
                                        "pages": format!("{}-{}", table.page_start, table.page_end),
                                    }));
                                }
                            }
                        }
                    }
                }

                if let Some(pb) = table_spinner {
                    if tables_extracted > 0 {
                        pb.finish_with_message(format!("Extracted {} table(s)", tables_extracted));
                    } else {
                        pb.finish_with_message("No tables found");
                    }
                }

                // Commit table frames
                if tables_extracted > 0 {
                    mem.commit()?;
                }
            }
        }
    }

    // Logic-Mesh: Extract entities and build relationship graph
    // Opt-in only: requires --logic-mesh flag
    #[cfg(feature = "logic_mesh")]
    let mut logic_mesh_entities = 0usize;
    #[cfg(feature = "logic_mesh")]
    {
        use memvid_core::{ner_model_path, ner_tokenizer_path, LogicMesh, MeshNode, NerModel};

        let models_dir = config.models_dir.clone();
        let model_path = ner_model_path(&models_dir);
        let tokenizer_path = ner_tokenizer_path(&models_dir);

        // Opt-in: only enable Logic-Mesh when explicitly requested with --logic-mesh
        let ner_available = model_path.exists() && tokenizer_path.exists();
        let should_run_ner = args.logic_mesh;

        if should_run_ner && !ner_available {
            // User explicitly requested --logic-mesh but model not installed
            if args.json {
                summary_warnings.push(
                    "logic_mesh_skipped: NER model not installed. Run 'memvid models install --ner distilbert-ner'".to_string()
                );
            } else {
                eprintln!("⚠️  Logic-Mesh skipped: NER model not installed.");
                eprintln!("   Run: memvid models install --ner distilbert-ner");
            }
        } else if should_run_ner {
            let ner_spinner = if args.json {
                None
            } else {
                Some(create_spinner("Building Logic-Mesh entity graph..."))
            };

            match NerModel::load(&model_path, &tokenizer_path, None) {
                Ok(mut ner_model) => {
                    let mut mesh = LogicMesh::new();
                    let mut filtered_count = 0usize;

                    // Process each ingested frame for entity extraction using cached search_text
                    for report in &reports {
                        let frame_id = report.frame_id;
                        // Use the cached search_text from ingestion
                        if let Some(ref content) = report.search_text {
                            if !content.is_empty() {
                                match ner_model.extract(content) {
                                    Ok(raw_entities) => {
                                        // Merge adjacent entities that were split by tokenization
                                        // e.g., "S" + "&" + "P Global" -> "S&P Global"
                                        let merged_entities =
                                            merge_adjacent_entities(raw_entities, content);

                                        for entity in &merged_entities {
                                            // Apply post-processing filter to remove noisy entities
                                            if !is_valid_entity(&entity.text, entity.confidence) {
                                                tracing::debug!(
                                                    "Filtered entity: '{}' (confidence: {:.0}%)",
                                                    entity.text,
                                                    entity.confidence * 100.0
                                                );
                                                filtered_count += 1;
                                                continue;
                                            }

                                            // Convert NER entity type to EntityKind
                                            let kind = entity.to_entity_kind();
                                            // Create mesh node with proper structure
                                            let node = MeshNode::new(
                                                entity.text.to_lowercase(),
                                                entity.text.trim().to_string(),
                                                kind,
                                                entity.confidence,
                                                frame_id,
                                                entity.byte_start as u32,
                                                (entity.byte_end - entity.byte_start)
                                                    .min(u16::MAX as usize)
                                                    as u16,
                                            );
                                            mesh.merge_node(node);
                                            logic_mesh_entities += 1;
                                        }
                                    }
                                    Err(e) => {
                                        warn!("NER extraction failed for frame {frame_id}: {e}");
                                    }
                                }
                            }
                        }
                    }

                    if logic_mesh_entities > 0 {
                        mesh.finalize();
                        let node_count = mesh.stats().node_count;
                        // Persist mesh to MV2 file
                        mem.set_logic_mesh(mesh);
                        mem.commit()?;
                        info!(
                            "Logic-Mesh persisted with {} entities ({} unique nodes, {} filtered)",
                            logic_mesh_entities, node_count, filtered_count
                        );
                    }

                    if let Some(pb) = ner_spinner {
                        pb.finish_with_message(format!(
                            "Logic-Mesh: {} entities ({} unique)",
                            logic_mesh_entities,
                            mem.mesh_node_count()
                        ));
                    }
                }
                Err(e) => {
                    if let Some(pb) = ner_spinner {
                        pb.finish_with_message(format!("Logic-Mesh failed: {e}"));
                    }
                    if args.json {
                        summary_warnings.push(format!("logic_mesh_failed: {e}"));
                    }
                }
            }
        }
    }

    // Rules-based enrichment: extract memory cards from ingested frames
    let mut enrichment_cards = 0usize;
    if args.enrich && !args.no_enrich && !reports.is_empty() {
        let enrich_spinner = if args.json {
            None
        } else {
            Some(create_spinner(&format!(
                "Extracting memories from {} frame(s)...",
                reports.len()
            )))
        };

        let mut rules_engine = RulesEngine::new();
        rules_engine.init().ok(); // Rules engine doesn't need external init

        let frame_ids: Vec<u64> = reports.iter().map(|r| r.frame_id).collect();
        for frame_id in &frame_ids {
            if let Ok(frame) = mem.frame_by_id(*frame_id) {
                let text = frame.search_text.as_deref().unwrap_or("");
                if !text.is_empty() {
                    let ctx = memvid_core::enrich::EnrichmentContext::new(
                        *frame_id,
                        frame.uri.clone().unwrap_or_default(),
                        text.to_string(),
                        frame.title.clone(),
                        frame.timestamp,
                        None,
                    );
                    let result = rules_engine.enrich(&ctx);
                    if !result.cards.is_empty() {
                        for card in result.cards {
                            if mem.put_memory_card(card).is_ok() {
                                enrichment_cards += 1;
                            }
                        }
                    }
                }
            }
        }

        if enrichment_cards > 0 {
            mem.commit()?;
        }

        if let Some(pb) = enrich_spinner {
            if enrichment_cards > 0 {
                pb.finish_with_message(format!("Extracted {} memory card(s)", enrichment_cards));
            } else {
                pb.finish_with_message("No memories extracted");
            }
        }
    }

    let stats = mem.stats()?;
    if args.json {
        if capacity_reached {
            summary_warnings.push("capacity_limit_reached".to_string());
        }
        let reports_json: Vec<serde_json::Value> = reports.iter().map(put_report_to_json).collect();
        let total_duration_ms: Option<u64> = {
            let sum: u64 = reports
                .iter()
                .filter_map(|r| r.extraction.duration_ms)
                .sum();
            if sum > 0 {
                Some(sum)
            } else {
                None
            }
        };
        let total_pages: Option<u32> = {
            let sum: u32 = reports
                .iter()
                .filter_map(|r| r.extraction.pages_processed)
                .sum();
            if sum > 0 {
                Some(sum)
            } else {
                None
            }
        };
        let embedding_mode_str = match embedding_mode {
            EmbeddingMode::None => "none",
            EmbeddingMode::Auto => "auto",
            EmbeddingMode::Explicit => "explicit",
        };
        let reduction_percent = if bytes_input > 0 && bytes_input > bytes_added {
            Some(((bytes_input - bytes_added) as f64 / bytes_input as f64) * 100.0)
        } else {
            None
        };
        let summary_json = json!({
            "processed": processed,
            "ingested": reports.len(),
            "skipped": skipped,
            "bytes_input": bytes_input,
            "bytes_input_human": format_bytes(bytes_input),
            "bytes_stored": bytes_added,
            "bytes_stored_human": format_bytes(bytes_added),
            "reduction_percent": reduction_percent,
            "embedded_documents": embedded_docs,
            "embedding": {
                "enabled": embedding_enabled,
                "mode": embedding_mode_str,
                "runtime_dimension": runtime_ref.map(|rt| rt.dimension()),
            },
            "tables": {
                "enabled": args.tables && !args.no_tables,
                "extracted": tables_extracted,
                "tables": table_summaries,
            },
            "enrichment": {
                "enabled": args.enrich && !args.no_enrich,
                "cards_extracted": enrichment_cards,
            },
            "capacity_reached": capacity_reached,
            "warnings": summary_warnings,
            "extraction": {
                "total_duration_ms": total_duration_ms,
                "total_pages_processed": total_pages,
            },
            "memory": {
                "path": args.file.display().to_string(),
                "frame_count": stats.frame_count,
                "size_bytes": stats.size_bytes,
                "tier": format!("{:?}", stats.tier),
            },
            "reports": reports_json,
        });
        println!("{}", serde_json::to_string_pretty(&summary_json)?);
    } else {
        println!(
            "Committed {} frame(s); total frames {}",
            reports.len(),
            stats.frame_count
        );
        println!(
            "Summary: processed {}, ingested {}, skipped {}",
            processed,
            reports.len(),
            skipped
        );
        // Show storage efficiency - use actual payload bytes for accuracy
        // When --no-raw is used, stored_bytes in reports doesn't reflect actual storage
        if bytes_input > 0 {
            let actual_stored = stats.payload_bytes;
            if actual_stored < bytes_input && no_raw_count > 0 {
                // --no-raw mode: show the dramatic reduction (raw files → extracted text)
                let reduction = ((bytes_input - actual_stored) as f64 / bytes_input as f64) * 100.0;
                println!(
                    "Storage: {} input → {} stored ({:.1}% smaller, text extracted)",
                    format_bytes(bytes_input),
                    format_bytes(actual_stored),
                    reduction
                );
            } else if bytes_added < bytes_input {
                // Normal compression
                let reduction = ((bytes_input - bytes_added) as f64 / bytes_input as f64) * 100.0;
                println!(
                    "Storage: {} input → {} stored ({:.1}% smaller)",
                    format_bytes(bytes_input),
                    format_bytes(bytes_added),
                    reduction
                );
            } else {
                println!("Storage: {} added", format_bytes(bytes_added));
            }
        }
        if tables_extracted > 0 {
            println!("Tables: {} extracted from PDF(s)", tables_extracted);
        }
        if enrichment_cards > 0 {
            println!("Enrichment: {} memory card(s) extracted", enrichment_cards);
        }
    }

    // Save active replay session if one exists
    #[cfg(feature = "replay")]
    let _ = mem.save_active_session();

    Ok(())
}

pub fn handle_update(config: &CliConfig, args: UpdateArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;
    ensure_cli_mutation_allowed(&mem)?;
    apply_lock_cli(&mut mem, &args.lock);
    let existing = select_frame(&mut mem, args.frame_id, args.uri.as_deref())?;
    let frame_id = existing.id;

    let payload_bytes = match args.input.as_ref() {
        Some(path) => Some(read_payload(Some(path))?),
        None => None,
    };
    let payload_slice = payload_bytes.as_deref();
    let payload_utf8 = payload_slice.and_then(|bytes| std::str::from_utf8(bytes).ok());
    let source_path = args.input.as_deref();

    let mut options = PutOptions::default();
    options.enable_embedding = false;
    options.auto_tag = payload_slice.is_some();
    options.extract_dates = payload_slice.is_some();
    options.timestamp = Some(existing.timestamp);
    options.track = existing.track.clone();
    options.kind = existing.kind.clone();
    options.uri = existing.uri.clone();
    options.title = existing.title.clone();
    options.metadata = existing.metadata.clone();
    options.search_text = existing.search_text.clone();
    options.tags = existing.tags.clone();
    options.labels = existing.labels.clone();
    options.extra_metadata = existing.extra_metadata.clone();

    if let Some(new_uri) = &args.set_uri {
        options.uri = Some(derive_uri(Some(new_uri), None));
    }

    if options.uri.is_none() && payload_slice.is_some() {
        options.uri = Some(derive_uri(None, source_path));
    }

    if let Some(title) = &args.title {
        options.title = Some(title.clone());
    }
    if let Some(ts) = args.timestamp {
        options.timestamp = Some(ts);
    }
    if let Some(track) = &args.track {
        options.track = Some(track.clone());
    }
    if let Some(kind) = &args.kind {
        options.kind = Some(kind.clone());
    }

    if !args.labels.is_empty() {
        options.labels = args.labels.clone();
    }

    if !args.tags.is_empty() {
        options.tags.clear();
        for (key, value) in &args.tags {
            if !key.is_empty() {
                options.tags.push(key.clone());
            }
            if !value.is_empty() && value != key {
                options.tags.push(value.clone());
            }
            options.extra_metadata.insert(key.clone(), value.clone());
        }
    }

    apply_metadata_overrides(&mut options, &args.metadata);

    let mut search_source = options.search_text.clone();

    if let Some(payload) = payload_slice {
        let inferred_title = derive_title(args.title.clone(), source_path, payload_utf8);
        if let Some(title) = inferred_title {
            options.title = Some(title);
        }

        if options.uri.is_none() {
            options.uri = Some(derive_uri(None, source_path));
        }

        let analysis = analyze_file(
            source_path,
            payload,
            payload_utf8,
            options.title.as_deref(),
            AudioAnalyzeOptions::default(),
            false,
        );
        if let Some(meta) = analysis.metadata {
            options.metadata = Some(meta);
        }
        if let Some(text) = analysis.search_text {
            search_source = Some(text.clone());
            options.search_text = Some(text);
        }
    } else {
        options.auto_tag = false;
        options.extract_dates = false;
    }

    let stats = mem.stats()?;
    options.enable_embedding = stats.has_vec_index;

    // Auto-detect embedding model from existing MV2 dimension
    let mv2_dimension = mem.effective_vec_index_dimension()?;
    let mut embedding: Option<Vec<f32>> = None;
    if args.embeddings {
        let inferred_embedding_model = match mem.embedding_identity_summary(10_000) {
            memvid_core::EmbeddingIdentitySummary::Single(identity) => {
                identity.model.map(String::from)
            }
            memvid_core::EmbeddingIdentitySummary::Mixed(identities) => {
                let models: Vec<_> = identities
                    .iter()
                    .filter_map(|entry| entry.identity.model.as_deref())
                    .collect();
                anyhow::bail!(
                    "memory contains mixed embedding models; refusing to recompute embeddings.\n\n\
                    Detected models: {:?}",
                    models
                );
            }
            memvid_core::EmbeddingIdentitySummary::Unknown => None,
        };

        let runtime = load_embedding_runtime_for_mv2(
            config,
            inferred_embedding_model.as_deref(),
            mv2_dimension,
        )?;
        if let Some(text) = search_source.as_ref() {
            if !text.trim().is_empty() {
                embedding = Some(runtime.embed_passage(text)?);
            }
        }
        if embedding.is_none() {
            if let Some(text) = payload_utf8 {
                if !text.trim().is_empty() {
                    embedding = Some(runtime.embed_passage(text)?);
                }
            }
        }
        if embedding.is_none() {
            warn!("no textual content available; embeddings not recomputed");
        }
        if embedding.is_some() {
            apply_embedding_identity_metadata(&mut options, &runtime);
        }
    }

    let mut effective_embedding = embedding;
    if effective_embedding.is_none() && stats.has_vec_index {
        effective_embedding = mem.frame_embedding(frame_id)?;
    }

    let final_uri = options.uri.clone();
    let replaced_payload = payload_slice.is_some();
    let seq = mem.update_frame(frame_id, payload_bytes, options, effective_embedding)?;
    mem.commit()?;

    let updated_frame =
        if let Some(uri) = final_uri.and_then(|u| if u.is_empty() { None } else { Some(u) }) {
            mem.frame_by_uri(&uri)?
        } else {
            let mut query = TimelineQueryBuilder::default();
            if let Some(limit) = NonZeroU64::new(1) {
                query = query.limit(limit).reverse(true);
            }
            let latest = mem.timeline(query.build())?;
            if let Some(entry) = latest.first() {
                mem.frame_by_id(entry.frame_id)?
            } else {
                mem.frame_by_id(frame_id)?
            }
        };

    if args.json {
        let json = json!({
            "sequence": seq,
            "previous_frame_id": frame_id,
            "frame": frame_to_json(&updated_frame),
            "replaced_payload": replaced_payload,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!(
            "Updated frame {} → new frame {} (seq {})",
            frame_id, updated_frame.id, seq
        );
        println!(
            "Payload: {}",
            if replaced_payload {
                "replaced"
            } else {
                "reused"
            }
        );
        print_frame_summary(&mut mem, &updated_frame)?;
    }

    Ok(())
}

fn derive_uri(provided: Option<&str>, source: Option<&Path>) -> String {
    if let Some(uri) = provided {
        let sanitized = sanitize_uri(uri, true);
        return format!("mv2://{}", sanitized);
    }

    if let Some(path) = source {
        let raw = path.to_string_lossy();
        let sanitized = sanitize_uri(&raw, false);
        return format!("mv2://{}", sanitized);
    }

    format!("mv2://frames/{}", Uuid::new_v4())
}

pub fn derive_video_uri(payload: &[u8], source: Option<&Path>, mime: &str) -> String {
    let digest = hash(payload).to_hex();
    let short = &digest[..32];
    let ext_from_path = source
        .and_then(|path| path.extension())
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
        .filter(|ext| !ext.is_empty());
    let ext = ext_from_path
        .or_else(|| extension_from_mime(mime).map(|ext| ext.to_string()))
        .unwrap_or_else(|| "bin".to_string());
    format!("mv2://video/{short}.{ext}")
}

fn derive_title(
    provided: Option<String>,
    source: Option<&Path>,
    payload_utf8: Option<&str>,
) -> Option<String> {
    if let Some(title) = provided {
        let trimmed = title.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    } else {
        if let Some(text) = payload_utf8 {
            if let Some(markdown_title) = extract_markdown_title(text) {
                return Some(markdown_title);
            }
        }
        source
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .map(to_title_case)
    }
}

fn extract_markdown_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            let title = trimmed.trim_start_matches('#').trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
    }
    None
}

fn to_title_case(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }
    let mut chars = input.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let prefix = first.to_uppercase().collect::<String>();
    prefix + chars.as_str()
}

fn should_skip_input(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".DS_Store")
    )
}

fn should_skip_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.starts_with('.')
    )
}

enum InputSet {
    Stdin,
    Files(Vec<PathBuf>),
}

/// Check if a file is an image based on extension
#[cfg(feature = "clip")]
fn is_image_file(path: &std::path::Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff" | "tif" | "ico"
    )
}

fn resolve_inputs(input: Option<&str>) -> Result<InputSet> {
    let Some(raw) = input else {
        return Ok(InputSet::Stdin);
    };

    if raw.contains('*') || raw.contains('?') || raw.contains('[') {
        let mut files = Vec::new();
        let mut matched_any = false;
        for entry in glob(raw)? {
            let path = entry?;
            if path.is_file() {
                matched_any = true;
                if should_skip_input(&path) {
                    continue;
                }
                files.push(path);
            }
        }
        files.sort();
        if files.is_empty() {
            if matched_any {
                anyhow::bail!(
                    "pattern '{raw}' only matched files that are ignored by default (e.g. .DS_Store)"
                );
            }
            anyhow::bail!("pattern '{raw}' did not match any files");
        }
        return Ok(InputSet::Files(files));
    }

    let path = PathBuf::from(raw);
    if path.is_dir() {
        let (mut files, skipped_any) = collect_directory_files(&path)?;
        files.sort();
        if files.is_empty() {
            if skipped_any {
                anyhow::bail!(
                    "directory '{}' contains only ignored files (e.g. .DS_Store/.git)",
                    path.display()
                );
            }
            anyhow::bail!(
                "directory '{}' contains no ingestible files",
                path.display()
            );
        }
        Ok(InputSet::Files(files))
    } else {
        Ok(InputSet::Files(vec![path]))
    }
}

fn collect_directory_files(root: &Path) -> Result<(Vec<PathBuf>, bool)> {
    let mut files = Vec::new();
    let mut skipped_any = false;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                if should_skip_dir(&path) {
                    skipped_any = true;
                    continue;
                }
                stack.push(path);
            } else if file_type.is_file() {
                if should_skip_input(&path) {
                    skipped_any = true;
                    continue;
                }
                files.push(path);
            }
        }
    }
    Ok((files, skipped_any))
}

struct IngestOutcome {
    report: PutReport,
    embedded: bool,
    #[cfg(feature = "parallel_segments")]
    parallel_input: Option<ParallelInput>,
}

struct PutReport {
    seq: u64,
    #[allow(dead_code)] // Used in feature-gated code (clip, logic_mesh)
    frame_id: u64,
    uri: String,
    title: Option<String>,
    original_bytes: usize,
    stored_bytes: usize,
    compressed: bool,
    /// --no-raw mode: raw binary not stored, only extracted text + hash
    no_raw: bool,
    source: Option<PathBuf>,
    mime: Option<String>,
    metadata: Option<DocMetadata>,
    extraction: ExtractionSummary,
    /// Text content for entity extraction (only populated when --logic-mesh is enabled)
    #[cfg(feature = "logic_mesh")]
    search_text: Option<String>,
}

struct CapacityGuard {
    #[allow(dead_code)]
    tier: Tier,
    capacity: u64,
    current_size: u64,
}

impl CapacityGuard {
    const MIN_OVERHEAD: u64 = 128 * 1024;
    const RELATIVE_OVERHEAD: f64 = 0.05;

    fn from_stats(stats: &Stats) -> Option<Self> {
        if stats.capacity_bytes == 0 {
            return None;
        }
        Some(Self {
            tier: stats.tier,
            capacity: stats.capacity_bytes,
            current_size: stats.size_bytes,
        })
    }

    fn ensure_capacity(&self, additional: u64) -> Result<()> {
        let projected = self
            .current_size
            .saturating_add(additional)
            .saturating_add(Self::estimate_overhead(additional));
        if projected > self.capacity {
            return Err(map_put_error(
                MemvidError::CapacityExceeded {
                    current: self.current_size,
                    limit: self.capacity,
                    required: projected,
                },
                Some(self.capacity),
            ));
        }
        Ok(())
    }

    fn update_after_commit(&mut self, stats: &Stats) {
        self.current_size = stats.size_bytes;
    }

    fn capacity_hint(&self) -> Option<u64> {
        Some(self.capacity)
    }

    // PHASE 2: Check capacity and warn at thresholds (50%, 75%, 90%)
    fn check_and_warn_capacity(&self) {
        if self.capacity == 0 {
            return;
        }

        let usage_pct = (self.current_size as f64 / self.capacity as f64) * 100.0;

        // Warn at key thresholds
        if usage_pct >= 90.0 && usage_pct < 91.0 {
            eprintln!(
                "⚠️  CRITICAL: 90% capacity used ({} / {})",
                format_bytes(self.current_size),
                format_bytes(self.capacity)
            );
            eprintln!("   Actions:");
            eprintln!("   1. Recreate with larger --size");
            eprintln!("   2. Delete old frames with: memvid delete <file> --uri <pattern>");
            eprintln!("   3. If using vectors, consider --no-embedding for new docs");
        } else if usage_pct >= 75.0 && usage_pct < 76.0 {
            eprintln!(
                "⚠️  WARNING: 75% capacity used ({} / {})",
                format_bytes(self.current_size),
                format_bytes(self.capacity)
            );
            eprintln!("   Consider planning for capacity expansion soon");
        } else if usage_pct >= 50.0 && usage_pct < 51.0 {
            eprintln!(
                "ℹ️  INFO: 50% capacity used ({} / {})",
                format_bytes(self.current_size),
                format_bytes(self.capacity)
            );
        }
    }

    fn estimate_overhead(additional: u64) -> u64 {
        let fractional = ((additional as f64) * Self::RELATIVE_OVERHEAD).ceil() as u64;
        fractional.max(Self::MIN_OVERHEAD)
    }
}

#[derive(Debug, Clone)]
pub struct FileAnalysis {
    pub mime: String,
    pub metadata: Option<DocMetadata>,
    pub search_text: Option<String>,
    pub extraction: ExtractionSummary,
}

#[derive(Clone, Copy)]
pub struct AudioAnalyzeOptions {
    force: bool,
    segment_secs: u32,
    transcribe: bool,
}

impl AudioAnalyzeOptions {
    const DEFAULT_SEGMENT_SECS: u32 = 30;

    fn normalised_segment_secs(self) -> u32 {
        self.segment_secs.max(1)
    }
}

impl Default for AudioAnalyzeOptions {
    fn default() -> Self {
        Self {
            force: false,
            segment_secs: Self::DEFAULT_SEGMENT_SECS,
            transcribe: false,
        }
    }
}

struct AudioAnalysis {
    metadata: DocAudioMetadata,
    caption: Option<String>,
    search_terms: Vec<String>,
    transcript: Option<String>,
}

struct ImageAnalysis {
    width: u32,
    height: u32,
    palette: Vec<String>,
    caption: Option<String>,
    exif: Option<DocExifMetadata>,
}

#[derive(Debug, Clone)]
pub struct ExtractionSummary {
    reader: Option<String>,
    status: ExtractionStatus,
    warnings: Vec<String>,
    duration_ms: Option<u64>,
    pages_processed: Option<u32>,
}

impl ExtractionSummary {
    fn record_warning<S: Into<String>>(&mut self, warning: S) {
        self.warnings.push(warning.into());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtractionStatus {
    Skipped,
    Ok,
    FallbackUsed,
    Empty,
    Failed,
}

impl ExtractionStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Skipped => "skipped",
            Self::Ok => "ok",
            Self::FallbackUsed => "fallback_used",
            Self::Empty => "empty",
            Self::Failed => "failed",
        }
    }
}

fn analyze_video(source_path: Option<&Path>, mime: &str, bytes: u64) -> MediaManifest {
    let filename = source_path
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .map(|value| value.to_string());
    MediaManifest {
        kind: "video".to_string(),
        mime: mime.to_string(),
        bytes,
        filename,
        duration_ms: None,
        width: None,
        height: None,
        codec: None,
    }
}

pub fn analyze_file(
    source_path: Option<&Path>,
    payload: &[u8],
    payload_utf8: Option<&str>,
    inferred_title: Option<&str>,
    audio_opts: AudioAnalyzeOptions,
    force_video: bool,
) -> FileAnalysis {
    let (mime, treat_as_text_base) = detect_mime(source_path, payload, payload_utf8);
    let is_video = force_video || mime.starts_with("video/");
    let treat_as_text = if is_video { false } else { treat_as_text_base };

    let mut metadata = DocMetadata::default();
    metadata.mime = Some(mime.clone());
    metadata.bytes = Some(payload.len() as u64);
    metadata.hash = Some(hash(payload).to_hex().to_string());

    let mut search_text = None;

    let mut extraction = ExtractionSummary {
        reader: None,
        status: ExtractionStatus::Skipped,
        warnings: Vec::new(),
        duration_ms: None,
        pages_processed: None,
    };

    let reader_registry = default_reader_registry();
    let magic = payload.get(..MAGIC_SNIFF_BYTES).and_then(|slice| {
        if slice.is_empty() {
            None
        } else {
            Some(slice)
        }
    });

    let apply_extracted_text = |text: &str,
                                reader_label: &str,
                                status_if_applied: ExtractionStatus,
                                extraction: &mut ExtractionSummary,
                                metadata: &mut DocMetadata,
                                search_text: &mut Option<String>|
     -> bool {
        if let Some(normalized) = normalize_text(text, MAX_SEARCH_TEXT_LEN) {
            let mut value = normalized.text;
            if normalized.truncated {
                value.push('…');
            }
            if metadata.caption.is_none() {
                if let Some(caption) = caption_from_text(&value) {
                    metadata.caption = Some(caption);
                }
            }
            *search_text = Some(value);
            extraction.reader = Some(reader_label.to_string());
            extraction.status = status_if_applied;
            true
        } else {
            false
        }
    };

    if is_video {
        metadata.media = Some(analyze_video(source_path, &mime, payload.len() as u64));
    }

    if treat_as_text {
        if let Some(text) = payload_utf8 {
            if !apply_extracted_text(
                text,
                "bytes",
                ExtractionStatus::Ok,
                &mut extraction,
                &mut metadata,
                &mut search_text,
            ) {
                extraction.status = ExtractionStatus::Empty;
                extraction.reader = Some("bytes".to_string());
                let msg = "text payload contained no searchable content after normalization";
                extraction.record_warning(msg);
                warn!("{msg}");
            }
        } else {
            extraction.reader = Some("bytes".to_string());
            extraction.status = ExtractionStatus::Failed;
            let msg = "payload reported as text but was not valid UTF-8";
            extraction.record_warning(msg);
            warn!("{msg}");
        }
    } else if mime == "application/pdf"
        || mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        || mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        || mime == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        || mime == "application/vnd.ms-excel"
        || mime == "application/msword"
    {
        // Use reader registry for PDFs and Office documents (XLSX, DOCX, PPTX, etc.)
        let mut applied = false;

        let format_hint = infer_document_format(Some(mime.as_str()), magic);
        let hint = ReaderHint::new(Some(mime.as_str()), format_hint)
            .with_uri(source_path.and_then(|p| p.to_str()))
            .with_magic(magic);

        if let Some(reader) = reader_registry.find_reader(&hint) {
            match reader.extract(payload, &hint) {
                Ok(output) => {
                    extraction.reader = Some(output.reader_name.clone());
                    if let Some(ms) = output.diagnostics.duration_ms {
                        extraction.duration_ms = Some(ms);
                    }
                    if let Some(pages) = output.diagnostics.pages_processed {
                        extraction.pages_processed = Some(pages);
                    }
                    if let Some(mime_type) = output.document.mime_type.clone() {
                        metadata.mime = Some(mime_type);
                    }

                    for warning in output.diagnostics.warnings.iter() {
                        extraction.record_warning(warning);
                        warn!("{warning}");
                    }

                    let status = if output.diagnostics.fallback {
                        ExtractionStatus::FallbackUsed
                    } else {
                        ExtractionStatus::Ok
                    };

                    if let Some(doc_text) = output.document.text.as_ref() {
                        applied = apply_extracted_text(
                            doc_text,
                            output.reader_name.as_str(),
                            status,
                            &mut extraction,
                            &mut metadata,
                            &mut search_text,
                        );
                        if !applied {
                            extraction.reader = Some(output.reader_name);
                            extraction.status = ExtractionStatus::Empty;
                            let msg = "primary reader returned no usable text";
                            extraction.record_warning(msg);
                            warn!("{msg}");
                        }
                    } else {
                        extraction.reader = Some(output.reader_name);
                        extraction.status = ExtractionStatus::Empty;
                        let msg = "primary reader returned empty text";
                        extraction.record_warning(msg);
                        warn!("{msg}");
                    }
                }
                Err(err) => {
                    let name = reader.name();
                    extraction.reader = Some(name.to_string());
                    extraction.status = ExtractionStatus::Failed;
                    let msg = format!("primary reader {name} failed: {err}");
                    extraction.record_warning(&msg);
                    warn!("{msg}");
                }
            }
        } else {
            extraction.record_warning("no registered reader matched this PDF payload");
            warn!("no registered reader matched this PDF payload");
        }

        if !applied {
            match extract_pdf_text(payload) {
                Ok(pdf_text) => {
                    if !apply_extracted_text(
                        &pdf_text,
                        "pdf_extract",
                        ExtractionStatus::FallbackUsed,
                        &mut extraction,
                        &mut metadata,
                        &mut search_text,
                    ) {
                        extraction.reader = Some("pdf_extract".to_string());
                        extraction.status = ExtractionStatus::Empty;
                        let msg = "PDF text extraction yielded no searchable content";
                        extraction.record_warning(msg);
                        warn!("{msg}");
                    }
                }
                Err(err) => {
                    extraction.reader = Some("pdf_extract".to_string());
                    extraction.status = ExtractionStatus::Failed;
                    let msg = format!("failed to extract PDF text via fallback: {err}");
                    extraction.record_warning(&msg);
                    warn!("{msg}");
                }
            }
        }
    }

    if !is_video && mime.starts_with("image/") {
        if let Some(image_meta) = analyze_image(payload) {
            let ImageAnalysis {
                width,
                height,
                palette,
                caption,
                exif,
            } = image_meta;
            metadata.width = Some(width);
            metadata.height = Some(height);
            if !palette.is_empty() {
                metadata.colors = Some(palette);
            }
            if metadata.caption.is_none() {
                metadata.caption = caption;
            }
            if let Some(exif) = exif {
                metadata.exif = Some(exif);
            }
        }
    }

    if !is_video && (audio_opts.force || mime.starts_with("audio/")) {
        if let Some(AudioAnalysis {
            metadata: audio_metadata,
            caption: audio_caption,
            mut search_terms,
            transcript,
        }) = analyze_audio(payload, source_path, &mime, audio_opts)
        {
            if metadata.audio.is_none()
                || metadata
                    .audio
                    .as_ref()
                    .map_or(true, DocAudioMetadata::is_empty)
            {
                metadata.audio = Some(audio_metadata);
            }
            if metadata.caption.is_none() {
                metadata.caption = audio_caption;
            }

            // Use transcript as primary search text if available
            if let Some(ref text) = transcript {
                extraction.reader = Some("whisper".to_string());
                extraction.status = ExtractionStatus::Ok;
                search_text = Some(text.clone());
            } else if !search_terms.is_empty() {
                match &mut search_text {
                    Some(existing) => {
                        for term in search_terms.drain(..) {
                            if !existing.contains(&term) {
                                if !existing.ends_with(' ') {
                                    existing.push(' ');
                                }
                                existing.push_str(&term);
                            }
                        }
                    }
                    None => {
                        search_text = Some(search_terms.join(" "));
                    }
                }
            }
        }
    }

    if metadata.caption.is_none() {
        if let Some(title) = inferred_title {
            let trimmed = title.trim();
            if !trimmed.is_empty() {
                metadata.caption = Some(truncate_to_boundary(trimmed, 240));
            }
        }
    }

    FileAnalysis {
        mime,
        metadata: if metadata.is_empty() {
            None
        } else {
            Some(metadata)
        },
        search_text,
        extraction,
    }
}

fn default_reader_registry() -> &'static ReaderRegistry {
    static REGISTRY: OnceLock<ReaderRegistry> = OnceLock::new();
    REGISTRY.get_or_init(ReaderRegistry::default)
}

fn infer_document_format(mime: Option<&str>, magic: Option<&[u8]>) -> Option<DocumentFormat> {
    if detect_pdf_magic(magic) {
        return Some(DocumentFormat::Pdf);
    }

    let mime = mime?.trim().to_ascii_lowercase();
    match mime.as_str() {
        "application/pdf" => Some(DocumentFormat::Pdf),
        "text/plain" => Some(DocumentFormat::PlainText),
        "text/markdown" => Some(DocumentFormat::Markdown),
        "text/html" | "application/xhtml+xml" => Some(DocumentFormat::Html),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Some(DocumentFormat::Docx)
        }
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => {
            Some(DocumentFormat::Pptx)
        }
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
            Some(DocumentFormat::Xlsx)
        }
        other if other.starts_with("text/") => Some(DocumentFormat::PlainText),
        _ => None,
    }
}

fn detect_pdf_magic(magic: Option<&[u8]>) -> bool {
    let mut slice = match magic {
        Some(slice) if !slice.is_empty() => slice,
        _ => return false,
    };
    if slice.starts_with(&[0xEF, 0xBB, 0xBF]) {
        slice = &slice[3..];
    }
    while let Some((first, rest)) = slice.split_first() {
        if first.is_ascii_whitespace() {
            slice = rest;
        } else {
            break;
        }
    }
    slice.starts_with(b"%PDF")
}

/// Robust PDF text extraction with multiple fallbacks for cross-platform support.
/// Tries all extractors and returns the one with the most text content.
fn extract_pdf_text(payload: &[u8]) -> Result<String, PdfExtractError> {
    let mut best_text = String::new();
    let mut best_source = "none";

    // Minimum chars to consider extraction "good" - scales with PDF size
    // For a 12MB PDF, expect at least ~1000 chars; for 1MB, ~100 chars
    let min_good_chars = (payload.len() / 10000).max(100).min(5000);

    // Try pdf_extract first
    match pdf_extract::extract_text_from_mem(payload) {
        Ok(text) => {
            let trimmed = text.trim();
            if trimmed.len() > best_text.len() {
                best_text = trimmed.to_string();
                best_source = "pdf_extract";
            }
            if trimmed.len() >= min_good_chars {
                info!("pdf_extract extracted {} chars (good)", trimmed.len());
                return Ok(best_text);
            } else if !trimmed.is_empty() {
                warn!(
                    "pdf_extract returned only {} chars, trying other extractors",
                    trimmed.len()
                );
            }
        }
        Err(err) => {
            warn!("pdf_extract failed: {err}");
        }
    }

    // Try lopdf - pure Rust, works on all platforms
    match extract_pdf_with_lopdf(payload) {
        Ok(text) => {
            let trimmed = text.trim();
            if trimmed.len() > best_text.len() {
                best_text = trimmed.to_string();
                best_source = "lopdf";
            }
            if trimmed.len() >= min_good_chars {
                info!("lopdf extracted {} chars (good)", trimmed.len());
                return Ok(best_text);
            } else if !trimmed.is_empty() {
                warn!(
                    "lopdf returned only {} chars, trying pdftotext",
                    trimmed.len()
                );
            }
        }
        Err(err) => {
            warn!("lopdf failed: {err}");
        }
    }

    // Try pdftotext binary (requires poppler installed)
    match fallback_pdftotext(payload) {
        Ok(text) => {
            let trimmed = text.trim();
            if trimmed.len() > best_text.len() {
                best_text = trimmed.to_string();
                best_source = "pdftotext";
            }
            if !trimmed.is_empty() {
                info!("pdftotext extracted {} chars", trimmed.len());
            }
        }
        Err(err) => {
            warn!("pdftotext failed: {err}");
        }
    }

    // Return the best result we found
    if !best_text.is_empty() {
        info!(
            "using {} extraction ({} chars)",
            best_source,
            best_text.len()
        );
        Ok(best_text)
    } else {
        warn!("all PDF extraction methods failed, storing without searchable text");
        Ok(String::new())
    }
}

/// Extract text from PDF using lopdf (pure Rust, no native dependencies)
fn extract_pdf_with_lopdf(payload: &[u8]) -> Result<String> {
    use lopdf::Document;

    let mut document =
        Document::load_mem(payload).map_err(|e| anyhow!("lopdf failed to load PDF: {e}"))?;

    // Try to decrypt if encrypted (empty password)
    if document.is_encrypted() {
        let _ = document.decrypt("");
    }

    // Decompress streams for text extraction
    let _ = document.decompress();

    // Get sorted page numbers
    let mut page_numbers: Vec<u32> = document.get_pages().keys().copied().collect();
    if page_numbers.is_empty() {
        return Err(anyhow!("PDF has no pages"));
    }
    page_numbers.sort_unstable();

    // Limit to 4096 pages max
    if page_numbers.len() > 4096 {
        page_numbers.truncate(4096);
        warn!("PDF has more than 4096 pages, truncating extraction");
    }

    // Extract text from all pages
    let text = document
        .extract_text(&page_numbers)
        .map_err(|e| anyhow!("lopdf text extraction failed: {e}"))?;

    Ok(text.trim().to_string())
}

fn fallback_pdftotext(payload: &[u8]) -> Result<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let pdftotext = which::which("pdftotext").context("pdftotext binary not found in PATH")?;

    let mut temp_pdf = tempfile::NamedTempFile::new().context("failed to create temp pdf file")?;
    temp_pdf
        .write_all(payload)
        .context("failed to write pdf payload to temp file")?;
    temp_pdf.flush().context("failed to flush temp pdf file")?;

    let child = Command::new(pdftotext)
        .arg("-layout")
        .arg(temp_pdf.path())
        .arg("-")
        .stdout(Stdio::piped())
        .spawn()
        .context("failed to spawn pdftotext")?;

    let output = child
        .wait_with_output()
        .context("failed to read pdftotext output")?;

    if !output.status.success() {
        return Err(anyhow!("pdftotext exited with status {}", output.status));
    }

    let text = String::from_utf8(output.stdout).context("pdftotext produced non-UTF8 output")?;
    Ok(text)
}

fn detect_mime(
    source_path: Option<&Path>,
    payload: &[u8],
    payload_utf8: Option<&str>,
) -> (String, bool) {
    // For ZIP-based formats (OOXML: docx, xlsx, pptx), check extension FIRST
    // because magic byte detection returns generic "application/zip"
    if let Some(path) = source_path {
        if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
            let ext_lower = ext.to_ascii_lowercase();
            // Office Open XML and legacy Office formats need extension-based detection
            if matches!(
                ext_lower.as_str(),
                "docx" | "xlsx" | "pptx" | "doc" | "xls" | "ppt"
            ) {
                if let Some((mime, treat_as_text)) = mime_from_extension(ext) {
                    return (mime.to_string(), treat_as_text);
                }
            }
        }
    }

    if let Some(kind) = infer::get(payload) {
        let mime = kind.mime_type().to_string();
        let treat_as_text = mime.starts_with("text/")
            || matches!(
                mime.as_str(),
                "application/json" | "application/xml" | "application/javascript" | "image/svg+xml"
            );
        return (mime, treat_as_text);
    }

    let magic = magic_from_u8(payload);
    if !magic.is_empty() && magic != "application/octet-stream" {
        let treat_as_text = magic.starts_with("text/")
            || matches!(
                magic,
                "application/json"
                    | "application/xml"
                    | "application/javascript"
                    | "image/svg+xml"
                    | "text/plain"
            );
        return (magic.to_string(), treat_as_text);
    }

    if let Some(path) = source_path {
        if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
            if let Some((mime, treat_as_text)) = mime_from_extension(ext) {
                return (mime.to_string(), treat_as_text);
            }
        }
    }

    if payload_utf8.is_some() {
        return ("text/plain".to_string(), true);
    }

    ("application/octet-stream".to_string(), false)
}

fn mime_from_extension(ext: &str) -> Option<(&'static str, bool)> {
    let ext_lower = ext.to_ascii_lowercase();
    match ext_lower.as_str() {
        "txt" | "text" | "log" | "cfg" | "conf" | "ini" | "properties" | "sql" | "rs" | "py"
        | "js" | "ts" | "tsx" | "jsx" | "c" | "h" | "cpp" | "hpp" | "go" | "rb" | "php" | "css"
        | "scss" | "sass" | "sh" | "bash" | "zsh" | "ps1" | "swift" | "kt" | "java" | "scala"
        | "lua" | "pl" | "pm" | "r" | "erl" | "ex" | "exs" | "dart" => Some(("text/plain", true)),
        "md" | "markdown" => Some(("text/markdown", true)),
        "rst" => Some(("text/x-rst", true)),
        "json" => Some(("application/json", true)),
        "csv" => Some(("text/csv", true)),
        "tsv" => Some(("text/tab-separated-values", true)),
        "yaml" | "yml" => Some(("application/yaml", true)),
        "toml" => Some(("application/toml", true)),
        "html" | "htm" => Some(("text/html", true)),
        "xml" => Some(("application/xml", true)),
        "svg" => Some(("image/svg+xml", true)),
        // Office Open XML formats (ZIP-based, need explicit extension mapping)
        "docx" => Some((
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            false,
        )),
        "xlsx" => Some((
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            false,
        )),
        "pptx" => Some((
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            false,
        )),
        // Legacy Office formats
        "doc" => Some(("application/msword", false)),
        "xls" => Some(("application/vnd.ms-excel", false)),
        "ppt" => Some(("application/vnd.ms-powerpoint", false)),
        "gif" => Some(("image/gif", false)),
        "jpg" | "jpeg" => Some(("image/jpeg", false)),
        "png" => Some(("image/png", false)),
        "bmp" => Some(("image/bmp", false)),
        "ico" => Some(("image/x-icon", false)),
        "tif" | "tiff" => Some(("image/tiff", false)),
        "webp" => Some(("image/webp", false)),
        _ => None,
    }
}

fn caption_from_text(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return Some(truncate_to_boundary(trimmed, 240));
        }
    }
    None
}

fn truncate_to_boundary(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    let mut end = max_len;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    if end == 0 {
        return String::new();
    }
    let mut truncated = text[..end].trim_end().to_string();
    truncated.push('…');
    truncated
}

fn analyze_image(payload: &[u8]) -> Option<ImageAnalysis> {
    let reader = ImageReader::new(Cursor::new(payload))
        .with_guessed_format()
        .map_err(|err| warn!("failed to guess image format: {err}"))
        .ok()?;
    let image = reader
        .decode()
        .map_err(|err| warn!("failed to decode image: {err}"))
        .ok()?;
    let width = image.width();
    let height = image.height();
    let rgb = image.to_rgb8();
    let palette = match get_palette(
        rgb.as_raw(),
        ColorFormat::Rgb,
        COLOR_PALETTE_SIZE,
        COLOR_PALETTE_QUALITY,
    ) {
        Ok(colors) => colors.into_iter().map(color_to_hex).collect(),
        Err(err) => {
            warn!("failed to compute colour palette: {err}");
            Vec::new()
        }
    };

    let (exif, exif_caption) = extract_exif_metadata(payload);

    Some(ImageAnalysis {
        width,
        height,
        palette,
        caption: exif_caption,
        exif,
    })
}

fn color_to_hex(color: Color) -> String {
    format!("#{:02x}{:02x}{:02x}", color.r, color.g, color.b)
}

fn analyze_audio(
    payload: &[u8],
    source_path: Option<&Path>,
    mime: &str,
    options: AudioAnalyzeOptions,
) -> Option<AudioAnalysis> {
    use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = Cursor::new(payload.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(path) = source_path {
        if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
            hint.with_extension(ext);
        }
    }
    if let Some(suffix) = mime.split('/').nth(1) {
        hint.with_extension(suffix);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| warn!("failed to probe audio stream: {err}"))
        .ok()?;

    let mut format = probed.format;
    let track = format.default_track()?;
    let track_id = track.id;
    let codec_params: CodecParameters = track.codec_params.clone();
    let sample_rate = codec_params.sample_rate;
    let channel_count = codec_params.channels.map(|channels| channels.count() as u8);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|err| warn!("failed to create audio decoder: {err}"))
        .ok()?;

    let mut decoded_frames: u64 = 0;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::DecodeError(err)) => {
                warn!("skipping undecodable audio packet: {err}");
                continue;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(other) => {
                warn!("stopping audio analysis due to error: {other}");
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                decoded_frames = decoded_frames.saturating_add(audio_buf.frames() as u64);
            }
            Err(SymphoniaError::DecodeError(err)) => {
                warn!("failed to decode audio packet: {err}");
            }
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
            }
            Err(other) => {
                warn!("decoder error: {other}");
                break;
            }
        }
    }

    let duration_secs = match sample_rate {
        Some(rate) if rate > 0 => decoded_frames as f64 / rate as f64,
        _ => 0.0,
    };

    let bitrate_kbps = if duration_secs > 0.0 {
        Some(((payload.len() as f64 * 8.0) / duration_secs / 1_000.0).round() as u32)
    } else {
        None
    };

    let tags = extract_lofty_tags(payload);
    let caption = tags
        .get("title")
        .cloned()
        .or_else(|| tags.get("tracktitle").cloned());

    let mut search_terms = Vec::new();
    if let Some(title) = tags.get("title").or_else(|| tags.get("tracktitle")) {
        search_terms.push(title.clone());
    }
    if let Some(artist) = tags.get("artist") {
        search_terms.push(artist.clone());
    }
    if let Some(album) = tags.get("album") {
        search_terms.push(album.clone());
    }
    if let Some(genre) = tags.get("genre") {
        search_terms.push(genre.clone());
    }

    let mut segments = Vec::new();
    if duration_secs > 0.0 {
        let segment_len = options.normalised_segment_secs() as f64;
        if segment_len > 0.0 {
            let mut start = 0.0;
            let mut idx = 1usize;
            while start < duration_secs {
                let end = (start + segment_len).min(duration_secs);
                segments.push(AudioSegmentMetadata {
                    start_seconds: start as f32,
                    end_seconds: end as f32,
                    label: Some(format!("Segment {}", idx)),
                });
                if end >= duration_secs {
                    break;
                }
                start = end;
                idx += 1;
            }
        }
    }

    let mut metadata = DocAudioMetadata::default();
    if duration_secs > 0.0 {
        metadata.duration_secs = Some(duration_secs as f32);
    }
    if let Some(rate) = sample_rate {
        metadata.sample_rate_hz = Some(rate);
    }
    if let Some(channels) = channel_count {
        metadata.channels = Some(channels);
    }
    if let Some(bitrate) = bitrate_kbps {
        metadata.bitrate_kbps = Some(bitrate);
    }
    if codec_params.codec != CODEC_TYPE_NULL {
        metadata.codec = Some(format!("{:?}", codec_params.codec));
    }
    if !segments.is_empty() {
        metadata.segments = segments;
    }
    if !tags.is_empty() {
        metadata.tags = tags
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<BTreeMap<_, _>>();
    }

    // Transcribe audio if requested
    let transcript = if options.transcribe {
        transcribe_audio(source_path)
    } else {
        None
    };

    Some(AudioAnalysis {
        metadata,
        caption,
        search_terms,
        transcript,
    })
}

/// Transcribe audio file using Whisper
#[cfg(feature = "whisper")]
fn transcribe_audio(source_path: Option<&Path>) -> Option<String> {
    use memvid_core::{WhisperConfig, WhisperTranscriber};

    let path = source_path?;

    tracing::info!(path = %path.display(), "Transcribing audio with Whisper");

    let config = WhisperConfig::default();
    let mut transcriber = match WhisperTranscriber::new(&config) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to initialize Whisper transcriber");
            return None;
        }
    };

    match transcriber.transcribe_file(path) {
        Ok(result) => {
            tracing::info!(
                duration = result.duration_secs,
                text_len = result.text.len(),
                "Audio transcription complete"
            );
            Some(result.text)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to transcribe audio");
            None
        }
    }
}

#[cfg(not(feature = "whisper"))]
fn transcribe_audio(_source_path: Option<&Path>) -> Option<String> {
    tracing::warn!(
        "Whisper feature not enabled. Rebuild with --features whisper to enable transcription."
    );
    None
}

fn extract_lofty_tags(payload: &[u8]) -> HashMap<String, String> {
    use lofty::{ItemKey, Probe as LoftyProbe, Tag, TaggedFileExt};

    fn collect_tag(tag: &Tag, out: &mut HashMap<String, String>) {
        if let Some(value) = tag.get_string(&ItemKey::TrackTitle) {
            out.entry("title".into())
                .or_insert_with(|| value.to_string());
            out.entry("tracktitle".into())
                .or_insert_with(|| value.to_string());
        }
        if let Some(value) = tag.get_string(&ItemKey::TrackArtist) {
            out.entry("artist".into())
                .or_insert_with(|| value.to_string());
        } else if let Some(value) = tag.get_string(&ItemKey::AlbumArtist) {
            out.entry("artist".into())
                .or_insert_with(|| value.to_string());
        }
        if let Some(value) = tag.get_string(&ItemKey::AlbumTitle) {
            out.entry("album".into())
                .or_insert_with(|| value.to_string());
        }
        if let Some(value) = tag.get_string(&ItemKey::Genre) {
            out.entry("genre".into())
                .or_insert_with(|| value.to_string());
        }
        if let Some(value) = tag.get_string(&ItemKey::TrackNumber) {
            out.entry("track_number".into())
                .or_insert_with(|| value.to_string());
        }
        if let Some(value) = tag.get_string(&ItemKey::DiscNumber) {
            out.entry("disc_number".into())
                .or_insert_with(|| value.to_string());
        }
    }

    let mut tags = HashMap::new();
    let probe = match LoftyProbe::new(Cursor::new(payload)).guess_file_type() {
        Ok(probe) => probe,
        Err(err) => {
            warn!("failed to guess audio tag file type: {err}");
            return tags;
        }
    };

    let tagged_file = match probe.read() {
        Ok(file) => file,
        Err(err) => {
            warn!("failed to read audio tags: {err}");
            return tags;
        }
    };

    if let Some(primary) = tagged_file.primary_tag() {
        collect_tag(primary, &mut tags);
    }
    for tag in tagged_file.tags() {
        collect_tag(tag, &mut tags);
    }

    tags
}

fn extract_exif_metadata(payload: &[u8]) -> (Option<DocExifMetadata>, Option<String>) {
    let mut cursor = Cursor::new(payload);
    let exif = match ExifReader::new().read_from_container(&mut cursor) {
        Ok(exif) => exif,
        Err(err) => {
            warn!("failed to read EXIF metadata: {err}");
            return (None, None);
        }
    };

    let mut doc = DocExifMetadata::default();
    let mut has_data = false;

    if let Some(make) = exif_string(&exif, Tag::Make) {
        doc.make = Some(make);
        has_data = true;
    }
    if let Some(model) = exif_string(&exif, Tag::Model) {
        doc.model = Some(model);
        has_data = true;
    }
    if let Some(lens) =
        exif_string(&exif, Tag::LensModel).or_else(|| exif_string(&exif, Tag::LensMake))
    {
        doc.lens = Some(lens);
        has_data = true;
    }
    if let Some(dt) =
        exif_string(&exif, Tag::DateTimeOriginal).or_else(|| exif_string(&exif, Tag::DateTime))
    {
        doc.datetime = Some(dt);
        has_data = true;
    }

    if let Some(gps) = extract_gps_metadata(&exif) {
        doc.gps = Some(gps);
        has_data = true;
    }

    let caption = exif_string(&exif, Tag::ImageDescription);

    let metadata = if has_data { Some(doc) } else { None };
    (metadata, caption)
}

fn exif_string(exif: &exif::Exif, tag: Tag) -> Option<String> {
    exif.fields()
        .find(|field| field.tag == tag)
        .and_then(|field| field_to_string(field, exif))
}

fn field_to_string(field: &exif::Field, exif: &exif::Exif) -> Option<String> {
    let value = field.display_value().with_unit(exif).to_string();
    let trimmed = value.trim_matches('\0').trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn extract_gps_metadata(exif: &exif::Exif) -> Option<DocGpsMetadata> {
    let latitude_field = exif.fields().find(|field| field.tag == Tag::GPSLatitude)?;
    let latitude_ref_field = exif
        .fields()
        .find(|field| field.tag == Tag::GPSLatitudeRef)?;
    let longitude_field = exif.fields().find(|field| field.tag == Tag::GPSLongitude)?;
    let longitude_ref_field = exif
        .fields()
        .find(|field| field.tag == Tag::GPSLongitudeRef)?;

    let mut latitude = gps_value_to_degrees(&latitude_field.value)?;
    let mut longitude = gps_value_to_degrees(&longitude_field.value)?;

    if let Some(reference) = value_to_ascii(&latitude_ref_field.value) {
        if reference.eq_ignore_ascii_case("S") {
            latitude = -latitude;
        }
    }
    if let Some(reference) = value_to_ascii(&longitude_ref_field.value) {
        if reference.eq_ignore_ascii_case("W") {
            longitude = -longitude;
        }
    }

    Some(DocGpsMetadata {
        latitude,
        longitude,
    })
}

fn gps_value_to_degrees(value: &ExifValue) -> Option<f64> {
    match value {
        ExifValue::Rational(values) if !values.is_empty() => {
            let deg = rational_to_f64_u(values.get(0)?)?;
            let min = values
                .get(1)
                .and_then(|v| rational_to_f64_u(v))
                .unwrap_or(0.0);
            let sec = values
                .get(2)
                .and_then(|v| rational_to_f64_u(v))
                .unwrap_or(0.0);
            Some(deg + (min / 60.0) + (sec / 3600.0))
        }
        ExifValue::SRational(values) if !values.is_empty() => {
            let deg = rational_to_f64_i(values.get(0)?)?;
            let min = values
                .get(1)
                .and_then(|v| rational_to_f64_i(v))
                .unwrap_or(0.0);
            let sec = values
                .get(2)
                .and_then(|v| rational_to_f64_i(v))
                .unwrap_or(0.0);
            Some(deg + (min / 60.0) + (sec / 3600.0))
        }
        _ => None,
    }
}

fn rational_to_f64_u(value: &exif::Rational) -> Option<f64> {
    if value.denom == 0 {
        None
    } else {
        Some(value.num as f64 / value.denom as f64)
    }
}

fn rational_to_f64_i(value: &exif::SRational) -> Option<f64> {
    if value.denom == 0 {
        None
    } else {
        Some(value.num as f64 / value.denom as f64)
    }
}

fn value_to_ascii(value: &ExifValue) -> Option<String> {
    if let ExifValue::Ascii(values) = value {
        values
            .first()
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .map(|s| s.trim_matches('\0').trim().to_string())
            .filter(|s| !s.is_empty())
    } else {
        None
    }
}

fn ingest_payload(
    mem: &mut Memvid,
    args: &PutArgs,
    config: &CliConfig,
    payload: Vec<u8>,
    source_path: Option<&Path>,
    capacity_guard: Option<&mut CapacityGuard>,
    enable_embedding: bool,
    runtime: Option<&EmbeddingRuntime>,
    parallel_mode: bool,
) -> Result<IngestOutcome> {
    let original_bytes = payload.len();
    let (stored_bytes, compressed) = canonical_storage_len(&payload)?;

    // Check if adding this payload would exceed free tier limit
    // Require API key for operations that would exceed 1GB total
    let current_size = mem.stats().map(|s| s.size_bytes).unwrap_or(0);
    crate::utils::ensure_capacity_with_api_key(current_size, stored_bytes as u64, config)?;

    let payload_text = std::str::from_utf8(&payload).ok();
    let inferred_title = derive_title(args.title.clone(), source_path, payload_text);

    let audio_opts = AudioAnalyzeOptions {
        force: args.audio || args.transcribe,
        segment_secs: args
            .audio_segment_seconds
            .unwrap_or(AudioAnalyzeOptions::DEFAULT_SEGMENT_SECS),
        transcribe: args.transcribe,
    };

    let mut analysis = analyze_file(
        source_path,
        &payload,
        payload_text,
        inferred_title.as_deref(),
        audio_opts,
        args.video,
    );

    if args.video && !analysis.mime.starts_with("video/") {
        anyhow::bail!(
            "--video requires a video input; detected MIME type {}",
            analysis.mime
        );
    }

    let mut search_text = analysis.search_text.clone();
    if let Some(ref mut text) = search_text {
        if text.len() > MAX_SEARCH_TEXT_LEN {
            let truncated = truncate_to_boundary(text, MAX_SEARCH_TEXT_LEN);
            *text = truncated;
        }
    }
    analysis.search_text = search_text.clone();

    let uri = if let Some(ref explicit) = args.uri {
        derive_uri(Some(explicit), None)
    } else if args.video {
        derive_video_uri(&payload, source_path, &analysis.mime)
    } else {
        derive_uri(None, source_path)
    };

    let mut options_builder = PutOptions::builder()
        .enable_embedding(enable_embedding)
        .auto_tag(!args.no_auto_tag)
        .extract_dates(!args.no_extract_dates)
        .extract_triplets(!args.no_extract_triplets);
    if let Some(ts) = args.timestamp {
        options_builder = options_builder.timestamp(ts);
    }
    if let Some(track) = &args.track {
        options_builder = options_builder.track(track.clone());
    }
    if args.video {
        options_builder = options_builder.kind("video");
        options_builder = options_builder.tag("kind", "video");
        options_builder = options_builder.push_tag("video");
    } else if let Some(kind) = &args.kind {
        options_builder = options_builder.kind(kind.clone());
    }
    options_builder = options_builder.uri(uri.clone());
    if let Some(ref title) = inferred_title {
        options_builder = options_builder.title(title.clone());
    }
    for (key, value) in &args.tags {
        options_builder = options_builder.tag(key.clone(), value.clone());
        if !key.is_empty() {
            options_builder = options_builder.push_tag(key.clone());
        }
        if !value.is_empty() && value != key {
            options_builder = options_builder.push_tag(value.clone());
        }
    }
    for label in &args.labels {
        options_builder = options_builder.label(label.clone());
    }
    if let Some(metadata) = analysis.metadata.clone() {
        if !metadata.is_empty() {
            options_builder = options_builder.metadata(metadata);
        }
    }
    for (idx, entry) in args.metadata.iter().enumerate() {
        match serde_json::from_str::<DocMetadata>(entry) {
            Ok(meta) => {
                options_builder = options_builder.metadata(meta);
            }
            Err(_) => match serde_json::from_str::<serde_json::Value>(entry) {
                Ok(value) => {
                    options_builder =
                        options_builder.metadata_entry(format!("custom_metadata_{}", idx), value);
                }
                Err(err) => {
                    warn!("failed to parse --metadata JSON: {err}");
                }
            },
        }
    }
    if let Some(text) = search_text.clone() {
        if !text.trim().is_empty() {
            options_builder = options_builder.search_text(text);
        }
    }
    // Default: no-raw mode (only store extracted text + BLAKE3 hash)
    // Use --raw to store raw binary content for later retrieval
    let effective_no_raw = !args.raw; // true by default unless --raw is passed
    if effective_no_raw {
        options_builder = options_builder.no_raw(true);
        if let Some(path) = source_path {
            options_builder = options_builder.source_path(path.display().to_string());
        }
    }
    // --dedup: skip if identical content already exists
    if args.dedup {
        options_builder = options_builder.dedup(true);
    }
    let mut options = options_builder.build();

    // Set extraction budget for progressive ingestion
    if let Some(budget_ms) = args.extraction_budget_ms {
        options.extraction_budget_ms = budget_ms;
    }

    let existing_frame = if args.update_existing || !args.allow_duplicate {
        match mem.frame_by_uri(&uri) {
            Ok(frame) => Some(frame),
            Err(MemvidError::FrameNotFoundByUri { .. }) => None,
            Err(err) => return Err(err.into()),
        }
    } else {
        None
    };

    // Compute frame_id: for updates it's the existing frame's id,
    // for new frames it's the current frame count (which becomes the next id)
    let frame_id = if let Some(ref existing) = existing_frame {
        existing.id
    } else {
        mem.stats()?.frame_count
    };

    let mut embedded = false;
    let allow_embedding = enable_embedding && !args.video;
    if enable_embedding && !allow_embedding && args.video {
        warn!("semantic embeddings are not generated for video payloads");
    }
    let seq = if let Some(existing) = existing_frame {
        if args.update_existing {
            let payload_for_update = payload.clone();
            if allow_embedding {
                if let Some(path) = args.embedding_vec.as_deref() {
                    let embedding = crate::utils::read_embedding(path)?;
                    if let Some(model_str) = args.embedding_vec_model.as_deref() {
                        let choice = model_str.parse::<EmbeddingModelChoice>()?;
                        let expected = choice.dimensions();
                        if expected != 0 && embedding.len() != expected {
                            anyhow::bail!(
                                "pre-computed embedding has {} dimensions but --embedding-vec-model {} expects {}",
                                embedding.len(),
                                choice.name(),
                                expected
                            );
                        }
                        apply_embedding_identity_metadata_from_choice(
                            &mut options,
                            choice,
                            embedding.len(),
                            Some(model_str),
                        );
                    } else {
                        options.extra_metadata.insert(
                            MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
                            embedding.len().to_string(),
                        );
                    }
                    embedded = true;
                    mem.update_frame(
                        existing.id,
                        Some(payload_for_update),
                        options.clone(),
                        Some(embedding),
                    )
                    .map_err(|err| {
                        map_put_error(err, capacity_guard.as_ref().and_then(|g| g.capacity_hint()))
                    })?
                } else {
                    let runtime = runtime.ok_or_else(|| anyhow!("semantic runtime unavailable"))?;
                    let embed_text = search_text
                        .clone()
                        .or_else(|| payload_text.map(|text| text.to_string()))
                        .unwrap_or_default();
                    if embed_text.trim().is_empty() {
                        warn!("no textual content available; embedding skipped");
                        mem.update_frame(
                            existing.id,
                            Some(payload_for_update),
                            options.clone(),
                            None,
                        )
                        .map_err(|err| {
                            map_put_error(
                                err,
                                capacity_guard.as_ref().and_then(|g| g.capacity_hint()),
                            )
                        })?
                    } else {
                        // Truncate to avoid token limits
                        let truncated_text = truncate_for_embedding(&embed_text);
                        if truncated_text.len() < embed_text.len() {
                            info!(
                                "Truncated text from {} to {} chars for embedding",
                                embed_text.len(),
                                truncated_text.len()
                            );
                        }
                        let embedding = runtime.embed_passage(&truncated_text)?;
                        embedded = true;
                        apply_embedding_identity_metadata(&mut options, runtime);
                        mem.update_frame(
                            existing.id,
                            Some(payload_for_update),
                            options.clone(),
                            Some(embedding),
                        )
                        .map_err(|err| {
                            map_put_error(
                                err,
                                capacity_guard.as_ref().and_then(|g| g.capacity_hint()),
                            )
                        })?
                    }
                }
            } else {
                mem.update_frame(existing.id, Some(payload_for_update), options.clone(), None)
                    .map_err(|err| {
                        map_put_error(err, capacity_guard.as_ref().and_then(|g| g.capacity_hint()))
                    })?
            }
        } else {
            return Err(DuplicateUriError::new(uri.as_str()).into());
        }
    } else {
        if let Some(guard) = capacity_guard.as_ref() {
            guard.ensure_capacity(stored_bytes as u64)?;
        }
        if parallel_mode {
            #[cfg(feature = "parallel_segments")]
            {
                let mut parent_embedding = None;
                let mut chunk_embeddings_vec = None;
                if allow_embedding {
                    if let Some(path) = args.embedding_vec.as_deref() {
                        let embedding = crate::utils::read_embedding(path)?;
                        if let Some(model_str) = args.embedding_vec_model.as_deref() {
                            let choice = model_str.parse::<EmbeddingModelChoice>()?;
                            let expected = choice.dimensions();
                            if expected != 0 && embedding.len() != expected {
                                anyhow::bail!(
                                    "pre-computed embedding has {} dimensions but --embedding-vec-model {} expects {}",
                                    embedding.len(),
                                    choice.name(),
                                    expected
                                );
                            }
                            apply_embedding_identity_metadata_from_choice(
                                &mut options,
                                choice,
                                embedding.len(),
                                Some(model_str),
                            );
                        } else {
                            options.extra_metadata.insert(
                                MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
                                embedding.len().to_string(),
                            );
                        }
                        parent_embedding = Some(embedding);
                        embedded = true;
                    } else {
                        let runtime =
                            runtime.ok_or_else(|| anyhow!("semantic runtime unavailable"))?;
                        let embed_text = search_text
                            .clone()
                            .or_else(|| payload_text.map(|text| text.to_string()))
                            .unwrap_or_default();
                        if embed_text.trim().is_empty() {
                            warn!("no textual content available; embedding skipped");
                        } else {
                            // Check if document will be chunked - if so, embed each chunk
                            info!(
                                "parallel mode: checking for chunks on {} bytes payload",
                                payload.len()
                            );
                            if let Some(chunk_texts) = mem.preview_chunks(&payload) {
                                // Document will be chunked - embed each chunk for full semantic coverage
                                info!("parallel mode: Document will be chunked into {} chunks, generating embeddings for each", chunk_texts.len());

                                // Apply temporal enrichment if enabled (before contextual)
                                #[cfg(feature = "temporal_enrich")]
                                let enriched_chunks = if args.temporal_enrich {
                                    info!(
                                        "Temporal enrichment enabled, processing {} chunks",
                                        chunk_texts.len()
                                    );
                                    // Use today's date as fallback document date
                                    let today = chrono::Local::now().date_naive();
                                    let results = temporal_enrich_chunks(&chunk_texts, Some(today));
                                    // Results are tuples of (enriched_text, TemporalEnrichment)
                                    let enriched: Vec<String> =
                                        results.iter().map(|(text, _)| text.clone()).collect();
                                    let resolved_count = results
                                        .iter()
                                        .filter(|(_, e)| {
                                            e.relative_phrases.iter().any(|p| p.resolved.is_some())
                                        })
                                        .count();
                                    info!(
                                    "Temporal enrichment: resolved {} chunks with temporal context",
                                    resolved_count
                                );
                                    enriched
                                } else {
                                    chunk_texts.clone()
                                };
                                #[cfg(not(feature = "temporal_enrich"))]
                                let enriched_chunks = {
                                    if args.temporal_enrich {
                                        warn!(
                                        "Temporal enrichment requested but feature not compiled in"
                                    );
                                    }
                                    chunk_texts.clone()
                                };

                                // Apply contextual retrieval if enabled
                                let embed_chunks = if args.contextual {
                                    info!("Contextual retrieval enabled, generating context prefixes for {} chunks", enriched_chunks.len());
                                    let engine = if args.contextual_model.as_deref()
                                        == Some("local")
                                    {
                                        #[cfg(feature = "llama-cpp")]
                                        {
                                            let model_path = get_local_contextual_model_path()?;
                                            ContextualEngine::local(model_path)
                                        }
                                        #[cfg(not(feature = "llama-cpp"))]
                                        {
                                            anyhow::bail!("Local contextual model requires the 'llama-cpp' feature. Use --contextual-model openai or omit the flag for OpenAI.");
                                        }
                                    } else if let Some(model) = &args.contextual_model {
                                        ContextualEngine::openai_with_model(model)?
                                    } else {
                                        ContextualEngine::openai()?
                                    };

                                    match engine
                                        .generate_contexts_batch(&embed_text, &enriched_chunks)
                                    {
                                        Ok(contexts) => {
                                            info!(
                                                "Generated {} contextual prefixes",
                                                contexts.len()
                                            );
                                            apply_contextual_prefixes(
                                                &embed_text,
                                                &enriched_chunks,
                                                &contexts,
                                            )
                                        }
                                        Err(e) => {
                                            warn!("Failed to generate contextual prefixes: {}. Using original chunks.", e);
                                            enriched_chunks.clone()
                                        }
                                    }
                                } else {
                                    enriched_chunks
                                };

                                let truncated_chunks: Vec<String> = embed_chunks
                                .iter()
                                .map(|chunk_text| {
                                    // Truncate chunk to avoid provider token limit issues (contextual prefixes can make chunks large)
                                    let truncated_chunk = truncate_for_embedding(chunk_text);
                                    if truncated_chunk.len() < chunk_text.len() {
                                        info!(
                                            "parallel mode: Truncated chunk from {} to {} chars for embedding",
                                            chunk_text.len(),
                                            truncated_chunk.len()
                                        );
                                    }
                                    truncated_chunk
                                })
                                .collect();
                                let truncated_refs: Vec<&str> = truncated_chunks
                                    .iter()
                                    .map(|chunk| chunk.as_str())
                                    .collect();
                                let chunk_embeddings =
                                    runtime.embed_batch_passages(&truncated_refs)?;
                                let num_chunks = chunk_embeddings.len();
                                chunk_embeddings_vec = Some(chunk_embeddings);
                                // Also embed the parent for the parent frame (truncate to avoid token limits)
                                let parent_text = truncate_for_embedding(&embed_text);
                                if parent_text.len() < embed_text.len() {
                                    info!("parallel mode: Truncated parent text from {} to {} chars for embedding", embed_text.len(), parent_text.len());
                                }
                                parent_embedding = Some(runtime.embed_passage(&parent_text)?);
                                embedded = true;
                                info!(
                                "parallel mode: Generated {} chunk embeddings + 1 parent embedding",
                                num_chunks
                            );
                            } else {
                                // No chunking - just embed the whole document (truncate to avoid token limits)
                                info!("parallel mode: No chunking (payload < 2400 chars or not UTF-8), using single embedding");
                                let truncated_text = truncate_for_embedding(&embed_text);
                                if truncated_text.len() < embed_text.len() {
                                    info!("parallel mode: Truncated text from {} to {} chars for embedding", embed_text.len(), truncated_text.len());
                                }
                                parent_embedding = Some(runtime.embed_passage(&truncated_text)?);
                                embedded = true;
                            }
                        }
                    }
                }
                if embedded {
                    if let Some(runtime) = runtime {
                        apply_embedding_identity_metadata(&mut options, runtime);
                    }
                }
                let payload_variant = if let Some(path) = source_path {
                    ParallelPayload::Path(path.to_path_buf())
                } else {
                    ParallelPayload::Bytes(payload)
                };
                let input = ParallelInput {
                    payload: payload_variant,
                    options: options.clone(),
                    embedding: parent_embedding,
                    chunk_embeddings: chunk_embeddings_vec,
                };
                return Ok(IngestOutcome {
                    report: PutReport {
                        seq: 0,
                        frame_id: 0, // Will be populated after parallel commit
                        uri,
                        title: inferred_title,
                        original_bytes,
                        stored_bytes,
                        compressed,
                        no_raw: effective_no_raw,
                        source: source_path.map(Path::to_path_buf),
                        mime: Some(analysis.mime),
                        metadata: analysis.metadata,
                        extraction: analysis.extraction,
                        #[cfg(feature = "logic_mesh")]
                        search_text: search_text.clone(),
                    },
                    embedded,
                    parallel_input: Some(input),
                });
            }
            #[cfg(not(feature = "parallel_segments"))]
            {
                mem.put_bytes_with_options(&payload, options)
                    .map_err(|err| {
                        map_put_error(err, capacity_guard.as_ref().and_then(|g| g.capacity_hint()))
                    })?
            }
        } else if allow_embedding {
            if let Some(path) = args.embedding_vec.as_deref() {
                let embedding = crate::utils::read_embedding(path)?;
                if let Some(model_str) = args.embedding_vec_model.as_deref() {
                    let choice = model_str.parse::<EmbeddingModelChoice>()?;
                    let expected = choice.dimensions();
                    if expected != 0 && embedding.len() != expected {
                        anyhow::bail!(
                            "pre-computed embedding has {} dimensions but --embedding-vec-model {} expects {}",
                            embedding.len(),
                            choice.name(),
                            expected
                        );
                    }
                    apply_embedding_identity_metadata_from_choice(
                        &mut options,
                        choice,
                        embedding.len(),
                        Some(model_str),
                    );
                } else {
                    options.extra_metadata.insert(
                        MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
                        embedding.len().to_string(),
                    );
                }
                embedded = true;
                mem.put_with_embedding_and_options(&payload, embedding, options)
                    .map_err(|err| {
                        map_put_error(err, capacity_guard.as_ref().and_then(|g| g.capacity_hint()))
                    })?
            } else {
                let runtime = runtime.ok_or_else(|| anyhow!("semantic runtime unavailable"))?;
                let embed_text = search_text
                    .clone()
                    .or_else(|| payload_text.map(|text| text.to_string()))
                    .unwrap_or_default();
                if embed_text.trim().is_empty() {
                    warn!("no textual content available; embedding skipped");
                    mem.put_bytes_with_options(&payload, options)
                        .map_err(|err| {
                            map_put_error(
                                err,
                                capacity_guard.as_ref().and_then(|g| g.capacity_hint()),
                            )
                        })?
                } else {
                    // Check if document will be chunked - if so, embed each chunk
                    if let Some(chunk_texts) = mem.preview_chunks(&payload) {
                        // Document will be chunked - embed each chunk for full semantic coverage
                        info!(
                        "Document will be chunked into {} chunks, generating embeddings for each",
                        chunk_texts.len()
                    );

                        // Apply temporal enrichment if enabled (before contextual)
                        #[cfg(feature = "temporal_enrich")]
                        let enriched_chunks = if args.temporal_enrich {
                            info!(
                                "Temporal enrichment enabled, processing {} chunks",
                                chunk_texts.len()
                            );
                            // Use today's date as fallback document date
                            let today = chrono::Local::now().date_naive();
                            let results = temporal_enrich_chunks(&chunk_texts, Some(today));
                            // Results are tuples of (enriched_text, TemporalEnrichment)
                            let enriched: Vec<String> =
                                results.iter().map(|(text, _)| text.clone()).collect();
                            let resolved_count = results
                                .iter()
                                .filter(|(_, e)| {
                                    e.relative_phrases.iter().any(|p| p.resolved.is_some())
                                })
                                .count();
                            info!(
                                "Temporal enrichment: resolved {} chunks with temporal context",
                                resolved_count
                            );
                            enriched
                        } else {
                            chunk_texts.clone()
                        };
                        #[cfg(not(feature = "temporal_enrich"))]
                        let enriched_chunks = {
                            if args.temporal_enrich {
                                warn!("Temporal enrichment requested but feature not compiled in");
                            }
                            chunk_texts.clone()
                        };

                        // Apply contextual retrieval if enabled
                        let embed_chunks = if args.contextual {
                            info!("Contextual retrieval enabled, generating context prefixes for {} chunks", enriched_chunks.len());
                            let engine = if args.contextual_model.as_deref() == Some("local") {
                                #[cfg(feature = "llama-cpp")]
                                {
                                    let model_path = get_local_contextual_model_path()?;
                                    ContextualEngine::local(model_path)
                                }
                                #[cfg(not(feature = "llama-cpp"))]
                                {
                                    anyhow::bail!("Local contextual model requires the 'llama-cpp' feature. Use --contextual-model openai or omit the flag for OpenAI.");
                                }
                            } else if let Some(model) = &args.contextual_model {
                                ContextualEngine::openai_with_model(model)?
                            } else {
                                ContextualEngine::openai()?
                            };

                            match engine.generate_contexts_batch(&embed_text, &enriched_chunks) {
                                Ok(contexts) => {
                                    info!("Generated {} contextual prefixes", contexts.len());
                                    apply_contextual_prefixes(
                                        &embed_text,
                                        &enriched_chunks,
                                        &contexts,
                                    )
                                }
                                Err(e) => {
                                    warn!("Failed to generate contextual prefixes: {}. Using original chunks.", e);
                                    enriched_chunks.clone()
                                }
                            }
                        } else {
                            enriched_chunks
                        };

                        let truncated_chunks: Vec<String> = embed_chunks
                            .iter()
                            .map(|chunk_text| {
                                // Truncate chunk to avoid provider token limit issues (contextual prefixes can make chunks large)
                                let truncated_chunk = truncate_for_embedding(chunk_text);
                                if truncated_chunk.len() < chunk_text.len() {
                                    info!(
                                        "Truncated chunk from {} to {} chars for embedding",
                                        chunk_text.len(),
                                        truncated_chunk.len()
                                    );
                                }
                                truncated_chunk
                            })
                            .collect();
                        let truncated_refs: Vec<&str> = truncated_chunks
                            .iter()
                            .map(|chunk| chunk.as_str())
                            .collect();
                        let chunk_embeddings = runtime.embed_batch_passages(&truncated_refs)?;
                        // Truncate parent text to avoid token limits
                        let parent_text = truncate_for_embedding(&embed_text);
                        if parent_text.len() < embed_text.len() {
                            info!(
                                "Truncated parent text from {} to {} chars for embedding",
                                embed_text.len(),
                                parent_text.len()
                            );
                        }
                        let parent_embedding = runtime.embed_passage(&parent_text)?;
                        embedded = true;
                        apply_embedding_identity_metadata(&mut options, runtime);
                        info!(
                            "Calling put_with_chunk_embeddings with {} chunk embeddings",
                            chunk_embeddings.len()
                        );
                        mem.put_with_chunk_embeddings(
                            &payload,
                            Some(parent_embedding),
                            chunk_embeddings,
                            options,
                        )
                        .map_err(|err| {
                            map_put_error(
                                err,
                                capacity_guard.as_ref().and_then(|g| g.capacity_hint()),
                            )
                        })?
                    } else {
                        // No chunking - just embed the whole document (truncate to avoid token limits)
                        info!("Document too small for chunking (< 2400 chars after normalization), using single embedding");
                        let truncated_text = truncate_for_embedding(&embed_text);
                        if truncated_text.len() < embed_text.len() {
                            info!(
                                "Truncated text from {} to {} chars for embedding",
                                embed_text.len(),
                                truncated_text.len()
                            );
                        }
                        let embedding = runtime.embed_passage(&truncated_text)?;
                        embedded = true;
                        apply_embedding_identity_metadata(&mut options, runtime);
                        mem.put_with_embedding_and_options(&payload, embedding, options)
                            .map_err(|err| {
                                map_put_error(
                                    err,
                                    capacity_guard.as_ref().and_then(|g| g.capacity_hint()),
                                )
                            })?
                    }
                }
            }
        } else {
            mem.put_bytes_with_options(&payload, options)
                .map_err(|err| {
                    map_put_error(err, capacity_guard.as_ref().and_then(|g| g.capacity_hint()))
                })?
        }
    };

    Ok(IngestOutcome {
        report: PutReport {
            seq,
            frame_id,
            uri,
            title: inferred_title,
            original_bytes,
            stored_bytes,
            compressed,
            no_raw: effective_no_raw,
            source: source_path.map(Path::to_path_buf),
            mime: Some(analysis.mime),
            metadata: analysis.metadata,
            extraction: analysis.extraction,
            #[cfg(feature = "logic_mesh")]
            search_text,
        },
        embedded,
        #[cfg(feature = "parallel_segments")]
        parallel_input: None,
    })
}

fn canonical_storage_len(payload: &[u8]) -> Result<(usize, bool)> {
    if std::str::from_utf8(payload).is_ok() {
        let compressed = zstd::encode_all(Cursor::new(payload), 3)?;
        Ok((compressed.len(), true))
    } else {
        Ok((payload.len(), false))
    }
}

fn print_report(report: &PutReport) {
    let name = report
        .source
        .as_ref()
        .map(|path| {
            let pretty = env::current_dir()
                .ok()
                .as_ref()
                .and_then(|cwd| diff_paths(path, cwd))
                .unwrap_or_else(|| path.to_path_buf());
            pretty.to_string_lossy().into_owned()
        })
        .unwrap_or_else(|| "stdin".to_string());

    let ratio = if report.original_bytes > 0 {
        (report.stored_bytes as f64 / report.original_bytes as f64) * 100.0
    } else {
        100.0
    };

    println!("• {name} → {}", report.uri);
    println!("  seq: {}", report.seq);
    if report.no_raw {
        println!(
            "  size: {} B (--no-raw: text only, hash stored)",
            report.original_bytes
        );
    } else if report.compressed {
        println!(
            "  size: {} B → {} B ({:.1}% of original, compressed)",
            report.original_bytes, report.stored_bytes, ratio
        );
    } else {
        println!("  size: {} B (stored as-is)", report.original_bytes);
    }
    if let Some(mime) = &report.mime {
        println!("  mime: {mime}");
    }
    if let Some(title) = &report.title {
        println!("  title: {title}");
    }
    let extraction_reader = report.extraction.reader.as_deref().unwrap_or("n/a");
    println!(
        "  extraction: status={} reader={}",
        report.extraction.status.label(),
        extraction_reader
    );
    if let Some(ms) = report.extraction.duration_ms {
        println!("  extraction-duration: {} ms", ms);
    }
    if let Some(pages) = report.extraction.pages_processed {
        println!("  extraction-pages: {}", pages);
    }
    for warning in &report.extraction.warnings {
        println!("  warning: {warning}");
    }
    if let Some(metadata) = &report.metadata {
        if let Some(caption) = &metadata.caption {
            println!("  caption: {caption}");
        }
        if let Some(audio) = metadata.audio.as_ref() {
            if audio.duration_secs.is_some()
                || audio.sample_rate_hz.is_some()
                || audio.channels.is_some()
            {
                let duration = audio
                    .duration_secs
                    .map(|secs| format!("{secs:.1}s"))
                    .unwrap_or_else(|| "unknown".into());
                let rate = audio
                    .sample_rate_hz
                    .map(|hz| format!("{} Hz", hz))
                    .unwrap_or_else(|| "? Hz".into());
                let channels = audio
                    .channels
                    .map(|ch| ch.to_string())
                    .unwrap_or_else(|| "?".into());
                println!("  audio: duration {duration}, {channels} ch, {rate}");
            }
        }
    }

    println!();
}

fn put_report_to_json(report: &PutReport) -> serde_json::Value {
    let source = report
        .source
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned());
    let extraction = json!({
        "status": report.extraction.status.label(),
        "reader": report.extraction.reader,
        "duration_ms": report.extraction.duration_ms,
        "pages_processed": report.extraction.pages_processed,
        "warnings": report.extraction.warnings,
    });
    json!({
        "seq": report.seq,
        "uri": report.uri,
        "title": report.title,
        "original_bytes": report.original_bytes,
        "original_bytes_human": format_bytes(report.original_bytes as u64),
        "stored_bytes": report.stored_bytes,
        "stored_bytes_human": format_bytes(report.stored_bytes as u64),
        "compressed": report.compressed,
        "mime": report.mime,
        "source": source,
        "metadata": report.metadata,
        "extraction": extraction,
    })
}

fn map_put_error(err: MemvidError, _capacity_hint: Option<u64>) -> anyhow::Error {
    match err {
        MemvidError::CapacityExceeded {
            current,
            limit,
            required,
        } => anyhow!(CapacityExceededMessage {
            current,
            limit,
            required,
        }),
        other => other.into(),
    }
}

fn apply_metadata_overrides(options: &mut PutOptions, entries: &[String]) {
    for (idx, entry) in entries.iter().enumerate() {
        match serde_json::from_str::<DocMetadata>(entry) {
            Ok(meta) => {
                options.metadata = Some(meta);
            }
            Err(_) => match serde_json::from_str::<serde_json::Value>(entry) {
                Ok(value) => {
                    options
                        .extra_metadata
                        .insert(format!("custom_metadata_{idx}"), value.to_string());
                }
                Err(err) => warn!("failed to parse --metadata JSON: {err}"),
            },
        }
    }
}

fn transcript_notice_message(mem: &mut Memvid) -> Result<String> {
    let stats = mem.stats()?;
    let message = match stats.tier {
        Tier::Free => "Transcript requires Dev/Enterprise tier. Apply a ticket to enable.".to_string(),
        Tier::Dev | Tier::Enterprise => "Transcript capture will attach in a future update; no transcript generated in this build.".to_string(),
    };
    Ok(message)
}

fn sanitize_uri(raw: &str, keep_path: bool) -> String {
    let trimmed = raw.trim().trim_start_matches("mv2://");
    let normalized = trimmed.replace('\\', "/");
    let mut segments: Vec<String> = Vec::new();
    for segment in normalized.split('/') {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".." {
            segments.pop();
            continue;
        }
        segments.push(segment.to_string());
    }

    if segments.is_empty() {
        return normalized
            .split('/')
            .last()
            .filter(|s| !s.is_empty())
            .unwrap_or("document")
            .to_string();
    }

    if keep_path {
        segments.join("/")
    } else {
        segments
            .last()
            .cloned()
            .unwrap_or_else(|| "document".to_string())
    }
}

fn create_task_progress_bar(total: Option<u64>, message: &str, quiet: bool) -> ProgressBar {
    if quiet {
        let pb = ProgressBar::hidden();
        pb.set_message(message.to_string());
        return pb;
    }

    match total {
        Some(len) => {
            let pb = ProgressBar::new(len);
            pb.set_draw_target(ProgressDrawTarget::stderr());
            let style = ProgressStyle::with_template("{msg:>9} {pos}/{len}")
                .unwrap_or_else(|_| ProgressStyle::default_bar());
            pb.set_style(style);
            pb.set_message(message.to_string());
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_draw_target(ProgressDrawTarget::stderr());
            pb.set_message(message.to_string());
            pb.enable_steady_tick(Duration::from_millis(120));
            pb
        }
    }
}

fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_draw_target(ProgressDrawTarget::stderr());
    let style = ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
        .tick_strings(&["-", "\\", "|", "/"]);
    pb.set_style(style);
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(120));
    pb
}

// ============================================================================
// put-many command - batch ingestion with pre-computed embeddings
// ============================================================================

/// Arguments for the `put-many` subcommand
#[cfg(feature = "parallel_segments")]
#[derive(Args)]
pub struct PutManyArgs {
    /// Path to the memory file to append to
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    /// Emit JSON instead of human-readable output
    #[arg(long)]
    pub json: bool,
    /// Path to JSON file containing batch requests (array of objects with title, label, text, uri, tags, labels, metadata, embedding)
    /// If not specified, reads from stdin
    #[arg(long, value_name = "PATH")]
    pub input: Option<PathBuf>,
    /// Zstd compression level (1-9, default: 3)
    #[arg(long, default_value = "3")]
    pub compression_level: i32,
    /// Store raw binary content along with extracted text.
    /// By default, only extracted text + BLAKE3 hash is stored (--no-raw behavior).
    #[arg(long)]
    pub raw: bool,
    /// Don't store raw binary content (DEFAULT behavior).
    /// Kept for backwards compatibility.
    #[arg(long, hide = true)]
    pub no_raw: bool,
    /// Skip ingestion if identical content (by BLAKE3 hash) already exists.
    /// Returns the existing frame's ID instead of creating a duplicate.
    #[arg(long)]
    pub dedup: bool,
    #[command(flatten)]
    pub lock: LockCliArgs,
}

/// JSON input format for put-many requests
#[cfg(feature = "parallel_segments")]
#[derive(Debug, serde::Deserialize)]
pub struct PutManyRequest {
    pub title: String,
    #[serde(default)]
    pub label: String,
    pub text: String,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Extra metadata (string key/value) stored on the frame.
    /// Use this to persist embedding identity keys like `memvid.embedding.model`.
    #[serde(default)]
    pub extra_metadata: BTreeMap<String, String>,
    /// Pre-computed embedding vector for the parent document (e.g., from OpenAI)
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    /// Pre-computed embeddings for each chunk (matched by index)
    /// If the document gets chunked, these embeddings are assigned to each chunk frame.
    /// The number of chunk embeddings should match the number of chunks that will be created.
    #[serde(default)]
    pub chunk_embeddings: Option<Vec<Vec<f32>>>,
    /// Per-item override for no_raw mode. If not specified, uses the global --no-raw flag.
    /// When true, stores only extracted text + BLAKE3 hash instead of raw content.
    #[serde(default)]
    pub no_raw: Option<bool>,
    /// Per-item override for dedup mode. If not specified, uses the global --dedup flag.
    /// When true, skips ingestion if identical content already exists.
    #[serde(default)]
    pub dedup: Option<bool>,
}

/// Batch input format (array of requests)
#[cfg(feature = "parallel_segments")]
#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
pub struct PutManyBatch {
    pub requests: Vec<PutManyRequest>,
}

#[cfg(feature = "parallel_segments")]
pub fn handle_put_many(config: &crate::config::CliConfig, args: PutManyArgs) -> Result<()> {
    use memvid_core::{BuildOpts, Memvid, ParallelInput, ParallelPayload, PutOptions};
    use std::io::Read;

    // Check if plan allows write operations (blocks expired subscriptions)
    crate::utils::require_active_plan(config, "put-many")?;

    let mut mem = Memvid::open(&args.file)?;
    crate::utils::ensure_cli_mutation_allowed(&mem)?;
    crate::utils::apply_lock_cli(&mut mem, &args.lock);

    // Ensure indexes are enabled
    let stats = mem.stats()?;
    if !stats.has_lex_index {
        mem.enable_lex()?;
    }
    if !stats.has_vec_index {
        mem.enable_vec()?;
    }

    // Check if current memory size exceeds free tier limit
    crate::utils::ensure_capacity_with_api_key(stats.size_bytes, 0, config)?;

    // Read batch input
    let batch_json: String = if let Some(input_path) = &args.input {
        fs::read_to_string(input_path)
            .with_context(|| format!("Failed to read input file: {}", input_path.display()))?
    } else {
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .context("Failed to read from stdin")?;
        buffer
    };

    let batch: PutManyBatch =
        serde_json::from_str(&batch_json).context("Failed to parse JSON input")?;

    if batch.requests.is_empty() {
        if args.json {
            println!("{{\"frame_ids\": [], \"count\": 0}}");
        } else {
            println!("No requests to process");
        }
        return Ok(());
    }

    let count = batch.requests.len();
    if !args.json {
        eprintln!("Processing {} documents...", count);
    }

    // Convert to ParallelInput
    // Default: no-raw mode (only store extracted text + BLAKE3 hash)
    // Use --raw to store raw binary content for later retrieval
    let global_no_raw = !args.raw; // true by default unless --raw is passed
    let global_dedup = args.dedup;
    let inputs: Vec<ParallelInput> = batch
        .requests
        .into_iter()
        .enumerate()
        .map(|(i, req)| {
            let uri = req.uri.unwrap_or_else(|| format!("mv2://batch/{}", i));
            let mut options = PutOptions::default();
            options.title = Some(req.title.clone());
            options.uri = Some(uri);
            options.tags = req.tags;
            options.labels = req.labels;
            options.extra_metadata = req.extra_metadata;
            // Use per-item no_raw if specified, otherwise use global --no-raw flag
            options.no_raw = req.no_raw.unwrap_or(global_no_raw);
            // Use per-item dedup if specified, otherwise use global --dedup flag
            options.dedup = req.dedup.unwrap_or(global_dedup);
            if !req.metadata.is_null() {
                match serde_json::from_value::<DocMetadata>(req.metadata.clone()) {
                    Ok(meta) => options.metadata = Some(meta),
                    Err(_) => {
                        options
                            .extra_metadata
                            .entry("memvid.metadata.json".to_string())
                            .or_insert_with(|| req.metadata.to_string());
                    }
                }
            }

            if let Some(embedding) = req
                .embedding
                .as_ref()
                .or_else(|| req.chunk_embeddings.as_ref().and_then(|emb| emb.first()))
            {
                options
                    .extra_metadata
                    .entry(MEMVID_EMBEDDING_DIMENSION_KEY.to_string())
                    .or_insert_with(|| embedding.len().to_string());
            }

            ParallelInput {
                payload: ParallelPayload::Bytes(req.text.into_bytes()),
                options,
                embedding: req.embedding,
                chunk_embeddings: req.chunk_embeddings,
            }
        })
        .collect();

    // Build options
    let build_opts = BuildOpts {
        zstd_level: args.compression_level.clamp(1, 9),
        ..BuildOpts::default()
    };

    // Ingest batch
    let frame_ids = mem
        .put_parallel_inputs(&inputs, build_opts)
        .context("Failed to ingest batch")?;

    if args.json {
        let output = serde_json::json!({
            "frame_ids": frame_ids,
            "count": frame_ids.len()
        });
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("Ingested {} documents", frame_ids.len());
        if frame_ids.len() <= 10 {
            for fid in &frame_ids {
                println!("  frame_id: {}", fid);
            }
        } else {
            println!("  first: {}", frame_ids.first().unwrap_or(&0));
            println!("  last:  {}", frame_ids.last().unwrap_or(&0));
        }
    }

    Ok(())
}

/// Arguments for the `correct` subcommand
#[derive(Args)]
pub struct CorrectArgs {
    /// Path to the memory file
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    /// The correction statement (e.g., "Ben Koenig reported to Chloe Nguyen before 2025")
    #[arg(value_name = "STATEMENT")]
    pub statement: String,
    /// Topics for better retrieval matching (comma-separated)
    #[arg(long, value_name = "TOPICS", value_delimiter = ',')]
    pub topics: Option<Vec<String>>,
    /// Source attribution for the correction
    #[arg(long, value_name = "SOURCE")]
    pub source: Option<String>,
    /// Retrieval boost factor (default: 2.0)
    #[arg(long, default_value = "2.0")]
    pub boost: f64,
    /// Emit JSON instead of human-readable output
    #[arg(long)]
    pub json: bool,
    #[command(flatten)]
    pub lock: LockCliArgs,
}

/// Store a correction with retrieval priority boost.
///
/// Corrections are stored as frames with special metadata that gives them
/// priority during retrieval. This ensures corrected information surfaces
/// first when relevant queries are made.
pub fn handle_correct(config: &CliConfig, args: CorrectArgs) -> Result<()> {
    crate::utils::require_active_plan(config, "correct")?;

    let mut mem = Memvid::open(&args.file)?;
    ensure_cli_mutation_allowed(&mem)?;
    apply_lock_cli(&mut mem, &args.lock);

    let stats = mem.stats()?;
    if stats.frame_count == 0 && !stats.has_lex_index {
        mem.enable_lex()?;
    }

    // Build extra_metadata for correction
    let mut extra_metadata = BTreeMap::new();
    extra_metadata.insert("memvid.correction".to_string(), "true".to_string());
    extra_metadata.insert(
        "memvid.correction.boost".to_string(),
        args.boost.to_string(),
    );
    if let Some(source) = &args.source {
        extra_metadata.insert("memvid.correction.source".to_string(), source.clone());
    }

    // Convert topics to tags for better retrieval matching
    let tags: Vec<String> = args.topics.clone().unwrap_or_default();

    // Build put options
    let mut options = PutOptions::default();
    options.title = Some(format!(
        "Correction: {}",
        truncate_title(&args.statement, 50)
    ));
    options.uri = Some(format!("mv2://correction/{}", Uuid::new_v4()));
    options.labels = vec!["correction".to_string()];
    options.tags = tags;
    options.extra_metadata = extra_metadata;
    options.no_raw = true; // Corrections are text-only

    // Store the correction
    let frame_id = mem.put_bytes_with_options(args.statement.as_bytes(), options)?;

    if args.json {
        let output = json!({
            "frame_id": frame_id,
            "type": "correction",
            "boost": args.boost,
            "topics": args.topics.unwrap_or_default(),
            "source": args.source,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("✓ Correction stored (frame_id: {})", frame_id);
        if let Some(topics) = &args.topics {
            println!("  Topics: {}", topics.join(", "));
        }
        if let Some(source) = &args.source {
            println!("  Source: {}", source);
        }
        println!("  Boost:  {}x", args.boost);
    }

    Ok(())
}

/// Truncate a string to a maximum length, adding "..." if truncated
fn truncate_title(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
