//! Enrichment command handler for extracting memory cards from frames.
//!
//! The `enrich` command runs enrichment engines over MV2 frames to extract
//! structured memory cards (facts, preferences, events, etc.).

use std::path::PathBuf;

#[cfg(feature = "llama-cpp")]
use anyhow::bail;
use anyhow::Result;
use clap::{Args, ValueEnum};
use memvid_core::{EnrichmentEngine, Memvid, RulesEngine};
use serde::Serialize;

#[cfg(feature = "llama-cpp")]
use crate::commands::{default_enrichment_model, get_installed_model_path, LlmModel};
use crate::config::CliConfig;
#[cfg(feature = "candle-llm")]
use crate::enrich::CandlePhiEngine;
#[cfg(feature = "llama-cpp")]
use crate::enrich::LlmEngine;
use crate::enrich::{
    ClaudeEngine, GeminiEngine, GroqEngine, MistralEngine, OpenAiEngine, XaiEngine,
};

/// Engine type for enrichment
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum EnrichEngine {
    /// Rules-based extraction using regex patterns (fast, no models)
    #[default]
    Rules,
    /// LLM-based extraction with Phi-3.5 Mini via llama.cpp (requires model installation)
    #[cfg(feature = "llama-cpp")]
    Llm,
    /// Candle-based Phi-3 extraction (downloads from Hugging Face)
    #[cfg(feature = "candle-llm")]
    Candle,
    /// OpenAI API-based extraction with GPT-4o-mini (requires OPENAI_API_KEY)
    Openai,
    /// Claude (Anthropic) API-based extraction with Claude 3.5 Haiku (requires ANTHROPIC_API_KEY)
    Claude,
    /// Gemini (Google) API-based extraction with Gemini 2.0 Flash (requires GOOGLE_API_KEY or GEMINI_API_KEY)
    Gemini,
    /// xAI API-based extraction with Grok-2 (requires XAI_API_KEY)
    Xai,
    /// Groq API-based extraction with Llama 3.3 70B (requires GROQ_API_KEY)
    Groq,
    /// Mistral API-based extraction with Mistral Large (requires MISTRAL_API_KEY)
    Mistral,
}

/// Arguments for the `enrich` subcommand
#[derive(Args)]
pub struct EnrichArgs {
    /// Path to the `.mv2` file
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Enrichment engine to use
    #[arg(long, value_enum, default_value_t = EnrichEngine::Rules)]
    pub engine: EnrichEngine,

    /// Only process frames that haven't been enriched yet (default)
    #[arg(long, default_value_t = true)]
    pub incremental: bool,

    /// Re-enrich all frames, ignoring previous enrichment records
    #[arg(long, conflicts_with = "incremental")]
    pub force: bool,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,

    /// Show extracted memory cards
    #[arg(long)]
    pub verbose: bool,

    /// Number of parallel workers for API calls (default: 20)
    #[arg(long, default_value_t = 20)]
    pub workers: usize,

    /// Number of frames to batch per API call (default: 10)
    #[arg(long, default_value_t = 10)]
    pub batch_size: usize,
}

/// Result of enrichment for JSON output
#[derive(Debug, Serialize)]
pub struct EnrichResult {
    pub engine: String,
    pub version: String,
    pub frames_processed: usize,
    pub cards_extracted: usize,
    pub total_cards: usize,
    pub total_entities: usize,
}

/// Handle the `enrich` command
#[allow(unused_variables)]
pub fn handle_enrich(config: &CliConfig, args: EnrichArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    // Get initial stats
    let initial_stats = mem.memories_stats();

    // If force mode, clear existing memories first
    if args.force {
        mem.clear_memories();
    }

    // Run the selected engine
    let (engine_kind, engine_version, frames, cards) = match args.engine {
        EnrichEngine::Rules => {
            let engine = RulesEngine::new();
            let kind = engine.kind().to_string();
            let version = engine.version().to_string();
            let (frames, cards) = mem.run_enrichment(&engine)?;
            (kind, version, frames, cards)
        }
        #[cfg(feature = "llama-cpp")]
        EnrichEngine::Llm => {
            // Check if model is installed
            let model = default_enrichment_model();
            let model_path = match get_installed_model_path(config, model) {
                Some(path) => path,
                None => {
                    bail!(
                        "LLM model not installed. Run `memvid models install {}` first.",
                        match model {
                            LlmModel::Phi35Mini => "phi-3.5-mini",
                            LlmModel::Phi35MiniQ8 => "phi-3.5-mini-q8",
                        }
                    );
                }
            };

            // Create and initialize the LLM engine
            let mut engine = LlmEngine::new(model_path);
            eprintln!("Loading LLM model...");
            engine.init()?;

            let kind = engine.kind().to_string();
            let version = engine.version().to_string();
            let (frames, cards) = mem.run_enrichment(&engine)?;
            (kind, version, frames, cards)
        }
        #[cfg(feature = "candle-llm")]
        EnrichEngine::Candle => {
            // Create and initialize the Candle Phi-3 engine
            // Uses Q4 quantized GGUF (~2.4GB) stored in ~/.memvid/models/llm/phi-3-mini-q4/
            eprintln!("Loading Phi-3-mini Q4 model via Candle (first run downloads ~2.4GB to ~/.memvid/models/llm/)...");
            let mut engine = CandlePhiEngine::from_memvid_models(config.models_dir.clone());
            engine.init()?;

            let kind = engine.kind().to_string();
            let version = engine.version().to_string();
            let (frames, cards) = mem.run_enrichment(&engine)?;
            (kind, version, frames, cards)
        }
        EnrichEngine::Openai => {
            // Create and initialize the OpenAI engine with parallel batch support
            eprintln!("Using OpenAI GPT-4o-mini for enrichment (parallel mode, {} workers, batch size {})...", args.workers, args.batch_size);
            let mut engine = OpenAiEngine::new()
                .with_parallelism(args.workers)
                .with_batch_size(args.batch_size);
            engine.init()?;

            let kind = engine.kind().to_string();
            let version = engine.version().to_string();

            // Use parallel batch processing for OpenAI
            let (frames, cards) = run_openai_parallel(&mut mem, &engine, args.workers)?;
            (kind, version, frames, cards)
        }
        EnrichEngine::Claude => {
            // Create and initialize the Claude engine with parallel support
            eprintln!(
                "Using Claude Haiku 4.5 for enrichment (parallel mode, {} workers)...",
                args.workers
            );
            let mut engine = ClaudeEngine::new().with_parallelism(args.workers);
            engine.init()?;

            let kind = engine.kind().to_string();
            let version = engine.version().to_string();

            // Use parallel batch processing for Claude
            let (frames, cards) = run_claude_parallel(&mut mem, &engine, args.workers)?;
            (kind, version, frames, cards)
        }
        EnrichEngine::Gemini => {
            // Create and initialize the Gemini engine with parallel support
            eprintln!(
                "Using Gemini 2.5 Flash for enrichment (parallel mode, {} workers)...",
                args.workers
            );
            let mut engine = GeminiEngine::new().with_parallelism(args.workers);
            engine.init()?;

            let kind = engine.kind().to_string();
            let version = engine.version().to_string();

            // Use parallel batch processing for Gemini
            let (frames, cards) = run_gemini_parallel(&mut mem, &engine, args.workers)?;
            (kind, version, frames, cards)
        }
        EnrichEngine::Xai => {
            // Create and initialize the xAI engine with parallel support
            eprintln!(
                "Using xAI Grok 4 Fast for enrichment (parallel mode, {} workers)...",
                args.workers
            );
            let mut engine = XaiEngine::new().with_parallelism(args.workers);
            engine.init()?;

            let kind = engine.kind().to_string();
            let version = engine.version().to_string();

            // Use parallel batch processing for xAI
            let (frames, cards) = run_xai_parallel(&mut mem, &engine, args.workers)?;
            (kind, version, frames, cards)
        }
        EnrichEngine::Groq => {
            // Create and initialize the Groq engine with parallel support
            eprintln!(
                "Using Groq Llama 3.3 70B for enrichment (parallel mode, {} workers)...",
                args.workers
            );
            let mut engine = GroqEngine::new().with_parallelism(args.workers);
            engine.init()?;

            let kind = engine.kind().to_string();
            let version = engine.version().to_string();

            // Use parallel batch processing for Groq
            let (frames, cards) = run_groq_parallel(&mut mem, &engine, args.workers)?;
            (kind, version, frames, cards)
        }
        EnrichEngine::Mistral => {
            // Create and initialize the Mistral engine with parallel support
            eprintln!(
                "Using Mistral Large for enrichment (parallel mode, {} workers)...",
                args.workers
            );
            let mut engine = MistralEngine::new().with_parallelism(args.workers);
            engine.init()?;

            let kind = engine.kind().to_string();
            let version = engine.version().to_string();

            // Use parallel batch processing for Mistral
            let (frames, cards) = run_mistral_parallel(&mut mem, &engine, args.workers)?;
            (kind, version, frames, cards)
        }
    };

    // Commit changes
    mem.commit()?;

    // Get final stats
    let final_stats = mem.memories_stats();

    if args.json {
        let result = EnrichResult {
            engine: engine_kind,
            version: engine_version,
            frames_processed: frames,
            cards_extracted: cards,
            total_cards: final_stats.card_count,
            total_entities: final_stats.entity_count,
        };
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Enrichment complete:");
        println!("  Engine: {} v{}", engine_kind, engine_version);
        println!("  Frames processed: {}", frames);
        println!("  Cards extracted: {}", cards);
        println!(
            "  Total cards: {} (+{})",
            final_stats.card_count,
            final_stats
                .card_count
                .saturating_sub(initial_stats.card_count)
        );
        println!("  Entities: {}", final_stats.entity_count);

        if args.verbose && cards > 0 {
            println!("\nExtracted memory cards:");
            for entity in mem.memory_entities() {
                println!("  {}:", entity);
                for card in mem.get_entity_memories(&entity) {
                    println!("    - {}: {} = \"{}\"", card.kind, card.slot, card.value);
                }
            }
        }
    }

    Ok(())
}

/// Run OpenAI enrichment with parallel batch processing.
///
/// This gathers all unenriched frames, sends them to OpenAI in parallel,
/// and stores the resulting memory cards.
fn run_openai_parallel(
    mem: &mut memvid_core::Memvid,
    engine: &OpenAiEngine,
    workers: usize,
) -> Result<(usize, usize)> {
    use memvid_core::enrich::EnrichmentContext;
    use memvid_core::EnrichmentEngine;

    let kind = engine.kind();
    let version = engine.version();

    // Get all unenriched frames
    let unenriched = mem.get_unenriched_frames(kind, version);
    let total_frames = unenriched.len();

    if total_frames == 0 {
        eprintln!("No unenriched frames found.");
        return Ok((0, 0));
    }

    eprintln!(
        "Gathering {} frames for parallel enrichment...",
        total_frames
    );

    // Build enrichment contexts for all frames
    let mut contexts = Vec::with_capacity(total_frames);
    for frame_id in &unenriched {
        let frame = match mem.frame_by_id(*frame_id) {
            Ok(f) => f,
            Err(_) => continue,
        };

        // Get full frame content (not truncated preview)
        let text = match mem.frame_text_by_id(*frame_id) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let uri = frame
            .uri
            .clone()
            .unwrap_or_else(|| format!("mv2://frame/{}", frame_id));
        let metadata_json = frame
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        let ctx = EnrichmentContext::new(
            *frame_id,
            uri,
            text,
            frame.title.clone(),
            frame.timestamp,
            metadata_json,
        );

        contexts.push(ctx);
    }

    eprintln!(
        "Starting parallel enrichment of {} frames with {} workers...",
        contexts.len(),
        workers
    );

    // Run parallel batch enrichment
    let results = engine.enrich_batch(contexts)?;

    // Store results back to MV2
    let mut total_cards = 0;
    for (frame_id, cards) in results {
        let card_count = cards.len();

        // Store cards
        let card_ids = if !cards.is_empty() {
            mem.put_memory_cards(cards)?
        } else {
            Vec::new()
        };

        // Record enrichment
        mem.record_enrichment(frame_id, kind, version, card_ids)?;

        total_cards += card_count;
    }

    Ok((total_frames, total_cards))
}

/// Run Claude enrichment with parallel batch processing.
fn run_claude_parallel(
    mem: &mut memvid_core::Memvid,
    engine: &ClaudeEngine,
    workers: usize,
) -> Result<(usize, usize)> {
    use memvid_core::enrich::EnrichmentContext;
    use memvid_core::EnrichmentEngine;

    let kind = engine.kind();
    let version = engine.version();

    let unenriched = mem.get_unenriched_frames(kind, version);
    let total_frames = unenriched.len();

    if total_frames == 0 {
        eprintln!("No unenriched frames found.");
        return Ok((0, 0));
    }

    eprintln!(
        "Gathering {} frames for parallel enrichment...",
        total_frames
    );

    let mut contexts = Vec::with_capacity(total_frames);
    for frame_id in &unenriched {
        let frame = match mem.frame_by_id(*frame_id) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let text = match mem.frame_text_by_id(*frame_id) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let uri = frame
            .uri
            .clone()
            .unwrap_or_else(|| format!("mv2://frame/{}", frame_id));
        let metadata_json = frame
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        let ctx = EnrichmentContext::new(
            *frame_id,
            uri,
            text,
            frame.title.clone(),
            frame.timestamp,
            metadata_json,
        );

        contexts.push(ctx);
    }

    eprintln!(
        "Starting parallel enrichment of {} frames with {} workers...",
        contexts.len(),
        workers
    );

    let results = engine.enrich_batch(contexts)?;

    let mut total_cards = 0;
    for (frame_id, cards) in results {
        let card_count = cards.len();

        let card_ids = if !cards.is_empty() {
            mem.put_memory_cards(cards)?
        } else {
            Vec::new()
        };

        mem.record_enrichment(frame_id, kind, version, card_ids)?;

        total_cards += card_count;
    }

    Ok((total_frames, total_cards))
}

/// Run Gemini enrichment with parallel batch processing.
fn run_gemini_parallel(
    mem: &mut memvid_core::Memvid,
    engine: &GeminiEngine,
    workers: usize,
) -> Result<(usize, usize)> {
    use memvid_core::enrich::EnrichmentContext;
    use memvid_core::EnrichmentEngine;

    let kind = engine.kind();
    let version = engine.version();

    let unenriched = mem.get_unenriched_frames(kind, version);
    let total_frames = unenriched.len();

    if total_frames == 0 {
        eprintln!("No unenriched frames found.");
        return Ok((0, 0));
    }

    eprintln!(
        "Gathering {} frames for parallel enrichment...",
        total_frames
    );

    let mut contexts = Vec::with_capacity(total_frames);
    for frame_id in &unenriched {
        let frame = match mem.frame_by_id(*frame_id) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let text = match mem.frame_text_by_id(*frame_id) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let uri = frame
            .uri
            .clone()
            .unwrap_or_else(|| format!("mv2://frame/{}", frame_id));
        let metadata_json = frame
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        let ctx = EnrichmentContext::new(
            *frame_id,
            uri,
            text,
            frame.title.clone(),
            frame.timestamp,
            metadata_json,
        );

        contexts.push(ctx);
    }

    eprintln!(
        "Starting parallel enrichment of {} frames with {} workers...",
        contexts.len(),
        workers
    );

    let results = engine.enrich_batch(contexts)?;

    let mut total_cards = 0;
    for (frame_id, cards) in results {
        let card_count = cards.len();

        let card_ids = if !cards.is_empty() {
            mem.put_memory_cards(cards)?
        } else {
            Vec::new()
        };

        mem.record_enrichment(frame_id, kind, version, card_ids)?;

        total_cards += card_count;
    }

    Ok((total_frames, total_cards))
}

/// Run xAI enrichment with parallel batch processing.
fn run_xai_parallel(
    mem: &mut memvid_core::Memvid,
    engine: &XaiEngine,
    workers: usize,
) -> Result<(usize, usize)> {
    use memvid_core::enrich::EnrichmentContext;
    use memvid_core::EnrichmentEngine;

    let kind = engine.kind();
    let version = engine.version();

    let unenriched = mem.get_unenriched_frames(kind, version);
    let total_frames = unenriched.len();

    if total_frames == 0 {
        eprintln!("No unenriched frames found.");
        return Ok((0, 0));
    }

    eprintln!(
        "Gathering {} frames for parallel enrichment...",
        total_frames
    );

    let mut contexts = Vec::with_capacity(total_frames);
    for frame_id in &unenriched {
        let frame = match mem.frame_by_id(*frame_id) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let text = match mem.frame_text_by_id(*frame_id) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let uri = frame
            .uri
            .clone()
            .unwrap_or_else(|| format!("mv2://frame/{}", frame_id));
        let metadata_json = frame
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        let ctx = EnrichmentContext::new(
            *frame_id,
            uri,
            text,
            frame.title.clone(),
            frame.timestamp,
            metadata_json,
        );

        contexts.push(ctx);
    }

    eprintln!(
        "Starting parallel enrichment of {} frames with {} workers...",
        contexts.len(),
        workers
    );

    let results = engine.enrich_batch(contexts)?;

    let mut total_cards = 0;
    for (frame_id, cards) in results {
        let card_count = cards.len();

        let card_ids = if !cards.is_empty() {
            mem.put_memory_cards(cards)?
        } else {
            Vec::new()
        };

        mem.record_enrichment(frame_id, kind, version, card_ids)?;

        total_cards += card_count;
    }

    Ok((total_frames, total_cards))
}

/// Run Groq enrichment with parallel batch processing.
fn run_groq_parallel(
    mem: &mut memvid_core::Memvid,
    engine: &GroqEngine,
    workers: usize,
) -> Result<(usize, usize)> {
    use memvid_core::enrich::EnrichmentContext;
    use memvid_core::EnrichmentEngine;

    let kind = engine.kind();
    let version = engine.version();

    let unenriched = mem.get_unenriched_frames(kind, version);
    let total_frames = unenriched.len();

    if total_frames == 0 {
        eprintln!("No unenriched frames found.");
        return Ok((0, 0));
    }

    eprintln!(
        "Gathering {} frames for parallel enrichment...",
        total_frames
    );

    let mut contexts = Vec::with_capacity(total_frames);
    for frame_id in &unenriched {
        let frame = match mem.frame_by_id(*frame_id) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let text = match mem.frame_text_by_id(*frame_id) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let uri = frame
            .uri
            .clone()
            .unwrap_or_else(|| format!("mv2://frame/{}", frame_id));
        let metadata_json = frame
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        let ctx = EnrichmentContext::new(
            *frame_id,
            uri,
            text,
            frame.title.clone(),
            frame.timestamp,
            metadata_json,
        );

        contexts.push(ctx);
    }

    eprintln!(
        "Starting parallel enrichment of {} frames with {} workers...",
        contexts.len(),
        workers
    );

    let results = engine.enrich_batch(contexts)?;

    let mut total_cards = 0;
    for (frame_id, cards) in results {
        let card_count = cards.len();

        let card_ids = if !cards.is_empty() {
            mem.put_memory_cards(cards)?
        } else {
            Vec::new()
        };

        mem.record_enrichment(frame_id, kind, version, card_ids)?;

        total_cards += card_count;
    }

    Ok((total_frames, total_cards))
}

/// Run Mistral enrichment with parallel batch processing.
fn run_mistral_parallel(
    mem: &mut memvid_core::Memvid,
    engine: &MistralEngine,
    workers: usize,
) -> Result<(usize, usize)> {
    use memvid_core::enrich::EnrichmentContext;
    use memvid_core::EnrichmentEngine;

    let kind = engine.kind();
    let version = engine.version();

    let unenriched = mem.get_unenriched_frames(kind, version);
    let total_frames = unenriched.len();

    if total_frames == 0 {
        eprintln!("No unenriched frames found.");
        return Ok((0, 0));
    }

    eprintln!(
        "Gathering {} frames for parallel enrichment...",
        total_frames
    );

    let mut contexts = Vec::with_capacity(total_frames);
    for frame_id in &unenriched {
        let frame = match mem.frame_by_id(*frame_id) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let text = match mem.frame_text_by_id(*frame_id) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let uri = frame
            .uri
            .clone()
            .unwrap_or_else(|| format!("mv2://frame/{}", frame_id));
        let metadata_json = frame
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        let ctx = EnrichmentContext::new(
            *frame_id,
            uri,
            text,
            frame.title.clone(),
            frame.timestamp,
            metadata_json,
        );

        contexts.push(ctx);
    }

    eprintln!(
        "Starting parallel enrichment of {} frames with {} workers...",
        contexts.len(),
        workers
    );

    let results = engine.enrich_batch(contexts)?;

    let mut total_cards = 0;
    for (frame_id, cards) in results {
        let card_count = cards.len();

        let card_ids = if !cards.is_empty() {
            mem.put_memory_cards(cards)?
        } else {
            Vec::new()
        };

        mem.record_enrichment(frame_id, kind, version, card_ids)?;

        total_cards += card_count;
    }

    Ok((total_frames, total_cards))
}

/// Handle the `memories` subcommand (view memory cards)
#[derive(Args)]
pub struct MemoriesArgs {
    /// Path to the `.mv2` file
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Filter by entity
    #[arg(long)]
    pub entity: Option<String>,

    /// Filter by slot
    #[arg(long)]
    pub slot: Option<String>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Memory card output for JSON serialization
#[derive(Debug, Serialize)]
pub struct MemoryOutput {
    pub id: u64,
    pub kind: String,
    pub entity: String,
    pub slot: String,
    pub value: String,
    pub polarity: Option<String>,
    pub document_date: Option<i64>,
    pub source_frame_id: u64,
}

pub fn handle_memories(_config: &CliConfig, args: MemoriesArgs) -> Result<()> {
    let mem = Memvid::open(&args.file)?;

    let stats = mem.memories_stats();

    if args.json {
        let mut cards: Vec<MemoryOutput> = Vec::new();

        if let Some(entity) = &args.entity {
            if let Some(slot) = &args.slot {
                // Specific entity:slot
                if let Some(card) = mem.get_current_memory(entity, slot) {
                    cards.push(card_to_output(card));
                }
            } else {
                // All cards for entity
                for card in mem.get_entity_memories(entity) {
                    cards.push(card_to_output(card));
                }
            }
        } else {
            // All entities
            for entity in mem.memory_entities() {
                for card in mem.get_entity_memories(&entity) {
                    cards.push(card_to_output(card));
                }
            }
        }

        println!("{}", serde_json::to_string_pretty(&cards)?);
    } else {
        println!(
            "Memory cards: {} total, {} entities",
            stats.card_count, stats.entity_count
        );
        println!();

        if let Some(entity) = &args.entity {
            if let Some(slot) = &args.slot {
                // Specific entity:slot
                if let Some(card) = mem.get_current_memory(entity, slot) {
                    println!("{}:{} = \"{}\"", entity, slot, card.value);
                } else {
                    println!("No memory found for {}:{}", entity, slot);
                }
            } else {
                // All cards for entity
                println!("{}:", entity);
                for card in mem.get_entity_memories(entity) {
                    println!("  {}: {} = \"{}\"", card.kind, card.slot, card.value);
                }
            }
        } else {
            // All entities
            for entity in mem.memory_entities() {
                println!("{}:", entity);
                for card in mem.get_entity_memories(&entity) {
                    let polarity = card
                        .polarity
                        .as_ref()
                        .map(|p| format!(" [{}]", p))
                        .unwrap_or_default();
                    println!(
                        "  {}: {} = \"{}\"{}",
                        card.kind, card.slot, card.value, polarity
                    );
                }
                println!();
            }
        }
    }

    Ok(())
}

fn card_to_output(card: &memvid_core::MemoryCard) -> MemoryOutput {
    MemoryOutput {
        id: card.id,
        kind: card.kind.to_string(),
        entity: card.entity.clone(),
        slot: card.slot.clone(),
        value: card.value.clone(),
        polarity: card.polarity.as_ref().map(|p| p.to_string()),
        document_date: card.document_date,
        source_frame_id: card.source_frame_id,
    }
}

// ============================================================================
// State Command - Entity-Centric Current State Queries
// ============================================================================

/// Handle the `state` subcommand (query current entity state)
#[derive(Args)]
pub struct StateArgs {
    /// Path to the `.mv2` file
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Entity to query (required)
    #[arg(long, short = 'e')]
    pub entity: String,

    /// Specific slot to query (optional, omit for full entity profile)
    #[arg(long, short = 's')]
    pub slot: Option<String>,

    /// Query state at a specific point in time (Unix timestamp)
    #[arg(long)]
    pub at_time: Option<i64>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// State output for JSON serialization
#[derive(Debug, Serialize)]
pub struct StateOutput {
    pub entity: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slot: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_time: Option<i64>,
    pub state: StateValue,
}

/// The state value - either a single slot or full profile
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum StateValue {
    /// Single slot value
    Single {
        value: String,
        kind: String,
        polarity: Option<String>,
        source_frame_id: u64,
        document_date: Option<i64>,
    },
    /// Full entity profile
    Profile(Vec<SlotState>),
}

#[derive(Debug, Serialize)]
pub struct SlotState {
    pub slot: String,
    pub value: String,
    pub kind: String,
    pub polarity: Option<String>,
    pub source_frame_id: u64,
    pub document_date: Option<i64>,
}

pub fn handle_state(_config: &CliConfig, args: StateArgs) -> Result<()> {
    let mem = Memvid::open(&args.file)?;

    let entity = args.entity.to_lowercase(); // Normalize entity name

    if let Some(slot) = &args.slot {
        // O(1) lookup for specific entity:slot
        let card = if let Some(ts) = args.at_time {
            mem.get_memory_at_time(&entity, slot, ts)
        } else {
            mem.get_current_memory(&entity, slot)
        };

        if args.json {
            if let Some(card) = card {
                let output = StateOutput {
                    entity: entity.clone(),
                    slot: Some(slot.clone()),
                    at_time: args.at_time,
                    state: StateValue::Single {
                        value: card.value.clone(),
                        kind: card.kind.to_string(),
                        polarity: card.polarity.as_ref().map(|p| p.to_string()),
                        source_frame_id: card.source_frame_id,
                        document_date: card.document_date,
                    },
                };
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("null");
            }
        } else {
            if let Some(card) = card {
                let time_info = if let Some(ts) = args.at_time {
                    format!(" (at {})", format_timestamp(ts))
                } else {
                    String::new()
                };
                let polarity = card
                    .polarity
                    .as_ref()
                    .map(|p| format!(" [{}]", p))
                    .unwrap_or_default();
                println!(
                    "{}:{} = \"{}\"{}{}",
                    entity, slot, card.value, polarity, time_info
                );
                println!("  kind: {}", card.kind);
                println!("  source: frame {}", card.source_frame_id);
                if let Some(date) = card.document_date {
                    println!("  date: {}", format_timestamp(date));
                }
            } else {
                let time_info = if let Some(ts) = args.at_time {
                    format!(" at {}", format_timestamp(ts))
                } else {
                    String::new()
                };
                println!("No value for {}:{}{}", entity, slot, time_info);
            }
        }
    } else {
        // Get full entity profile
        let cards = mem.get_entity_memories(&entity);

        if cards.is_empty() {
            if args.json {
                println!("null");
            } else {
                println!("No state found for entity: {}", entity);
            }
            return Ok(());
        }

        // Group by slot and get current value for each
        let mut slots: std::collections::HashMap<String, &memvid_core::MemoryCard> =
            std::collections::HashMap::new();

        for card in &cards {
            // Keep the most recent card for each slot
            let dominated = slots
                .get(&card.slot)
                .map(|existing| card.effective_timestamp() > existing.effective_timestamp())
                .unwrap_or(true);

            if dominated && !card.is_retracted() {
                slots.insert(card.slot.clone(), card);
            }
        }

        if args.json {
            let mut profile: Vec<SlotState> = slots
                .values()
                .map(|card| SlotState {
                    slot: card.slot.clone(),
                    value: card.value.clone(),
                    kind: card.kind.to_string(),
                    polarity: card.polarity.as_ref().map(|p| p.to_string()),
                    source_frame_id: card.source_frame_id,
                    document_date: card.document_date,
                })
                .collect();
            profile.sort_by(|a, b| a.slot.cmp(&b.slot));

            let output = StateOutput {
                entity: entity.clone(),
                slot: None,
                at_time: args.at_time,
                state: StateValue::Profile(profile),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("{}:", entity);
            let mut sorted_slots: Vec<_> = slots.into_iter().collect();
            sorted_slots.sort_by(|a, b| a.0.cmp(&b.0));

            for (slot, card) in sorted_slots {
                let polarity = card
                    .polarity
                    .as_ref()
                    .map(|p| format!(" [{}]", p))
                    .unwrap_or_default();
                println!(
                    "  {}: \"{}\"{}  ({})",
                    slot, card.value, polarity, card.kind
                );
            }
        }
    }

    Ok(())
}

fn format_timestamp(ts: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let datetime = UNIX_EPOCH + Duration::from_secs(ts as u64);
    let datetime: chrono::DateTime<chrono::Utc> = datetime.into();
    datetime.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

/// Arguments for the `facts` (entity audit) command.
#[derive(Debug, Args)]
pub struct FactsArgs {
    /// Path to the `.mv2` file
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Filter by entity (optional)
    #[arg(long, short = 'e')]
    pub entity: Option<String>,

    /// Filter by predicate/slot (optional)
    #[arg(long, short = 'p')]
    pub predicate: Option<String>,

    /// Filter by value (optional)
    #[arg(long, short = 'v')]
    pub value: Option<String>,

    /// Show full history including superseded values
    #[arg(long)]
    pub history: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Audit log entry for JSON output.
#[derive(Debug, Serialize)]
pub struct AuditLogEntry {
    pub frame_id: u64,
    pub timestamp: Option<i64>,
    pub entity: String,
    pub slot: String,
    pub value: String,
    pub relation: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polarity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    pub engine: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<u64>,
}

/// Audit output for JSON serialization.
#[derive(Debug, Serialize)]
pub struct AuditLogOutput {
    pub total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate_filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_filter: Option<String>,
    pub entries: Vec<AuditLogEntry>,
}

/// Format a Unix timestamp as ISO 8601 (without chrono).
fn format_audit_timestamp(ts: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};

    let datetime = UNIX_EPOCH + Duration::from_secs(ts.unsigned_abs() as u64);
    let secs = datetime
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    let mut year = 1970i32;
    let mut remaining_days = days as i32;

    loop {
        let days_in_year = if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
            366
        } else {
            365
        };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let mut month = 1u32;
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_months = if is_leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    for days_in_month in days_in_months {
        if remaining_days < days_in_month {
            break;
        }
        remaining_days -= days_in_month;
        month += 1;
    }

    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

pub fn handle_facts(_config: &CliConfig, args: FactsArgs) -> Result<()> {
    let mem = Memvid::open(&args.file)?;

    // Collect all memory cards matching the filters
    let mut entries: Vec<AuditLogEntry> = Vec::new();

    // Get entities to iterate
    let entities: Vec<String> = if let Some(entity) = &args.entity {
        vec![entity.to_lowercase()]
    } else {
        mem.memory_entities()
    };

    for entity in entities {
        let cards = mem.get_entity_memories(&entity);

        for card in cards {
            // Filter by predicate
            if let Some(pred) = &args.predicate {
                if !card.slot.eq_ignore_ascii_case(pred) {
                    continue;
                }
            }

            // Filter by value
            if let Some(val) = &args.value {
                if !card.value.to_lowercase().contains(&val.to_lowercase()) {
                    continue;
                }
            }

            entries.push(AuditLogEntry {
                frame_id: card.source_frame_id,
                timestamp: card.document_date.or(Some(card.created_at)),
                entity: card.entity.clone(),
                slot: card.slot.clone(),
                value: card.value.clone(),
                relation: card.version_relation.as_str().to_string(),
                kind: card.kind.to_string(),
                polarity: card.polarity.as_ref().map(|p| p.to_string()),
                confidence: card.confidence,
                engine: card.engine.clone(),
                supersedes: None, // TODO: track superseded frame IDs
            });
        }
    }

    // Sort by timestamp (oldest first for audit trail)
    entries.sort_by(|a, b| {
        let ts_a = a.timestamp.unwrap_or(0);
        let ts_b = b.timestamp.unwrap_or(0);
        ts_a.cmp(&ts_b)
    });

    if args.json {
        let output = AuditLogOutput {
            total: entries.len(),
            entity_filter: args.entity.clone(),
            predicate_filter: args.predicate.clone(),
            value_filter: args.value.clone(),
            entries,
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        if entries.is_empty() {
            println!("No matching facts found.");
            return Ok(());
        }

        println!("Audit Trail ({} entries):", entries.len());
        println!();

        for entry in entries {
            let ts_str = entry
                .timestamp
                .map(format_audit_timestamp)
                .unwrap_or_else(|| "unknown".to_string());

            let polarity_suffix = entry
                .polarity
                .as_ref()
                .map(|p| format!(" [{}]", p))
                .unwrap_or_default();

            let conf = entry
                .confidence
                .map(|c| format!(" (conf: {:.2})", c))
                .unwrap_or_default();

            let _polarity_prefix = if entry.polarity.is_some() {
                if entry.polarity.as_deref() == Some("negative") {
                    "-"
                } else if entry.polarity.as_deref() == Some("positive") {
                    "+"
                } else {
                    ""
                }
            } else {
                ""
            };

            println!(
                "Frame {} ({}): {} {}:{}=\"{}\"{}  [{}]{}",
                entry.frame_id,
                ts_str,
                entry.relation.to_uppercase(),
                entry.entity,
                entry.slot,
                entry.value,
                polarity_suffix,
                entry.engine,
                conf,
            );
        }
    }

    Ok(())
}

// ============================================================================
// Export Command - Export facts to standard formats (N-Triples, JSON, CSV)
// ============================================================================

/// Export format for the `export` command.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum ExportFormat {
    /// N-Triples RDF format (.nt)
    #[default]
    Ntriples,
    /// JSON format with full metadata
    Json,
    /// CSV format for spreadsheet import
    Csv,
}

/// Arguments for the `export` subcommand.
#[derive(Debug, Args)]
pub struct ExportArgs {
    /// Path to the `.mv2` file
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Export format
    #[arg(long, short = 'f', value_enum, default_value_t = ExportFormat::Ntriples)]
    pub format: ExportFormat,

    /// Filter by entity (optional)
    #[arg(long, short = 'e')]
    pub entity: Option<String>,

    /// Filter by predicate/slot (optional)
    #[arg(long, short = 'p')]
    pub predicate: Option<String>,

    /// Base URI for N-Triples output (default: mv2://entity/)
    #[arg(long, default_value = "mv2://entity/")]
    pub base_uri: String,

    /// Include provenance metadata in output
    #[arg(long)]
    pub with_provenance: bool,
}

/// Export entry for JSON output.
#[derive(Debug, Serialize)]
pub struct ExportEntry {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Escape a string for N-Triples format.
fn escape_ntriples(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            _ => result.push(c),
        }
    }
    result
}

/// Escape a string for CSV format.
fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Normalize an entity name into a valid URI component.
fn normalize_uri_component(s: &str) -> String {
    s.replace(' ', "_")
        .replace('/', "_")
        .replace(':', "_")
        .replace('#', "_")
        .replace('?', "_")
        .replace('&', "_")
}

pub fn handle_export(_config: &CliConfig, args: ExportArgs) -> Result<()> {
    let mem = Memvid::open(&args.file)?;

    // Collect all memory cards matching the filters
    let entities: Vec<String> = if let Some(entity) = &args.entity {
        vec![entity.to_lowercase()]
    } else {
        mem.memory_entities()
    };

    // Build list of triplets
    let mut triplets: Vec<ExportEntry> = Vec::new();

    for entity in entities {
        let cards = mem.get_entity_memories(&entity);

        for card in cards {
            // Filter by predicate
            if let Some(pred) = &args.predicate {
                if !card.slot.eq_ignore_ascii_case(pred) {
                    continue;
                }
            }

            // Skip retracted cards
            if card.is_retracted() {
                continue;
            }

            triplets.push(ExportEntry {
                subject: card.entity.clone(),
                predicate: card.slot.clone(),
                object: card.value.clone(),
                source_frame_id: if args.with_provenance {
                    Some(card.source_frame_id)
                } else {
                    None
                },
                timestamp: if args.with_provenance {
                    card.document_date.or(Some(card.created_at))
                } else {
                    None
                },
                engine: if args.with_provenance {
                    Some(card.engine.clone())
                } else {
                    None
                },
                confidence: if args.with_provenance {
                    card.confidence
                } else {
                    None
                },
            });
        }
    }

    match args.format {
        ExportFormat::Ntriples => {
            // Output N-Triples format
            // Format: <subject> <predicate> "object" .
            for t in &triplets {
                let subject_uri =
                    format!("<{}{}>", args.base_uri, normalize_uri_component(&t.subject));
                let predicate_uri = format!(
                    "<{}pred/{}>",
                    args.base_uri,
                    normalize_uri_component(&t.predicate)
                );
                let object_literal = format!("\"{}\"", escape_ntriples(&t.object));

                println!("{} {} {} .", subject_uri, predicate_uri, object_literal);
            }
        }
        ExportFormat::Json => {
            // Output JSON format
            println!("{}", serde_json::to_string_pretty(&triplets)?);
        }
        ExportFormat::Csv => {
            // Output CSV format
            if args.with_provenance {
                println!("subject,predicate,object,source_frame_id,timestamp,engine,confidence");
            } else {
                println!("subject,predicate,object");
            }

            for t in &triplets {
                if args.with_provenance {
                    println!(
                        "{},{},{},{},{},{},{}",
                        escape_csv(&t.subject),
                        escape_csv(&t.predicate),
                        escape_csv(&t.object),
                        t.source_frame_id
                            .map(|id| id.to_string())
                            .unwrap_or_default(),
                        t.timestamp.map(|ts| ts.to_string()).unwrap_or_default(),
                        t.engine.as_deref().map(escape_csv).unwrap_or_default(),
                        t.confidence
                            .map(|c| format!("{:.2}", c))
                            .unwrap_or_default(),
                    );
                } else {
                    println!(
                        "{},{},{}",
                        escape_csv(&t.subject),
                        escape_csv(&t.predicate),
                        escape_csv(&t.object),
                    );
                }
            }
        }
    }

    // Output count to stderr so it doesn't pollute stdout piping
    eprintln!("Exported {} triplets", triplets.len());

    Ok(())
}

// ============================================================================
// Schema Command - Infer and manage predicate schemas
// ============================================================================

/// Arguments for the `schema` subcommand.
#[derive(Debug, Args)]
pub struct SchemaArgs {
    #[command(subcommand)]
    pub command: SchemaCommand,
}

/// Schema subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum SchemaCommand {
    /// Infer schemas from existing memory cards
    Infer(SchemaInferArgs),
    /// List registered schemas (built-in + custom)
    List(SchemaListArgs),
}

/// Arguments for `schema infer`.
#[derive(Debug, Args)]
pub struct SchemaInferArgs {
    /// Path to the `.mv2` file
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Register inferred schemas to the memory file
    #[arg(long)]
    pub register: bool,

    /// Overwrite existing schemas when registering
    #[arg(long, requires = "register")]
    pub overwrite: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `schema list`.
#[derive(Debug, Args)]
pub struct SchemaListArgs {
    /// Path to the `.mv2` file (optional, shows built-in schemas if omitted)
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: Option<PathBuf>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Show only built-in schemas
    #[arg(long)]
    pub builtin_only: bool,
}

/// Schema list entry for JSON output.
#[derive(Debug, Serialize)]
pub struct SchemaListEntry {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub value_type: String,
    pub cardinality: String,
    pub domain: Vec<String>,
    pub builtin: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inverse: Option<String>,
}

pub fn handle_schema(_config: &CliConfig, args: SchemaArgs) -> Result<()> {
    match args.command {
        SchemaCommand::Infer(infer_args) => handle_schema_infer(_config, infer_args),
        SchemaCommand::List(list_args) => handle_schema_list(_config, list_args),
    }
}

fn handle_schema_infer(_config: &CliConfig, args: SchemaInferArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    // Get schema summary (which includes inference)
    let summary = mem.schema_summary();

    if summary.is_empty() {
        if args.json {
            println!("[]");
        } else {
            println!("No predicates found in memory.");
        }
        return Ok(());
    }

    if args.register {
        let count = mem.register_inferred_schemas(args.overwrite);
        mem.commit()?;
        eprintln!("Registered {} inferred schemas", count);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("Inferred Schemas ({} predicates):", summary.len());
        println!();
        println!(
            "{:<20} {:<12} {:<10} {:<8} {:<8} {:<8} {}",
            "PREDICATE", "TYPE", "CARDINAL", "ENTITIES", "VALUES", "UNIQUE", "BUILTIN"
        );
        println!("{}", "-".repeat(80));

        for entry in &summary {
            let cardinality = match entry.cardinality {
                memvid_core::Cardinality::Single => "single",
                memvid_core::Cardinality::Multiple => "multiple",
            };
            let builtin = if entry.is_builtin { "yes" } else { "-" };

            println!(
                "{:<20} {:<12} {:<10} {:<8} {:<8} {:<8} {}",
                truncate(&entry.predicate, 20),
                truncate(&entry.inferred_type, 12),
                cardinality,
                entry.entity_count,
                entry.value_count,
                entry.unique_values,
                builtin
            );
        }
    }

    Ok(())
}

fn handle_schema_list(_config: &CliConfig, args: SchemaListArgs) -> Result<()> {
    // If file is provided, open it to get custom schemas; otherwise use default registry
    let registry = if let Some(ref path) = args.file {
        let mem = Memvid::open(path)?;
        mem.schema_registry().clone()
    } else {
        memvid_core::SchemaRegistry::new()
    };

    let mut entries: Vec<SchemaListEntry> = registry
        .all()
        .filter(|s| !args.builtin_only || s.builtin)
        .map(|schema| SchemaListEntry {
            id: schema.id.clone(),
            name: schema.name.clone(),
            description: schema.description.clone(),
            value_type: schema.range.description(),
            cardinality: match schema.cardinality {
                memvid_core::Cardinality::Single => "single".to_string(),
                memvid_core::Cardinality::Multiple => "multiple".to_string(),
            },
            domain: schema
                .domain
                .iter()
                .map(|k| k.as_str().to_string())
                .collect(),
            builtin: schema.builtin,
            inverse: schema.inverse.clone(),
        })
        .collect();

    entries.sort_by(|a, b| a.id.cmp(&b.id));

    if entries.is_empty() {
        if args.json {
            println!("[]");
        } else {
            println!("No schemas found.");
        }
        return Ok(());
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        let title = if args.builtin_only {
            "Built-in Schemas"
        } else {
            "Registered Schemas"
        };
        println!("{} ({} total):", title, entries.len());
        println!();
        println!(
            "{:<20} {:<15} {:<12} {:<10} {}",
            "ID", "NAME", "TYPE", "CARDINAL", "DOMAIN"
        );
        println!("{}", "-".repeat(70));

        for entry in &entries {
            let domain = if entry.domain.is_empty() {
                "*".to_string()
            } else {
                entry.domain.join(", ")
            };
            let cardinality = if entry.cardinality == "multiple" {
                "multiple"
            } else {
                "single"
            };

            println!(
                "{:<20} {:<15} {:<12} {:<10} {}",
                truncate(&entry.id, 20),
                truncate(&entry.name, 15),
                truncate(&entry.value_type, 12),
                cardinality,
                truncate(&domain, 20)
            );
        }
    }

    Ok(())
}

/// Truncate a string to max length, adding "..." if needed.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
