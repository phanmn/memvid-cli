#![cfg_attr(check_cfg, allow(unexpected_cfgs))]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::process::ExitCode;

use anyhow::Result;
use clap::{ArgAction, Parser, Subcommand};

use memvid_cli::analytics::{force_flush_sync, init_analytics, track_command_with_tier};
use memvid_cli::commands::{
    handle_api_fetch, handle_ask, handle_audit, handle_binding, handle_config, handle_correct,
    handle_create, handle_debug_segment, handle_delete, handle_doctor, handle_enrich,
    handle_export, handle_facts, handle_find, handle_follow, handle_memories, handle_models,
    handle_nudge, handle_open, handle_plan, handle_process_queue, handle_put, handle_schema,
    handle_sketch, handle_state, handle_stats, handle_status, handle_tables, handle_tickets,
    handle_timeline, handle_update, handle_vec_search, handle_verify, handle_verify_single_file,
    handle_version, handle_view, handle_who, ApiFetchArgs, AskArgs, AuditArgs, BindingArgs,
    ConfigArgs, CorrectArgs, CreateArgs, DebugSegmentArgs, DeleteArgs, DoctorArgs, EnrichArgs,
    ExportArgs, FactsArgs, FindArgs, FollowArgs, MemoriesArgs, ModelsArgs, NudgeArgs, OpenArgs,
    PlanArgs, ProcessQueueArgs, PutArgs, SchemaArgs, SketchArgs, StateArgs, StatsArgs, StatusArgs,
    TablesArgs, TicketsArgs, TimelineArgs, UpdateArgs, VecSearchArgs, VerifyArgs,
    VerifySingleFileArgs, ViewArgs, WhoArgs,
};
#[cfg(feature = "encryption")]
use memvid_cli::commands::{handle_lock, handle_unlock, LockArgs, UnlockArgs};
#[cfg(feature = "parallel_segments")]
use memvid_cli::commands::{handle_put_many, PutManyArgs};
#[cfg(feature = "replay")]
use memvid_cli::commands::{handle_session, SessionArgs};
#[cfg(feature = "temporal_track")]
use memvid_cli::commands::{handle_when, WhenArgs};
use memvid_cli::config::{init_tracing, CliConfig, EmbeddingModelChoice};
use memvid_cli::org_ticket_cache;

#[cfg(feature = "parallel_segments")]
#[derive(Parser)]
#[command(name = "memvid", author, version, about = "Memvid single-file memory CLI", long_about = None)]
struct Cli {
    /// Increase logging verbosity (use multiple times for more detail)
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,

    /// Default embedding model (used by `put --embedding` and by semantic queries when needed):
    /// bge-small (fast, default), bge-base, nomic (high accuracy), gte-large, openai (requires OPENAI_API_KEY), openai-small, openai-ada, nvidia (requires NVIDIA_API_KEY)
    #[arg(
        short = 'm',
        long,
        global = true,
        value_name = "MODEL",
        alias = "model"
    )]
    embedding_model: Option<EmbeddingModelChoice>,

    /// Enable the parallel segment builder globally (requires --features parallel_segments)
    #[cfg(feature = "parallel_segments")]
    #[arg(long, global = true, action = ArgAction::SetTrue, conflicts_with = "global_no_parallel_segments")]
    parallel_segments: bool,
    /// Force the legacy ingestion path globally
    #[cfg(feature = "parallel_segments")]
    #[arg(
        long = "global-no-parallel-segments",
        global = true,
        action = ArgAction::SetTrue,
        conflicts_with = "parallel_segments"
    )]
    global_no_parallel_segments: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new `.mv2` memory file
    Create(CreateArgs),
    /// Inspect metadata and manifests for an existing memory
    Open(OpenArgs),
    /// Append a frame to the memory, optionally with metadata
    Put(PutArgs),
    /// Store a correction with retrieval priority boost
    Correct(CorrectArgs),
    /// Batch ingest multiple documents with pre-computed embeddings
    #[cfg(feature = "parallel_segments")]
    PutMany(PutManyArgs),
    /// Fetch remote content and ingest it as frames
    ApiFetch(ApiFetchArgs),
    /// View a single frame
    View(ViewArgs),
    /// Update an existing frame
    Update(UpdateArgs),
    /// Delete a frame from the memory
    Delete(DeleteArgs),
    /// View the timeline of frames
    Timeline(TimelineArgs),
    /// Ask questions with retrieval + synthesis
    Ask(AskArgs),
    /// Generate an audit report with full source provenance
    Audit(AuditArgs),
    /// Perform lexical search over the memory
    Find(FindArgs),
    /// Perform vector similarity search
    VecSearch(VecSearchArgs),
    /// Dump raw vector segment bytes for debugging
    DebugSegment(DebugSegmentArgs),
    #[cfg(feature = "temporal_track")]
    /// Resolve temporal phrases and list matching frames
    When(WhenArgs),
    /// Display statistics about the memory
    Stats(StatsArgs),
    /// Run integrity verification checks
    Verify(VerifyArgs),
    /// Run doctor workflows to repair or optimise the memory
    Doctor(DoctorArgs),
    /// Process the enrichment queue (re-extract skim frames, update indexes)
    ProcessQueue(ProcessQueueArgs),
    /// Ensure no auxiliary files exist alongside the memory
    VerifySingleFile(VerifySingleFileArgs),
    /// Extract, list, export, and view tables from documents
    Tables(TablesArgs),
    /// Manage access tickets via the API
    Tickets(TicketsArgs),
    /// View and manage your plan/subscription
    Plan(PlanArgs),
    /// Show memory binding information
    Binding(BindingArgs),
    /// Manage persistent CLI configuration (API keys, settings)
    Config(ConfigArgs),
    /// Show configuration and system status
    Status(StatusArgs),
    /// Show the active writer holding the lock
    Who(WhoArgs),
    /// Request the active writer flush and release when safe
    Nudge(NudgeArgs),
    /// Run enrichment engines to extract memory cards from frames
    Enrich(EnrichArgs),
    /// View extracted memory cards
    Memories(MemoriesArgs),
    /// Query current entity state (O(1) lookup)
    State(StateArgs),
    /// Audit fact changes with provenance and filtering
    Facts(FactsArgs),
    /// Export facts to N-Triples, JSON, or CSV format
    Export(ExportArgs),
    /// Infer and manage predicate schemas
    Schema(SchemaArgs),
    /// Manage LLM models for enrichment
    Models(ModelsArgs),
    /// Traverse the Logic-Mesh entity graph
    Follow(FollowArgs),
    /// Build and manage sketch track for fast candidate generation
    Sketch(SketchArgs),
    /// Manage time-travel replay sessions
    #[cfg(feature = "replay")]
    Session(SessionArgs),
    /// Encrypt a memory file into an encrypted capsule (.mv2e)
    #[cfg(feature = "encryption")]
    Lock(LockArgs),
    /// Decrypt an encrypted capsule (.mv2e) back to a `.mv2` file
    #[cfg(feature = "encryption")]
    Unlock(UnlockArgs),
    /// Print version information for debugging scripts
    Version,
}

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let (code, message) = memvid_cli::error::render_error(&err);
            eprintln!("{message}");
            ExitCode::from(code as u8)
        }
    }
}

fn try_main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose)?;
    #[cfg(feature = "parallel_segments")]
    apply_global_parallel_flags(&cli);
    let mut config = CliConfig::load()?;

    // Initialize anonymous telemetry (opt-out: MEMVID_TELEMETRY=0)
    init_analytics();

    // Apply embedding model override from CLI flag
    if let Some(model) = cli.embedding_model {
        config = config.with_embedding_model(model);
    }

    let result = if let Some(command) = cli.command {
        dispatch(&config, command)
    } else {
        print_command_overview();
        Ok(())
    };

    // Flush analytics before exit (fire-and-forget, errors ignored)
    let _ = force_flush_sync();

    result
}

fn print_command_overview() {
    println!("Memvid CLI – common commands:");
    println!("  create <file>         Create an empty .mv2 memory");
    println!("  put <file>            Append documents or folders to a memory");
    println!("  find <file> --query   Run lexical + vector search");
    println!("  view <file> --frame   Inspect the contents/metadata of a frame");
    println!("  ask <file> --question Ask retrieval-augmented questions");
    println!("  timeline <file>       Browse the chronological frame list");
    println!("  stats <file>          Show storage and index utilization");
    println!("  doctor <file>         Repair or vacuum a memory");
    println!("  plan show             Show your current plan and capacity");
    println!("  plan sync             Sync your plan ticket from the dashboard");
    println!("  config set <key> <v>  Set a configuration value (e.g., api_key)");
    println!("  config list           List all configuration values");
    #[cfg(feature = "encryption")]
    {
        println!("  lock <file>           Encrypt a .mv2 into a .mv2e capsule");
        println!("  unlock <file>         Decrypt a .mv2e capsule back to .mv2");
    }
    println!();
    println!("Run `memvid <command> --help` for detailed flags.");
}

#[cfg(feature = "parallel_segments")]
fn apply_global_parallel_flags(cli: &Cli) {
    if cli.parallel_segments {
        unsafe {
            std::env::set_var("MEMVID_PARALLEL_SEGMENTS", "1");
        }
    } else if cli.global_no_parallel_segments {
        unsafe {
            std::env::set_var("MEMVID_PARALLEL_SEGMENTS", "0");
        }
    }
}

fn dispatch(config: &CliConfig, command: Commands) -> Result<()> {
    // Extract file path and command name for analytics
    let (cmd_name, file_path, is_create, is_open) = get_command_info(&command);

    // Execute command and track result
    let result = match command {
        Commands::Create(args) => handle_create(config, args),
        Commands::Open(args) => handle_open(config, args),
        Commands::Put(args) => handle_put(config, args),
        Commands::Correct(args) => handle_correct(config, args),
        #[cfg(feature = "parallel_segments")]
        Commands::PutMany(args) => handle_put_many(config, args),
        Commands::ApiFetch(args) => handle_api_fetch(config, args),
        Commands::View(args) => handle_view(args),
        Commands::Update(args) => handle_update(config, args),
        Commands::Delete(args) => handle_delete(config, args),
        Commands::Timeline(args) => handle_timeline(config, args),
        Commands::Ask(args) => handle_ask(config, args),
        Commands::Audit(args) => handle_audit(config, args),
        Commands::Find(args) => handle_find(config, args),
        Commands::VecSearch(args) => handle_vec_search(config, args),
        Commands::DebugSegment(args) => handle_debug_segment(args),
        #[cfg(feature = "temporal_track")]
        Commands::When(args) => handle_when(config, args),
        Commands::Stats(args) => handle_stats(config, args),
        Commands::Verify(args) => handle_verify(config, args),
        Commands::Doctor(args) => handle_doctor(config, args),
        Commands::ProcessQueue(args) => handle_process_queue(config, args),
        Commands::VerifySingleFile(args) => handle_verify_single_file(config, args),
        Commands::Tables(args) => handle_tables(config, args),
        Commands::Tickets(cmd) => handle_tickets(config, cmd),
        Commands::Plan(args) => handle_plan(config, args),
        Commands::Binding(args) => handle_binding(args),
        Commands::Config(args) => handle_config(args),
        Commands::Status(args) => handle_status(args),
        Commands::Who(args) => handle_who(args),
        Commands::Nudge(args) => handle_nudge(args),
        Commands::Enrich(args) => handle_enrich(config, args),
        Commands::Memories(args) => handle_memories(config, args),
        Commands::State(args) => handle_state(config, args),
        Commands::Facts(args) => handle_facts(config, args),
        Commands::Export(args) => handle_export(config, args),
        Commands::Schema(args) => handle_schema(config, args),
        Commands::Models(args) => handle_models(config, args),
        Commands::Follow(args) => handle_follow(config, args),
        Commands::Sketch(args) => handle_sketch(args),
        #[cfg(feature = "replay")]
        Commands::Session(args) => handle_session(args),
        #[cfg(feature = "encryption")]
        Commands::Lock(args) => handle_lock(args),
        #[cfg(feature = "encryption")]
        Commands::Unlock(args) => handle_unlock(args),
        Commands::Version => handle_version(),
    };

    // Get user tier for analytics (free tier if no valid ticket)
    let user_tier = org_ticket_cache::get_optional(config)
        .map(|t| t.plan_name)
        .unwrap_or_else(|| "free".to_string());

    // Track command execution (fire-and-forget)
    track_command_with_tier(
        file_path.as_deref(),
        cmd_name,
        result.is_ok(),
        is_create && result.is_ok(),
        is_open && result.is_ok(),
        &user_tier,
    );

    result
}

/// Helper to convert PathBuf to String for analytics
fn path_to_string(path: &std::path::Path) -> String {
    path.display().to_string()
}

/// Extract command info for analytics tracking
fn get_command_info(command: &Commands) -> (&'static str, Option<String>, bool, bool) {
    match command {
        Commands::Create(args) => ("create", Some(path_to_string(&args.file)), true, false),
        Commands::Open(args) => ("open", Some(path_to_string(&args.file)), false, true),
        Commands::Put(args) => ("put", Some(path_to_string(&args.file)), false, false),
        Commands::Correct(args) => ("correct", Some(path_to_string(&args.file)), false, false),
        #[cfg(feature = "parallel_segments")]
        Commands::PutMany(args) => ("put_many", Some(path_to_string(&args.file)), false, false),
        Commands::ApiFetch(args) => ("api_fetch", Some(path_to_string(&args.file)), false, false),
        Commands::View(args) => ("view", Some(path_to_string(&args.file)), false, true),
        Commands::Update(args) => ("update", Some(path_to_string(&args.file)), false, false),
        Commands::Delete(args) => ("delete", Some(path_to_string(&args.file)), false, false),
        Commands::Timeline(args) => ("timeline", Some(path_to_string(&args.file)), false, true),
        Commands::Ask(args) => ("ask", args.targets.first().cloned(), false, true),
        Commands::Audit(args) => ("audit", Some(path_to_string(&args.file)), false, true),
        Commands::Find(args) => ("find", Some(path_to_string(&args.file)), false, true),
        Commands::VecSearch(args) => ("vec_search", Some(path_to_string(&args.file)), false, true),
        Commands::DebugSegment(args) => (
            "debug_segment",
            Some(path_to_string(&args.file)),
            false,
            false,
        ),
        #[cfg(feature = "temporal_track")]
        Commands::When(args) => ("when", Some(path_to_string(&args.file)), false, true),
        Commands::Stats(args) => ("stats", Some(path_to_string(&args.file)), false, true),
        Commands::Verify(args) => ("verify", Some(path_to_string(&args.file)), false, true),
        Commands::Doctor(args) => ("doctor", Some(path_to_string(&args.file)), false, false),
        Commands::ProcessQueue(args) => (
            "process_queue",
            Some(path_to_string(&args.file)),
            false,
            false,
        ),
        Commands::VerifySingleFile(args) => (
            "verify_single_file",
            Some(path_to_string(&args.file)),
            false,
            true,
        ),
        Commands::Tables(_) => ("tables", None, false, false),
        Commands::Tickets(_) => ("tickets", None, false, false),
        Commands::Plan(_) => ("plan", None, false, false),
        Commands::Binding(args) => ("binding", Some(path_to_string(&args.file)), false, true),
        Commands::Config(_) => ("config", None, false, false),
        Commands::Status(_) => ("status", None, false, false),
        Commands::Who(args) => ("who", Some(path_to_string(&args.file)), false, true),
        Commands::Nudge(args) => ("nudge", Some(path_to_string(&args.file)), false, false),
        Commands::Enrich(args) => ("enrich", Some(path_to_string(&args.file)), false, false),
        Commands::Memories(args) => ("memories", Some(path_to_string(&args.file)), false, true),
        Commands::State(args) => ("state", Some(path_to_string(&args.file)), false, true),
        Commands::Facts(args) => ("facts", Some(path_to_string(&args.file)), false, true),
        Commands::Export(args) => ("export", Some(path_to_string(&args.file)), false, true),
        Commands::Schema(_) => ("schema", None, false, false),
        Commands::Models(_) => ("models", None, false, false),
        Commands::Follow(_) => ("follow", None, false, false),
        Commands::Sketch(_) => ("sketch", None, false, false),
        #[cfg(feature = "replay")]
        Commands::Session(_) => ("session", None, false, false),
        #[cfg(feature = "encryption")]
        Commands::Lock(args) => ("lock", Some(path_to_string(&args.file)), false, true),
        #[cfg(feature = "encryption")]
        Commands::Unlock(args) => ("unlock", Some(path_to_string(&args.file)), false, true),
        Commands::Version => ("version", None, false, false),
    }
}
