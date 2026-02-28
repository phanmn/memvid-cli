//! Creation command handlers (create, open)

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{ArgAction, Args, ValueEnum};
use memvid_core::{Memvid, Ticket};

use crate::commands::tickets::MemoryId;
use crate::config::CliConfig;
use crate::utils::{
    format_bytes, open_read_only_mem, parse_size, yes_no, FREE_TIER_MAX_FILE_SIZE, MIN_FILE_SIZE,
};

/// Tier argument for CLI
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TierArg {
    Free,
    Dev,
    Enterprise,
}

impl From<TierArg> for memvid_core::Tier {
    fn from(value: TierArg) -> Self {
        match value {
            TierArg::Free => memvid_core::Tier::Free,
            TierArg::Dev => memvid_core::Tier::Dev,
            TierArg::Enterprise => memvid_core::Tier::Enterprise,
        }
    }
}

/// Arguments for the `create` subcommand
#[derive(Args)]
pub struct CreateArgs {
    /// Path to the memory file to create
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    /// Tier to apply when creating the memory
    #[arg(long, value_enum)]
    pub tier: Option<TierArg>,
    /// Set the maximum memory size (e.g. 512MB); defaults to plan capacity
    #[arg(long = "size", alias = "capacity", value_name = "SIZE", value_parser = parse_size)]
    pub size: Option<u64>,
    /// Bind to a dashboard memory ID (UUID or 24-char ObjectId) for capacity and tracking
    #[arg(long = "memory-id", value_name = "ID")]
    pub memory_id: Option<MemoryId>,
    /// Use a named memory from config (e.g., --memory work uses memory.work)
    #[arg(long = "memory", value_name = "NAME")]
    pub memory_name: Option<String>,
    /// Disable lexical index
    #[arg(long = "no-lex", action = ArgAction::SetTrue)]
    pub no_lex: bool,
    /// Disable vector index
    #[arg(long = "no-vector", aliases = ["no-vec"], action = ArgAction::SetTrue)]
    pub no_vector: bool,
}

/// Arguments for the `open` subcommand
#[derive(Args)]
pub struct OpenArgs {
    /// Path to the memory file to open
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    /// Emit JSON instead of human-readable output
    #[arg(long)]
    pub json: bool,
}

/// Handler for `memvid create`
pub fn handle_create(config: &CliConfig, args: CreateArgs) -> Result<()> {
    if let Some(parent) = args.file.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    // Capacity is determined by binding:
    // - With --memory-id: capacity comes from dashboard memory (paid plan)
    // - Without --memory-id: always free tier (1GB), regardless of API key
    // This ensures paid capacity is only used for dashboard-tracked memories
    //
    // Memory ID priority: --memory-id flag > --memory name lookup > config default > none
    let effective_memory_id: Option<MemoryId> = args.memory_id.clone().or_else(|| {
        // If --memory <name> is provided, look it up in config
        if let Some(ref name) = args.memory_name {
            if let Ok(persistent_config) = crate::commands::config::PersistentConfig::load() {
                if let Some(id) = persistent_config.get_memory(name) {
                    return id.parse::<MemoryId>().ok();
                } else {
                    eprintln!(
                        "⚠️  Memory '{}' not found in config. Available memories:",
                        name
                    );
                    for (mem_name, _) in &persistent_config.memory {
                        eprintln!("   - {}", mem_name);
                    }
                    eprintln!(
                        "   Set it with: memvid config set memory.{} <MEMORY_ID>",
                        name
                    );
                }
            }
            return None;
        }
        // Fall back to config default (memory.default or legacy memory_id)
        config
            .memory_id
            .as_ref()
            .and_then(|id| id.parse::<MemoryId>().ok())
    });
    let has_memory_id = effective_memory_id.is_some();

    // Validate --size if provided
    if let Some(requested) = args.size {
        // Check minimum size
        if requested < MIN_FILE_SIZE {
            anyhow::bail!(
                "Requested capacity {} is below minimum ({}).\n\
                 Minimum memory size is {} to ensure reasonable capacity for basic usage.",
                format_bytes(requested),
                format_bytes(MIN_FILE_SIZE),
                format_bytes(MIN_FILE_SIZE)
            );
        }

        // Check maximum size (only for free tier without memory-id)
        if requested > FREE_TIER_MAX_FILE_SIZE && !has_memory_id {
            anyhow::bail!(
                "Requested capacity {} exceeds free tier limit ({}).\n\
                 To unlock more capacity, bind to a dashboard memory:\n\
                   memvid create {} --memory-id <MEMORY_ID>\n\
                 Or bind an existing file:\n\
                   memvid tickets sync {} --memory-id <MEMORY_ID>",
                format_bytes(requested),
                format_bytes(FREE_TIER_MAX_FILE_SIZE),
                args.file.display(),
                args.file.display()
            );
        }
    }

    // For initial creation, always use free tier capacity
    // If --memory-id is provided, capacity will be upgraded after binding
    let initial_capacity = args.size.unwrap_or(FREE_TIER_MAX_FILE_SIZE);
    let capacity_bytes = initial_capacity.min(FREE_TIER_MAX_FILE_SIZE);

    let lexical_enabled = !args.no_lex;
    let vector_enabled = !args.no_vector;

    let mut mem = Memvid::create(&args.file)?;
    apply_capacity_override(&mut mem, capacity_bytes)?;
    if lexical_enabled {
        mem.enable_lex()?;
    }

    if vector_enabled {
        mem.enable_vec()?;
    }
    mem.commit()?;

    // If memory-id is provided (CLI or config), bind to the dashboard memory (this upgrades capacity)
    let binding_info = if let Some(memory_id) = &effective_memory_id {
        match bind_to_dashboard_memory(config, &mut mem, &args.file, memory_id) {
            Ok(info) => Some(info),
            Err(e) => {
                // Show root cause if available for better error messages
                let root_cause = e.root_cause();
                eprintln!("⚠️  Failed to bind to dashboard memory: {}", root_cause);
                eprintln!("   File created with free tier capacity. You can bind later with:");
                eprintln!(
                    "   memvid tickets sync {} --memory-id {}",
                    args.file.display(),
                    memory_id
                );
                None
            }
        }
    } else {
        None
    };

    let stats = mem.stats()?;

    // Format output with next steps
    let filename = args.file.display();
    println!("✓ Created memory at {}", filename);
    if let Some((bound_id, bound_capacity)) = &binding_info {
        println!("  Bound to: {}", bound_id);
        println!(
            "  Capacity: {} (from dashboard)",
            format_bytes(*bound_capacity)
        );
    } else {
        println!("  Tier: Free");
        println!(
            "  Capacity: {} ({} bytes)",
            format_bytes(stats.capacity_bytes),
            stats.capacity_bytes
        );
    }
    println!("  Size: {}", format_bytes(stats.size_bytes));
    println!(
        "  Indexes: {} | {}",
        if lexical_enabled { "lexical" } else { "no-lex" },
        if vector_enabled { "vector" } else { "no-vec" }
    );
    println!();
    println!("Next steps:");
    println!("  memvid put {} --input <file>     # Add content", filename);
    println!("  memvid find {} --query <text>    # Search", filename);
    println!("  memvid stats {}                  # View stats", filename);
    println!();
    if binding_info.is_none() {
        println!("Tip: Bind to a dashboard memory to unlock your plan's capacity:");
        println!(
            "     memvid tickets sync {} --memory-id <MEMORY_ID>",
            filename
        );
    }
    println!("Documentation: https://docs.memvid.com/cli/tickets-and-capacity");
    Ok(())
}

/// Bind a memory file to a dashboard memory, syncing capacity ticket
pub fn bind_to_dashboard_memory(
    config: &CliConfig,
    mem: &mut Memvid,
    file_path: &PathBuf,
    memory_id: &MemoryId,
) -> Result<(String, u64)> {
    use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
    use chrono::Utc;
    use memvid_core::types::MemoryBinding;
    use memvid_core::verify_ticket_signature;

    use crate::api::{fetch_ticket as api_fetch_ticket, register_file, RegisterFileRequest};
    use crate::ticket_cache::{store as store_cached_ticket, CachedTicket};

    // Require API key for binding
    if config.api_key.is_none() {
        anyhow::bail!("API key required for dashboard binding. Set MEMVID_API_KEY.");
    }

    let pubkey = config
        .ticket_pubkey
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Ticket verification key not available"))?;

    // Fetch ticket from API
    let response =
        api_fetch_ticket(config, memory_id).context("failed to fetch ticket from dashboard")?;

    // Verify signature
    let signature_bytes = BASE64_STANDARD
        .decode(response.payload.signature.as_bytes())
        .context("ticket signature is not valid base64")?;

    verify_ticket_signature(
        pubkey,
        memory_id,
        &response.payload.issuer,
        response.payload.sequence,
        response.payload.expires_in,
        response.payload.capacity_bytes,
        &signature_bytes,
    )
    .context("ticket signature verification failed")?;

    // Create ticket
    let mut ticket = Ticket::new(&response.payload.issuer, response.payload.sequence)
        .expires_in_secs(response.payload.expires_in);
    if let Some(capacity) = response.payload.capacity_bytes {
        ticket = ticket.capacity_bytes(capacity);
    }

    // Create memory binding
    let binding = MemoryBinding {
        memory_id: **memory_id,
        memory_name: response.payload.issuer.clone(),
        bound_at: Utc::now(),
        api_url: config.api_url.clone(),
    };

    // Register file with dashboard FIRST to check for duplicate bindings
    let api_key = config
        .api_key
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("API key required for file registration"))?;
    let abs_path = std::fs::canonicalize(file_path).unwrap_or_else(|_| file_path.clone());
    let file_metadata = std::fs::metadata(file_path)?;
    let file_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown.mv2");
    let machine_id = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let abs_path_str = abs_path.to_string_lossy();
    let register_request = RegisterFileRequest {
        file_name,
        file_path: &abs_path_str,
        file_size: file_metadata.len() as i64,
        machine_id: &machine_id,
    };
    // File registration will fail with 409 if memory is already bound to a different file
    register_file(config, memory_id, &register_request, api_key)
        .context("failed to register file with dashboard")?;

    // Bind memory (stores both ticket and binding)
    mem.bind_memory(binding, ticket)
        .context("failed to bind memory")?;
    mem.commit()?;

    // Cache the ticket
    let cache_entry = CachedTicket {
        memory_id: **memory_id,
        issuer: response.payload.issuer.clone(),
        seq_no: response.payload.sequence,
        expires_in: response.payload.expires_in,
        capacity_bytes: response.payload.capacity_bytes,
        signature: response.payload.signature.clone(),
    };
    store_cached_ticket(config, &cache_entry)?;

    let capacity = mem.get_capacity();
    Ok((memory_id.to_string(), capacity))
}

/// Handler for `memvid open`
pub fn handle_open(_config: &CliConfig, args: OpenArgs) -> Result<()> {
    let mem = open_read_only_mem(&args.file)?;
    let stats = mem.stats()?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("Memory: {}", args.file.display());
        println!("Frames: {}", stats.frame_count);
        println!("Size: {} bytes", stats.size_bytes);
        println!("Tier: {:?}", stats.tier);
        println!(
            "Indices → lex: {}, vec: {}, time: {}",
            yes_no(stats.has_lex_index),
            yes_no(stats.has_vec_index),
            yes_no(stats.has_time_index)
        );
        if let Some(seq) = stats.seq_no {
            println!("Ticket sequence: {seq}");
        }
    }
    Ok(())
}

// Helper functions

pub fn apply_capacity_override(mem: &mut Memvid, capacity_bytes: u64) -> Result<()> {
    let current = mem.current_ticket();
    if current.capacity_bytes == capacity_bytes {
        return Ok(());
    }

    let seq = current.seq_no.saturating_add(1).max(1);
    let mut ticket = Ticket::new(current.issuer.clone(), seq).capacity_bytes(capacity_bytes);
    if current.expires_in_secs != 0 {
        ticket = ticket.expires_in_secs(current.expires_in_secs);
    }
    apply_ticket_with_warning(mem, ticket)?;
    Ok(())
}

/// Apply a ticket with a warning if capacity is reduced.
/// Uses the deprecated `apply_ticket` method for local/dev use cases
/// where cryptographic verification is not required.
#[allow(deprecated)]
pub fn apply_ticket_with_warning(mem: &mut Memvid, ticket: Ticket) -> Result<()> {
    let before = mem.stats()?.capacity_bytes;
    mem.apply_ticket(ticket)?;
    let after = mem.stats()?.capacity_bytes;
    if after < before {
        println!(
            "Warning: capacity reduced from {} to {}",
            format_bytes(before),
            format_bytes(after)
        );
    }
    Ok(())
}
