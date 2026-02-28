//! Ticket management command handlers (list, issue, revoke, sync, apply)

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use chrono::Utc;
use clap::{ArgAction, Args, Subcommand};
use memvid_core::types::{MemoryBinding, SignedTicket};
use memvid_core::{verify_ticket_signature, Memvid, Ticket};
use serde_json::json;
use uuid::Uuid;

/// A memory ID that accepts both UUID (32 chars) and MongoDB ObjectId (24 chars) formats
#[derive(Debug, Clone, Copy)]
pub struct MemoryId(Uuid);

impl serde::Serialize for MemoryId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl MemoryId {
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl std::ops::Deref for MemoryId {
    type Target = Uuid;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for MemoryId {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        // Try parsing as standard UUID first
        if let Ok(uuid) = Uuid::parse_str(s) {
            return Ok(MemoryId(uuid));
        }

        // If it's 24 chars (MongoDB ObjectId), pad with zeros to make it a valid UUID
        if s.len() == 24 && s.chars().all(|c| c.is_ascii_hexdigit()) {
            // Pad to 32 chars by appending zeros
            let padded = format!("{}00000000", s);
            if let Ok(uuid) = Uuid::parse_str(&padded) {
                return Ok(MemoryId(uuid));
            }
        }

        Err(format!(
            "invalid memory ID '{}': expected UUID (32 chars) or ObjectId (24 chars)",
            s
        ))
    }
}

use crate::api::{
    apply_ticket as api_apply_ticket, fetch_ticket as api_fetch_ticket,
    ApplyTicketRequest as ApiApplyTicketRequest,
};
use crate::commands::{apply_ticket_with_warning, LockCliArgs};
use crate::config::CliConfig;
use crate::ticket_cache::{load as load_cached_ticket, store as store_cached_ticket, CachedTicket};
use crate::utils::{apply_lock_cli, open_read_only_mem};

/// Subcommands for the `tickets` command
#[derive(Subcommand)]
pub enum TicketsCommand {
    /// Show the currently applied ticket
    List(TicketsStatusArgs),
    /// Apply a new ticket
    Issue(TicketsIssueArgs),
    /// Clear ticket metadata while advancing the sequence
    Revoke(TicketsRevokeArgs),
    /// Synchronise the ticket state with the Memvid API
    Sync(TicketsSyncArgs),
    /// Submit a ticket fetched from the API back to the control plane
    Apply(TicketsApplyArgs),
}

/// Arguments for the `tickets` command
#[derive(Args)]
pub struct TicketsArgs {
    #[command(subcommand)]
    pub command: TicketsCommand,
}

/// Arguments for the `tickets sync` subcommand
#[derive(Args)]
pub struct TicketsSyncArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(
        long = "memory-id",
        value_name = "ID",
        help = "Memory ID (UUID or 24-char ObjectId)"
    )]
    pub memory_id: MemoryId,
    #[arg(long)]
    pub json: bool,

    #[command(flatten)]
    pub lock: LockCliArgs,
}

/// Arguments for the `tickets apply` subcommand
#[derive(Args)]
pub struct TicketsApplyArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(
        long = "memory-id",
        value_name = "ID",
        help = "Memory ID (UUID or 24-char ObjectId)"
    )]
    pub memory_id: MemoryId,
    #[arg(long = "from-api", action = ArgAction::SetTrue)]
    pub from_api: bool,
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the `tickets issue` subcommand
#[derive(Args)]
pub struct TicketsIssueArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long)]
    pub issuer: String,
    #[arg(long, value_name = "SEQ")]
    pub seq: i64,
    #[arg(long = "expires-in", value_name = "SECS")]
    pub expires_in: Option<u64>,
    #[arg(long, value_name = "BYTES")]
    pub capacity: Option<u64>,
    #[arg(long)]
    pub json: bool,

    #[command(flatten)]
    pub lock: LockCliArgs,
}

/// Arguments for the `tickets revoke` subcommand
#[derive(Args)]
pub struct TicketsRevokeArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long)]
    pub json: bool,

    #[command(flatten)]
    pub lock: LockCliArgs,
}

/// Arguments for the `tickets list` subcommand
#[derive(Args)]
pub struct TicketsStatusArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long)]
    pub json: bool,
}

// ============================================================================
// Tickets command handlers
// ============================================================================

pub fn handle_tickets(config: &CliConfig, args: TicketsArgs) -> Result<()> {
    match args.command {
        TicketsCommand::List(status) => handle_ticket_status(status),
        TicketsCommand::Issue(issue) => handle_ticket_issue(issue),
        TicketsCommand::Revoke(revoke) => handle_ticket_revoke(revoke),
        TicketsCommand::Sync(sync) => handle_ticket_sync(config, sync),
        TicketsCommand::Apply(apply) => handle_ticket_apply(config, apply),
    }
}

fn ticket_public_key(config: &CliConfig) -> Result<&ed25519_dalek::VerifyingKey> {
    config
        .ticket_pubkey
        .as_ref()
        .ok_or_else(|| anyhow!("MEMVID_TICKET_PUBKEY is not set"))
}

fn ticket_to_json(ticket: &memvid_core::TicketRef) -> serde_json::Value {
    json!({
        "issuer": ticket.issuer,
        "seq_no": ticket.seq_no,
        "expires_in_secs": ticket.expires_in_secs,
        "capacity_bytes": ticket.capacity_bytes,
        "verified": ticket.verified,
    })
}

pub fn handle_ticket_status(args: TicketsStatusArgs) -> Result<()> {
    let mem = open_read_only_mem(&args.file)?;
    let ticket = mem.current_ticket();
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&ticket_to_json(&ticket))?
        );
    } else {
        println!("Ticket issuer: {}", ticket.issuer);
        println!("Sequence: {}", ticket.seq_no);
        println!("Expires in (secs): {}", ticket.expires_in_secs);
        if ticket.capacity_bytes != 0 {
            println!("Capacity bytes: {}", ticket.capacity_bytes);
        } else {
            println!("Capacity bytes: (tier default)");
        }
    }
    Ok(())
}

pub fn handle_ticket_issue(args: TicketsIssueArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;
    apply_lock_cli(&mut mem, &args.lock);
    let mut ticket = Ticket::new(&args.issuer, args.seq);
    if let Some(expires) = args.expires_in {
        ticket = ticket.expires_in_secs(expires);
    }
    if let Some(capacity) = args.capacity {
        ticket = ticket.capacity_bytes(capacity);
    }
    apply_ticket_with_warning(&mut mem, ticket)?;
    let ticket = mem.current_ticket();
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&ticket_to_json(&ticket))?
        );
    } else {
        println!(
            "Applied ticket seq={} issuer={}",
            ticket.seq_no, ticket.issuer
        );
    }
    Ok(())
}

pub fn handle_ticket_revoke(args: TicketsRevokeArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;
    apply_lock_cli(&mut mem, &args.lock);
    let current = mem.current_ticket();
    let next_seq = current.seq_no.saturating_add(1).max(1);
    let ticket = Ticket::new("", next_seq);
    apply_ticket_with_warning(&mut mem, ticket)?;
    let ticket = mem.current_ticket();
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&ticket_to_json(&ticket))?
        );
    } else {
        println!("Ticket cleared; seq advanced to {}", ticket.seq_no);
    }
    Ok(())
}

pub fn handle_ticket_sync(config: &CliConfig, args: TicketsSyncArgs) -> Result<()> {
    let pubkey = ticket_public_key(config)?;
    let response = api_fetch_ticket(config, &args.memory_id)?;

    // Get file info for registration
    let file_path = std::fs::canonicalize(&args.file)
        .with_context(|| format!("failed to get absolute path for {:?}", args.file))?;
    let file_metadata = std::fs::metadata(&args.file)
        .with_context(|| format!("failed to get file metadata for {:?}", args.file))?;
    let file_size = file_metadata.len() as i64;
    let file_name = args
        .file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown.mv2");
    let machine_id = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let signature_bytes = BASE64_STANDARD
        .decode(response.payload.signature.as_bytes())
        .context("ticket signature is not valid base64")?;

    // Extract values from response
    let issuer = response.payload.issuer.clone();
    let seq_no = response.payload.sequence;
    let expires_in = response.payload.expires_in;
    let capacity_bytes = response.payload.capacity_bytes;

    verify_ticket_signature(
        pubkey,
        &args.memory_id,
        &issuer,
        seq_no,
        expires_in,
        capacity_bytes,
        &signature_bytes,
    )
    .context("ticket signature verification failed")?;

    let mut mem = Memvid::open(&args.file)?;
    apply_lock_cli(&mut mem, &args.lock);

    // Check if the file already has this ticket or a newer one
    let current = mem.current_ticket();
    if current.seq_no >= seq_no {
        // Already bound with same or newer ticket
        let capacity = mem.get_capacity();

        // Still register/update file with the dashboard
        let file_path_str = file_path.to_string_lossy();
        let register_request = crate::api::RegisterFileRequest {
            file_name,
            file_path: &file_path_str,
            file_size,
            machine_id: &machine_id,
        };
        if let Some(api_key) = config.api_key.as_deref() {
            if let Err(e) =
                crate::api::register_file(config, &args.memory_id, &register_request, api_key)
            {
                log::warn!("Failed to register file with dashboard: {}", e);
            }
        }

        if args.json {
            let json = json!({
                "issuer": current.issuer,
                "seq_no": current.seq_no,
                "expires_in": current.expires_in_secs,
                "capacity_bytes": if current.capacity_bytes > 0 { Some(current.capacity_bytes) } else { None },
                "memory_id": args.memory_id,
                "verified": current.verified,
                "already_bound": true,
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        } else {
            let verified_str = if current.verified { " ✓" } else { "" };
            println!(
                "Already bound to memory {} (seq={}, issuer={}{})",
                args.memory_id, current.seq_no, current.issuer, verified_str
            );
            println!(
                "Current capacity: {:.2} GB",
                capacity as f64 / 1024.0 / 1024.0 / 1024.0
            );
        }
        return Ok(());
    }

    // Register file with the dashboard FIRST (before binding locally)
    // This ensures only one file can be bound to each memory
    let file_path_str = file_path.to_string_lossy();
    let register_request = crate::api::RegisterFileRequest {
        file_name,
        file_path: &file_path_str,
        file_size,
        machine_id: &machine_id,
    };
    if let Some(api_key) = config.api_key.as_deref() {
        crate::api::register_file(config, &args.memory_id, &register_request, api_key)
            .context("failed to register file with dashboard - this memory may already be bound to another file")?;
    } else {
        bail!("API key required for binding. Set MEMVID_API_KEY.");
    }

    // Create memory binding first if not already bound
    let binding = MemoryBinding {
        memory_id: *args.memory_id,
        memory_name: issuer.clone(), // Use issuer as name for now
        bound_at: Utc::now(),
        api_url: config.api_url.clone(),
    };

    // Ensure memory is bound before applying signed ticket
    if mem.get_memory_binding().is_none() {
        // Create a temporary ticket for binding (seq-1 to allow signed ticket to apply)
        let temp_ticket =
            Ticket::new(&issuer, seq_no.saturating_sub(1)).expires_in_secs(expires_in);
        mem.bind_memory(binding, temp_ticket)
            .context("failed to bind memory")?;
    }

    // Create and apply signed ticket with cryptographic verification
    let signed_ticket = SignedTicket::new(
        &issuer,
        seq_no,
        expires_in,
        capacity_bytes,
        *args.memory_id,
        signature_bytes,
    );

    mem.apply_signed_ticket(signed_ticket)
        .context("failed to apply signed ticket")?;
    mem.commit()?;

    let cache_entry = CachedTicket {
        memory_id: *args.memory_id,
        issuer: issuer.clone(),
        seq_no,
        expires_in,
        capacity_bytes,
        signature: response.payload.signature.clone(),
    };
    store_cached_ticket(config, &cache_entry)?;

    let capacity = mem.get_capacity();

    if args.json {
        let json = json!({
            "issuer": issuer,
            "seq_no": seq_no,
            "expires_in": expires_in,
            "capacity_bytes": capacity_bytes,
            "memory_id": args.memory_id,
            "request_id": response.request_id,
            "verified": true,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!(
            "Bound to memory {} (seq={}, issuer={}) ✓",
            args.memory_id, seq_no, issuer
        );
        println!(
            "New capacity: {:.2} GB",
            capacity as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }
    Ok(())
}

pub fn handle_ticket_apply(config: &CliConfig, args: TicketsApplyArgs) -> Result<()> {
    if !args.from_api {
        bail!("use --from-api to submit the ticket fetched from the API");
    }
    let cached = load_cached_ticket(config, &args.memory_id)
        .context("no cached ticket available; run `tickets sync` first")?;
    let request = ApiApplyTicketRequest {
        issuer: &cached.issuer,
        seq_no: cached.seq_no,
        expires_in: cached.expires_in,
        capacity_bytes: cached.capacity_bytes,
        signature: &cached.signature,
    };
    let request_id = api_apply_ticket(config, &args.memory_id, &request)?;
    if args.json {
        let json = json!({
            "memory_id": args.memory_id,
            "seq_no": cached.seq_no,
            "request_id": request_id,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!(
            "Submitted ticket seq={} for memory {} (request {})",
            cached.seq_no, args.memory_id, request_id
        );
    }
    Ok(())
}
