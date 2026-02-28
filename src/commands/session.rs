//! CLI commands for time-travel replay session management.
//!
//! These commands enable recording and managing agent sessions for
//! debugging and testing purposes.

use anyhow::Result;
use clap::{Args, Subcommand};
use colored::Colorize;
#[cfg(feature = "replay")]
use memvid_ask_model::run_model_inference;
#[cfg(feature = "replay")]
use memvid_core::replay::ActionType;
use memvid_core::Memvid;

/// Session management commands for time-travel replay
#[derive(Args, Debug)]
pub struct SessionArgs {
    #[command(subcommand)]
    pub command: SessionCommand,
}

#[derive(Subcommand, Debug)]
pub enum SessionCommand {
    /// Start recording a new session
    Start(SessionStartArgs),
    /// End the current recording session
    End(SessionEndArgs),
    /// List all recorded sessions
    List(SessionListArgs),
    /// View details of a specific session
    View(SessionViewArgs),
    /// Create a checkpoint in the current session
    Checkpoint(SessionCheckpointArgs),
    /// Delete a recorded session
    Delete(SessionDeleteArgs),
    /// Save sessions to the memory file
    Save(SessionSaveArgs),
    /// Load sessions from the memory file
    Load(SessionLoadArgs),
    /// Replay a recorded session and verify consistency
    Replay(SessionReplayArgs),
    /// Compare two recorded sessions
    Compare(SessionCompareArgs),
}

#[derive(Args, Debug)]
pub struct SessionStartArgs {
    /// Path to the memory file
    pub file: String,

    /// Optional name for the session
    #[arg(short, long)]
    pub name: Option<String>,

    /// Auto-checkpoint interval (number of actions between checkpoints, 0 = disabled)
    #[arg(long, default_value = "0")]
    pub auto_checkpoint: u64,
}

#[derive(Args, Debug)]
pub struct SessionEndArgs {
    /// Path to the memory file
    pub file: String,
}

#[derive(Args, Debug)]
pub struct SessionListArgs {
    /// Path to the memory file
    pub file: String,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: String,
}

#[derive(Args, Debug)]
pub struct SessionViewArgs {
    /// Path to the memory file
    pub file: String,

    /// Session ID to view
    #[arg(short, long)]
    pub session: String,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: String,
}

#[derive(Args, Debug)]
pub struct SessionCheckpointArgs {
    /// Path to the memory file
    pub file: String,
}

#[derive(Args, Debug)]
pub struct SessionDeleteArgs {
    /// Path to the memory file
    pub file: String,

    /// Session ID to delete
    #[arg(short, long)]
    pub session: String,
}

#[derive(Args, Debug)]
pub struct SessionSaveArgs {
    /// Path to the memory file
    pub file: String,
}

#[derive(Args, Debug)]
pub struct SessionLoadArgs {
    /// Path to the memory file
    pub file: String,
}

#[derive(Args, Debug)]
pub struct SessionReplayArgs {
    /// Path to the memory file
    pub file: String,

    /// Session ID to replay
    #[arg(short, long)]
    pub session: String,

    /// Start replay from a specific checkpoint
    #[arg(long)]
    pub from_checkpoint: Option<u64>,

    /// Number of results to retrieve during replay (default: same as original)
    /// Use higher values to discover documents that were missed due to low top-k
    #[arg(short = 'k', long)]
    pub top_k: Option<usize>,

    /// Use adaptive retrieval that adjusts top-k based on score distribution
    #[arg(long)]
    pub adaptive: bool,

    /// Minimum relevancy score for adaptive retrieval (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub min_relevancy: f32,

    /// Skip put/update/delete actions
    #[arg(long)]
    pub skip_mutations: bool,

    /// Skip find/search actions
    #[arg(long)]
    pub skip_finds: bool,

    /// Skip ask actions (LLM calls)
    #[arg(long)]
    pub skip_asks: bool,

    /// Stop on first mismatch
    #[arg(long)]
    pub stop_on_mismatch: bool,

    /// AUDIT MODE: Use frozen retrieval for ASK actions
    /// Replays use recorded frame IDs instead of re-executing search
    #[arg(long)]
    pub audit: bool,

    /// Override model for audit replay (format: "provider:model")
    /// When set, prepares for LLM re-execution with frozen context
    #[arg(long)]
    pub use_model: Option<String>,

    /// Generate diff report comparing original vs new answers
    #[arg(long)]
    pub diff: bool,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: String,

    /// Launch the Time Machine web UI for visual replay
    #[cfg(feature = "web")]
    #[arg(long)]
    pub web: bool,

    /// Port for the Time Machine web server (default: 4242)
    #[cfg(feature = "web")]
    #[arg(long, default_value = "4242")]
    pub port: u16,

    /// Don't auto-open browser when starting web UI
    #[cfg(feature = "web")]
    #[arg(long)]
    pub no_open: bool,
}

#[derive(Args, Debug)]
pub struct SessionCompareArgs {
    /// Path to the memory file
    pub file: String,

    /// First session ID
    #[arg(short = 'a', long)]
    pub session_a: String,

    /// Second session ID
    #[arg(short = 'b', long)]
    pub session_b: String,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: String,
}

/// Handle session subcommands
#[cfg(feature = "replay")]
pub fn handle_session(args: SessionArgs) -> Result<()> {
    match args.command {
        SessionCommand::Start(args) => handle_session_start(args),
        SessionCommand::End(args) => handle_session_end(args),
        SessionCommand::List(args) => handle_session_list(args),
        SessionCommand::View(args) => handle_session_view(args),
        SessionCommand::Checkpoint(args) => handle_session_checkpoint(args),
        SessionCommand::Delete(args) => handle_session_delete(args),
        SessionCommand::Save(args) => handle_session_save(args),
        SessionCommand::Load(args) => handle_session_load(args),
        SessionCommand::Replay(args) => handle_session_replay(args),
        SessionCommand::Compare(args) => handle_session_compare(args),
    }
}

#[cfg(not(feature = "replay"))]
pub fn handle_session(_args: SessionArgs) -> Result<()> {
    bail!("Session recording requires the 'replay' feature. Rebuild with --features replay")
}

#[cfg(feature = "replay")]
fn handle_session_start(args: SessionStartArgs) -> Result<()> {
    use memvid_core::replay::ReplayConfig;

    let mut mem = Memvid::open(&args.file)?;

    // Check if there's already an active session
    if mem.load_active_session()? {
        let session_id = mem.active_session_id().unwrap();
        anyhow::bail!(
            "Session {} is already active. End it first with `session end`.",
            session_id
        );
    }

    let config = ReplayConfig {
        auto_checkpoint_interval: args.auto_checkpoint,
        ..Default::default()
    };

    let session_id = mem.start_session(args.name, Some(config))?;

    // Save the active session to a sidecar file so it persists across commands
    mem.save_active_session()?;

    println!("Started session: {}", session_id);
    println!("Recording is now active. Run operations like put, find, ask.");
    println!("End the session with: memvid session end {}", args.file);

    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_end(args: SessionEndArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    // Load the active session from the sidecar file
    if !mem.load_active_session()? {
        anyhow::bail!("No active session found. Start one with `session start`.");
    }

    let session = mem.end_session()?;
    println!("Ended session: {}", session.session_id);
    println!("  Actions recorded: {}", session.actions.len());
    println!("  Checkpoints: {}", session.checkpoints.len());
    println!("  Duration: {}s", session.duration_secs());

    // First commit to finalize any pending Tantivy/index data
    // This ensures data_end is accurate before saving replay segment
    mem.commit()?;

    // Now save the replay sessions after all other data is finalized
    mem.save_replay_sessions()?;

    // Commit again to persist the replay manifest in TOC
    mem.commit()?;

    // Remove the sidecar file
    mem.clear_active_session_file()?;

    println!("Session saved to file.");
    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_list(args: SessionListArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    // Check for active session
    let has_active = mem.load_active_session()?;

    // Load completed sessions from file
    mem.load_replay_sessions()?;

    let sessions = mem.list_sessions();

    if !has_active && sessions.is_empty() {
        println!("No recorded sessions.");
        return Ok(());
    }

    if args.format == "json" {
        let json = serde_json::to_string_pretty(&sessions)?;
        println!("{}", json);
    } else {
        // Show active session first if present
        if has_active {
            if let Some(session_id) = mem.active_session_id() {
                println!("Active session (recording):");
                println!("  {} - currently recording", session_id);
                println!();
            }
        }

        if !sessions.is_empty() {
            println!("Completed sessions ({}):", sessions.len());
            for summary in sessions {
                let name = summary.name.as_deref().unwrap_or("(unnamed)");
                let status = if summary.ended_secs.is_some() {
                    "completed"
                } else {
                    "recording"
                };
                println!(
                    "  {} - {} [{}, {} actions, {} checkpoints]",
                    summary.session_id,
                    name,
                    status,
                    summary.action_count,
                    summary.checkpoint_count
                );
            }
        }
    }

    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_view(args: SessionViewArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    // Load sessions from file
    mem.load_replay_sessions()?;

    let session_id: uuid::Uuid = args.session.parse()?;
    let session = mem
        .get_session(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;

    if args.format == "json" {
        let json = serde_json::to_string_pretty(&session)?;
        println!("{}", json);
    } else {
        let name = session.name.as_deref().unwrap_or("(unnamed)");
        println!("Session: {} - {}", session.session_id, name);
        println!("  Created: {}", session.created_secs);
        if let Some(ended) = session.ended_secs {
            println!("  Ended: {}", ended);
            println!("  Duration: {}s", session.duration_secs());
        } else {
            println!("  Status: recording");
        }
        println!("  Actions: {}", session.actions.len());
        println!("  Checkpoints: {}", session.checkpoints.len());

        if !session.actions.is_empty() {
            println!("\nActions:");
            for (i, action) in session.actions.iter().enumerate().take(20) {
                println!(
                    "  [{}] {} - {:?}",
                    i,
                    action.action_type.name(),
                    action.action_type
                );
            }
            if session.actions.len() > 20 {
                println!("  ... and {} more", session.actions.len() - 20);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_checkpoint(args: SessionCheckpointArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    let checkpoint_id = mem.create_checkpoint()?;
    println!("Created checkpoint: {}", checkpoint_id);

    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_delete(args: SessionDeleteArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    // Load sessions from file
    mem.load_replay_sessions()?;

    let session_id: uuid::Uuid = args.session.parse()?;
    mem.delete_session(session_id)?;

    // Save updated sessions
    mem.save_replay_sessions()?;
    mem.commit()?;

    println!("Deleted session: {}", session_id);
    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_save(args: SessionSaveArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    // Load any existing sessions from file first
    mem.load_replay_sessions()?;

    // Check if there are any sessions to save
    let session_count = mem.list_sessions().len();
    if session_count == 0 {
        println!(
            "No sessions to save. Sessions are automatically saved when you run `session end`."
        );
        return Ok(());
    }

    // Re-save all sessions (useful if you added sessions via API or need to re-persist)
    mem.save_replay_sessions()?;
    mem.commit()?;

    println!("Saved {} session(s) to file.", session_count);
    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_load(args: SessionLoadArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    mem.load_replay_sessions()?;

    let sessions = mem.list_sessions();
    println!("Loaded {} sessions from file.", sessions.len());

    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_replay(args: SessionReplayArgs) -> Result<()> {
    // Check if web UI is requested
    #[cfg(feature = "web")]
    if args.web {
        use crate::commands::session_web::start_web_server;
        use std::path::PathBuf;

        let rt = tokio::runtime::Runtime::new()?;
        return rt.block_on(start_web_server(
            args.session.clone(),
            PathBuf::from(&args.file),
            args.port,
            !args.no_open,
        ));
    }

    use memvid_core::replay::{ReplayEngine, ReplayExecutionConfig};

    // Use open_read_only to match how find command opens files
    // This ensures lex_enabled is properly set from has_lex_index()
    let mut mem = Memvid::open_read_only(&args.file)?;

    // Load sessions from file
    mem.load_replay_sessions()?;

    let session_id: uuid::Uuid = args.session.parse()?;
    let session = mem
        .get_session(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?
        .clone();

    let config = ReplayExecutionConfig {
        skip_puts: args.skip_mutations,
        skip_finds: args.skip_finds,
        skip_asks: args.skip_asks,
        stop_on_mismatch: args.stop_on_mismatch,
        verbose: true,
        top_k: args.top_k,
        adaptive: args.adaptive,
        min_relevancy: args.min_relevancy,
        audit_mode: args.audit,
        use_model: args.use_model.clone(),
        generate_diff: args.diff,
    };

    let mut engine = ReplayEngine::new(&mut mem, config);
    let result = engine.replay_session_from(&session, args.from_checkpoint)?;

    if args.format == "json" {
        let json = serde_json::to_string_pretty(&result)?;
        println!("{}", json);
    } else {
        println!();
        println!("{}", "━".repeat(60).dimmed());
        println!(
            "{} {}",
            "Replaying session:".bold(),
            session_id.to_string().cyan()
        );
        println!("{} {}", "File:".dimmed(), args.file);

        // Show mode based on flags
        let mode_str = if args.audit {
            "Audit (frozen retrieval)".green().to_string()
        } else {
            "Debug (showing recorded actions)".white().to_string()
        };
        println!("{} {}", "Mode:".dimmed(), mode_str);

        // Show model override if set
        if let Some(ref model) = args.use_model {
            println!("{} {}", "Model Override:".dimmed(), model.yellow());
        }

        println!("{}", "━".repeat(60).dimmed());
        println!();

        // Show all actions with their details
        let total = result.action_results.len();
        for (action_idx, action_result) in result.action_results.iter().enumerate() {
            let seq = action_result.sequence;
            let action_type = &action_result.action_type;
            let diff = action_result.diff.as_deref().unwrap_or("");

            // Skip "skipped" actions in display unless they're ASK
            if diff == "skipped" && action_type != "ASK" {
                continue;
            }

            // Format based on action type
            if action_type == "ASK" {
                // Rich output for ASK actions
                let status = if action_result.matched {
                    "✓".green()
                } else {
                    "✗".red()
                };
                println!(
                    "{} Step {}/{} {} {}",
                    status,
                    (seq + 1).to_string().yellow(),
                    total,
                    "ask".cyan().bold(),
                    ""
                );
                // Print multiline diff with proper indentation
                for line in diff.lines() {
                    println!("         {}", line);
                }

                // If audit mode with use_model, re-execute LLM with frozen context
                if args.audit && args.use_model.is_some() {
                    if let Some(original_action) = session.actions.get(action_idx) {
                        if let ActionType::Ask { query, .. } = &original_action.action_type {
                            let frame_ids = &original_action.affected_frames;
                            if !frame_ids.is_empty() {
                                // Build context from frozen frames
                                let mut context_parts = Vec::new();
                                for (idx, frame_id) in frame_ids.iter().enumerate() {
                                    if let Ok(content) = mem.frame_text_by_id(*frame_id) {
                                        // Truncate to reasonable size
                                        let snippet = if content.len() > 1000 {
                                            format!("{}...", &content[..1000])
                                        } else {
                                            content
                                        };
                                        context_parts.push(format!("[{}] {}", idx + 1, snippet));
                                    }
                                }
                                let frozen_context = context_parts.join("\n\n");

                                // Call LLM with override model
                                let model_name = args.use_model.as_ref().unwrap();
                                match run_model_inference(
                                    model_name,
                                    query,
                                    &frozen_context,
                                    &[], // No hits needed
                                    None,
                                    None,
                                    None,
                                ) {
                                    Ok(inference) => {
                                        let new_answer = &inference.answer.answer;
                                        let original_answer = &original_action.output_preview;

                                        // Show the new answer
                                        println!(
                                            "         {} \"{}\"",
                                            "New Answer:".green().bold(),
                                            if new_answer.len() > 150 {
                                                format!("{}...", &new_answer[..150])
                                            } else {
                                                new_answer.clone()
                                            }
                                        );

                                        // Show diff if requested
                                        if args.diff {
                                            let same = new_answer.trim() == original_answer.trim();
                                            if same {
                                                println!(
                                                    "         {} {}",
                                                    "Diff:".dimmed(),
                                                    "IDENTICAL".green()
                                                );
                                            } else {
                                                println!(
                                                    "         {} {}",
                                                    "Diff:".dimmed(),
                                                    "CHANGED".yellow()
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        println!("         {} {}", "LLM Error:".red(), e);
                                    }
                                }
                            }
                        }
                    }
                }
                println!();
            } else if action_type == "FIND" && diff.contains("DISCOVERY:")
                || diff.contains("FILTER:")
            {
                // Rich output for FIND with discoveries
                let colored_diff = if diff.starts_with("FILTER:") {
                    diff.replace("FILTER:", &"FILTER:".red().bold().to_string())
                        .replace("would be MISSED", &"would be MISSED".red().to_string())
                } else if diff.starts_with("DISCOVERY:") {
                    diff.replace("DISCOVERY:", &"DISCOVERY:".green().bold().to_string())
                        .replace("[NEW]", &"[NEW]".green().to_string())
                } else {
                    diff.to_string()
                };
                println!(
                    "  Step {}/{} {} - {}",
                    (seq + 1).to_string().yellow(),
                    total,
                    action_type.cyan(),
                    colored_diff
                );
            } else if !diff.is_empty() && diff != "skipped" {
                // Standard output for other actions with diffs
                let status = if action_result.matched { "✓" } else { "✗" };
                println!(
                    "  {} Step {}/{} {} - {}",
                    if action_result.matched {
                        status.green()
                    } else {
                        status.red()
                    },
                    (seq + 1).to_string().yellow(),
                    total,
                    action_type.cyan(),
                    diff.dimmed()
                );
            }
        }

        println!("{}", "━".repeat(60).dimmed());
        println!();
        println!("{}", "Summary:".bold());
        println!(
            "  Total actions: {}",
            result.total_actions.to_string().white()
        );
        println!("  Matched: {}", result.matched_actions.to_string().green());
        println!(
            "  Mismatched: {}",
            result.mismatched_actions.to_string().red()
        );
        println!("  Skipped: {}", result.skipped_actions.to_string().yellow());
        println!(
            "  Match rate: {}",
            format!("{:.1}%", result.match_rate()).cyan()
        );
        println!("  Duration: {}ms", result.total_duration_ms);

        if let Some(cp) = result.from_checkpoint {
            println!("  Started from checkpoint: {}", cp);
        }

        println!();
        if result.is_success() {
            println!("{}", "✓ Replay completed successfully".green().bold());
        } else {
            println!("{}", "✗ Replay found mismatches".red().bold());
        }
    }

    Ok(())
}

#[cfg(feature = "replay")]
fn handle_session_compare(args: SessionCompareArgs) -> Result<()> {
    use memvid_core::replay::ReplayEngine;

    let mut mem = Memvid::open(&args.file)?;

    // Load sessions from file
    mem.load_replay_sessions()?;

    let session_a_id: uuid::Uuid = args.session_a.parse()?;
    let session_b_id: uuid::Uuid = args.session_b.parse()?;

    let session_a = mem
        .get_session(session_a_id)
        .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_a_id))?
        .clone();

    let session_b = mem
        .get_session(session_b_id)
        .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_b_id))?
        .clone();

    let comparison = ReplayEngine::compare_sessions(&session_a, &session_b);

    if args.format == "json" {
        let json = serde_json::to_string_pretty(&comparison)?;
        println!("{}", json);
    } else {
        println!("Session Comparison:");
        println!("  Session A: {}", session_a_id);
        println!("  Session B: {}", session_b_id);
        println!();

        if comparison.is_identical() {
            println!("✓ Sessions are identical");
            println!("  Matching actions: {}", comparison.matching_actions);
        } else {
            println!("✗ Sessions differ:");
            println!("  Matching actions: {}", comparison.matching_actions);

            if !comparison.actions_only_in_a.is_empty() {
                println!("  Actions only in A: {:?}", comparison.actions_only_in_a);
            }

            if !comparison.actions_only_in_b.is_empty() {
                println!("  Actions only in B: {:?}", comparison.actions_only_in_b);
            }

            if !comparison.differing_actions.is_empty() {
                println!("  Differing actions:");
                for diff in &comparison.differing_actions {
                    println!(
                        "    [{}] {} vs {} - {}",
                        diff.sequence, diff.action_type_a, diff.action_type_b, diff.description
                    );
                }
            }
        }
    }

    Ok(())
}
