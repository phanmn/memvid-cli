//! Maintenance command handlers (verify, doctor, verify-single-file, nudge, process-queue).
//! These commands surface integrity checks, healing workflows, and lock inspection
//! for `.mv2` files without mutating unrelated state.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use clap::{ArgAction, Args};
use memvid_core::{
    lockfile, DoctorActionDetail, DoctorActionKind, DoctorFindingCode, DoctorOptions,
    DoctorPhaseKind, DoctorReport, DoctorSeverity, DoctorStatus, Memvid, VerificationStatus,
};
use serde::Serialize;

use crate::config::CliConfig;

/// Arguments for the `nudge` subcommand
#[derive(Args)]
pub struct NudgeArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
}

/// Arguments for the `verify` subcommand
#[derive(Args)]
pub struct VerifyArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long)]
    pub deep: bool,
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the `doctor` subcommand
#[derive(Args)]
pub struct DoctorArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long = "rebuild-time-index", action = ArgAction::SetTrue)]
    pub rebuild_time_index: bool,
    #[arg(long = "rebuild-lex-index", action = ArgAction::SetTrue)]
    pub rebuild_lex_index: bool,
    #[arg(long = "rebuild-vec-index", action = ArgAction::SetTrue)]
    pub rebuild_vec_index: bool,
    #[arg(long = "vacuum", action = ArgAction::SetTrue)]
    pub vacuum: bool,
    #[arg(long = "plan-only", action = ArgAction::SetTrue)]
    pub plan_only: bool,
    #[arg(long = "json", action = ArgAction::SetTrue)]
    pub json: bool,
}

/// Arguments for the `verify-single-file` subcommand
#[derive(Args)]
pub struct VerifySingleFileArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
}

// ============================================================================
// Maintenance command handlers
// ============================================================================

pub fn handle_verify(_config: &CliConfig, args: VerifyArgs) -> Result<()> {
    let report = Memvid::verify(&args.file, args.deep)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Verification report for {}", args.file.display());
        for check in &report.checks {
            match &check.details {
                Some(details) => println!("- {}: {:?} ({details})", check.name, check.status),
                None => println!("- {}: {:?}", check.name, check.status),
            }
        }
        println!("Overall: {:?}", report.overall_status);
    }

    if report.overall_status == VerificationStatus::Failed {
        anyhow::bail!("verification failed");
    }
    Ok(())
}

pub fn handle_doctor(_config: &CliConfig, args: DoctorArgs) -> Result<()> {
    let options = DoctorOptions {
        rebuild_time_index: args.rebuild_time_index,
        rebuild_lex_index: args.rebuild_lex_index,
        rebuild_vec_index: args.rebuild_vec_index,
        vacuum: args.vacuum,
        dry_run: args.plan_only,
        quiet: false,
    };

    let report = Memvid::doctor(&args.file, options)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_doctor_report(&args.file, &report);
    }

    match report.status {
        DoctorStatus::Failed => anyhow::bail!("doctor failed; see findings for details"),
        DoctorStatus::Partial => {
            anyhow::bail!("doctor completed with partial repairs; rerun or restore from backup")
        }
        DoctorStatus::PlanOnly => {
            if report.plan.is_noop() && !args.json {
                println!(
                    "No repairs required for {} (plan-only run)",
                    args.file.display()
                );
            } else if !args.json {
                println!("Plan generated. Re-run without --plan-only to apply repairs.");
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn print_doctor_report(path: &Path, report: &DoctorReport) {
    println!("Doctor status for {}: {:?}", path.display(), report.status);

    if !report.plan.findings.is_empty() {
        println!("Findings:");
        for finding in &report.plan.findings {
            let severity = format_severity(finding.severity);
            let code = format_finding_code(finding.code);
            match &finding.detail {
                Some(detail) => println!(
                    "  - [{}] {}: {} ({detail})",
                    severity, code, finding.message
                ),
                None => println!("  - [{}] {}: {}", severity, code, finding.message),
            }
        }
    }

    if report.plan.phases.is_empty() {
        println!("Planned phases: (none)");
    } else {
        println!("Planned phases:");
        for phase in &report.plan.phases {
            println!("  - {}", label_phase(phase.phase));
            for action in &phase.actions {
                let mut notes: Vec<String> = Vec::new();
                if action.required {
                    notes.push("required".into());
                }
                if !action.reasons.is_empty() {
                    let reasons: Vec<String> = action
                        .reasons
                        .iter()
                        .map(|code| format_finding_code(*code))
                        .collect();
                    notes.push(format!("reasons: {}", reasons.join(", ")));
                }
                if let Some(detail) = &action.detail {
                    notes.push(format_action_detail(detail));
                }
                if let Some(note) = &action.note {
                    notes.push(note.clone());
                }
                let suffix = if notes.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", notes.join(" | "))
                };
                println!("    * {}{}", label_action(action.action), suffix);
            }
        }
    }

    if report.phases.is_empty() {
        println!("Execution: (skipped)");
    } else {
        println!("Execution:");
        for phase in &report.phases {
            println!("  - {}: {:?}", label_phase(phase.phase), phase.status);
            if let Some(duration) = phase.duration_ms {
                println!("      duration: {} ms", duration);
            }
            for action in &phase.actions {
                match &action.detail {
                    Some(detail) => println!(
                        "    * {}: {:?} ({detail})",
                        label_action(action.action),
                        action.status
                    ),
                    None => println!("    * {}: {:?}", label_action(action.action), action.status),
                }
            }
        }
    }

    if report.metrics.total_duration_ms > 0 {
        println!("Total duration: {} ms", report.metrics.total_duration_ms);
    }
}

fn format_severity(severity: DoctorSeverity) -> &'static str {
    match severity {
        DoctorSeverity::Info => "info",
        DoctorSeverity::Warning => "warning",
        DoctorSeverity::Error => "error",
    }
}

fn format_finding_code(code: DoctorFindingCode) -> String {
    serde_json::to_string(&code)
        .map(|value| value.trim_matches('"').replace('_', " "))
        .unwrap_or_else(|_| format!("{code:?}"))
}

fn label_phase(kind: DoctorPhaseKind) -> &'static str {
    match kind {
        DoctorPhaseKind::Probe => "probe",
        DoctorPhaseKind::HeaderHealing => "header healing",
        DoctorPhaseKind::WalReplay => "wal replay",
        DoctorPhaseKind::IndexRebuild => "index rebuild",
        DoctorPhaseKind::Vacuum => "vacuum",
        DoctorPhaseKind::Finalize => "finalize",
        DoctorPhaseKind::Verify => "verify",
    }
}

fn label_action(kind: DoctorActionKind) -> &'static str {
    match kind {
        DoctorActionKind::HealHeaderPointer => "heal header pointer",
        DoctorActionKind::HealTocChecksum => "heal toc checksum",
        DoctorActionKind::ReplayWal => "replay wal",
        DoctorActionKind::DiscardWal => "discard wal",
        DoctorActionKind::RebuildTimeIndex => "rebuild time index",
        DoctorActionKind::RebuildLexIndex => "rebuild lex index",
        DoctorActionKind::RebuildVecIndex => "rebuild vector index",
        DoctorActionKind::VacuumCompaction => "vacuum compaction",
        DoctorActionKind::RecomputeToc => "recompute toc",
        DoctorActionKind::UpdateHeader => "update header",
        DoctorActionKind::DeepVerify => "deep verify",
        DoctorActionKind::NoOp => "no-op",
    }
}

fn format_action_detail(detail: &DoctorActionDetail) -> String {
    match detail {
        DoctorActionDetail::HeaderPointer {
            target_footer_offset,
        } => {
            format!("target offset: {}", target_footer_offset)
        }
        DoctorActionDetail::TocChecksum { expected } => {
            let checksum: String = expected.iter().map(|b| format!("{:02x}", b)).collect();
            format!("expected checksum: {}", checksum)
        }
        DoctorActionDetail::WalReplay {
            from_sequence,
            to_sequence,
            pending_records,
        } => format!(
            "apply wal records {}→{} ({} pending)",
            from_sequence, to_sequence, pending_records
        ),
        DoctorActionDetail::TimeIndex { expected_entries } => {
            format!("expected entries: {}", expected_entries)
        }
        DoctorActionDetail::LexIndex { expected_docs } => {
            format!("expected docs: {}", expected_docs)
        }
        DoctorActionDetail::VecIndex {
            expected_vectors,
            dimension,
        } => format!(
            "expected vectors: {}, dimension: {}",
            expected_vectors, dimension
        ),
        DoctorActionDetail::VacuumStats { active_frames } => {
            format!("active frames: {}", active_frames)
        }
    }
}

pub fn handle_verify_single_file(_config: &CliConfig, args: VerifySingleFileArgs) -> Result<()> {
    let offenders = find_auxiliary_files(&args.file)?;
    if offenders.is_empty() {
        println!(
            "\u{2713} Single file guarantee maintained ({})",
            args.file.display()
        );
        Ok(())
    } else {
        println!("Found auxiliary files:");
        for path in &offenders {
            println!("- {}", path.display());
        }
        anyhow::bail!("auxiliary files detected")
    }
}

pub fn handle_nudge(args: NudgeArgs) -> Result<()> {
    match lockfile::current_owner(&args.file)? {
        Some(owner) => {
            if let Some(pid) = owner.pid {
                #[cfg(unix)]
                {
                    let result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGUSR1) };
                    if result == 0 {
                        println!("Sent SIGUSR1 to process {pid}");
                    } else {
                        return Err(std::io::Error::last_os_error().into());
                    }
                }
                #[cfg(not(unix))]
                {
                    println!(
                        "Active writer pid {pid}; nudging is not supported on this platform. Notify the process manually."
                    );
                }
            } else {
                bail!("Active writer does not expose a pid; cannot nudge");
            }
        }
        None => {
            println!("No active writer for {}", args.file.display());
        }
    }
    Ok(())
}

fn find_auxiliary_files(memory: &Path) -> Result<Vec<PathBuf>> {
    let parent = memory
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let name = memory
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("memory path must be a valid file name"))?;

    let mut offenders = Vec::new();
    let forbidden = ["-wal", "-shm", "-lock", "-journal"];
    for suffix in &forbidden {
        let candidate = parent.join(format!("{name}{suffix}"));
        if candidate.exists() {
            offenders.push(candidate);
        }
    }
    let hidden_forbidden = [".wal", ".shm", ".lock", ".journal"];
    for suffix in &hidden_forbidden {
        let candidate = parent.join(format!(".{name}{suffix}"));
        if candidate.exists() {
            offenders.push(candidate);
        }
    }
    Ok(offenders)
}

// ============================================================================
// Process Queue Command - Progressive Ingestion Enrichment
// ============================================================================

/// Arguments for the `process-queue` subcommand
#[derive(Args)]
pub struct ProcessQueueArgs {
    /// Path to the `.mv2` file
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Only show queue status without processing
    #[arg(long)]
    pub status: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Result of processing the enrichment queue
#[derive(Debug, Serialize)]
pub struct ProcessQueueResult {
    /// Number of frames in queue before processing
    pub queue_before: usize,
    /// Number of frames processed
    pub frames_processed: usize,
    /// Number of frames remaining in queue
    pub queue_after: usize,
    /// Total active frames
    pub total_frames: usize,
    /// Frames fully enriched
    pub enriched_frames: usize,
    /// Frames that are searchable but not enriched
    pub searchable_only: usize,
}

/// Handle the `process-queue` command for progressive ingestion enrichment.
///
/// This command processes frames in the enrichment queue:
/// - Re-extracts full text for skim (time-limited) extractions
/// - Updates the Tantivy index with enriched content
/// - Marks frames as Enriched when complete
pub fn handle_process_queue(_config: &CliConfig, args: ProcessQueueArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    // Get initial stats
    let initial_stats = mem.enrichment_stats();
    let queue_before = mem.enrichment_queue_len();

    if args.status {
        // Just show status
        if args.json {
            let result = ProcessQueueResult {
                queue_before,
                frames_processed: 0,
                queue_after: queue_before,
                total_frames: initial_stats.total_frames,
                enriched_frames: initial_stats.enriched_frames,
                searchable_only: initial_stats.searchable_only,
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("Enrichment Queue Status:");
            println!("  Pending: {} frames", queue_before);
            println!("  Total frames: {}", initial_stats.total_frames);
            println!("  Enriched: {}", initial_stats.enriched_frames);
            println!("  Searchable only: {}", initial_stats.searchable_only);

            if queue_before == 0 {
                println!("\n✓ No frames pending enrichment");
            } else {
                println!(
                    "\nRun without --status to process {} pending frames",
                    queue_before
                );
            }
        }
        return Ok(());
    }

    if queue_before == 0 {
        if args.json {
            let result = ProcessQueueResult {
                queue_before: 0,
                frames_processed: 0,
                queue_after: 0,
                total_frames: initial_stats.total_frames,
                enriched_frames: initial_stats.enriched_frames,
                searchable_only: initial_stats.searchable_only,
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("✓ No frames pending enrichment");
        }
        return Ok(());
    }

    // Process all pending enrichment tasks
    if !args.json {
        eprintln!("Processing {} pending frames...", queue_before);
    }

    let start = std::time::Instant::now();
    let frames_processed = mem.process_all_enrichment();
    let elapsed = start.elapsed();

    // Commit changes
    mem.commit()?;

    // Get final stats
    let final_stats = mem.enrichment_stats();
    let queue_after = mem.enrichment_queue_len();

    if args.json {
        let result = ProcessQueueResult {
            queue_before,
            frames_processed,
            queue_after,
            total_frames: final_stats.total_frames,
            enriched_frames: final_stats.enriched_frames,
            searchable_only: final_stats.searchable_only,
        };
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Enrichment complete:");
        println!("  Frames processed: {}", frames_processed);
        println!("  Time: {:.2}s", elapsed.as_secs_f64());
        println!(
            "  Throughput: {:.1} frames/sec",
            frames_processed as f64 / elapsed.as_secs_f64().max(0.001)
        );
        println!();
        println!("Status:");
        println!("  Total frames: {}", final_stats.total_frames);
        println!("  Enriched: {}", final_stats.enriched_frames);
        println!("  Searchable only: {}", final_stats.searchable_only);
        println!("  Queue remaining: {}", queue_after);
    }

    Ok(())
}
