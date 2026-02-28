//! Sketch track commands for fast candidate generation.
//!
//! Commands:
//! - `sketch build`: Build sketches for all frames without existing sketches
//! - `sketch info`: Show statistics about the sketch track

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use memvid_core::{Memvid, SketchVariant};
use serde_json::json;

/// Arguments for the `sketch` command
#[derive(Args)]
pub struct SketchArgs {
    #[command(subcommand)]
    pub command: SketchCommand,
}

/// Sketch subcommands
#[derive(Subcommand)]
pub enum SketchCommand {
    /// Build sketches for all frames that don't have one
    Build(SketchBuildArgs),
    /// Show sketch track statistics
    Info(SketchInfoArgs),
}

/// Arguments for `sketch build`
#[derive(Args)]
pub struct SketchBuildArgs {
    /// Path to the memory file
    #[arg(value_name = "FILE")]
    pub file: PathBuf,

    /// Sketch variant: small (32 bytes), medium (64 bytes), large (96 bytes)
    #[arg(long, default_value = "small")]
    pub variant: SketchVariantArg,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `sketch info`
#[derive(Args)]
pub struct SketchInfoArgs {
    /// Path to the memory file
    #[arg(value_name = "FILE")]
    pub file: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Sketch variant argument
#[derive(Clone, Debug)]
pub struct SketchVariantArg(pub SketchVariant);

impl Default for SketchVariantArg {
    fn default() -> Self {
        Self(SketchVariant::Small)
    }
}

impl std::str::FromStr for SketchVariantArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "small" | "s" | "32" => Ok(Self(SketchVariant::Small)),
            "medium" | "m" | "64" => Ok(Self(SketchVariant::Medium)),
            "large" | "l" | "96" => Ok(Self(SketchVariant::Large)),
            _ => Err(format!(
                "Unknown variant '{}'. Use: small (32 bytes), medium (64 bytes), or large (96 bytes)",
                s
            )),
        }
    }
}

/// Handle the sketch command
pub fn handle_sketch(args: SketchArgs) -> Result<()> {
    match args.command {
        SketchCommand::Build(build_args) => handle_sketch_build(build_args),
        SketchCommand::Info(info_args) => handle_sketch_info(info_args),
    }
}

/// Handle `sketch build`
fn handle_sketch_build(args: SketchBuildArgs) -> Result<()> {
    let start = Instant::now();

    let mut mem = Memvid::open(&args.file)
        .with_context(|| format!("Failed to open memory: {}", args.file.display()))?;

    let frame_count = mem.frame_count();
    let existing_sketches = mem.sketches().len();

    // Build sketches for frames that don't have one
    let new_sketches = mem.build_all_sketches(args.variant.0);

    // Commit if we added any sketches
    if new_sketches > 0 {
        mem.commit()
            .context("Failed to commit sketch track changes")?;
    }

    let elapsed = start.elapsed();
    let stats = mem.sketch_stats();

    if args.json {
        let output = json!({
            "file": args.file.display().to_string(),
            "variant": format!("{:?}", args.variant.0),
            "frames_total": frame_count,
            "sketches_existing": existing_sketches,
            "sketches_new": new_sketches,
            "sketches_total": stats.entry_count,
            "size_bytes": stats.size_bytes,
            "elapsed_ms": elapsed.as_millis(),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Sketch build complete for {}", args.file.display());
        println!();
        println!("  Variant:          {:?}", args.variant.0);
        println!("  Total frames:     {}", frame_count);
        println!("  Existing sketches: {}", existing_sketches);
        println!("  New sketches:     {}", new_sketches);
        println!("  Total sketches:   {}", stats.entry_count);
        println!("  Size on disk:     {}", format_bytes(stats.size_bytes));
        println!("  Time:             {:.2}s", elapsed.as_secs_f64());

        if new_sketches == 0 && existing_sketches == stats.entry_count as usize {
            println!();
            println!("All frames already have sketches.");
        } else if new_sketches > 0 {
            println!();
            let rate = new_sketches as f64 / elapsed.as_secs_f64();
            println!("Rate: {:.0} sketches/sec", rate);
        }
    }

    Ok(())
}

/// Handle `sketch info`
fn handle_sketch_info(args: SketchInfoArgs) -> Result<()> {
    let mem = Memvid::open_read_only(&args.file)
        .with_context(|| format!("Failed to open memory: {}", args.file.display()))?;

    let stats = mem.sketch_stats();
    let frame_count = mem.frame_count();
    let coverage = if frame_count > 0 {
        stats.entry_count as f64 / frame_count as f64 * 100.0
    } else {
        0.0
    };

    if args.json {
        let output = json!({
            "file": args.file.display().to_string(),
            "enabled": !mem.sketches().is_empty(),
            "variant": format!("{:?}", stats.variant),
            "entry_count": stats.entry_count,
            "frame_count": frame_count,
            "coverage_percent": coverage,
            "size_bytes": stats.size_bytes,
            "short_text_count": stats.short_text_count,
            "entry_size_bytes": stats.variant.entry_size(),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Sketch Track Info: {}", args.file.display());
        println!();

        if mem.sketches().is_empty() {
            println!("  Status: No sketches built");
            println!();
            println!(
                "  Run `memvid sketch build {}` to enable fast candidate generation.",
                args.file.display()
            );
        } else {
            println!("  Status:           Enabled");
            println!(
                "  Variant:          {:?} ({} bytes/entry)",
                stats.variant,
                stats.variant.entry_size()
            );
            println!("  Entries:          {}", stats.entry_count);
            println!(
                "  Coverage:         {:.1}% ({}/{})",
                coverage, stats.entry_count, frame_count
            );
            println!("  Size:             {}", format_bytes(stats.size_bytes));
            println!("  Short text:       {} entries", stats.short_text_count);
            println!();
            println!(
                "Sketch-enabled queries will filter {} candidates before BM25 rerank.",
                stats.entry_count
            );
        }
    }

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}
