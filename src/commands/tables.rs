//! Table extraction and management commands.
//!
//! Provides CLI commands for:
//! - Importing tables from PDF documents
//! - Listing tables stored in an MV2 file
//! - Exporting tables to CSV/JSON
//! - Viewing individual tables

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Args, Subcommand, ValueEnum};
use memvid_core::table::{
    export_to_csv, export_to_json, extract_tables, get_table, list_tables, store_table,
    ExtractionMode, TableExtractionOptions, TableQuality,
};
use memvid_core::Memvid;
use serde_json::json;

use crate::config::CliConfig;

/// Arguments for the `tables` command
#[derive(Args)]
pub struct TablesArgs {
    #[command(subcommand)]
    pub command: TablesCommand,
}

/// Tables subcommands
#[derive(Subcommand)]
pub enum TablesCommand {
    /// Import tables from a document (PDF, DOCX, XLSX)
    Import(TablesImportArgs),
    /// List all tables stored in the memory
    List(TablesListArgs),
    /// Export a table to CSV or JSON
    Export(TablesExportArgs),
    /// View a specific table
    View(TablesViewArgs),
}

/// Arguments for `tables import`
#[derive(Args)]
pub struct TablesImportArgs {
    /// MV2 file to store tables in
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Input document (PDF, DOCX, XLSX)
    #[arg(long = "input", short = 'i', value_name = "PATH", value_parser = clap::value_parser!(PathBuf))]
    pub input: PathBuf,

    /// Extraction mode
    #[arg(long = "mode", value_enum, default_value = "conservative")]
    pub mode: ExtractionModeArg,

    /// Minimum number of rows required
    #[arg(long = "min-rows", default_value = "2")]
    pub min_rows: usize,

    /// Minimum number of columns required
    #[arg(long = "min-cols", default_value = "2")]
    pub min_cols: usize,

    /// Minimum quality threshold
    #[arg(long = "min-quality", value_enum, default_value = "medium")]
    pub min_quality: QualityArg,

    /// Enable multi-page table merging
    #[arg(long = "merge-multi-page", default_value = "true")]
    pub merge_multi_page: bool,

    /// Maximum pages to process (0 = all)
    #[arg(long = "max-pages", default_value = "0")]
    pub max_pages: usize,

    /// Embed table rows for search indexing
    #[arg(long = "embed-rows", default_value = "true")]
    pub embed_rows: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `tables list`
#[derive(Args)]
pub struct TablesListArgs {
    /// MV2 file to list tables from
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `tables export`
#[derive(Args)]
pub struct TablesExportArgs {
    /// MV2 file containing the table
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Table ID to export
    #[arg(long = "table-id", value_name = "ID")]
    pub table_id: String,

    /// Output file path
    #[arg(long = "out", short = 'o', value_name = "PATH", value_parser = clap::value_parser!(PathBuf))]
    pub out: Option<PathBuf>,

    /// Output format
    #[arg(long = "format", value_enum, default_value = "csv")]
    pub format: ExportFormatArg,

    /// For JSON: output as array of records instead of table object
    #[arg(long = "as-records")]
    pub as_records: bool,
}

/// Arguments for `tables view`
#[derive(Args)]
pub struct TablesViewArgs {
    /// MV2 file containing the table
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,

    /// Table ID to view
    #[arg(long = "table-id", value_name = "ID")]
    pub table_id: String,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Maximum rows to display (0 = all)
    #[arg(long = "limit", default_value = "50")]
    pub limit: usize,
}

/// Extraction mode CLI argument
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum ExtractionModeArg {
    /// Only detect tables with visible grid lines
    LatticeOnly,
    /// Only detect tables from text alignment
    StreamOnly,
    /// Try lattice first, fall back to stream
    Conservative,
    /// Aggressive detection (both methods, more false positives)
    Aggressive,
}

impl From<ExtractionModeArg> for ExtractionMode {
    fn from(value: ExtractionModeArg) -> Self {
        match value {
            ExtractionModeArg::LatticeOnly => ExtractionMode::LatticeOnly,
            ExtractionModeArg::StreamOnly => ExtractionMode::StreamOnly,
            ExtractionModeArg::Conservative => ExtractionMode::Conservative,
            ExtractionModeArg::Aggressive => ExtractionMode::Aggressive,
        }
    }
}

/// Quality threshold CLI argument
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum QualityArg {
    High,
    Medium,
    Low,
}

impl From<QualityArg> for TableQuality {
    fn from(value: QualityArg) -> Self {
        match value {
            QualityArg::High => TableQuality::High,
            QualityArg::Medium => TableQuality::Medium,
            QualityArg::Low => TableQuality::Low,
        }
    }
}

/// Export format CLI argument
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum ExportFormatArg {
    Csv,
    Json,
}

// ============================================================================
// Command handlers
// ============================================================================

/// Main dispatcher for tables commands
pub fn handle_tables(_config: &CliConfig, args: TablesArgs) -> Result<()> {
    match args.command {
        TablesCommand::Import(import_args) => handle_tables_import(import_args),
        TablesCommand::List(list_args) => handle_tables_list(list_args),
        TablesCommand::Export(export_args) => handle_tables_export(export_args),
        TablesCommand::View(view_args) => handle_tables_view(view_args),
    }
}

/// Handle `tables import` command
fn handle_tables_import(args: TablesImportArgs) -> Result<()> {
    // Read input document
    let input_bytes = fs::read(&args.input)?;
    let filename = args
        .input
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Build extraction options
    let options = TableExtractionOptions::builder()
        .mode(args.mode.into())
        .min_rows(args.min_rows)
        .min_cols(args.min_cols)
        .min_quality(args.min_quality.into())
        .merge_multi_page(args.merge_multi_page)
        .max_pages(args.max_pages)
        .build();

    // Extract tables
    let result = extract_tables(&input_bytes, filename, &options)?;

    if result.tables.is_empty() {
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "tables_found": 0,
                    "tables_stored": 0,
                    "warnings": result.warnings,
                }))?
            );
        } else {
            println!("No tables found in {}", filename);
            if !result.warnings.is_empty() {
                println!("\nWarnings:");
                for warning in &result.warnings {
                    println!("  - {}", warning);
                }
            }
        }
        return Ok(());
    }

    // Open MV2 file and store tables
    let mut mem = Memvid::open(&args.file)?;
    let mut stored_tables = Vec::new();

    for table in &result.tables {
        let (meta_id, row_ids) = store_table(&mut mem, table, args.embed_rows)?;
        stored_tables.push(json!({
            "table_id": table.table_id,
            "meta_frame_id": meta_id,
            "row_frame_ids": row_ids,
            "rows": table.n_rows,
            "cols": table.n_cols,
            "quality": format!("{:?}", table.quality),
            "detection_mode": format!("{:?}", table.detection_mode),
            "pages": format!("{}-{}", table.page_start, table.page_end),
        }));
    }

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "tables_found": result.tables.len(),
                "tables_stored": stored_tables.len(),
                "extraction_ms": result.total_ms,
                "tables": stored_tables,
                "warnings": result.warnings,
            }))?
        );
    } else {
        println!(
            "Extracted {} tables from {} in {} ms",
            result.tables.len(),
            filename,
            result.total_ms
        );
        println!();
        for (i, table) in result.tables.iter().enumerate() {
            println!(
                "Table {}: {} rows × {} cols ({:?}, {:?})",
                i + 1,
                table.n_rows,
                table.n_cols,
                table.quality,
                table.detection_mode
            );
            println!(
                "  Pages: {}-{}, Confidence: {:.2}",
                table.page_start, table.page_end, table.confidence_score
            );
            if !table.headers.is_empty() {
                let header_preview: Vec<_> = table
                    .headers
                    .iter()
                    .take(5)
                    .map(|s| truncate_string(s, 20))
                    .collect();
                let suffix = if table.headers.len() > 5 {
                    format!(" ... ({} more)", table.headers.len() - 5)
                } else {
                    String::new()
                };
                println!("  Headers: [{}]{}", header_preview.join(", "), suffix);
            }
        }
        if !result.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &result.warnings {
                println!("  - {}", warning);
            }
        }
    }

    Ok(())
}

/// Handle `tables list` command
fn handle_tables_list(args: TablesListArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;
    let tables = list_tables(&mut mem)?;

    if args.json {
        let json_tables: Vec<_> = tables
            .iter()
            .map(|t| {
                json!({
                    "table_id": t.table_id,
                    "frame_id": t.frame_id,
                    "source_file": t.source_file,
                    "n_rows": t.n_rows,
                    "n_cols": t.n_cols,
                    "pages": format!("{}-{}", t.page_start, t.page_end),
                    "quality": format!("{:?}", t.quality),
                    "headers": t.headers,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "count": tables.len(),
                "tables": json_tables,
            }))?
        );
    } else if tables.is_empty() {
        println!("No tables stored in this memory.");
    } else {
        println!("Tables in memory ({}):", tables.len());
        println!();
        for table in &tables {
            println!(
                "  {} — {} rows × {} cols",
                table.table_id, table.n_rows, table.n_cols
            );
            println!(
                "    Source: {}, Pages: {}-{}, Quality: {:?}",
                table.source_file, table.page_start, table.page_end, table.quality
            );
            if !table.headers.is_empty() {
                let header_preview: Vec<_> = table
                    .headers
                    .iter()
                    .take(4)
                    .map(|s| truncate_string(s, 15))
                    .collect();
                let suffix = if table.headers.len() > 4 {
                    format!(" ... (+{})", table.headers.len() - 4)
                } else {
                    String::new()
                };
                println!("    Headers: [{}]{}", header_preview.join(", "), suffix);
            }
            println!();
        }
    }

    Ok(())
}

/// Handle `tables export` command
fn handle_tables_export(args: TablesExportArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    let table = get_table(&mut mem, &args.table_id)?;
    let table = match table {
        Some(t) => t,
        None => bail!("Table '{}' not found", args.table_id),
    };

    let output = match args.format {
        ExportFormatArg::Csv => export_to_csv(&table),
        ExportFormatArg::Json => export_to_json(&table, args.as_records)?,
    };

    if let Some(out_path) = args.out {
        fs::write(&out_path, &output)?;
        println!(
            "Exported table '{}' to {}",
            args.table_id,
            out_path.display()
        );
    } else {
        println!("{}", output);
    }

    Ok(())
}

/// Handle `tables view` command
fn handle_tables_view(args: TablesViewArgs) -> Result<()> {
    let mut mem = Memvid::open(&args.file)?;

    let table = get_table(&mut mem, &args.table_id)?;
    let table = match table {
        Some(t) => t,
        None => bail!("Table '{}' not found", args.table_id),
    };

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "table_id": table.table_id,
                "source_file": table.source_file,
                "page_start": table.page_start,
                "page_end": table.page_end,
                "n_rows": table.n_rows,
                "n_cols": table.n_cols,
                "quality": format!("{:?}", table.quality),
                "detection_mode": format!("{:?}", table.detection_mode),
                "confidence_score": table.confidence_score,
                "headers": table.headers,
                "rows": table.rows.iter().take(if args.limit == 0 { usize::MAX } else { args.limit }).map(|r| {
                    json!({
                        "row_index": r.row_index,
                        "page": r.page,
                        "is_header": r.is_header_row,
                        "cells": r.cells.iter().map(|c| {
                            json!({
                                "text": c.text,
                                "col_index": c.col_index,
                                "is_header": c.is_header,
                                "col_span": c.col_span,
                                "row_span": c.row_span,
                            })
                        }).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<_>>(),
                "warnings": table.warnings,
            }))?
        );
    } else {
        println!("Table: {}", table.table_id);
        println!("Source: {}", table.source_file);
        println!(
            "Pages: {}-{}, Quality: {:?}, Mode: {:?}",
            table.page_start, table.page_end, table.quality, table.detection_mode
        );
        println!(
            "Size: {} rows × {} cols, Confidence: {:.2}",
            table.n_rows, table.n_cols, table.confidence_score
        );
        println!();

        // Calculate column widths
        let mut col_widths: Vec<usize> = vec![0; table.n_cols];
        for (i, header) in table.headers.iter().enumerate() {
            if i < col_widths.len() {
                col_widths[i] = col_widths[i].max(header.len().min(30));
            }
        }
        for row in &table.rows {
            for cell in &row.cells {
                if cell.col_index < col_widths.len() {
                    col_widths[cell.col_index] =
                        col_widths[cell.col_index].max(cell.text.len().min(30));
                }
            }
        }

        // Print headers
        if !table.headers.is_empty() {
            let header_line: Vec<String> = table
                .headers
                .iter()
                .enumerate()
                .map(|(i, h)| {
                    let width = col_widths.get(i).copied().unwrap_or(10);
                    format!("{:width$}", truncate_string(h, width), width = width)
                })
                .collect();
            println!("| {} |", header_line.join(" | "));
            let separator: Vec<String> = col_widths.iter().map(|w| "-".repeat(*w)).collect();
            println!("|-{}-|", separator.join("-|-"));
        }

        // Print rows
        let limit = if args.limit == 0 {
            usize::MAX
        } else {
            args.limit
        };
        let rows_to_show: Vec<_> = table
            .rows
            .iter()
            .filter(|r| !r.is_header_row)
            .take(limit)
            .collect();

        for row in &rows_to_show {
            let mut cell_texts: Vec<String> = vec![String::new(); table.n_cols];
            for cell in &row.cells {
                if cell.col_index < cell_texts.len() {
                    cell_texts[cell.col_index] = cell.text.clone();
                }
            }
            let row_line: Vec<String> = cell_texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    let width = col_widths.get(i).copied().unwrap_or(10);
                    format!("{:width$}", truncate_string(text, width), width = width)
                })
                .collect();
            println!("| {} |", row_line.join(" | "));
        }

        let total_data_rows = table.rows.iter().filter(|r| !r.is_header_row).count();
        if rows_to_show.len() < total_data_rows {
            println!(
                "\n... showing {} of {} rows (use --limit 0 to show all)",
                rows_to_show.len(),
                total_data_rows
            );
        }

        if !table.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &table.warnings {
                println!("  - {}", warning);
            }
        }
    }

    Ok(())
}

/// Truncate a string to a maximum length
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        s.chars().take(max_len).collect()
    } else {
        format!("{}...", s.chars().take(max_len - 3).collect::<String>())
    }
}
