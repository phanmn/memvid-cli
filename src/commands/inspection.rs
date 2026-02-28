//! Inspection command handlers (view, stats, who)

#[cfg(feature = "audio-playback")]
use std::io::Cursor;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(feature = "audio-playback")]
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use hex;
use memvid_core::table::list_tables;
use memvid_core::{
    lockfile, normalize_text, Frame, FrameRole, MediaManifest, Memvid, TextChunkManifest,
    TextChunkRange,
};
use serde_json::{json, Value};
use tempfile::Builder;
use tracing::warn;
use uuid::Uuid;

use crate::config::CliConfig;
use crate::utils::{
    format_bytes, format_percent, format_timestamp_ms, frame_status_str, open_read_only_mem,
    owner_hint_to_json, parse_timecode, round_percent, select_frame, yes_no,
};

const DEFAULT_VIEW_PAGE_CHARS: usize = 1_200;
const CHUNK_MANIFEST_KEY: &str = "memvid_chunks_v1";

/// Arguments for the `view` subcommand
#[derive(Args)]
pub struct ViewArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long = "frame-id", value_name = "ID", conflicts_with = "uri")]
    pub frame_id: Option<u64>,
    #[arg(long, value_name = "URI", conflicts_with = "frame_id")]
    pub uri: Option<String>,
    #[arg(long)]
    pub json: bool,
    #[arg(long, conflicts_with = "json")]
    pub binary: bool,
    #[arg(long, conflicts_with_all = ["json", "binary"])]
    pub preview: bool,
    /// Optional start time for video previews (HH:MM:SS[.mmm])
    #[arg(
        long = "start",
        value_name = "HH:MM:SS",
        requires = "preview",
        conflicts_with_all = ["json", "binary", "play"]
    )]
    pub preview_start: Option<String>,
    /// Optional end time for video previews (HH:MM:SS[.mmm])
    #[arg(
        long = "end",
        value_name = "HH:MM:SS",
        requires = "preview",
        conflicts_with_all = ["json", "binary", "play"]
    )]
    pub preview_end: Option<String>,
    #[arg(long = "play", conflicts_with_all = ["json", "binary", "preview"])]
    pub play: bool,
    #[arg(long = "start-seconds", requires = "play")]
    pub start_seconds: Option<f32>,
    #[arg(long = "end-seconds", requires = "play")]
    pub end_seconds: Option<f32>,
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub page: usize,
    #[arg(long = "page-size", value_name = "CHARS")]
    pub page_size: Option<usize>,
}

/// Arguments for the `stats` subcommand
#[derive(Args)]
pub struct StatsArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long)]
    pub json: bool,
    /// Replay: Show stats for frames with ID <= AS_OF_FRAME (time-travel view)
    #[arg(long = "as-of-frame", value_name = "FRAME_ID")]
    pub as_of_frame: Option<u64>,
    /// Replay: Show stats for frames with timestamp <= AS_OF_TS (time-travel view)
    #[arg(long = "as-of-ts", value_name = "UNIX_TIMESTAMP")]
    pub as_of_ts: Option<i64>,
}

/// Arguments for the `who` subcommand
#[derive(Args)]
pub struct WhoArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long)]
    pub json: bool,
}

/// Handler for `memvid stats`
pub fn handle_stats(_config: &CliConfig, args: StatsArgs) -> Result<()> {
    let mut mem = Memvid::open_read_only(&args.file)?;
    let stats = mem.stats()?;
    let tables = list_tables(&mut mem).unwrap_or_default();
    let vec_dimension = mem.effective_vec_index_dimension()?;
    let embedding_identity = mem.embedding_identity_summary(10_000);

    // Note: Replay filtering for stats is currently not implemented
    // The stats show the full memory state
    if args.as_of_frame.is_some() || args.as_of_ts.is_some() {
        eprintln!("Note: Replay filtering (--as-of-frame/--as-of-ts) shows current stats.");
        eprintln!("      Use 'find' or 'timeline' commands for filtered results.");
    }
    let overhead_bytes = stats.size_bytes.saturating_sub(stats.payload_bytes);
    let payload_share_percent: f64 = if stats.size_bytes > 0 {
        round_percent((stats.payload_bytes as f64 / stats.size_bytes as f64) * 100.0)
    } else {
        0.0
    };
    let overhead_share_percent: f64 = if stats.size_bytes > 0 {
        round_percent((100.0 - payload_share_percent).max(0.0))
    } else {
        0.0
    };
    let maintenance_command = format!(
        "memvid doctor {} --vacuum --rebuild-time-index --rebuild-lex-index",
        args.file.display()
    );

    if args.json {
        let mut raw_json = serde_json::to_value(&stats)?;
        if let Value::Object(ref mut obj) = raw_json {
            obj.remove("tier");
        }

        // Build tables list for JSON output
        let tables_json: Vec<serde_json::Value> = tables
            .iter()
            .map(|t| {
                json!({
                    "table_id": t.table_id,
                    "source_file": t.source_file,
                    "n_rows": t.n_rows,
                    "n_cols": t.n_cols,
                    "pages": format!("{}-{}", t.page_start, t.page_end),
                    "quality": format!("{:?}", t.quality),
                    "headers": t.headers,
                })
            })
            .collect();

        // Compute embedding quality for JSON output
        let embedding_quality_json = if stats.has_vec_index {
            mem.embedding_quality().ok().flatten().map(|eq| {
                json!({
                    "vector_count": eq.vector_count,
                    "dimension": eq.dimension,
                    "avg_similarity": eq.avg_similarity,
                    "min_similarity": eq.min_similarity,
                    "max_similarity": eq.max_similarity,
                    "std_similarity": eq.std_similarity,
                    "clustering_coefficient": eq.clustering_coefficient,
                    "estimated_clusters": eq.estimated_clusters,
                    "recommended_threshold": eq.recommended_threshold,
                    "quality_rating": eq.quality_rating,
                    "quality_explanation": eq.quality_explanation,
                })
            })
        } else {
            None
        };

        let embedding_identity_json = match &embedding_identity {
            memvid_core::EmbeddingIdentitySummary::Unknown => Value::Null,
            memvid_core::EmbeddingIdentitySummary::Single(identity) => json!({
                "provider": identity.provider.as_deref(),
                "model": identity.model.as_deref(),
                "dimension": identity.dimension.or(vec_dimension),
                "normalized": identity.normalized,
            }),
            memvid_core::EmbeddingIdentitySummary::Mixed(identities) => {
                let values: Vec<Value> = identities
                    .iter()
                    .map(|entry| {
                        json!({
                            "provider": entry.identity.provider.as_deref(),
                            "model": entry.identity.model.as_deref(),
                            "dimension": entry.identity.dimension.or(vec_dimension),
                            "normalized": entry.identity.normalized,
                            "count": entry.count,
                        })
                    })
                    .collect();
                json!({ "mixed": values })
            }
        };

        // Get enrichment stats for JSON output
        let enrichment_stats = mem.enrichment_stats();
        let enrichment_json = json!({
            "total_frames": enrichment_stats.total_frames,
            "enriched_frames": enrichment_stats.enriched_frames,
            "pending_frames": enrichment_stats.pending_frames,
            "searchable_only": enrichment_stats.searchable_only,
        });

        // Get ticket info for JSON output
        let ticket = mem.current_ticket();
        let ticket_json = json!({
            "issuer": ticket.issuer,
            "seq_no": ticket.seq_no,
            "expires_in_secs": ticket.expires_in_secs,
            "capacity_bytes": ticket.capacity_bytes,
            "verified": ticket.verified,
        });

        let report = json!({
            "summary": {
                "sequence": stats.seq_no,
                "frames": format!("{} total ({} active)", stats.frame_count, stats.active_frame_count),
                "usage": format!(
                    "{} used / {} total ({})",
                    format_bytes(stats.size_bytes),
                    format_bytes(stats.capacity_bytes),
                    format_percent(stats.storage_utilisation_percent)
                ),
                "remaining": format!("{} free", format_bytes(stats.remaining_capacity_bytes)),
            },
            "storage": {
                "payload": format!("{} ({})", format_bytes(stats.payload_bytes), format_percent(payload_share_percent)),
                "overhead": format!("{} ({}) - WAL + indexes", format_bytes(overhead_bytes), format_percent(overhead_share_percent)),
                "logical_payload": format!("{} before compression", format_bytes(stats.logical_bytes)),
                "compression_savings": format!("{} saved ({})", format_bytes(stats.saved_bytes), format_percent(stats.savings_percent)),
                "compression_ratio": format_percent(stats.compression_ratio_percent),
            },
            "frames": {
                "average_stored": format_bytes(stats.average_frame_payload_bytes),
                "average_logical": format_bytes(stats.average_frame_logical_bytes),
                "clip_images": stats.clip_image_count,
            },
            "indexes": {
                "lexical": yes_no(stats.has_lex_index),
                "vector": yes_no(stats.has_vec_index),
                "time": yes_no(stats.has_time_index),
            },
            "enrichment": enrichment_json,
            "ticket": ticket_json,
            "embedding_identity": embedding_identity_json,
            "embedding_quality": embedding_quality_json,
            "tables": {
                "count": tables.len(),
                "tables": tables_json,
            },
            "maintenance": maintenance_command,
            "raw": raw_json,
        });

        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let seq_display = stats
            .seq_no
            .map(|seq| seq.to_string())
            .unwrap_or_else(|| "n/a".to_string());

        println!("Memory: {}", args.file.display());
        println!("Sequence: {}", seq_display);
        println!(
            "Frames: {} total ({} active)",
            stats.frame_count, stats.active_frame_count
        );

        println!("\nCapacity:");
        println!(
            "  Usage: {} used / {} total ({})",
            format_bytes(stats.size_bytes),
            format_bytes(stats.capacity_bytes),
            format_percent(stats.storage_utilisation_percent)
        );
        println!(
            "  Remaining: {}",
            format_bytes(stats.remaining_capacity_bytes)
        );

        // Show ticket verification status
        let ticket = mem.current_ticket();
        if ticket.seq_no > 0 {
            let verified_str = if ticket.verified {
                "✓ verified"
            } else {
                "⚠ unverified"
            };
            println!(
                "  Ticket: seq={} issuer={} ({})",
                ticket.seq_no, ticket.issuer, verified_str
            );
        }

        println!("\nStorage breakdown:");
        println!(
            "  Payload: {} ({})",
            format_bytes(stats.payload_bytes),
            format_percent(payload_share_percent)
        );
        println!(
            "  Overhead: {} ({})",
            format_bytes(overhead_bytes),
            format_percent(overhead_share_percent)
        );
        // PHASE 2: Detailed overhead breakdown for observability
        println!("    ├─ WAL: {}", format_bytes(stats.wal_bytes));
        println!(
            "    ├─ Lexical index: {}",
            format_bytes(stats.lex_index_bytes)
        );
        println!(
            "    ├─ Vector index: {}",
            format_bytes(stats.vec_index_bytes)
        );
        println!(
            "    └─ Time index: {}",
            format_bytes(stats.time_index_bytes)
        );
        println!(
            "  Logical payload: {} before compression",
            format_bytes(stats.logical_bytes)
        );

        if stats.has_vec_index {
            println!("\nEmbeddings:");
            if let Some(dim) = vec_dimension {
                println!("  Dimension: {}", dim);
            }
            match &embedding_identity {
                memvid_core::EmbeddingIdentitySummary::Unknown => {
                    println!("  Model: unknown (no persisted embedding identity)");
                }
                memvid_core::EmbeddingIdentitySummary::Single(identity) => {
                    if let Some(provider) = identity.provider.as_deref() {
                        println!("  Provider: {}", provider);
                    }
                    if let Some(model) = identity.model.as_deref() {
                        println!("  Model: {}", model);
                    }
                }
                memvid_core::EmbeddingIdentitySummary::Mixed(identities) => {
                    println!("  Model: mixed ({} identities detected)", identities.len());
                    for entry in identities.iter().take(5) {
                        let provider = entry.identity.provider.as_deref().unwrap_or("unknown");
                        let model = entry.identity.model.as_deref().unwrap_or("unknown");
                        println!("    - {} / {} ({} frames)", provider, model, entry.count);
                    }
                    if identities.len() > 5 {
                        println!("    - ...");
                    }
                }
            }
        }
        println!(
            "  Compression savings: {} ({})",
            format_bytes(stats.saved_bytes),
            format_percent(stats.savings_percent)
        );

        println!("\nAverage frame:");
        println!(
            "  Stored: {}   Logical: {}",
            format_bytes(stats.average_frame_payload_bytes),
            format_bytes(stats.average_frame_logical_bytes)
        );
        if stats.clip_image_count > 0 {
            println!("  CLIP images: {}", stats.clip_image_count);
        }

        // PHASE 2: Per-document cost analysis
        if stats.active_frame_count > 0 {
            let overhead_per_doc = overhead_bytes / stats.active_frame_count;
            let lex_per_doc = stats.lex_index_bytes / stats.active_frame_count;
            let vec_per_doc = stats.vec_index_bytes / stats.active_frame_count;

            println!("\nPer-document overhead:");
            println!("  Total: {}", format_bytes(overhead_per_doc));
            if stats.has_lex_index {
                println!("  Lexical: {}", format_bytes(lex_per_doc));
            }
            if stats.has_vec_index {
                let vec_ratio = if stats.average_frame_payload_bytes > 0 {
                    vec_per_doc as f64 / stats.average_frame_payload_bytes as f64
                } else {
                    0.0
                };
                println!(
                    "  Vector: {} ({:.0}x text size)",
                    format_bytes(vec_per_doc),
                    vec_ratio
                );
            }
        }

        println!("\nIndexes:");
        println!(
            "  Lexical: {}   Vector: {}   Time: {}",
            yes_no(stats.has_lex_index),
            yes_no(stats.has_vec_index),
            yes_no(stats.has_time_index)
        );

        // Show enrichment queue stats
        let enrichment_stats = mem.enrichment_stats();
        if enrichment_stats.pending_frames > 0 || enrichment_stats.searchable_only > 0 {
            println!("\nEnrichment:");
            println!(
                "  Enriched: {} / {}",
                enrichment_stats.enriched_frames, enrichment_stats.total_frames
            );
            if enrichment_stats.pending_frames > 0 {
                println!("  Pending: {} frames", enrichment_stats.pending_frames);
                println!(
                    "  Run `memvid process-queue {}` to complete enrichment",
                    args.file.display()
                );
            }
        }

        // Show embedding quality stats if vector index is available
        if stats.has_vec_index {
            if let Ok(Some(eq)) = mem.embedding_quality() {
                println!("\nEmbedding Quality:");
                println!(
                    "  Vectors: {}   Dimension: {}",
                    eq.vector_count, eq.dimension
                );
                println!(
                    "  Similarity: avg={:.3}  min={:.3}  max={:.3}  std={:.3}",
                    eq.avg_similarity, eq.min_similarity, eq.max_similarity, eq.std_similarity
                );
                println!(
                    "  Clusters: ~{}   Quality: {}",
                    eq.estimated_clusters, eq.quality_rating
                );
                println!(
                    "  Recommended --min-relevancy: {:.1}",
                    eq.recommended_threshold
                );
                println!("  {}", eq.quality_explanation);
            }
        }

        if !tables.is_empty() {
            println!("\nTables: {} extracted", tables.len());
            for t in &tables {
                println!(
                    "  {} — {} rows × {} cols ({})",
                    t.table_id, t.n_rows, t.n_cols, t.source_file
                );
            }
        }

        println!("\nMaintenance:");
        println!(
            "  Run `{}` to rebuild indexes and reclaim space.",
            maintenance_command
        );
    }
    Ok(())
}

/// Handler for `memvid who`
pub fn handle_who(args: WhoArgs) -> Result<()> {
    match lockfile::current_owner(&args.file)? {
        Some(owner) => {
            if args.json {
                let output = json!({
                    "locked": true,
                    "owner": owner_hint_to_json(&owner),
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("{} is locked by:", args.file.display());
                if let Some(pid) = owner.pid {
                    println!("  pid: {pid}");
                }
                if let Some(cmd) = owner.cmd.as_deref() {
                    println!("  cmd: {cmd}");
                }
                if let Some(started) = owner.started_at.as_deref() {
                    println!("  started_at: {started}");
                }
                if let Some(last) = owner.last_heartbeat.as_deref() {
                    println!("  last_heartbeat: {last}");
                }
                if let Some(interval) = owner.heartbeat_ms {
                    println!("  heartbeat_interval_ms: {interval}");
                }
                if let Some(file_id) = owner.file_id.as_deref() {
                    println!("  file_id: {file_id}");
                }
                if let Some(path) = owner.file_path.as_ref() {
                    println!("  file_path: {}", path.display());
                }
            }
        }
        None => {
            if args.json {
                let output = json!({"locked": false});
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("No active writer for {}", args.file.display());
            }
        }
    }
    Ok(())
}

// ============================================================================
// View command handler and helpers
// ============================================================================

/// Handler for `memvid view`
pub fn handle_view(args: ViewArgs) -> Result<()> {
    if args.page == 0 {
        bail!("page must be greater than zero");
    }
    if let Some(size) = args.page_size {
        if size == 0 {
            bail!("page-size must be greater than zero");
        }
    }

    let mut mem = open_read_only_mem(&args.file)?;
    let frame = select_frame(&mut mem, args.frame_id, args.uri.as_deref())?;

    if args.play {
        #[cfg(feature = "audio-playback")]
        {
            play_frame_audio(&mut mem, &frame, args.start_seconds, args.end_seconds)?;
            return Ok(());
        }
        #[cfg(not(feature = "audio-playback"))]
        {
            bail!("Audio playback requires the 'audio-playback' feature (only available on macOS)");
        }
    }

    if args.preview {
        let bounds = parse_preview_bounds(args.preview_start.as_ref(), args.preview_end.as_ref())?;
        preview_frame_media(&mut mem, &frame, args.uri.as_deref(), bounds)?;
        return Ok(());
    }

    if args.binary {
        let bytes = mem.frame_canonical_payload(frame.id)?;
        let mut stdout = io::stdout();
        stdout.write_all(&bytes)?;
        stdout.flush()?;
        return Ok(());
    }

    let canonical_text = canonical_text_for_view(&mut mem, &frame)?;
    let manifest_from_meta = canonical_manifest_from_frame(&canonical_text, &frame);

    let page_size = args
        .page_size
        .or_else(|| manifest_from_meta.as_ref().map(|m| m.chunk_chars))
        .unwrap_or(DEFAULT_VIEW_PAGE_CHARS);

    let mut manifest = if args.page_size.is_none() {
        manifest_from_meta.unwrap_or_else(|| compute_chunk_manifest(&canonical_text, page_size))
    } else {
        compute_chunk_manifest(&canonical_text, page_size)
    };
    if manifest.chunks.is_empty() {
        manifest = TextChunkManifest {
            chunk_chars: page_size,
            chunks: vec![TextChunkRange {
                start: 0,
                end: canonical_text.chars().count(),
            }],
        };
    }

    if frame.role == FrameRole::DocumentChunk && args.page_size.is_none() {
        let total_chars = canonical_text.chars().count();
        manifest = TextChunkManifest {
            chunk_chars: total_chars.max(1),
            chunks: vec![TextChunkRange {
                start: 0,
                end: total_chars,
            }],
        };
    }

    let total_pages = manifest.chunks.len().max(1);
    if args.page > total_pages {
        bail!(
            "page {} is out of range (total pages: {})",
            args.page,
            total_pages
        );
    }

    let chunk = &manifest.chunks[args.page - 1];
    let content = extract_chunk_slice(&canonical_text, chunk);

    if args.json {
        let mut frame_json = frame_to_json(&frame);
        if let Some(obj) = frame_json.as_object_mut() {
            // Note: Do NOT overwrite search_text - it contains the extracted text from the document.
            // The "content" field shows the paginated payload view.
            if let Some(manifest_json) = obj.get_mut("chunk_manifest") {
                if let Some(manifest_obj) = manifest_json.as_object_mut() {
                    let total = manifest.chunks.len();
                    if total > 0 {
                        let mut window = serde_json::Map::new();
                        let idx = args.page.saturating_sub(1).min(total - 1);
                        if idx > 0 {
                            let prev = &manifest.chunks[idx - 1];
                            window.insert("prev".into(), json!([prev.start, prev.end]));
                        }
                        let current = &manifest.chunks[idx];
                        window.insert("current".into(), json!([current.start, current.end]));
                        if idx + 1 < total {
                            let next = &manifest.chunks[idx + 1];
                            window.insert("next".into(), json!([next.start, next.end]));
                        }
                        manifest_obj.insert("chunks".into(), Value::Object(window));
                    }
                }
            }
        }
        let json = json!({
            "frame": frame_json,
            "page": args.page,
            "page_size": manifest.chunk_chars,
            "page_count": total_pages,
            "has_prev": args.page > 1,
            "has_next": args.page < total_pages,
            "content": content,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        print_frame_summary(&mut mem, &frame)?;
        println!(
            "Page {}/{} ({} chars per page)",
            args.page, total_pages, manifest.chunk_chars
        );
        println!();
        println!("{}", content);
    }
    Ok(())
}

#[derive(Debug)]
pub struct PreviewBounds {
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

pub fn parse_preview_bounds(
    start: Option<&String>,
    end: Option<&String>,
) -> Result<Option<PreviewBounds>> {
    let start_ms = match start {
        Some(value) => Some(parse_timecode(value)?),
        None => None,
    };
    let end_ms = match end {
        Some(value) => Some(parse_timecode(value)?),
        None => None,
    };

    if let (Some(s), Some(e)) = (start_ms, end_ms) {
        if e <= s {
            anyhow::bail!("--end must be greater than --start");
        }
    }

    if start_ms.is_none() && end_ms.is_none() {
        Ok(None)
    } else {
        Ok(Some(PreviewBounds { start_ms, end_ms }))
    }
}

fn preview_frame_media(
    mem: &mut Memvid,
    frame: &Frame,
    cli_uri: Option<&str>,
    bounds: Option<PreviewBounds>,
) -> Result<()> {
    let manifest = mem.media_manifest(frame.id)?;
    let mut mime = manifest
        .as_ref()
        .map(|m| m.mime.clone())
        .or_else(|| frame.metadata.as_ref().and_then(|meta| meta.mime.clone()))
        .unwrap_or_else(|| "application/octet-stream".to_string());

    // If mime is generic, try to detect from payload bytes
    if mime == "application/octet-stream" {
        if let Ok(bytes) = mem.frame_canonical_payload(frame.id) {
            if let Some(kind) = infer::get(&bytes) {
                mime = kind.mime_type().to_string();
            }
        }
    }

    let is_video = manifest
        .as_ref()
        .map(|media| media.kind.eq_ignore_ascii_case("video"))
        .unwrap_or_else(|| mime.starts_with("video/"));

    if is_video {
        preview_frame_video(mem, frame, cli_uri, bounds, manifest, &mime)?;
    } else {
        if bounds.is_some() {
            anyhow::bail!("--start/--end are only supported for video previews");
        }
        if is_image_mime(&mime) {
            preview_frame_image(mem, frame, cli_uri)?;
        } else if is_audio_mime(&mime) {
            preview_frame_audio_file(mem, frame, cli_uri, manifest.as_ref(), &mime)?;
        } else {
            preview_frame_document(mem, frame, cli_uri, manifest.as_ref(), &mime)?;
        }
    }
    Ok(())
}

fn preview_frame_video(
    mem: &mut Memvid,
    frame: &Frame,
    cli_uri: Option<&str>,
    bounds: Option<PreviewBounds>,
    manifest: Option<MediaManifest>,
    mime: &str,
) -> Result<()> {
    let extension = manifest
        .as_ref()
        .and_then(|m| m.filename.as_deref())
        .and_then(|name| Path::new(name).extension().and_then(|ext| ext.to_str()))
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
        .or_else(|| extension_from_mime(mime).map(|ext| ext.to_string()))
        .unwrap_or_else(|| "mp4".to_string());

    let mut temp_file = Builder::new()
        .prefix("memvid-preview-")
        .suffix(&format!(".{extension}"))
        .tempfile_in(std::env::temp_dir())
        .context("failed to create temporary preview file")?;

    let mut reader = mem
        .blob_reader(frame.id)
        .context("failed to stream payload for preview")?;
    io::copy(&mut reader, &mut temp_file).context("failed to write video data to preview file")?;
    temp_file
        .flush()
        .context("failed to flush video preview to disk")?;

    let (file, preview_path) = temp_file.keep().context("failed to persist preview file")?;
    drop(file);

    let mut display_path = preview_path.clone();
    if let Some(ref span) = bounds {
        let needs_trim = span.start_ms.is_some() || span.end_ms.is_some();
        if needs_trim {
            if let Some(trimmed) = maybe_trim_with_ffmpeg(&preview_path, &extension, span)? {
                display_path = trimmed;
            }
        }
    }

    println!("Opening preview...");
    open::that(&display_path).with_context(|| {
        format!(
            "failed to launch default video player for {}",
            display_path.display()
        )
    })?;

    let display_uri = cli_uri
        .or_else(|| frame.uri.as_deref())
        .unwrap_or("<unknown>");
    println!(
        "Opened preview for {} (frame {}) -> {} ({})",
        display_uri,
        frame.id,
        display_path.display(),
        mime
    );
    Ok(())
}

fn maybe_trim_with_ffmpeg(
    source: &Path,
    extension: &str,
    bounds: &PreviewBounds,
) -> Result<Option<PathBuf>> {
    if bounds.start_ms.is_none() && bounds.end_ms.is_none() {
        return Ok(None);
    }

    let ffmpeg = match which::which("ffmpeg") {
        Ok(path) => path,
        Err(_) => {
            warn!("ffmpeg binary not found on PATH; opening full video");
            return Ok(None);
        }
    };

    let target = std::env::temp_dir().join(format!(
        "memvid-preview-clip-{}.{}",
        Uuid::new_v4(),
        extension
    ));

    let mut command = Command::new(ffmpeg);
    command.arg("-y");
    if let Some(start) = bounds.start_ms {
        command.arg("-ss").arg(format_timestamp_ms(start));
    }
    command.arg("-i").arg(source);
    if let Some(end) = bounds.end_ms {
        command.arg("-to").arg(format_timestamp_ms(end));
    }
    command.arg("-c").arg("copy");
    command.arg(&target);

    let status = command
        .status()
        .context("failed to run ffmpeg for preview trimming")?;
    if status.success() {
        return Ok(Some(target));
    }

    let details = status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated".to_string());
    warn!("ffmpeg exited with status {details}; opening full video");
    Ok(None)
}

fn preview_frame_image(mem: &mut Memvid, frame: &Frame, cli_uri: Option<&str>) -> Result<()> {
    let bytes = mem
        .frame_canonical_payload(frame.id)
        .context("failed to load canonical payload for frame")?;
    if bytes.is_empty() {
        bail!("frame payload is empty; nothing to preview");
    }

    let detected_kind = infer::get(&bytes);
    let mut mime = frame
        .metadata
        .as_ref()
        .and_then(|meta| meta.mime.clone())
        .filter(|value| is_image_mime(value));

    if mime.is_none() {
        if let Some(kind) = &detected_kind {
            let candidate = kind.mime_type();
            if is_image_mime(candidate) {
                mime = Some(candidate.to_string());
            }
        }
    }

    let mime = mime.ok_or_else(|| anyhow!("frame does not contain an image payload"))?;
    if !is_image_mime(&mime) {
        bail!("frame mime type {mime} is not an image");
    }

    let extension = detected_kind
        .as_ref()
        .map(|kind| kind.extension().to_string())
        .or_else(|| extension_from_mime(&mime).map(|ext| ext.to_string()))
        .unwrap_or_else(|| "img".to_string());

    let suffix = format!(".{extension}");
    let mut temp_file = Builder::new()
        .prefix("memvid-preview-")
        .suffix(&suffix)
        .tempfile_in(std::env::temp_dir())
        .context("failed to create temporary preview file")?;
    temp_file
        .write_all(&bytes)
        .context("failed to write image data to preview file")?;
    temp_file
        .flush()
        .context("failed to flush preview file to disk")?;

    let (file, preview_path) = temp_file.keep().context("failed to persist preview file")?;
    drop(file);

    println!("Opening preview...");
    open::that(&preview_path).with_context(|| {
        format!(
            "failed to launch default image viewer for {}",
            preview_path.display()
        )
    })?;

    let display_uri = cli_uri
        .or_else(|| frame.uri.as_deref())
        .unwrap_or("<unknown>");
    println!(
        "Opened preview for {} (frame {}) -> {} ({})",
        display_uri,
        frame.id,
        preview_path.display(),
        mime
    );
    Ok(())
}

fn preview_frame_document(
    mem: &mut Memvid,
    frame: &Frame,
    cli_uri: Option<&str>,
    manifest: Option<&MediaManifest>,
    mime: &str,
) -> Result<()> {
    let display_uri = cli_uri
        .or_else(|| frame.uri.as_deref())
        .unwrap_or("<unknown>");

    // For documents (PDFs, etc.), prefer opening the original source file if it still exists.
    // The canonical payload for document frames contains extracted text, not the original binary.
    if let Some(source_path) = &frame.source_path {
        let source = Path::new(source_path);
        if source.exists() {
            println!("Opening preview...");
            open::that(source).with_context(|| {
                format!("failed to launch default viewer for {}", source.display())
            })?;
            println!(
                "Opened preview for {} (frame {}) -> {} ({})",
                display_uri, frame.id, source_path, mime
            );
            return Ok(());
        } else {
            warn!(
                "Original source file no longer exists: {}. Falling back to extracted content.",
                source_path
            );
        }
    }

    // Fall back to extracted content (text for PDFs, raw bytes for other documents)
    let bytes = mem
        .frame_canonical_payload(frame.id)
        .context("failed to load canonical payload for frame")?;
    if bytes.is_empty() {
        bail!("frame payload is empty; nothing to preview");
    }

    let mut extension = manifest
        .and_then(|m| m.filename.as_deref())
        .and_then(|name| Path::new(name).extension().and_then(|ext| ext.to_str()))
        .map(|ext| ext.trim_start_matches('.').to_string())
        .or_else(|| extension_from_mime(mime).map(|ext| ext.to_string()))
        .unwrap_or_else(|| "bin".to_string());

    // For documents where we only have extracted text, use .txt extension
    if frame.chunk_manifest.is_some() {
        extension = "txt".to_string();
    } else if extension == "bin" && std::str::from_utf8(&bytes).is_ok() {
        extension = "txt".to_string();
    }

    let suffix = format!(".{extension}");
    let mut temp_file = Builder::new()
        .prefix("memvid-preview-")
        .suffix(&suffix)
        .tempfile_in(std::env::temp_dir())
        .context("failed to create temporary preview file")?;
    temp_file
        .write_all(&bytes)
        .context("failed to write document data to preview file")?;
    temp_file
        .flush()
        .context("failed to flush preview file to disk")?;

    let (file, preview_path) = temp_file.keep().context("failed to persist preview file")?;
    drop(file);

    println!("Opening preview...");
    open::that(&preview_path).with_context(|| {
        format!(
            "failed to launch default viewer for {}",
            preview_path.display()
        )
    })?;

    println!(
        "Opened preview for {} (frame {}) -> {} ({})",
        display_uri,
        frame.id,
        preview_path.display(),
        if frame.chunk_manifest.is_some() {
            "text/plain (extracted)"
        } else {
            mime
        }
    );
    Ok(())
}

fn preview_frame_audio_file(
    mem: &mut Memvid,
    frame: &Frame,
    cli_uri: Option<&str>,
    manifest: Option<&MediaManifest>,
    mime: &str,
) -> Result<()> {
    let bytes = mem
        .frame_canonical_payload(frame.id)
        .context("failed to load canonical payload for frame")?;
    if bytes.is_empty() {
        bail!("frame payload is empty; nothing to preview");
    }

    let mut extension = manifest
        .and_then(|m| m.filename.as_deref())
        .and_then(|name| Path::new(name).extension().and_then(|ext| ext.to_str()))
        .map(|ext| ext.trim_start_matches('.').to_string())
        .or_else(|| extension_from_mime(mime).map(|ext| ext.to_string()))
        .unwrap_or_else(|| "audio".to_string());

    if extension == "bin" {
        extension = "audio".to_string();
    }

    let suffix = format!(".{extension}");
    let mut temp_file = Builder::new()
        .prefix("memvid-preview-")
        .suffix(&suffix)
        .tempfile_in(std::env::temp_dir())
        .context("failed to create temporary preview file")?;
    temp_file
        .write_all(&bytes)
        .context("failed to write audio data to preview file")?;
    temp_file
        .flush()
        .context("failed to flush preview file to disk")?;

    let (file, preview_path) = temp_file.keep().context("failed to persist preview file")?;
    drop(file);

    println!("Opening preview...");
    open::that(&preview_path).with_context(|| {
        format!(
            "failed to launch default audio player for {}",
            preview_path.display()
        )
    })?;

    let display_uri = cli_uri
        .or_else(|| frame.uri.as_deref())
        .unwrap_or("<unknown>");
    println!(
        "Opened preview for {} (frame {}) -> {} ({})",
        display_uri,
        frame.id,
        preview_path.display(),
        mime
    );
    Ok(())
}

#[cfg(feature = "audio-playback")]
fn play_frame_audio(
    mem: &mut Memvid,
    frame: &Frame,
    start_seconds: Option<f32>,
    end_seconds: Option<f32>,
) -> Result<()> {
    use rodio::Source;

    if let (Some(start), Some(end)) = (start_seconds, end_seconds) {
        if end <= start {
            bail!("--end-seconds must be greater than --start-seconds");
        }
    }

    let bytes = mem
        .frame_canonical_payload(frame.id)
        .context("failed to load canonical payload for frame")?;
    if bytes.is_empty() {
        bail!("frame payload is empty; nothing to play");
    }

    let start = start_seconds.unwrap_or(0.0).max(0.0);
    let duration_meta = frame
        .metadata
        .as_ref()
        .and_then(|meta| meta.audio.as_ref())
        .and_then(|audio| audio.duration_secs)
        .unwrap_or(0.0);

    if duration_meta > 0.0 && start >= duration_meta {
        bail!("start-seconds ({start:.2}) exceeds audio duration ({duration_meta:.2})");
    }

    if let Some(end) = end_seconds {
        if duration_meta > 0.0 && end > duration_meta + f32::EPSILON {
            warn!(
                "requested end-seconds {:.2} exceeds known duration {:.2}; clamping",
                end, duration_meta
            );
        }
    }

    let cursor = Cursor::new(bytes);
    let decoder = rodio::Decoder::new(cursor).context("failed to decode audio stream")?;
    let (_stream, stream_handle) =
        rodio::OutputStream::try_default().context("failed to open default audio output")?;
    let sink = rodio::Sink::try_new(&stream_handle).context("failed to create audio sink")?;
    let display_uri = frame.uri.as_deref().unwrap_or("<unknown>");

    if let Some(end) = end_seconds {
        let effective_end = if duration_meta > 0.0 {
            end.min(duration_meta)
        } else {
            end
        };
        let duration = (effective_end - start).max(0.0);
        if duration <= 0.0 {
            bail!("playback duration is zero; adjust start/end seconds");
        }
        let source = decoder
            .skip_duration(Duration::from_secs_f32(start))
            .take_duration(Duration::from_secs_f32(duration));
        sink.append(source);
        let segment_desc = format!("{start:.2}s → {effective_end:.2}s");
        announce_playback(display_uri, &segment_desc);
    } else {
        let source = decoder.skip_duration(Duration::from_secs_f32(start));
        sink.append(source);
        let segment_desc = format!("{start:.2}s → end");
        announce_playback(display_uri, &segment_desc);
    }
    sink.sleep_until_end();
    Ok(())
}

#[cfg(feature = "audio-playback")]
fn announce_playback(uri: &str, segment_desc: &str) {
    println!("Playing {uri} ({segment_desc})");
}

fn is_image_mime(value: &str) -> bool {
    let normalized = value.split(';').next().unwrap_or(value).trim();
    normalized.to_ascii_lowercase().starts_with("image/")
}

fn is_audio_mime(value: &str) -> bool {
    let normalized = value.split(';').next().unwrap_or(value).trim();
    normalized.to_ascii_lowercase().starts_with("audio/")
}

pub fn extension_from_mime(mime: &str) -> Option<&'static str> {
    let normalized = mime
        .split(';')
        .next()
        .unwrap_or(mime)
        .trim()
        .to_ascii_lowercase();
    match normalized.as_str() {
        "image/jpeg" | "image/jpg" => Some("jpg"),
        "image/png" => Some("png"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        "image/bmp" => Some("bmp"),
        "image/tiff" => Some("tiff"),
        "image/x-icon" | "image/vnd.microsoft.icon" => Some("ico"),
        "image/svg+xml" => Some("svg"),
        "video/mp4" | "video/iso.segment" => Some("mp4"),
        "video/quicktime" => Some("mov"),
        "video/webm" => Some("webm"),
        "video/x-matroska" | "video/matroska" => Some("mkv"),
        "video/x-msvideo" => Some("avi"),
        "video/mpeg" => Some("mpg"),
        "application/pdf" => Some("pdf"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/wav" | "audio/x-wav" => Some("wav"),
        "audio/x-flac" | "audio/flac" => Some("flac"),
        "audio/ogg" | "audio/vorbis" => Some("ogg"),
        "audio/x-m4a" | "audio/mp4" => Some("m4a"),
        "audio/aac" => Some("aac"),
        "audio/x-aiff" | "audio/aiff" => Some("aiff"),
        "text/plain" => Some("txt"),
        "text/markdown" | "text/x-markdown" => Some("md"),
        "text/html" => Some("html"),
        "application/xhtml+xml" => Some("xhtml"),
        "application/json" | "text/json" | "application/vnd.api+json" => Some("json"),
        "application/xml" | "text/xml" => Some("xml"),
        "text/csv" | "application/csv" => Some("csv"),
        "application/javascript" | "text/javascript" => Some("js"),
        "text/css" => Some("css"),
        "application/yaml" | "application/x-yaml" | "text/yaml" => Some("yaml"),
        "application/rtf" => Some("rtf"),
        "application/msword" => Some("doc"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Some("docx"),
        "application/vnd.ms-powerpoint" => Some("ppt"),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => Some("pptx"),
        "application/vnd.ms-excel" => Some("xls"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Some("xlsx"),
        "application/zip" => Some("zip"),
        "application/x-tar" => Some("tar"),
        "application/x-7z-compressed" => Some("7z"),
        _ => None,
    }
}
pub fn search_snippet(text: Option<&String>) -> Option<String> {
    text.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.chars().take(160).collect())
        }
    })
}
pub fn frame_to_json(frame: &Frame) -> serde_json::Value {
    json!({
        "id": frame.id,
        "status": frame_status_str(frame.status),
        "timestamp": frame.timestamp,
        "kind": frame.kind,
        "track": frame.track,
        "uri": frame.uri,
        "title": frame.title,
        "payload_length": frame.payload_length,
        "canonical_encoding": format!("{:?}", frame.canonical_encoding),
        "canonical_length": frame.canonical_length,
        "role": format!("{:?}", frame.role),
        "parent_id": frame.parent_id,
        "chunk_index": frame.chunk_index,
        "chunk_count": frame.chunk_count,
        "tags": frame.tags,
        "labels": frame.labels,
        "search_text": frame.search_text,
        "metadata": frame.metadata,
        "extra_metadata": frame.extra_metadata,
        "content_dates": frame.content_dates,
        "chunk_manifest": frame.chunk_manifest,
        "supersedes": frame.supersedes,
        "superseded_by": frame.superseded_by,
        "source_sha256": frame.source_sha256.map(|h| hex::encode(h)),
        "source_path": frame.source_path,
    })
}
pub fn print_frame_summary(mem: &mut Memvid, frame: &Frame) -> Result<()> {
    println!("Frame {} [{}]", frame.id, frame_status_str(frame.status));
    println!("Timestamp: {}", frame.timestamp);
    if let Some(uri) = &frame.uri {
        println!("URI: {uri}");
    }
    if let Some(title) = &frame.title {
        println!("Title: {title}");
    }
    if let Some(kind) = &frame.kind {
        println!("Kind: {kind}");
    }
    if let Some(track) = &frame.track {
        println!("Track: {track}");
    }
    if let Some(supersedes) = frame.supersedes {
        println!("Supersedes frame: {supersedes}");
    }
    if let Some(successor) = frame.superseded_by {
        println!("Superseded by frame: {successor}");
    }
    println!(
        "Payload: {} bytes (canonical {:?}, logical {:?})",
        frame.payload_length, frame.canonical_encoding, frame.canonical_length
    );
    if !frame.tags.is_empty() {
        println!("Tags: {}", frame.tags.join(", "));
    }
    if !frame.labels.is_empty() {
        println!("Labels: {}", frame.labels.join(", "));
    }
    if let Some(snippet) = search_snippet(frame.search_text.as_ref()) {
        println!("Search text: {snippet}");
    }
    if let Some(meta) = &frame.metadata {
        let rendered = serde_json::to_string_pretty(meta)?;
        println!("Metadata: {rendered}");
    }
    if !frame.extra_metadata.is_empty() {
        let mut entries: Vec<_> = frame.extra_metadata.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        println!("Extra metadata:");
        for (key, value) in entries {
            println!("  {key}: {value}");
        }
    }
    if !frame.content_dates.is_empty() {
        println!("Content dates: {}", frame.content_dates.join(", "));
    }
    // Show no-raw mode info if applicable
    if let Some(hash) = frame.source_sha256 {
        println!(
            "Source SHA256: {} (raw binary not stored)",
            hex::encode(hash)
        );
        if let Some(path) = &frame.source_path {
            println!("Source path: {path}");
        }
    }
    match mem.frame_embedding(frame.id) {
        Ok(Some(embedding)) => println!("Embedding: {} dimensions", embedding.len()),
        Ok(None) => println!("Embedding: none"),
        Err(err) => println!("Embedding: unavailable ({err})"),
    }
    Ok(())
}
fn canonical_text_for_view(mem: &mut Memvid, frame: &Frame) -> Result<String> {
    let bytes = mem.frame_canonical_payload(frame.id)?;
    let raw = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(err) => {
            let bytes = err.into_bytes();
            String::from_utf8_lossy(&bytes).into_owned()
        }
    };

    Ok(normalize_text(&raw, usize::MAX)
        .map(|n| n.text)
        .unwrap_or_default())
}

fn manifests_match_text(text: &str, manifest: &TextChunkManifest) -> bool {
    if manifest.chunk_chars == 0 || manifest.chunks.is_empty() {
        return false;
    }
    let total_chars = text.chars().count();
    manifest
        .chunks
        .iter()
        .all(|chunk| chunk.start <= chunk.end && chunk.end <= total_chars)
}

fn canonical_manifest_from_frame(text: &str, frame: &Frame) -> Option<TextChunkManifest> {
    let primary = frame
        .chunk_manifest
        .clone()
        .filter(|manifest| manifests_match_text(text, manifest));
    if primary.is_some() {
        return primary;
    }

    frame
        .extra_metadata
        .get(CHUNK_MANIFEST_KEY)
        .and_then(|raw| serde_json::from_str::<TextChunkManifest>(raw).ok())
        .filter(|manifest| manifests_match_text(text, manifest))
}

fn compute_chunk_manifest(text: &str, chunk_chars: usize) -> TextChunkManifest {
    let normalized = normalize_text(text, usize::MAX)
        .map(|n| n.text)
        .unwrap_or_default();

    let effective_chunk = chunk_chars.max(1);
    let total_chars = normalized.chars().count();
    if total_chars == 0 {
        return TextChunkManifest {
            chunk_chars: effective_chunk,
            chunks: vec![TextChunkRange { start: 0, end: 0 }],
        };
    }
    if total_chars <= effective_chunk {
        return TextChunkManifest {
            chunk_chars: effective_chunk,
            chunks: vec![TextChunkRange {
                start: 0,
                end: total_chars,
            }],
        };
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total_chars {
        let end = (start + effective_chunk).min(total_chars);
        chunks.push(TextChunkRange { start, end });
        start = end;
    }
    TextChunkManifest {
        chunk_chars: effective_chunk,
        chunks,
    }
}

fn extract_chunk_slice(text: &str, range: &TextChunkRange) -> String {
    if range.start >= range.end || text.is_empty() {
        return String::new();
    }
    let mut start_byte = text.len();
    let mut end_byte = text.len();
    let mut idx = 0usize;
    for (byte_offset, _) in text.char_indices() {
        if idx == range.start {
            start_byte = byte_offset;
        }
        if idx == range.end {
            end_byte = byte_offset;
            break;
        }
        idx += 1;
    }
    if start_byte == text.len() {
        return String::new();
    }
    if end_byte == text.len() {
        end_byte = text.len();
    }
    text[start_byte..end_byte].to_string()
}
