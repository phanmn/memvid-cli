//! Debug command handler

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{ArgAction, Args};

use crate::utils::open_read_only_mem;

/// Arguments for the `debug-segment` subcommand
#[derive(Args)]
pub struct DebugSegmentArgs {
    #[arg(value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    pub file: PathBuf,
    #[arg(long = "segment-id", value_name = "ID")]
    pub segment_id: u64,
    #[arg(long = "hex-dump", action = ArgAction::SetTrue)]
    pub hex_dump: bool,
    #[arg(long = "max-bytes", value_name = "N", default_value = "256")]
    pub max_bytes: usize,
}

/// Handler for `memvid debug-segment`
pub fn handle_debug_segment(args: DebugSegmentArgs) -> Result<()> {
    let mut mem = open_read_only_mem(&args.file)?;
    let Some((descriptor, bytes)) = mem.read_vec_segment(args.segment_id)? else {
        bail!(
            "vector segment {} not found in {}",
            args.segment_id,
            args.file.display()
        );
    };

    println!("Vector segment {}", descriptor.common.segment_id);
    println!("  offset: {}", descriptor.common.bytes_offset);
    println!("  length: {} bytes", descriptor.common.bytes_length);
    println!("  checksum: {}", hex::encode(descriptor.common.checksum));
    println!("  vectors: {}", descriptor.vector_count);
    println!("  dimension: {}", descriptor.dimension);
    println!("  compression: {:?}", descriptor.vector_compression);
    println!("  bytes loaded: {}", bytes.len());

    if args.hex_dump {
        if args.max_bytes == 0 {
            println!("Hex dump skipped: --max-bytes is 0");
            return Ok(());
        }
        if bytes.is_empty() {
            println!("Hex dump skipped: segment contains no bytes");
            return Ok(());
        }
        let limit = bytes.len().min(args.max_bytes);
        println!("Hex dump (showing {} of {} bytes):", limit, bytes.len());
        for (row, chunk) in bytes[..limit].chunks(16).enumerate() {
            let offset = row * 16;
            print!("{offset:08x}: ");
            for i in 0..16 {
                if let Some(byte) = chunk.get(i) {
                    print!("{byte:02x} ");
                } else {
                    print!("   ");
                }
            }
            print!("|");
            for byte in chunk {
                let ch = if (0x20..=0x7e).contains(byte) {
                    *byte as char
                } else {
                    '.'
                };
                print!("{ch}");
            }
            println!("|");
        }
    }

    Ok(())
}
