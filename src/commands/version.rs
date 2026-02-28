//! Version command handler

use anyhow::Result;

/// Handle the version command
pub fn handle_version() -> Result<()> {
    println!("memvid-cli {}", env!("CARGO_PKG_VERSION"));
    Ok(())
}
