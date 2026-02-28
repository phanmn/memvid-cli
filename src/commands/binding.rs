//! Commands for managing memory bindings to dashboard

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use memvid_core::Memvid;

/// Arguments for the binding command
#[derive(Args)]
pub struct BindingArgs {
    /// Path to the MV2 file
    pub file: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Handle the binding command
pub fn handle_binding(args: BindingArgs) -> Result<()> {
    let mem = Memvid::open(&args.file)?;

    if let Some(binding) = mem.get_memory_binding() {
        if args.json {
            let json = serde_json::json!({
                "memory_id": binding.memory_id.to_string(),
                "memory_name": binding.memory_name,
                "bound_at": binding.bound_at.to_rfc3339(),
                "api_url": binding.api_url,
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        } else {
            println!("Memory Binding:");
            println!("  Memory ID:   {}", binding.memory_id);
            println!("  Memory Name: {}", binding.memory_name);
            println!("  Bound At:    {}", binding.bound_at.to_rfc3339());
            println!("  API URL:     {}", binding.api_url);
        }
    } else if args.json {
        println!("null");
    } else {
        println!("This file is not bound to any memory.");
    }

    Ok(())
}
