//! Local ticket caching
//!
//! This module provides functionality for caching tickets locally to avoid
//! repeated API calls and enable offline usage.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::CliConfig;

const CACHE_DIR_NAME: &str = "tickets";

/// A cached ticket entry stored locally
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedTicket {
    pub memory_id: Uuid,
    pub issuer: String,
    pub seq_no: i64,
    pub expires_in: u64,
    #[serde(default)]
    pub capacity_bytes: Option<u64>,
    pub signature: String,
}

pub fn store(config: &CliConfig, entry: &CachedTicket) -> Result<()> {
    let path = cache_path(config, &entry.memory_id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create ticket cache directory: {}",
                parent.display()
            )
        })?;
    }
    let tmp = path.with_extension("json.tmp");
    let data = serde_json::to_vec_pretty(entry)?;
    fs::write(&tmp, data)
        .with_context(|| format!("failed to write ticket cache file: {}", tmp.display()))?;
    fs::rename(&tmp, &path).with_context(|| {
        format!(
            "failed to atomically persist ticket cache file: {}",
            path.display()
        )
    })?;
    Ok(())
}

pub fn load(config: &CliConfig, memory_id: &Uuid) -> Result<CachedTicket> {
    let path = cache_path(config, memory_id);
    let data = fs::read(&path)
        .with_context(|| format!("no cached ticket available for memory {}", memory_id))?;
    let entry = serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse cached ticket: {}", path.display()))?;
    Ok(entry)
}

fn cache_path(config: &CliConfig, memory_id: &Uuid) -> PathBuf {
    config
        .cache_dir
        .join(CACHE_DIR_NAME)
        .join(format!("{memory_id}.json"))
}
