//! Analytics Module for Memvid CLI
//!
//! Provides anonymous telemetry tracking with:
//! - Zero latency impact (async fire-and-forget)
//! - Local JSONL queue with background flush
//! - SHA256-based anonymous IDs
//! - Opt-out via MEMVID_TELEMETRY=0

mod flush;
mod id;
mod queue;

pub use self::track_command_with_tier as track_command_tier;
pub use flush::{flush_analytics, force_flush_sync};
pub use id::{generate_anon_id, generate_file_hash};
pub use queue::{track_event, AnalyticsEvent};

use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to check if telemetry is enabled
static TELEMETRY_ENABLED: AtomicBool = AtomicBool::new(true);

/// Check if telemetry is enabled
/// Set MEMVID_TELEMETRY=0 to disable
pub fn is_telemetry_enabled() -> bool {
    TELEMETRY_ENABLED.load(Ordering::Relaxed)
}

/// Initialize analytics (called once at startup)
/// Checks environment variable and starts background flush
pub fn init_analytics() {
    // Check opt-out
    if let Ok(val) = std::env::var("MEMVID_TELEMETRY") {
        if val == "0" || val.to_lowercase() == "false" {
            TELEMETRY_ENABLED.store(false, Ordering::Relaxed);
            return;
        }
    }

    // Start background flush task
    flush::start_background_flush();
}

/// Convenience function to track a command execution
pub fn track_command(
    file_path: Option<&str>,
    command: &str,
    success: bool,
    file_created: bool,
    file_opened: bool,
) {
    track_command_with_tier(
        file_path,
        command,
        success,
        file_created,
        file_opened,
        "free",
    )
}

/// Track a command execution with user tier info
pub fn track_command_with_tier(
    file_path: Option<&str>,
    command: &str,
    success: bool,
    file_created: bool,
    file_opened: bool,
    user_tier: &str,
) {
    if !is_telemetry_enabled() {
        return;
    }

    let anon_id = generate_anon_id(file_path);
    let file_hash = file_path
        .map(|p| generate_file_hash(p))
        .unwrap_or_else(|| "none".to_string());

    let event = AnalyticsEvent {
        anon_id,
        file_hash,
        client: "cli".to_string(),
        command: command.to_string(),
        success,
        timestamp: chrono::Utc::now().to_rfc3339(),
        file_created,
        file_opened,
        user_tier: user_tier.to_string(),
    };

    // Fire-and-forget queue
    track_event(event);
}
