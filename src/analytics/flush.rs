//! Background flush for analytics events
//!
//! Sends queued events to the server in batches.
//! Runs asynchronously to not block CLI commands.

#[cfg(test)]
use super::queue::AnalyticsEvent;
use super::queue::{clear_queue, pending_count, read_pending_events};
use std::sync::OnceLock;
use std::time::Duration;

/// Analytics API endpoint
const ANALYTICS_ENDPOINT: &str = "https://memvid.com/api/analytics/ingest";

/// Development endpoint (for testing)
#[cfg(debug_assertions)]
const DEV_ANALYTICS_ENDPOINT: &str = "http://localhost:3001/api/analytics/ingest";

/// Maximum events per batch
const MAX_BATCH_SIZE: usize = 100;

/// Minimum interval between flush attempts (seconds)
const FLUSH_INTERVAL_SECS: u64 = 30;

/// Last flush timestamp
static LAST_FLUSH: OnceLock<std::sync::Mutex<std::time::Instant>> = OnceLock::new();

/// Get the analytics endpoint URL
fn get_endpoint() -> String {
    // Check for explicit analytics endpoint override
    if let Ok(url) = std::env::var("MEMVID_ANALYTICS_URL") {
        return url;
    }

    // Derive from dashboard URL if set (env var first, then config file)
    if let Ok(dashboard_url) = std::env::var("MEMVID_DASHBOARD_URL") {
        return format!(
            "{}/api/analytics/ingest",
            dashboard_url.trim_end_matches('/')
        );
    }

    // Check config file for dashboard_url
    if let Ok(config) = crate::commands::config::PersistentConfig::load() {
        if let Some(dashboard_url) = config.dashboard_url {
            return format!(
                "{}/api/analytics/ingest",
                dashboard_url.trim_end_matches('/')
            );
        }
    }

    // Use dev endpoint in debug builds if MEMVID_DEV is set
    #[cfg(debug_assertions)]
    if std::env::var("MEMVID_DEV").is_ok() {
        return DEV_ANALYTICS_ENDPOINT.to_string();
    }

    ANALYTICS_ENDPOINT.to_string()
}

/// Initialize analytics (no-op, kept for API compatibility)
pub fn start_background_flush() {
    // Initialize last flush time
    LAST_FLUSH.get_or_init(|| std::sync::Mutex::new(std::time::Instant::now()));
}

/// Force flush analytics synchronously (ignores interval check)
/// Call this at the end of main() to ensure events are sent
pub fn force_flush_sync() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    do_flush_sync(true)
}

/// Flush analytics synchronously (for background thread)
fn flush_analytics_sync() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    do_flush_sync(false)
}

/// Internal flush implementation
fn do_flush_sync(force: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Check if enough time has passed (skip if force=true)
    if !force {
        if let Some(last) = LAST_FLUSH.get() {
            if let Ok(last_time) = last.lock() {
                if last_time.elapsed() < Duration::from_secs(FLUSH_INTERVAL_SECS) {
                    return Ok(()); // Too soon
                }
            }
        }
    }

    // Check if there are events to flush
    let count = pending_count();
    if count == 0 {
        return Ok(());
    }

    // Read events
    let events = read_pending_events();
    if events.is_empty() {
        return Ok(());
    }

    // Send in batches using reqwest blocking client
    let endpoint = get_endpoint();
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    for chunk in events.chunks(MAX_BATCH_SIZE) {
        let payload = serde_json::json!({
            "events": chunk
        });

        match client
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
        {
            Ok(response) => {
                if response.status().is_success() {
                    #[cfg(debug_assertions)]
                    eprintln!("[analytics] Flushed {} events", chunk.len());
                } else {
                    #[cfg(debug_assertions)]
                    eprintln!("[analytics] Server returned {}", response.status());
                }
            }
            Err(e) => {
                // Log but don't fail - we'll retry next time
                #[cfg(debug_assertions)]
                eprintln!("[analytics] Flush error: {}", e);

                // Update last flush time even on error to prevent hammering
                if let Some(last) = LAST_FLUSH.get() {
                    if let Ok(mut last_time) = last.lock() {
                        *last_time = std::time::Instant::now();
                    }
                }

                return Err(Box::new(e));
            }
        }
    }

    // Clear queue on success
    clear_queue();

    // Update last flush time
    if let Some(last) = LAST_FLUSH.get() {
        if let Ok(mut last_time) = last.lock() {
            *last_time = std::time::Instant::now();
        }
    }

    Ok(())
}

/// Public function to trigger a flush (async spawn)
pub fn flush_analytics() {
    std::thread::spawn(|| {
        let _ = flush_analytics_sync();
    });
}

/// Send events directly without queuing (for testing)
#[cfg(test)]
#[allow(dead_code)]
pub fn send_events_direct(events: Vec<AnalyticsEvent>) -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = get_endpoint();
    let payload = serde_json::json!({ "events": events });

    reqwest::blocking::Client::new()
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()?;

    Ok(())
}
