//! Local JSONL queue for analytics events
//!
//! Events are appended to a local file and periodically flushed
//! to the server in batches.

use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Mutex;

/// Analytics event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEvent {
    pub anon_id: String,
    pub file_hash: String,
    pub client: String,
    pub command: String,
    pub success: bool,
    pub timestamp: String,
    #[serde(default)]
    pub file_created: bool,
    #[serde(default)]
    pub file_opened: bool,
    /// User tier: "free", or plan name like "starter", "teams", "enterprise"
    #[serde(default = "default_tier")]
    pub user_tier: String,
}

fn default_tier() -> String {
    "free".to_string()
}

/// Global queue file lock
static QUEUE_LOCK: Mutex<()> = Mutex::new(());

/// Get the analytics queue directory
fn get_analytics_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("memvid").join("analytics"))
}

/// Get the queue file path
fn get_queue_path() -> Option<PathBuf> {
    get_analytics_dir().map(|d| d.join("queue.jsonl"))
}

/// Ensure analytics directory exists
fn ensure_dir() -> Option<PathBuf> {
    let dir = get_analytics_dir()?;
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Track an analytics event (append to local queue)
/// This is fire-and-forget - errors are silently ignored
pub fn track_event(event: AnalyticsEvent) {
    if let Err(_e) = track_event_inner(event) {
        // Silently ignore errors - analytics should never impact UX
        #[cfg(debug_assertions)]
        eprintln!("[analytics] Failed to queue event: {}", _e);
    }
}

fn track_event_inner(event: AnalyticsEvent) -> std::io::Result<()> {
    let _lock = QUEUE_LOCK.lock().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::Other, "Failed to acquire queue lock")
    })?;

    let queue_path = get_queue_path()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Cannot find data dir"))?;

    ensure_dir();

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&queue_path)?;

    let json = serde_json::to_string(&event)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    writeln!(file, "{}", json)?;
    Ok(())
}

/// Read all pending events from the queue
pub fn read_pending_events() -> Vec<AnalyticsEvent> {
    let _lock = match QUEUE_LOCK.lock() {
        Ok(lock) => lock,
        Err(_) => return vec![],
    };

    let queue_path = match get_queue_path() {
        Some(p) => p,
        None => return vec![],
    };

    if !queue_path.exists() {
        return vec![];
    }

    let file = match fs::File::open(&queue_path) {
        Ok(f) => f,
        Err(_) => return vec![],
    };

    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for line in reader.lines() {
        if let Ok(line) = line {
            if let Ok(event) = serde_json::from_str::<AnalyticsEvent>(&line) {
                events.push(event);
            }
        }
    }

    events
}

/// Clear the queue after successful flush
pub fn clear_queue() {
    let _lock = match QUEUE_LOCK.lock() {
        Ok(lock) => lock,
        Err(_) => return,
    };

    if let Some(queue_path) = get_queue_path() {
        let _ = fs::remove_file(&queue_path);
    }
}

/// Get the number of pending events
pub fn pending_count() -> usize {
    read_pending_events().len()
}

/// Get the queue file size in bytes
#[allow(dead_code)]
pub fn queue_size_bytes() -> u64 {
    get_queue_path()
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0)
}
