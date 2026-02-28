//! Utility functions for the Memvid CLI

use std::path::Path;

use anyhow::{bail, Result};
use memvid_core::{error::LockOwnerHint, Memvid, Tier};
use serde_json::json;

use crate::config::CliConfig;
use crate::org_ticket_cache;

/// Free tier file size limit: 50 MB
/// Files larger than this require a paid plan with API key authentication
/// With --no-raw default, 50 MB fits ~700 small documents or ~4 large PDFs
pub const FREE_TIER_MAX_FILE_SIZE: u64 = 50 * 1024 * 1024; // 50 MB

/// Minimum file size: 10 MB
/// Ensures memory files have reasonable capacity for basic usage
pub const MIN_FILE_SIZE: u64 = 10 * 1024 * 1024; // 10 MB

/// Check if the user's plan allows write/query operations
///
/// Returns Ok(()) if:
/// - No API key (free tier user, not bound to dashboard)
/// - Subscription is active, trialing, or past_due
/// - Subscription is canceled but still in grace period (planEndDate > now)
///
/// Returns error if:
/// - Subscription is canceled AND grace period has expired (planEndDate < now)
///
/// This enables strict access control: expired plans can only view data,
/// not add new content or run queries.
pub fn require_active_plan(config: &CliConfig, operation: &str) -> Result<()> {
    // No API key = free tier user, not bound to dashboard - allow
    if config.api_key.is_none() {
        return Ok(());
    }

    // Get fresh org ticket for write operations
    // This ensures we have up-to-date subscription status (max 5 min old)
    // to prevent users from using CLI after cancellation
    let ticket = match org_ticket_cache::get_fresh_for_writes(config) {
        Some(t) => t,
        None => return Ok(()), // No ticket = allow (free tier fallback)
    };

    // Active subscriptions are always allowed
    let status = &ticket.subscription_status;
    if status == "active" || status == "trialing" || status == "past_due" {
        return Ok(());
    }

    // Canceled subscription - check if still in grace period
    if status == "canceled" {
        if ticket.is_in_grace_period() {
            // Still in grace period - allow
            return Ok(());
        }

        // Grace period expired - block write/query operations
        bail!(
            "Your subscription has expired.\n\n\
             The '{}' operation requires an active subscription.\n\
             Your plan ended on: {}\n\n\
             You can still view your data with:\n\
             • memvid stats <file>     - View file statistics\n\
             • memvid timeline <file>  - View timeline\n\
             • memvid view <file>      - View frames\n\n\
             To restore full access, reactivate your subscription:\n\
             https://memvid.com/dashboard/plan",
            operation,
            ticket.plan_end_date.as_deref().unwrap_or("unknown")
        );
    }

    // Inactive or other status - allow (fallback to capacity-based)
    Ok(())
}

/// Get the effective capacity limit for the current user
///
/// If an API key is configured and a valid org ticket exists, uses the
/// ticket's capacity_bytes. Otherwise falls back to free tier limit.
pub fn get_effective_capacity(config: &CliConfig) -> u64 {
    if let Some(ticket) = org_ticket_cache::get_optional(config) {
        ticket.capacity_bytes()
    } else {
        FREE_TIER_MAX_FILE_SIZE
    }
}

/// Check if a file exceeds the free tier limit and require API key if so
///
/// Returns Ok(()) if:
/// - File is under 1GB (no API key required)
/// - File is over 1GB AND API key is configured with valid paid plan
///
/// Returns error if file is over 1GB and no API key/paid plan
pub fn ensure_api_key_for_large_file(file_size: u64, config: &CliConfig) -> Result<()> {
    if file_size <= FREE_TIER_MAX_FILE_SIZE {
        return Ok(());
    }

    // File exceeds 1GB - require API key
    if config.api_key.is_none() {
        let size_str = format_bytes(file_size);
        let limit_str = format_bytes(FREE_TIER_MAX_FILE_SIZE);
        bail!(
            "File size ({}) exceeds free tier limit ({}).\n\n\
             To work with files larger than 1GB, you need a paid plan.\n\
             1. Sign up or log in at https://memvid.com/dashboard\n\
             2. Get your API key from the dashboard\n\
             3. Set it: export MEMVID_API_KEY=your_api_key\n\n\
             Learn more: https://memvid.com/pricing",
            size_str,
            limit_str
        );
    }

    // API key is set, check plan capacity
    if let Some(ticket) = org_ticket_cache::get_optional(config) {
        if file_size > ticket.capacity_bytes() {
            let size_str = format_bytes(file_size);
            let capacity_str = format_bytes(ticket.capacity_bytes());
            bail!(
                "File size ({}) exceeds your {} plan capacity ({}).\n\n\
                 Upgrade to a higher plan to work with larger files.\n\
                 Visit: https://memvid.com/dashboard/plan",
                size_str,
                ticket.plan_name,
                capacity_str
            );
        }
    }

    Ok(())
}

/// Check total memory size against plan capacity limit
/// Called before operations that would increase memory size
pub fn ensure_capacity_with_api_key(
    current_size: u64,
    additional_size: u64,
    config: &CliConfig,
) -> Result<()> {
    let total = current_size.saturating_add(additional_size);

    // Get effective capacity limit (from ticket or free tier)
    let capacity_limit = get_effective_capacity(config);

    if total <= capacity_limit {
        return Ok(());
    }

    // Total would exceed capacity limit
    let current_str = format_bytes(current_size);
    let additional_str = format_bytes(additional_size);
    let total_str = format_bytes(total);
    let limit_str = format_bytes(capacity_limit);

    if config.api_key.is_none() {
        bail!(
            "This operation would exceed the free tier limit.\n\n\
             Current size:    {}\n\
             Adding:          {}\n\
             Total:           {}\n\
             Free tier limit: {}\n\n\
             To store more than 1GB, you need a paid plan.\n\
             1. Sign up or log in at https://memvid.com/dashboard\n\
             2. Get your API key from the dashboard\n\
             3. Set it: export MEMVID_API_KEY=your_api_key\n\n\
             Learn more: https://memvid.com/pricing",
            current_str,
            additional_str,
            total_str,
            limit_str
        );
    }

    // API key is set but exceeding plan capacity
    let plan_name = org_ticket_cache::get_optional(config)
        .map(|t| t.plan_name.clone())
        .unwrap_or_else(|| "current".to_string());

    bail!(
        "This operation would exceed your {} plan capacity.\n\n\
         Current size:  {}\n\
         Adding:        {}\n\
         Total:         {}\n\
         Plan capacity: {}\n\n\
         Upgrade to a higher plan to store more data.\n\
         Visit: https://memvid.com/dashboard/plan",
        plan_name,
        current_str,
        additional_str,
        total_str,
        limit_str
    );
}

/// Check if the current plan allows a feature
pub fn ensure_feature_access(feature: &str, config: &CliConfig) -> Result<()> {
    let ticket = match org_ticket_cache::get_optional(config) {
        Some(t) => t,
        None => {
            // No API key - check if feature is in free tier
            let free_features = [
                "core",
                "temporal_track",
                "clip",
                "whisper",
                "temporal_enrich",
            ];
            if free_features.contains(&feature) {
                return Ok(());
            }
            bail!(
                "The '{}' feature requires a paid plan.\n\n\
                 1. Sign up or log in at https://memvid.com/dashboard\n\
                 2. Subscribe to a paid plan\n\
                 3. Get your API key from the dashboard\n\
                 4. Set it: export MEMVID_API_KEY=your_api_key\n\n\
                 Learn more: https://memvid.com/pricing",
                feature
            );
        }
    };

    // Check if feature is in the plan's feature list
    // Enterprise gets all features
    if ticket.plan_id == "enterprise" || ticket.ticket.features.contains(&"*".to_string()) {
        return Ok(());
    }

    if ticket.ticket.features.contains(&feature.to_string()) {
        return Ok(());
    }

    bail!(
        "The '{}' feature is not available on your {} plan.\n\n\
         Upgrade to access this feature.\n\
         Visit: https://memvid.com/dashboard/plan",
        feature,
        ticket.plan_name
    );
}

/// Open a memory file in read-only mode
pub fn open_read_only_mem(path: &Path) -> Result<Memvid> {
    Ok(Memvid::open_read_only(path)?)
}

/// Format bytes in a human-readable format (B, KB, MB, GB, TB)
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Round a percentage value to one decimal place
pub fn round_percent(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (value * 10.0).round() / 10.0
}

/// Format a percentage value for display
pub fn format_percent(value: f64) -> String {
    if !value.is_finite() {
        return "n/a".to_string();
    }
    let rounded = round_percent(value);
    let normalized = if rounded.abs() < 0.05 { 0.0 } else { rounded };
    if normalized.fract().abs() < 0.05 {
        format!("{:.0}%", normalized.round())
    } else {
        format!("{normalized:.1}%")
    }
}

/// Convert boolean to "yes" or "no" string
pub fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

/// Convert lock owner hint to JSON format
pub fn owner_hint_to_json(owner: &LockOwnerHint) -> serde_json::Value {
    json!({
        "pid": owner.pid,
        "cmd": owner.cmd,
        "started_at": owner.started_at,
        "file_path": owner
            .file_path
            .as_ref()
            .map(|path| path.display().to_string()),
        "file_id": owner.file_id,
        "last_heartbeat": owner.last_heartbeat,
        "heartbeat_ms": owner.heartbeat_ms,
    })
}

/// Parse a size string (e.g., "6MB", "1.5 GB") into bytes
pub fn parse_size(input: &str) -> Result<u64> {
    use anyhow::bail;

    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("size must not be empty");
    }

    let mut number = String::new();
    let mut suffix = String::new();
    let mut seen_unit = false;
    for ch in trimmed.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            if seen_unit {
                bail!("invalid size '{input}': unexpected digit after unit");
            }
            number.push(ch);
        } else if ch.is_ascii_whitespace() {
            if !number.is_empty() {
                seen_unit = true;
            }
        } else {
            seen_unit = true;
            suffix.push(ch);
        }
    }

    if number.is_empty() {
        bail!("invalid size '{input}': missing numeric value");
    }

    let value: f64 = number
        .parse()
        .map_err(|err| anyhow::anyhow!("invalid size '{input}': {err}"))?;
    let unit = suffix.trim().to_ascii_lowercase();

    let multiplier = match unit.as_str() {
        "" | "b" | "bytes" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => bail!("unsupported size unit '{other}'"),
    };

    let bytes = value * multiplier;
    if bytes <= 0.0 {
        bail!("size must be greater than zero");
    }
    if bytes > u64::MAX as f64 {
        bail!("size '{input}' exceeds supported maximum");
    }

    Ok(bytes.round() as u64)
}

/// Ensure that CLI mutations are allowed for the given memory
pub fn ensure_cli_mutation_allowed(mem: &Memvid) -> Result<()> {
    let ticket = mem.current_ticket();
    if ticket.issuer == "free-tier" {
        return Ok(());
    }
    let stats = mem.stats()?;
    if stats.tier == Tier::Free {
        return Ok(());
    }
    if ticket.issuer.trim().is_empty() {
        bail!(
            "Apply a ticket before mutating this memory (tier {:?})",
            stats.tier
        );
    }
    Ok(())
}

/// Apply lock CLI settings to a memory instance
pub fn apply_lock_cli(mem: &mut Memvid, opts: &crate::commands::LockCliArgs) {
    let settings = mem.lock_settings_mut();
    settings.timeout_ms = opts.lock_timeout;
    settings.force_stale = opts.force;
}

/// Select a frame by ID or URI
pub fn select_frame(
    mem: &mut Memvid,
    frame_id: Option<u64>,
    uri: Option<&str>,
) -> Result<memvid_core::Frame> {
    match (frame_id, uri) {
        (Some(id), None) => Ok(mem.frame_by_id(id)?),
        (None, Some(target_uri)) => Ok(mem.frame_by_uri(target_uri)?),
        (Some(_), Some(_)) => bail!("specify only one of --frame-id or --uri"),
        (None, None) => bail!("specify --frame-id or --uri to select a frame"),
    }
}

/// Convert frame status to string representation
pub fn frame_status_str(status: memvid_core::FrameStatus) -> &'static str {
    match status {
        memvid_core::FrameStatus::Active => "active",
        memvid_core::FrameStatus::Superseded => "superseded",
        memvid_core::FrameStatus::Deleted => "deleted",
    }
}

/// Check if a string looks like a memory file path
pub fn looks_like_memory(candidate: &str) -> bool {
    let path = std::path::Path::new(candidate);
    looks_like_memory_path(path) || candidate.trim().to_ascii_lowercase().ends_with(".mv2")
}

/// Check if a path looks like a memory file
pub fn looks_like_memory_path(path: &std::path::Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("mv2"))
        .unwrap_or(false)
}

/// Auto-detect a memory file in the current directory
pub fn autodetect_memory_file() -> Result<std::path::PathBuf> {
    let mut matches = Vec::new();
    for entry in std::fs::read_dir(".")? {
        let path = entry?.path();
        if path.is_file() && looks_like_memory_path(&path) {
            matches.push(path);
        }
    }

    match matches.len() {
        0 => bail!(
            "no .mv2 file detected in the current directory; specify the memory file explicitly"
        ),
        1 => Ok(matches.remove(0)),
        _ => bail!("multiple .mv2 files detected; specify the memory file explicitly"),
    }
}

/// Parse a timecode string (HH:MM:SS or MM:SS or SS) into milliseconds
pub fn parse_timecode(value: &str) -> Result<u64> {
    use anyhow::Context;

    let parts: Vec<&str> = value.split(':').collect();
    if parts.is_empty() || parts.len() > 3 {
        bail!("invalid time value `{value}`; expected SS, MM:SS, or HH:MM:SS");
    }
    let mut multiplier = 1_f64;
    let mut total_seconds = 0_f64;
    for part in parts.iter().rev() {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            bail!("invalid time value `{value}`");
        }
        let component: f64 = trimmed
            .parse()
            .with_context(|| format!("invalid time component `{trimmed}`"))?;
        total_seconds += component * multiplier;
        multiplier *= 60.0;
    }
    if total_seconds < 0.0 {
        bail!("time values must be positive");
    }
    Ok((total_seconds * 1000.0).round() as u64)
}

/// Format a Unix timestamp to ISO 8601 string
#[cfg(feature = "temporal_track")]
pub fn format_timestamp(ts: i64) -> Option<String> {
    use time::format_description::well_known::Rfc3339;
    use time::OffsetDateTime;

    OffsetDateTime::from_unix_timestamp(ts)
        .ok()
        .and_then(|dt| dt.format(&Rfc3339).ok())
}

/// Format milliseconds as HH:MM:SS.mmm
pub fn format_timestamp_ms(ms: u64) -> String {
    let total_seconds = ms / 1000;
    let millis = ms % 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

/// Read payload bytes from a file or stdin
pub fn read_payload(path: Option<&Path>) -> Result<Vec<u8>> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    match path {
        Some(p) => {
            let mut reader = BufReader::new(File::open(p)?);
            let mut buffer = Vec::new();
            if let Ok(meta) = std::fs::metadata(p) {
                if let Ok(len) = usize::try_from(meta.len()) {
                    buffer.reserve(len.saturating_add(1));
                }
            }
            reader.read_to_end(&mut buffer)?;
            Ok(buffer)
        }
        None => {
            let stdin = std::io::stdin();
            let mut reader = BufReader::new(stdin.lock());
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer)?;
            Ok(buffer)
        }
    }
}

/// Read an embedding vector from a file.
///
/// Supports:
/// - Whitespace-separated floats: `0.1 0.2 0.3`
/// - JSON array: `[0.1, 0.2, 0.3]`
pub fn read_embedding(path: &Path) -> Result<Vec<f32>> {
    use anyhow::anyhow;

    let text = std::fs::read_to_string(path)?;
    let trimmed = text.trim();
    if trimmed.starts_with('[') {
        let values: Vec<f32> = serde_json::from_str(trimmed).map_err(|err| {
            anyhow!(
                "failed to parse embedding JSON array from `{}`: {err}",
                path.display()
            )
        })?;
        if values.is_empty() {
            bail!("embedding file `{}` contained no values", path.display());
        }
        return Ok(values);
    }
    let mut values = Vec::new();
    for token in trimmed.split_whitespace() {
        let value: f32 = token
            .parse()
            .map_err(|err| anyhow!("invalid embedding value {token}: {err}"))?;
        values.push(value);
    }
    if values.is_empty() {
        bail!("embedding file `{}` contained no values", path.display());
    }
    Ok(values)
}

/// Parse a comma-separated vector string into f32 values
pub fn parse_vector(input: &str) -> Result<Vec<f32>> {
    use anyhow::anyhow;

    let mut values = Vec::new();
    for token in input.split(|c: char| c == ',' || c.is_whitespace()) {
        if token.is_empty() {
            continue;
        }
        let value: f32 = token
            .parse()
            .map_err(|err| anyhow!("invalid vector value {token}: {err}"))?;
        values.push(value);
    }
    if values.is_empty() {
        bail!("vector must contain at least one value");
    }
    Ok(values)
}

/// Parse a date boundary string (YYYY-MM-DD)
pub fn parse_date_boundary(raw: Option<&String>, end_of_day: bool) -> Result<Option<i64>> {
    use anyhow::anyhow;
    use time::macros::format_description;
    use time::{Date, PrimitiveDateTime, Time};

    let Some(value) = raw else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let format = format_description!("[year]-[month]-[day]");
    let date =
        Date::parse(trimmed, &format).map_err(|err| anyhow!("invalid date '{trimmed}': {err}"))?;
    let time = if end_of_day {
        Time::from_hms_milli(23, 59, 59, 999)
            .map_err(|err| anyhow!("unable to interpret end-of-day boundary: {err}"))?
    } else {
        Time::MIDNIGHT
    };
    let timestamp = PrimitiveDateTime::new(date, time)
        .assume_utc()
        .unix_timestamp();
    Ok(Some(timestamp))
}
