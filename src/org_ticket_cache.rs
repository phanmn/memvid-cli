//! Organisation-level ticket caching
//!
//! Caches org-level tickets locally to avoid repeated API calls.
//! Tickets are automatically refreshed when expired.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::Utc;

use crate::api::{fetch_org_ticket, OrgTicket, OrgTicketResponse};
use crate::config::CliConfig;

const ORG_TICKET_FILE: &str = "org_ticket.json";

/// How often to refresh ticket for write operations (5 minutes)
const WRITE_OP_REFRESH_SECS: i64 = 300;

/// Cached org ticket with metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedOrgTicket {
    pub ticket: OrgTicket,
    pub plan_id: String,
    pub plan_name: String,
    pub org_id: String,
    pub org_name: String,
    pub total_storage_bytes: u64,
    pub subscription_status: String,
    /// ISO date when paid plan started
    #[serde(default)]
    pub plan_start_date: Option<String>,
    /// ISO date when current billing period ends (renews)
    #[serde(default)]
    pub current_period_end: Option<String>,
    /// ISO date when paid plan ends (for canceled subscriptions in grace period)
    #[serde(default)]
    pub plan_end_date: Option<String>,
    /// Unix timestamp when this ticket was cached (for freshness checks)
    #[serde(default)]
    pub cached_at: i64,
}

impl CachedOrgTicket {
    /// Check if the cached ticket has expired
    pub fn is_expired(&self) -> bool {
        self.ticket.is_expired()
    }

    /// Get capacity in bytes from the ticket
    pub fn capacity_bytes(&self) -> u64 {
        self.ticket.capacity_bytes
    }

    /// Check if the plan is a paid plan
    pub fn is_paid(&self) -> bool {
        self.ticket.is_paid()
    }

    /// Check if the subscription is canceled but still in grace period
    pub fn is_in_grace_period(&self) -> bool {
        if self.subscription_status != "canceled" {
            return false;
        }
        if let Some(ref end_date) = self.plan_end_date {
            // Parse ISO date and check if still valid
            if let Ok(end) = chrono::DateTime::parse_from_rfc3339(end_date) {
                return end > Utc::now();
            }
        }
        false
    }

    /// Get days remaining in grace period (None if not in grace period)
    pub fn grace_period_days_remaining(&self) -> Option<i64> {
        if self.subscription_status != "canceled" {
            return None;
        }
        if let Some(ref end_date) = self.plan_end_date {
            if let Ok(end) = chrono::DateTime::parse_from_rfc3339(end_date) {
                let now = Utc::now();
                if end > now {
                    let duration = end.signed_duration_since(now);
                    return Some(duration.num_days());
                }
            }
        }
        None
    }

    /// Check if the ticket is too old for write operations
    /// Write operations require fresher subscription status to prevent
    /// users from continuing to use the CLI after cancellation
    pub fn is_stale_for_writes(&self) -> bool {
        if self.cached_at == 0 {
            // Old cache format without cached_at - treat as stale
            return true;
        }
        let now = Utc::now().timestamp();
        now - self.cached_at > WRITE_OP_REFRESH_SECS
    }
}

/// Get the path to the org ticket cache file
fn cache_path(config: &CliConfig) -> PathBuf {
    config.cache_dir.join(ORG_TICKET_FILE)
}

/// Load a cached org ticket from disk
pub fn load(config: &CliConfig) -> Result<CachedOrgTicket> {
    let path = cache_path(config);
    let data = fs::read(&path).with_context(|| "no cached org ticket available")?;
    let cached: CachedOrgTicket = serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse cached org ticket: {}", path.display()))?;
    Ok(cached)
}

/// Store an org ticket response to disk
pub fn store(config: &CliConfig, response: &OrgTicketResponse) -> Result<()> {
    let path = cache_path(config);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create org ticket cache directory: {}",
                parent.display()
            )
        })?;
    }

    let cached = CachedOrgTicket {
        ticket: response.ticket.clone(),
        plan_id: response.plan.id.clone(),
        plan_name: response.plan.name.clone(),
        org_id: response.organisation.id.clone(),
        org_name: response.organisation.name.clone(),
        total_storage_bytes: response.organisation.total_storage_bytes,
        subscription_status: response.subscription.status.clone(),
        plan_start_date: response.subscription.plan_start_date.clone(),
        current_period_end: response.subscription.current_period_end.clone(),
        plan_end_date: response.subscription.plan_end_date.clone(),
        cached_at: Utc::now().timestamp(),
    };

    let tmp = path.with_extension("json.tmp");
    let data = serde_json::to_vec_pretty(&cached)?;
    fs::write(&tmp, data)
        .with_context(|| format!("failed to write org ticket cache file: {}", tmp.display()))?;
    fs::rename(&tmp, &path).with_context(|| {
        format!(
            "failed to atomically persist org ticket cache file: {}",
            path.display()
        )
    })?;
    Ok(())
}

/// Clear the cached org ticket
pub fn clear(config: &CliConfig) -> Result<()> {
    let path = cache_path(config);
    if path.exists() {
        fs::remove_file(&path).with_context(|| {
            format!("failed to remove org ticket cache file: {}", path.display())
        })?;
    }
    Ok(())
}

/// Get a valid org ticket, fetching from API if needed
///
/// This function:
/// 1. Tries to load a cached ticket
/// 2. If cached ticket is valid (not expired), returns it
/// 3. If no cache or expired, fetches a new ticket from the API
/// 4. Caches the new ticket and returns it
pub fn get_or_refresh(config: &CliConfig) -> Result<CachedOrgTicket> {
    // Try cached ticket first
    if let Ok(cached) = load(config) {
        if !cached.is_expired() {
            return Ok(cached);
        }
        log::debug!("Cached org ticket expired, refreshing...");
    }

    // Fetch new ticket
    refresh(config)
}

/// Force refresh the org ticket from the API
pub fn refresh(config: &CliConfig) -> Result<CachedOrgTicket> {
    log::debug!("Fetching org ticket from API...");
    let response = fetch_org_ticket(config)?;
    store(config, &response)?;
    load(config)
}

/// Get org ticket if API key is configured, otherwise return None
///
/// This is useful for optional ticket validation - if no API key is set,
/// we fall back to free tier limits without requiring authentication.
///
/// When API key is set, this automatically syncs the plan ticket from the
/// dashboard if needed (first use or expired). This enables seamless
/// capacity validation without explicit `memvid plan sync`.
pub fn get_optional(config: &CliConfig) -> Option<CachedOrgTicket> {
    // No API key = free tier, silent fallback
    if config.api_key.is_none() {
        log::debug!("No API key set, using free tier limits");
        return None;
    }

    // Check if we need to fetch (no cache or expired)
    let needs_fetch = match load(config) {
        Ok(cached) => cached.is_expired(),
        Err(_) => true,
    };

    if needs_fetch {
        // Silent auto-sync on first use or when expired
        log::debug!("Auto-syncing plan ticket from dashboard...");
        match refresh(config) {
            Ok(ticket) => {
                log::info!(
                    "Plan synced: {} ({})",
                    ticket.plan_name,
                    format_capacity(ticket.capacity_bytes())
                );
                return Some(ticket);
            }
            Err(err) => {
                // Check if it's an auth error - silently fall back to free tier
                let err_str = err.to_string();
                if err_str.contains("Invalid API key") || err_str.contains("401") {
                    log::debug!("API key invalid, using free tier limits");
                } else {
                    // Only warn for non-auth errors (network issues, etc.)
                    log::warn!("Failed to sync plan (using free tier limits): {}", err);
                }
                return None;
            }
        }
    }

    // Use cached ticket
    match load(config) {
        Ok(ticket) => Some(ticket),
        Err(_) => None,
    }
}

/// Format capacity bytes for display
fn format_capacity(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 * 1024 {
        format!(
            "{:.1} TB",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0)
        )
    } else if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Get a fresh org ticket for write operations
///
/// Write operations (put, find, ask) require up-to-date subscription status
/// to prevent users from continuing to use the CLI after cancellation.
///
/// Smart refresh logic:
/// - If cached status is "active" → only refresh if older than 5 minutes
/// - If cached status is "canceled"/"inactive" → always refresh (instant unblock on reactivation)
///
/// This ensures:
/// - Active users get fast operations (minimal API calls)
/// - Canceled users can't abuse the CLI (checked every operation)
/// - Reactivated users get instant access (no waiting for cache expiry)
pub fn get_fresh_for_writes(config: &CliConfig) -> Option<CachedOrgTicket> {
    // No API key = free tier, no ticket needed
    if config.api_key.is_none() {
        return None;
    }

    // Check if we have a cached ticket and determine refresh strategy
    let needs_refresh = match load(config) {
        Ok(cached) => {
            if cached.is_expired() {
                true
            } else {
                // Smart refresh: active users get 5-min cache, others always refresh
                let status = cached.subscription_status.as_str();
                if status == "active" || status == "trialing" {
                    // Active subscription: only refresh if stale (> 5 min)
                    cached.is_stale_for_writes()
                } else {
                    // Canceled/inactive: always refresh to catch reactivation instantly
                    log::debug!(
                        "Non-active status '{}', refreshing to check for reactivation",
                        status
                    );
                    true
                }
            }
        }
        Err(_) => true,
    };

    if needs_refresh {
        log::debug!("Refreshing ticket for write operation...");
        match refresh(config) {
            Ok(ticket) => return Some(ticket),
            Err(err) => {
                let err_str = err.to_string();
                if err_str.contains("Invalid API key") || err_str.contains("401") {
                    log::debug!("API key invalid, using free tier limits");
                } else {
                    log::warn!("Failed to refresh ticket: {}", err);
                }
                return None;
            }
        }
    }

    // Use cached ticket (it's fresh enough)
    load(config).ok()
}
