//! API client for Memvid control plane
//!
//! This module provides HTTP client functionality for interacting with the Memvid
//! control plane API, specifically for ticket synchronization and application.

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use reqwest::blocking::{Client, Response};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::commands::config::PersistentConfig;
use crate::config::CliConfig;

const API_KEY_HEADER: &str = "X-API-KEY";
const JSON_CONTENT_TYPE: &str = "application/json";
const DEFAULT_DASHBOARD_URL: &str = "https://memvid.com";

/// Get the dashboard URL from environment or config file
fn get_dashboard_url() -> String {
    // Priority: env var > config file > default
    if let Ok(url) = std::env::var("MEMVID_DASHBOARD_URL") {
        if !url.trim().is_empty() {
            return url;
        }
    }
    if let Ok(config) = PersistentConfig::load() {
        if let Some(url) = config.dashboard_url {
            return url;
        }
    }
    DEFAULT_DASHBOARD_URL.to_string()
}

/// Wrapper for ticket data in sync response
#[derive(Debug, Deserialize)]
pub struct TicketSyncData {
    pub ticket: TicketSyncPayload,
}

/// Payload for ticket synchronization from the control plane
#[derive(Debug, Deserialize)]
pub struct TicketSyncPayload {
    pub memory_id: Uuid,
    #[serde(alias = "seq_no")]
    pub sequence: i64,
    pub issuer: String,
    pub expires_in: u64,
    #[serde(default)]
    pub capacity_bytes: Option<u64>,
    pub signature: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ApiEnvelope<T> {
    status: String,
    request_id: String,
    data: Option<T>,
    error: Option<ApiErrorBody>,
    signature: String,
}

#[derive(Debug, Deserialize)]
struct ApiErrorBody {
    code: String,
    message: String,
}

pub struct TicketSyncResponse {
    pub payload: TicketSyncPayload,
    pub request_id: String,
}

/// Dashboard API response format (simpler than control plane)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DashboardTicketResponse {
    data: TicketSyncData,
    request_id: String,
}

pub fn fetch_ticket(config: &CliConfig, memory_id: &Uuid) -> Result<TicketSyncResponse> {
    let api_key = require_api_key(config)?;
    let client = http_client()?;

    // Use the dashboard URL for per-memory tickets
    let base_url = get_dashboard_url();
    // Handle both http://localhost:3001 and http://localhost:3001/api
    let base = base_url.trim_end_matches('/').trim_end_matches("/api");
    let url = format!("{}/api/memories/{}/tickets/sync", base, memory_id);

    let response = client
        .post(&url)
        .headers(auth_headers(api_key)?)
        .header(CONTENT_TYPE, JSON_CONTENT_TYPE)
        .body("{}")
        .send()
        .with_context(|| format!("failed to contact ticket sync endpoint at {}", url))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .unwrap_or_else(|_| "unknown error".to_string());
        if status.as_u16() == 401 {
            bail!("Invalid API key. Get a valid key at https://memvid.com/dashboard/api-keys");
        }
        if status.as_u16() == 404 {
            bail!("Memory not found. Create it first at https://memvid.com/dashboard");
        }
        if status.as_u16() == 403 {
            bail!("You don't have access to this memory");
        }
        bail!("Ticket sync error ({}): {}", status, error_text);
    }

    let dashboard_response: DashboardTicketResponse = response
        .json()
        .context("failed to parse ticket sync response")?;

    Ok(TicketSyncResponse {
        payload: dashboard_response.data.ticket,
        request_id: dashboard_response.request_id,
    })
}

#[derive(serde::Serialize)]
pub struct ApplyTicketRequest<'a> {
    pub issuer: &'a str,
    pub seq_no: i64,
    pub expires_in: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity_bytes: Option<u64>,
    pub signature: &'a str,
}

pub fn apply_ticket(
    config: &CliConfig,
    memory_id: &Uuid,
    request: &ApplyTicketRequest<'_>,
) -> Result<String> {
    let api_key = require_api_key(config)?;
    let client = http_client()?;
    let url = format!(
        "{}/memories/{}/tickets/apply",
        config.api_url.trim_end_matches('/'),
        memory_id
    );
    let response = client
        .post(url)
        .headers(auth_headers(api_key)?)
        .json(request)
        .send()
        .with_context(|| "failed to contact ticket apply endpoint")?;
    let envelope: ApiEnvelope<serde_json::Value> = parse_envelope(response)?;
    Ok(envelope.request_id)
}

#[derive(Debug, Serialize)]
pub struct RegisterFileRequest<'a> {
    pub file_name: &'a str,
    pub file_path: &'a str,
    pub file_size: i64,
    pub machine_id: &'a str,
}

/// Response from file registration
#[derive(Debug, Deserialize)]
pub struct RegisterFileResponse {
    pub id: String,
    pub memory_id: String,
    pub file_name: String,
    pub file_path: String,
    pub file_size: i64,
    pub machine_id: String,
    pub last_synced: String,
    pub created_at: String,
}

/// Dashboard file registration response format
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct DashboardFileResponse {
    data: RegisterFileResponse,
    request_id: String,
}

/// Register a file with the dashboard
pub fn register_file(
    _config: &CliConfig,
    memory_id: &Uuid,
    request: &RegisterFileRequest<'_>,
    api_key: &str,
) -> Result<RegisterFileResponse> {
    let client = http_client()?;

    // Use the dashboard URL for file registration
    let base_url = get_dashboard_url();
    let url = format!(
        "{}/api/memories/{}/files",
        base_url.trim_end_matches('/'),
        memory_id
    );

    let response = client
        .post(&url)
        .headers(auth_headers(api_key)?)
        .json(request)
        .send()
        .with_context(|| "failed to contact file registration endpoint")?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .unwrap_or_else(|_| "unknown error".to_string());
        if status.as_u16() == 409 {
            // Try to extract the specific message from the API response
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&error_text) {
                if let Some(msg) = json
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                {
                    bail!("{}", msg);
                }
            }
            bail!("This memory is already bound to another file. Each memory can only be bound to one MV2 file.");
        }
        bail!("File registration error ({}): {}", status, error_text);
    }

    let dashboard_response: DashboardFileResponse = response
        .json()
        .context("failed to parse file registration response")?;

    Ok(dashboard_response.data)
}

fn http_client() -> Result<Client> {
    crate::http::blocking_client(Duration::from_secs(15))
}

fn auth_headers(api_key: &str) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let value = HeaderValue::from_str(api_key)
        .map_err(|_| anyhow!("API key contains invalid characters"))?;
    headers.insert(API_KEY_HEADER, value);
    Ok(headers)
}

fn require_api_key(config: &CliConfig) -> Result<&str> {
    config
        .api_key
        .as_deref()
        .ok_or_else(|| anyhow!("MEMVID_API_KEY is not set"))
}

fn parse_envelope<T: DeserializeOwned>(response: Response) -> Result<ApiEnvelope<T>> {
    let status = response.status();
    let envelope = response.json::<ApiEnvelope<T>>()?;
    if envelope.status == "ok" {
        return Ok(envelope);
    }

    let message = envelope
        .error
        .map(|err| format!("{}: {}", err.code, err.message))
        .unwrap_or_else(|| format!("request failed with status {}", status));
    bail!(message);
}

// ===========================================================================
// Org-level ticket API (for plan-based capacity validation)
// ===========================================================================

/// Organisation-level signed ticket from the dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrgTicket {
    pub version: i32,
    pub user_id: String,
    pub org_id: String,
    pub plan_id: String,
    pub capacity_bytes: u64,
    pub features: Vec<String>,
    pub expires_at: i64,
    pub signature: String,
}

impl OrgTicket {
    /// Check if this ticket has expired
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.expires_at < now
    }

    /// Check if the plan is a paid plan
    pub fn is_paid(&self) -> bool {
        matches!(self.plan_id.as_str(), "starter" | "pro" | "enterprise")
    }

    /// Get remaining time in seconds (0 if expired)
    pub fn expires_in_secs(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if self.expires_at > now {
            (self.expires_at - now) as u64
        } else {
            0
        }
    }
}

/// Plan information from the ticket response
#[derive(Debug, Clone, Deserialize)]
pub struct PlanInfo {
    pub id: String,
    pub name: String,
    pub limits: PlanLimits,
    pub features: Vec<String>,
}

/// Plan limits
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanLimits {
    pub capacity_bytes: u64,
    #[serde(default)]
    pub memory_files: Option<u64>,
    #[serde(default)]
    pub max_file_size: Option<u64>,
}

/// Subscription status information
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscriptionInfo {
    pub status: String,
    pub expires_at: i64,
    /// ISO date when paid plan started
    #[serde(default)]
    pub plan_start_date: Option<String>,
    /// ISO date when current billing period ends (renews)
    #[serde(default)]
    pub current_period_end: Option<String>,
    /// ISO date when paid plan ends (for canceled subscriptions in grace period)
    #[serde(default)]
    pub plan_end_date: Option<String>,
}

/// Organisation information
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrgInfo {
    pub id: String,
    pub name: String,
    pub total_storage_bytes: u64,
}

/// Response from the /api/ticket endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct OrgTicketResponse {
    pub ticket: OrgTicket,
    pub plan: PlanInfo,
    pub subscription: SubscriptionInfo,
    pub organisation: OrgInfo,
}

/// Fetch an org-level ticket from the dashboard API
///
/// This endpoint returns a signed ticket containing the user's plan capacity
/// and features. The ticket can be used to validate large file operations
/// and feature access.
pub fn fetch_org_ticket(config: &CliConfig) -> Result<OrgTicketResponse> {
    let api_key = require_api_key(config)?;
    let client = http_client()?;

    // Use the dashboard API URL (different from control plane)
    let base_url = get_dashboard_url();
    // Handle both http://localhost:3001 and http://localhost:3001/api
    let base = base_url.trim_end_matches('/').trim_end_matches("/api");
    let url = format!("{}/api/ticket", base);

    let response = client
        .get(&url)
        .headers(auth_headers(api_key)?)
        .send()
        .with_context(|| format!("failed to contact ticket endpoint at {}", url))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .unwrap_or_else(|_| "unknown error".to_string());
        if status.as_u16() == 401 {
            bail!("Invalid API key. Get a valid key at https://memvid.com/dashboard/api-keys");
        }
        bail!("Ticket API error ({}): {}", status, error_text);
    }

    let wrapper: serde_json::Value = response.json().context("failed to parse ticket response")?;

    // Handle the API response wrapper
    // The dashboard API returns: { "data": { "ticket": ..., "plan": ..., ... }, "requestId": ... }
    let data = if let Some(data_field) = wrapper.get("data") {
        data_field.clone()
    } else if wrapper.get("status").and_then(|s| s.as_str()) == Some("ok") {
        wrapper.get("data").cloned().unwrap_or(wrapper.clone())
    } else if wrapper.get("ticket").is_some() {
        // Direct response without wrapper
        wrapper
    } else {
        wrapper
    };

    let ticket_response: OrgTicketResponse =
        serde_json::from_value(data).context("failed to parse ticket data")?;

    Ok(ticket_response)
}

// ===========================================================================
// Query Usage Tracking
// ===========================================================================

/// Response from the /api/v1/query endpoint
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryTrackingResponse {
    pub success: bool,
    #[serde(default)]
    pub queries_used: Option<u64>,
    #[serde(default)]
    pub queries_limit: Option<u64>,
    #[serde(default)]
    pub queries_remaining: Option<u64>,
}

/// Track a query against the user's plan quota.
///
/// This is called before find() and ask() operations to record usage.
/// Returns Ok(()) on success, or an error if quota is exceeded.
/// Network errors are logged but don't fail the operation (best-effort tracking).
pub fn track_query_usage(config: &CliConfig, count: u64) -> Result<()> {
    let api_key = match require_api_key(config) {
        Ok(key) => key,
        Err(_) => {
            // No API key configured - skip tracking
            return Ok(());
        }
    };

    let client = http_client()?;
    let base_url = get_dashboard_url();
    let url = format!("{}/api/v1/query", base_url);

    let body = serde_json::json!({ "count": count });

    let response = match client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("X-API-Key", &*api_key)
        .timeout(std::time::Duration::from_secs(5))
        .body(body.to_string())
        .send()
    {
        Ok(resp) => resp,
        Err(e) => {
            // Network error - log but don't fail the query
            log::warn!("Query tracking failed: {}", e);
            return Ok(());
        }
    };

    let status = response.status();

    if status.as_u16() == 429 {
        // Quota exceeded - this is a hard error
        let body_text = response.text().unwrap_or_default();

        // Try to parse error details
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&body_text) {
            let message = data
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Monthly query quota exceeded");
            let limit = data.get("limit").and_then(|v| v.as_u64());
            let used = data.get("used").and_then(|v| v.as_u64());
            let reset_date = data.get("resetDate").and_then(|v| v.as_str());

            let mut error_msg = format!("{}", message);
            if let (Some(used), Some(limit)) = (used, limit) {
                error_msg.push_str(&format!("\nUsed: {} / {}", used, limit));
            }
            if let Some(reset) = reset_date {
                error_msg.push_str(&format!("\nResets: {}", reset));
            }
            error_msg.push_str(&format!("\nUpgrade at: {}/dashboard/plan", base_url));

            bail!(error_msg);
        } else {
            bail!(
                "Monthly query quota exceeded. Upgrade at: {}/dashboard/plan",
                base_url
            );
        }
    }

    if !status.is_success() {
        // Other error - log but don't fail
        log::warn!("Query tracking returned status {}", status);
    }

    Ok(())
}
