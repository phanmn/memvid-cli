//! Status command - Show configuration and system status
//!
//! Provides a comprehensive overview of:
//! - API key configuration
//! - Plan/subscription status
//! - Local memory files
//! - LLM provider keys

use anyhow::{Context, Result};
use clap::Args;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use super::config::PersistentConfig;

/// Arguments for the `status` command
#[derive(Args)]
pub struct StatusArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Directory to scan for .mv2 files (defaults to current directory)
    #[arg(short, long)]
    pub dir: Option<PathBuf>,

    /// Skip API validation (offline mode)
    #[arg(long)]
    pub offline: bool,
}

/// Subscription info from API
#[derive(Debug, Default)]
struct SubscriptionInfo {
    status: String,
    plan_name: String,
    capacity_gb: f64,
    renews_at: Option<String>,
}

/// Local memory file info
#[derive(Debug)]
struct LocalMemory {
    path: PathBuf,
    size_bytes: u64,
    name: String,
}

/// Handle the status command
pub fn handle_status(args: StatusArgs) -> Result<()> {
    let config = PersistentConfig::load()?;
    let config_path = PersistentConfig::config_path()?;

    // Check environment variables
    let env_api_key = std::env::var("MEMVID_API_KEY").ok();
    let env_dashboard_url = std::env::var("MEMVID_DASHBOARD_URL")
        .or_else(|_| std::env::var("MEMVID_API_URL"))
        .ok();

    // Effective values (env takes precedence)
    let effective_api_key = env_api_key.clone().or(config.api_key.clone());
    let effective_dashboard_url = env_dashboard_url
        .clone()
        .or(config.dashboard_url.clone())
        .or(config.api_url.clone())
        .unwrap_or_else(|| "https://memvid.com".to_string());

    // API key status
    let has_api_key = effective_api_key.is_some();
    let api_key_source = if env_api_key.is_some() {
        "environment"
    } else if config.api_key.is_some() {
        "config file"
    } else {
        "not set"
    };

    // Named memories from config
    let named_memories: Vec<(String, String)> = config
        .memory
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Check LLM provider keys (env vars or config file)
    let groq_key = std::env::var("GROQ_API_KEY")
        .ok()
        .or_else(|| config.get("groq_api_key"));
    let openai_key = std::env::var("OPENAI_API_KEY")
        .ok()
        .or_else(|| config.get("openai_api_key"));
    let gemini_key = std::env::var("GEMINI_API_KEY")
        .ok()
        .or_else(|| config.get("gemini_api_key"));
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .or_else(|| config.get("anthropic_api_key"));

    // Fetch subscription info (if API key set and not offline)
    let subscription = if has_api_key && !args.offline {
        fetch_subscription_info(
            &effective_api_key.as_ref().unwrap(),
            &effective_dashboard_url,
        )
        .ok()
    } else {
        None
    };

    // Scan for local .mv2 files
    let scan_dir = args.dir.clone().unwrap_or_else(|| PathBuf::from("."));
    let local_memories = scan_local_memories(&scan_dir);
    let total_size: u64 = local_memories.iter().map(|m| m.size_bytes).sum();

    if args.json {
        output_json(
            &config_path,
            has_api_key,
            api_key_source,
            &effective_api_key,
            &effective_dashboard_url,
            &named_memories,
            &subscription,
            &local_memories,
            total_size,
            groq_key.is_some(),
            openai_key.is_some(),
            gemini_key.is_some(),
            anthropic_key.is_some(),
        )?;
    } else {
        output_pretty(
            &config_path,
            has_api_key,
            api_key_source,
            &effective_api_key,
            &effective_dashboard_url,
            &named_memories,
            &subscription,
            &local_memories,
            total_size,
            groq_key.is_some(),
            openai_key.is_some(),
            gemini_key.is_some(),
            anthropic_key.is_some(),
        );
    }

    Ok(())
}

fn fetch_subscription_info(api_key: &str, dashboard_url: &str) -> Result<SubscriptionInfo> {
    let url = format!("{}/api/ticket", dashboard_url.trim_end_matches('/'));

    let mut headers = HeaderMap::new();
    headers.insert(
        "x-api-key",
        HeaderValue::from_str(api_key).context("Invalid API key format")?,
    );

    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("Failed to create HTTP client")?;

    let response = client
        .get(&url)
        .headers(headers)
        .send()
        .context("Failed to fetch subscription info")?;

    let body: serde_json::Value = response.json().context("Failed to parse response")?;

    let data = body.get("data").unwrap_or(&body);
    let ticket = data.get("ticket").unwrap_or(data);
    let subscription = data.get("subscription");

    let capacity_bytes = ticket
        .get("capacity_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(1_073_741_824); // 1 GB default

    let capacity_gb = capacity_bytes as f64 / 1_073_741_824.0;

    let status = subscription
        .and_then(|s| s.get("status"))
        .and_then(|v| v.as_str())
        .unwrap_or("active")
        .to_string();

    let plan_name = ticket
        .get("issuer")
        .and_then(|v| v.as_str())
        .unwrap_or("Free")
        .to_string();

    let renews_at = subscription
        .and_then(|s| s.get("planEndDate").or_else(|| s.get("ends_at")))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(SubscriptionInfo {
        status,
        plan_name,
        capacity_gb,
        renews_at,
    })
}

fn scan_local_memories(dir: &PathBuf) -> Vec<LocalMemory> {
    let mut memories = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "mv2") {
                if let Ok(metadata) = fs::metadata(&path) {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    memories.push(LocalMemory {
                        path,
                        size_bytes: metadata.len(),
                        name,
                    });
                }
            }
        }
    }

    // Sort by name
    memories.sort_by(|a, b| a.name.cmp(&b.name));
    memories
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

#[allow(clippy::too_many_arguments)]
fn output_pretty(
    config_path: &PathBuf,
    has_api_key: bool,
    api_key_source: &str,
    effective_api_key: &Option<String>,
    dashboard_url: &str,
    named_memories: &[(String, String)],
    subscription: &Option<SubscriptionInfo>,
    local_memories: &[LocalMemory],
    total_size: u64,
    has_groq: bool,
    has_openai: bool,
    has_gemini: bool,
    has_anthropic: bool,
) {
    println!();
    println!("Memvid Status");
    println!("{}", "━".repeat(50));
    println!();

    // API Key status
    if has_api_key {
        let key = effective_api_key.as_ref().unwrap();
        let masked = PersistentConfig::mask_value(key);
        println!("✓ API Key: configured ({})", masked);
        println!("  Source: {}", api_key_source);
    } else {
        println!("✗ API Key: not configured");
        println!("  Fix: memvid config set api_key <your-key>");
    }

    // Subscription/Plan status
    if let Some(sub) = subscription {
        println!();
        println!("✓ Plan: {} ({:.1} GB)", sub.plan_name, sub.capacity_gb);
        println!("✓ Subscription: {}", sub.status);
        if let Some(renews) = &sub.renews_at {
            println!("  Renews: {}", renews);
        }
    } else if has_api_key {
        println!();
        println!("⚠ Plan: Could not fetch (use --offline to skip)");
    }

    // Dashboard URL
    println!();
    println!("Dashboard: {}", dashboard_url);

    // Named memories
    if !named_memories.is_empty() {
        println!();
        println!("Named Memories:");
        for (name, id) in named_memories {
            let short_id = if id.len() > 12 {
                format!("{}...", &id[..12])
            } else {
                id.clone()
            };
            println!("  {} → {}", name, short_id);
        }
    }

    // LLM Provider keys
    println!();
    println!("LLM Providers:");
    print_key_status("  Groq", has_groq);
    print_key_status("  OpenAI", has_openai);
    print_key_status("  Gemini", has_gemini);
    print_key_status("  Anthropic", has_anthropic);

    // Local memories
    println!();
    if local_memories.is_empty() {
        println!("Local Memories: None found in current directory");
    } else {
        println!(
            "Local Memories: {} files ({})",
            local_memories.len(),
            format_bytes(total_size)
        );
        for mem in local_memories.iter().take(5) {
            println!("  {} ({})", mem.name, format_bytes(mem.size_bytes));
        }
        if local_memories.len() > 5 {
            println!("  ... and {} more", local_memories.len() - 5);
        }
    }

    // Config file location
    println!();
    println!("Config: {}", config_path.display());
    println!();
}

fn print_key_status(name: &str, configured: bool) {
    if configured {
        println!("{}: ✓ configured", name);
    } else {
        println!("{}: ✗ not set", name);
    }
}

#[allow(clippy::too_many_arguments)]
fn output_json(
    config_path: &PathBuf,
    has_api_key: bool,
    api_key_source: &str,
    effective_api_key: &Option<String>,
    dashboard_url: &str,
    named_memories: &[(String, String)],
    subscription: &Option<SubscriptionInfo>,
    local_memories: &[LocalMemory],
    total_size: u64,
    has_groq: bool,
    has_openai: bool,
    has_gemini: bool,
    has_anthropic: bool,
) -> Result<()> {
    let memories_json: Vec<serde_json::Value> = local_memories
        .iter()
        .map(|m| {
            json!({
                "name": m.name,
                "path": m.path.display().to_string(),
                "size_bytes": m.size_bytes,
            })
        })
        .collect();

    let named_memories_json: serde_json::Map<String, serde_json::Value> = named_memories
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();

    let output = json!({
        "config_path": config_path.display().to_string(),
        "api_key": {
            "configured": has_api_key,
            "source": api_key_source,
            "value": effective_api_key.as_ref().map(|k| PersistentConfig::mask_value(k)),
        },
        "dashboard_url": dashboard_url,
        "subscription": subscription.as_ref().map(|s| json!({
            "status": s.status,
            "plan": s.plan_name,
            "capacity_gb": s.capacity_gb,
            "renews_at": s.renews_at,
        })),
        "named_memories": named_memories_json,
        "llm_providers": {
            "groq": has_groq,
            "openai": has_openai,
            "gemini": has_gemini,
            "anthropic": has_anthropic,
        },
        "local_memories": {
            "count": local_memories.len(),
            "total_bytes": total_size,
            "files": memories_json,
        },
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
