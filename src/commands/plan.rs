//! Plan management commands (show, sync)
//!
//! Commands for viewing and syncing plan/subscription information.

use anyhow::Result;
use chrono::DateTime;
use clap::{Args, Subcommand};
use serde_json::json;

use crate::config::CliConfig;
use crate::org_ticket_cache;
use crate::utils::{format_bytes, FREE_TIER_MAX_FILE_SIZE};

/// Plan management commands
#[derive(Subcommand)]
pub enum PlanCommand {
    /// Show current plan and capacity
    Show(PlanShowArgs),
    /// Sync plan ticket from the dashboard
    Sync(PlanSyncArgs),
    /// Clear cached plan ticket
    Clear(PlanClearArgs),
}

/// Arguments for the `plan` command
#[derive(Args)]
pub struct PlanArgs {
    #[command(subcommand)]
    pub command: PlanCommand,
}

/// Arguments for the `plan show` subcommand
#[derive(Args)]
pub struct PlanShowArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the `plan sync` subcommand
#[derive(Args)]
pub struct PlanSyncArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the `plan clear` subcommand
#[derive(Args)]
pub struct PlanClearArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn handle_plan(config: &CliConfig, args: PlanArgs) -> Result<()> {
    match args.command {
        PlanCommand::Show(show) => handle_plan_show(config, show),
        PlanCommand::Sync(sync) => handle_plan_sync(config, sync),
        PlanCommand::Clear(clear) => handle_plan_clear(config, clear),
    }
}

fn handle_plan_show(config: &CliConfig, args: PlanShowArgs) -> Result<()> {
    if config.api_key.is_none() {
        if args.json {
            let output = json!({
                "plan": "free",
                "capacity_bytes": FREE_TIER_MAX_FILE_SIZE,
                "capacity": format_bytes(FREE_TIER_MAX_FILE_SIZE),
                "features": ["core", "temporal_track", "clip", "whisper", "temporal_enrich"],
                "authenticated": false,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("Plan: Free (not authenticated)");
            println!("Capacity: {}", format_bytes(FREE_TIER_MAX_FILE_SIZE));
            println!();
            println!("To unlock more capacity, set your API key:");
            println!("  export MEMVID_API_KEY=your_api_key");
            println!();
            println!("Get your API key at: https://memvid.com/dashboard/api-keys");
        }
        return Ok(());
    }

    // Try to get cached ticket or show info to sync
    match org_ticket_cache::get_or_refresh(config) {
        Ok(cached) => {
            let in_grace_period = cached.is_in_grace_period();
            let grace_days = cached.grace_period_days_remaining();

            if args.json {
                let mut output = json!({
                    "plan": cached.plan_id,
                    "plan_name": cached.plan_name,
                    "capacity_bytes": cached.capacity_bytes(),
                    "capacity": format_bytes(cached.capacity_bytes()),
                    "features": cached.ticket.features,
                    "org_id": cached.org_id,
                    "org_name": cached.org_name,
                    "total_storage_bytes": cached.total_storage_bytes,
                    "total_storage": format_bytes(cached.total_storage_bytes),
                    "subscription_status": cached.subscription_status,
                    "expires_at": cached.ticket.expires_at,
                    "expires_in_secs": cached.ticket.expires_in_secs(),
                    "authenticated": true,
                });
                // Add plan dates if available
                if let Some(ref start_date) = cached.plan_start_date {
                    output["plan_start_date"] = json!(start_date);
                }
                if let Some(ref period_end) = cached.current_period_end {
                    output["current_period_end"] = json!(period_end);
                }
                if let Some(ref end_date) = cached.plan_end_date {
                    output["plan_end_date"] = json!(end_date);
                }
                if in_grace_period {
                    output["in_grace_period"] = json!(true);
                    output["grace_period_days_remaining"] = json!(grace_days);
                }
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("Plan: {}", cached.plan_name);
                println!("Organisation: {}", cached.org_name);
                println!("Subscription: {}", cached.subscription_status);
                if let Some(ref start_date) = cached.plan_start_date {
                    // Format the ISO date to a more readable format
                    if let Ok(dt) = DateTime::parse_from_rfc3339(start_date) {
                        println!("Plan Started: {}", dt.format("%B %d, %Y"));
                    }
                }
                // Only show "Renews On" for active subscriptions (not canceled)
                if cached.subscription_status != "canceled" {
                    if let Some(ref period_end) = cached.current_period_end {
                        if let Ok(dt) = DateTime::parse_from_rfc3339(period_end) {
                            println!("Renews On: {}", dt.format("%B %d, %Y"));
                        }
                    }
                }

                // Show grace period warning if applicable
                if in_grace_period {
                    println!();
                    if let Some(days) = grace_days {
                        println!(
                            "⚠️  Your subscription was canceled but you have {} days remaining.",
                            days
                        );
                        println!(
                            "    After that, you'll be downgraded to Free tier ({} limit).",
                            format_bytes(FREE_TIER_MAX_FILE_SIZE)
                        );
                    }
                    if let Some(ref end_date) = cached.plan_end_date {
                        println!("    Plan ends: {}", end_date);
                    }
                }

                println!();
                println!("Capacity: {}", format_bytes(cached.capacity_bytes()));
                println!("Storage Used: {}", format_bytes(cached.total_storage_bytes));
                println!();
                println!("Features: {}", cached.ticket.features.join(", "));
                println!();
                let expires_in = cached.ticket.expires_in_secs();
                if expires_in > 0 {
                    let hours = expires_in / 3600;
                    let mins = (expires_in % 3600) / 60;
                    println!("Ticket expires in: {}h {}m", hours, mins);
                } else {
                    println!("Ticket expired (run `memvid plan sync` to refresh)");
                }
            }
        }
        Err(err) => {
            if args.json {
                let output = json!({
                    "error": err.to_string(),
                    "plan": "unknown",
                    "authenticated": true,
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("Failed to get plan information: {}", err);
                println!();
                println!("Try syncing your plan ticket:");
                println!("  memvid plan sync");
            }
        }
    }

    Ok(())
}

fn handle_plan_sync(config: &CliConfig, args: PlanSyncArgs) -> Result<()> {
    if config.api_key.is_none() {
        anyhow::bail!(
            "API key required. Set it with:\n  export MEMVID_API_KEY=your_api_key\n\n\
             Get your API key at: https://memvid.com/dashboard/api-keys"
        );
    }

    let cached = org_ticket_cache::refresh(config)?;
    let in_grace_period = cached.is_in_grace_period();
    let grace_days = cached.grace_period_days_remaining();

    if args.json {
        let mut output = json!({
            "success": true,
            "plan": cached.plan_id,
            "plan_name": cached.plan_name,
            "capacity_bytes": cached.capacity_bytes(),
            "capacity": format_bytes(cached.capacity_bytes()),
            "features": cached.ticket.features,
            "org_id": cached.org_id,
            "org_name": cached.org_name,
            "subscription_status": cached.subscription_status,
            "expires_at": cached.ticket.expires_at,
        });
        if let Some(ref end_date) = cached.plan_end_date {
            output["plan_end_date"] = json!(end_date);
        }
        if in_grace_period {
            output["in_grace_period"] = json!(true);
            output["grace_period_days_remaining"] = json!(grace_days);
        }
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("✓ Plan synced successfully");
        println!();
        println!("Plan: {}", cached.plan_name);
        println!("Organisation: {}", cached.org_name);
        println!("Capacity: {}", format_bytes(cached.capacity_bytes()));
        println!("Subscription: {}", cached.subscription_status);

        // Show grace period warning if applicable
        if in_grace_period {
            println!();
            if let Some(days) = grace_days {
                println!(
                    "⚠️  Subscription canceled - {} days remaining before downgrade.",
                    days
                );
            }
            if let Some(ref end_date) = cached.plan_end_date {
                println!("    Plan ends: {}", end_date);
            }
        }

        println!();
        let expires_in = cached.ticket.expires_in_secs();
        let hours = expires_in / 3600;
        let mins = (expires_in % 3600) / 60;
        println!("Ticket valid for: {}h {}m", hours, mins);
    }

    Ok(())
}

fn handle_plan_clear(config: &CliConfig, args: PlanClearArgs) -> Result<()> {
    org_ticket_cache::clear(config)?;

    if args.json {
        let output = json!({
            "success": true,
            "message": "Plan ticket cache cleared",
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("✓ Plan ticket cache cleared");
    }

    Ok(())
}
