//! Config management commands (set, get, list, unset, check)
//!
//! Commands for managing persistent CLI configuration including API keys.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Config management commands
#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Set a configuration value
    Set(ConfigSetArgs),
    /// Get a configuration value
    Get(ConfigGetArgs),
    /// List all configuration values
    List(ConfigListArgs),
    /// Remove a configuration value
    Unset(ConfigUnsetArgs),
    /// Check configuration validity (verifies API key with server)
    Check(ConfigCheckArgs),
}

/// Arguments for the `config` command
#[derive(Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub command: ConfigCommand,
}

/// Arguments for `config set`
#[derive(Args)]
pub struct ConfigSetArgs {
    /// Configuration key (e.g., api_key, api_url)
    pub key: String,
    /// Value to set
    pub value: String,
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `config get`
#[derive(Args)]
pub struct ConfigGetArgs {
    /// Configuration key to retrieve
    pub key: String,
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `config list`
#[derive(Args)]
pub struct ConfigListArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
    /// Show values (by default, sensitive values are masked)
    #[arg(long)]
    pub show_values: bool,
}

/// Arguments for `config unset`
#[derive(Args)]
pub struct ConfigUnsetArgs {
    /// Configuration key to remove
    pub key: String,
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `config check`
#[derive(Args)]
pub struct ConfigCheckArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Persistent configuration stored in config file
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistentConfig {
    /// API key for authentication
    pub api_key: Option<String>,
    /// API URL (defaults to https://memvid.com) - legacy, use dashboard_url
    pub api_url: Option<String>,
    /// Dashboard URL (defaults to https://memvid.com)
    pub dashboard_url: Option<String>,
    /// Default memory ID for dashboard sync (legacy, use [memory] section)
    pub memory_id: Option<String>,
    /// Named memory IDs (e.g., memory.default, memory.work)
    #[serde(default)]
    pub memory: HashMap<String, String>,
    /// Default embedding provider
    pub default_embedding_provider: Option<String>,
    /// Default LLM provider
    pub default_llm_provider: Option<String>,
    /// Additional custom settings
    #[serde(flatten)]
    pub extra: HashMap<String, String>,
}

impl PersistentConfig {
    /// Get the config file path (~/.config/memvid/config.toml)
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .context("Could not determine config directory")?
            .join("memvid");
        Ok(config_dir.join("config.toml"))
    }

    /// Load config from file, or return default if file doesn't exist
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: PersistentConfig = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        Ok(config)
    }

    /// Save config to file
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }
        let content = toml::to_string_pretty(self).context("Failed to serialize config")?;
        fs::write(&path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;
        Ok(())
    }

    /// Get a value by key (supports memory.<name> syntax)
    pub fn get(&self, key: &str) -> Option<String> {
        // Handle memory.<name> syntax
        if let Some(name) = key.strip_prefix("memory.") {
            return self.memory.get(name).cloned();
        }

        match key {
            "api_key" => self.api_key.clone(),
            "api_url" => self.api_url.clone(),
            "dashboard_url" => self.dashboard_url.clone(),
            "memory_id" => self.memory_id.clone(),
            "default_embedding_provider" => self.default_embedding_provider.clone(),
            "default_llm_provider" => self.default_llm_provider.clone(),
            _ => self.extra.get(key).cloned(),
        }
    }

    /// Set a value by key (supports memory.<name> syntax)
    pub fn set(&mut self, key: &str, value: String) {
        // Handle memory.<name> syntax
        if let Some(name) = key.strip_prefix("memory.") {
            self.memory.insert(name.to_string(), value);
            return;
        }

        match key {
            "api_key" => self.api_key = Some(value),
            "api_url" => self.api_url = Some(value),
            "dashboard_url" => self.dashboard_url = Some(value),
            "memory_id" => self.memory_id = Some(value),
            "default_embedding_provider" => self.default_embedding_provider = Some(value),
            "default_llm_provider" => self.default_llm_provider = Some(value),
            _ => {
                self.extra.insert(key.to_string(), value);
            }
        }
    }

    /// Unset a value by key (supports memory.<name> syntax)
    pub fn unset(&mut self, key: &str) -> bool {
        // Handle memory.<name> syntax
        if let Some(name) = key.strip_prefix("memory.") {
            return self.memory.remove(name).is_some();
        }

        match key {
            "api_key" => {
                let had_value = self.api_key.is_some();
                self.api_key = None;
                had_value
            }
            "api_url" => {
                let had_value = self.api_url.is_some();
                self.api_url = None;
                had_value
            }
            "dashboard_url" => {
                let had_value = self.dashboard_url.is_some();
                self.dashboard_url = None;
                had_value
            }
            "memory_id" => {
                let had_value = self.memory_id.is_some();
                self.memory_id = None;
                had_value
            }
            "default_embedding_provider" => {
                let had_value = self.default_embedding_provider.is_some();
                self.default_embedding_provider = None;
                had_value
            }
            "default_llm_provider" => {
                let had_value = self.default_llm_provider.is_some();
                self.default_llm_provider = None;
                had_value
            }
            _ => self.extra.remove(key).is_some(),
        }
    }

    /// Get the default memory ID (from memory.default or legacy memory_id)
    pub fn default_memory_id(&self) -> Option<String> {
        self.memory
            .get("default")
            .cloned()
            .or_else(|| self.memory_id.clone())
    }

    /// Get a named memory ID
    pub fn get_memory(&self, name: &str) -> Option<String> {
        self.memory.get(name).cloned()
    }

    /// Get all keys and values as a map
    pub fn to_map(&self) -> HashMap<String, Option<String>> {
        let mut map = HashMap::new();
        map.insert("api_key".to_string(), self.api_key.clone());
        map.insert("dashboard_url".to_string(), self.dashboard_url.clone());
        // Include named memories
        for (name, id) in &self.memory {
            map.insert(format!("memory.{}", name), Some(id.clone()));
        }
        map.insert(
            "default_embedding_provider".to_string(),
            self.default_embedding_provider.clone(),
        );
        map.insert(
            "default_llm_provider".to_string(),
            self.default_llm_provider.clone(),
        );
        for (k, v) in &self.extra {
            map.insert(k.clone(), Some(v.clone()));
        }
        map
    }

    /// Check if a key is sensitive (should be masked in output)
    pub fn is_sensitive(key: &str) -> bool {
        key.contains("key")
            || key.contains("secret")
            || key.contains("token")
            || key.contains("password")
    }

    /// Mask a sensitive value for display
    pub fn mask_value(value: &str) -> String {
        if value.len() <= 8 {
            "*".repeat(value.len())
        } else {
            format!("{}...{}", &value[..4], &value[value.len() - 4..])
        }
    }
}

/// Known configuration keys for help text
const KNOWN_KEYS: &[(&str, &str)] = &[
    ("api_key", "Memvid API key for authentication"),
    (
        "dashboard_url",
        "Dashboard URL (default: https://memvid.com)",
    ),
    (
        "memory.<name>",
        "Named memory ID (e.g., memory.default, memory.work)",
    ),
    (
        "default_embedding_provider",
        "Default embedding provider (e.g., openai, local)",
    ),
    (
        "default_llm_provider",
        "Default LLM provider for ask commands",
    ),
];

pub fn handle_config(args: ConfigArgs) -> Result<()> {
    match args.command {
        ConfigCommand::Set(set_args) => handle_config_set(set_args),
        ConfigCommand::Get(get_args) => handle_config_get(get_args),
        ConfigCommand::List(list_args) => handle_config_list(list_args),
        ConfigCommand::Unset(unset_args) => handle_config_unset(unset_args),
        ConfigCommand::Check(check_args) => handle_config_check(check_args),
    }
}

fn handle_config_set(args: ConfigSetArgs) -> Result<()> {
    let mut config = PersistentConfig::load()?;
    config.set(&args.key, args.value.clone());
    config.save()?;

    let display_value = if PersistentConfig::is_sensitive(&args.key) {
        PersistentConfig::mask_value(&args.value)
    } else {
        args.value.clone()
    };

    if args.json {
        let output = json!({
            "success": true,
            "key": args.key,
            "value": display_value,
            "message": format!("Configuration '{}' has been set", args.key),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Set {} = {}", args.key, display_value);
    }

    Ok(())
}

fn handle_config_get(args: ConfigGetArgs) -> Result<()> {
    let config = PersistentConfig::load()?;

    match config.get(&args.key) {
        Some(value) => {
            let display_value = if PersistentConfig::is_sensitive(&args.key) {
                PersistentConfig::mask_value(&value)
            } else {
                value.clone()
            };

            if args.json {
                let output = json!({
                    "key": args.key,
                    "value": display_value,
                    "found": true,
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("{}", display_value);
            }
        }
        None => {
            if args.json {
                let output = json!({
                    "key": args.key,
                    "value": null,
                    "found": false,
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                // Check if it's a known key
                if let Some((_, description)) = KNOWN_KEYS.iter().find(|(k, _)| *k == args.key) {
                    println!("'{}' is not set", args.key);
                    println!("Description: {}", description);
                } else {
                    println!("'{}' is not set", args.key);
                }
            }
        }
    }

    Ok(())
}

fn handle_config_list(args: ConfigListArgs) -> Result<()> {
    let config = PersistentConfig::load()?;
    let map = config.to_map();
    let config_path = PersistentConfig::config_path()?;

    if args.json {
        let mut values = serde_json::Map::new();
        for (key, value) in &map {
            if let Some(v) = value {
                let display_value = if !args.show_values && PersistentConfig::is_sensitive(&key) {
                    PersistentConfig::mask_value(v)
                } else {
                    v.clone()
                };
                values.insert(key.clone(), json!(display_value));
            }
        }
        let output = json!({
            "config_path": config_path.display().to_string(),
            "values": values,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Config file: {}", config_path.display());
        println!();

        let mut has_values = false;
        for (key, description) in KNOWN_KEYS {
            if let Some(Some(value)) = map.get(*key) {
                has_values = true;
                let display_value = if !args.show_values && PersistentConfig::is_sensitive(key) {
                    PersistentConfig::mask_value(value)
                } else {
                    value.clone()
                };
                println!("  {} = {}", key, display_value);
                println!("    {}", description);
            }
        }

        // Show named memories
        if !config.memory.is_empty() {
            has_values = true;
            println!();
            println!("  [memory]");
            for (name, id) in &config.memory {
                println!("    {} = {}", name, id);
            }
        }

        // Show any extra custom keys
        for (key, value) in &config.extra {
            has_values = true;
            let display_value = if !args.show_values && PersistentConfig::is_sensitive(&key) {
                PersistentConfig::mask_value(value)
            } else {
                value.clone()
            };
            println!("  {} = {}", key, display_value);
        }

        if !has_values {
            println!("  (no configuration set)");
            println!();
            println!("Available keys:");
            for (key, description) in KNOWN_KEYS {
                println!("  {} - {}", key, description);
            }
        }

        if !args.show_values {
            println!();
            println!("Use --show-values to display full values");
        }
    }

    Ok(())
}

fn handle_config_unset(args: ConfigUnsetArgs) -> Result<()> {
    let mut config = PersistentConfig::load()?;
    let was_set = config.unset(&args.key);
    config.save()?;

    if args.json {
        let output = json!({
            "success": true,
            "key": args.key,
            "was_set": was_set,
            "message": if was_set {
                format!("Configuration '{}' has been removed", args.key)
            } else {
                format!("Configuration '{}' was not set", args.key)
            },
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        if was_set {
            println!("Removed '{}'", args.key);
        } else {
            println!("'{}' was not set", args.key);
        }
    }

    Ok(())
}

fn handle_config_check(args: ConfigCheckArgs) -> Result<()> {
    let config = PersistentConfig::load()?;
    let config_path = PersistentConfig::config_path()?;

    // Also check environment variables
    let env_api_key = std::env::var("MEMVID_API_KEY").ok();
    let env_api_url = std::env::var("MEMVID_API_URL").ok();

    // Determine effective values (env takes precedence)
    let effective_api_key = env_api_key.clone().or(config.api_key.clone());
    let effective_api_url = env_api_url
        .clone()
        .or(config.api_url.clone())
        .unwrap_or_else(|| "https://memvid.com".to_string());

    let has_api_key = effective_api_key.is_some();
    let api_key_source = if env_api_key.is_some() {
        "environment"
    } else if config.api_key.is_some() {
        "config file"
    } else {
        "not set"
    };

    let api_url_source = if env_api_url.is_some() {
        "environment"
    } else if config.api_url.is_some() {
        "config file"
    } else {
        "default"
    };

    // TODO: Actually verify the API key with the server
    // For now, just check if it's set
    let api_key_valid = has_api_key; // Placeholder - should verify with server

    if args.json {
        let output = json!({
            "config_path": config_path.display().to_string(),
            "api_key": {
                "set": has_api_key,
                "source": api_key_source,
                "valid": api_key_valid,
            },
            "api_url": {
                "value": effective_api_url,
                "source": api_url_source,
            },
            "ready": has_api_key && api_key_valid,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Configuration Check");
        println!("===================");
        println!();
        println!("Config file: {}", config_path.display());
        println!();

        // API Key status
        if has_api_key {
            println!(
                "API Key: {} (source: {})",
                if api_key_valid { "valid" } else { "invalid" },
                api_key_source
            );
            if let Some(key) = &effective_api_key {
                println!("  Value: {}", PersistentConfig::mask_value(key));
            }
        } else {
            println!("API Key: not configured");
            println!();
            println!("To set your API key:");
            println!("  memvid config set api_key <your-key>");
            println!("  OR");
            println!("  export MEMVID_API_KEY=<your-key>");
            println!();
            println!("Get your API key at: https://memvid.com/dashboard/api-keys");
        }

        println!();
        println!(
            "API URL: {} (source: {})",
            effective_api_url, api_url_source
        );

        println!();
        if has_api_key && api_key_valid {
            println!("Status: Ready to use");
        } else {
            println!("Status: API key required");
        }
    }

    Ok(())
}
