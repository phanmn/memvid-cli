//! CLI configuration and environment handling
//!
//! This module provides configuration loading from environment variables,
//! tracing initialization, and embedding runtime management for semantic search.

use std::env;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Result};
use ed25519_dalek::VerifyingKey;
use memvid_core::EmbeddingProvider;

const DEFAULT_API_URL: &str = "https://memvid.com";
const DEFAULT_CACHE_DIR: &str = "~/.cache/memvid";

/// Supported embedding models for semantic search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingModelChoice {
    /// BGE-small-en-v1.5: Fast, 384-dim, ~78% accuracy (default)
    #[default]
    BgeSmall,
    /// BGE-base-en-v1.5: Balanced, 768-dim, ~85% accuracy
    BgeBase,
    /// Nomic-embed-text-v1.5: High accuracy, 768-dim, ~86% accuracy
    Nomic,
    /// GTE-large-en-v1.5: Best semantic depth, 1024-dim
    GteLarge,
    /// OpenAI text-embedding-3-large: Highest quality, 3072-dim (requires OPENAI_API_KEY)
    OpenAILarge,
    /// OpenAI text-embedding-3-small: Good quality, 1536-dim (requires OPENAI_API_KEY)
    OpenAISmall,
    /// OpenAI text-embedding-ada-002: Legacy model, 1536-dim (requires OPENAI_API_KEY)
    OpenAIAda,
    /// NVIDIA nv-embed-v1: High quality, remote embeddings (requires NVIDIA_API_KEY)
    Nvidia,
    /// Gemini text-embedding-004: Google AI embeddings, 768-dim (requires GOOGLE_API_KEY or GEMINI_API_KEY)
    Gemini,
    /// Mistral mistral-embed: Mistral AI embeddings, 1024-dim (requires MISTRAL_API_KEY)
    Mistral,
    /// OpenRouter embeddings: Access 100+ embedding models via OpenRouter API (requires OPENROUTER_API_KEY)
    OpenRouter,
}

impl EmbeddingModelChoice {
    /// Check if this is an OpenAI model (requires OPENAI_API_KEY)
    pub fn is_openai(&self) -> bool {
        matches!(
            self,
            EmbeddingModelChoice::OpenAILarge
                | EmbeddingModelChoice::OpenAISmall
                | EmbeddingModelChoice::OpenAIAda
        )
    }

    /// Check if this is a remote/cloud model (not local fastembed)
    pub fn is_remote(&self) -> bool {
        matches!(
            self,
            EmbeddingModelChoice::OpenAILarge
                | EmbeddingModelChoice::OpenAISmall
                | EmbeddingModelChoice::OpenAIAda
                | EmbeddingModelChoice::Nvidia
                | EmbeddingModelChoice::Gemini
                | EmbeddingModelChoice::Mistral
                | EmbeddingModelChoice::OpenRouter
        )
    }

    /// Get the fastembed EmbeddingModel enum value (only for local models)
    ///
    /// # Panics
    /// Panics if called on an OpenAI model. Use `is_openai()` to check first.
    #[cfg(feature = "local-embeddings")]
    pub fn to_fastembed_model(&self) -> fastembed::EmbeddingModel {
        match self {
            EmbeddingModelChoice::BgeSmall => fastembed::EmbeddingModel::BGESmallENV15,
            EmbeddingModelChoice::BgeBase => fastembed::EmbeddingModel::BGEBaseENV15,
            EmbeddingModelChoice::Nomic => fastembed::EmbeddingModel::NomicEmbedTextV15,
            EmbeddingModelChoice::GteLarge => fastembed::EmbeddingModel::GTELargeENV15,
            EmbeddingModelChoice::OpenAILarge
            | EmbeddingModelChoice::OpenAISmall
            | EmbeddingModelChoice::OpenAIAda => {
                panic!("OpenAI models don't use fastembed. Check is_remote() first.")
            }
            EmbeddingModelChoice::Nvidia => {
                panic!("NVIDIA embeddings don't use fastembed. Check is_remote() first.")
            }
            EmbeddingModelChoice::Gemini => {
                panic!("Gemini embeddings don't use fastembed. Check is_remote() first.")
            }
            EmbeddingModelChoice::Mistral => {
                panic!("Mistral embeddings don't use fastembed. Check is_remote() first.")
            }
            EmbeddingModelChoice::OpenRouter => {
                panic!("OpenRouter embeddings don't use fastembed. Check is_remote() first.")
            }
        }
    }

    /// Get human-readable model name
    pub fn name(&self) -> &'static str {
        match self {
            EmbeddingModelChoice::BgeSmall => "bge-small",
            EmbeddingModelChoice::BgeBase => "bge-base",
            EmbeddingModelChoice::Nomic => "nomic",
            EmbeddingModelChoice::GteLarge => "gte-large",
            EmbeddingModelChoice::OpenAILarge => "openai-large",
            EmbeddingModelChoice::OpenAISmall => "openai-small",
            EmbeddingModelChoice::OpenAIAda => "openai-ada",
            EmbeddingModelChoice::Nvidia => "nvidia",
            EmbeddingModelChoice::Gemini => "gemini",
            EmbeddingModelChoice::Mistral => "mistral",
            EmbeddingModelChoice::OpenRouter => "openrouter",
        }
    }

    /// Get the canonical provider model identifier used for persisted metadata.
    ///
    /// This is intended to match upstream provider IDs (OpenAI) and HuggingFace-style IDs
    /// (fastembed/ONNX) so that memories can record an embedding "identity" that other
    /// runtimes can select deterministically.
    pub fn canonical_model_id(&self) -> &'static str {
        match self {
            EmbeddingModelChoice::BgeSmall => "BAAI/bge-small-en-v1.5",
            EmbeddingModelChoice::BgeBase => "BAAI/bge-base-en-v1.5",
            EmbeddingModelChoice::Nomic => "nomic-embed-text-v1.5",
            EmbeddingModelChoice::GteLarge => "thenlper/gte-large",
            EmbeddingModelChoice::OpenAILarge => "text-embedding-3-large",
            EmbeddingModelChoice::OpenAISmall => "text-embedding-3-small",
            EmbeddingModelChoice::OpenAIAda => "text-embedding-ada-002",
            EmbeddingModelChoice::Nvidia => "nvidia/nv-embed-v1",
            EmbeddingModelChoice::Gemini => "text-embedding-004",
            EmbeddingModelChoice::Mistral => "mistral-embed",
            EmbeddingModelChoice::OpenRouter => "openrouter",
        }
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        match self {
            EmbeddingModelChoice::BgeSmall => 384,
            EmbeddingModelChoice::BgeBase => 768,
            EmbeddingModelChoice::Nomic => 768,
            EmbeddingModelChoice::GteLarge => 1024,
            EmbeddingModelChoice::OpenAILarge => 3072,
            EmbeddingModelChoice::OpenAISmall => 1536,
            EmbeddingModelChoice::OpenAIAda => 1536,
            // Remote model; infer from the first embedding response.
            EmbeddingModelChoice::Nvidia => 0,
            EmbeddingModelChoice::Gemini => 768,
            EmbeddingModelChoice::Mistral => 1024,
            // OpenRouter - dimension depends on model, infer from first response
            EmbeddingModelChoice::OpenRouter => 0,
        }
    }
}

impl FromStr for EmbeddingModelChoice {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let lowered = s.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "bge-small" | "bge_small" | "bgesmall" | "small" => Ok(EmbeddingModelChoice::BgeSmall),
            "baai/bge-small-en-v1.5" => Ok(EmbeddingModelChoice::BgeSmall),
            "bge-base" | "bge_base" | "bgebase" | "base" => Ok(EmbeddingModelChoice::BgeBase),
            "baai/bge-base-en-v1.5" => Ok(EmbeddingModelChoice::BgeBase),
            "nomic" | "nomic-embed" | "nomic_embed" => Ok(EmbeddingModelChoice::Nomic),
            "nomic-embed-text-v1.5" => Ok(EmbeddingModelChoice::Nomic),
            "gte-large" | "gte_large" | "gtelarge" | "gte" => Ok(EmbeddingModelChoice::GteLarge),
            "thenlper/gte-large" => Ok(EmbeddingModelChoice::GteLarge),
            // OpenAI models - default "openai" maps to "openai-large" for highest quality
            "openai" | "openai-large" | "openai_large" | "text-embedding-3-large" => {
                Ok(EmbeddingModelChoice::OpenAILarge)
            }
            "openai-small" | "openai_small" | "text-embedding-3-small" => {
                Ok(EmbeddingModelChoice::OpenAISmall)
            }
            "openai-ada" | "openai_ada" | "text-embedding-ada-002" | "ada" => {
                Ok(EmbeddingModelChoice::OpenAIAda)
            }
            "nvidia" | "nv" | "nv-embed-v1" | "nvidia/nv-embed-v1" => Ok(EmbeddingModelChoice::Nvidia),
            _ if lowered.starts_with("nvidia/") || lowered.starts_with("nvidia:") || lowered.starts_with("nv:") => {
                Ok(EmbeddingModelChoice::Nvidia)
            }
            // Gemini embeddings
            "gemini" | "gemini-embed" | "text-embedding-004" | "gemini-embedding-001" => {
                Ok(EmbeddingModelChoice::Gemini)
            }
            _ if lowered.starts_with("gemini/") || lowered.starts_with("gemini:") || lowered.starts_with("google:") => {
                Ok(EmbeddingModelChoice::Gemini)
            }
            // Mistral embeddings
            "mistral" | "mistral-embed" => Ok(EmbeddingModelChoice::Mistral),
            _ if lowered.starts_with("mistral/") || lowered.starts_with("mistral:") => {
                Ok(EmbeddingModelChoice::Mistral)
            }
            // OpenRouter embeddings
            "openrouter" | "openrouter-embed" => Ok(EmbeddingModelChoice::OpenRouter),
            _ if lowered.starts_with("openrouter/") || lowered.starts_with("openrouter:") => {
                Ok(EmbeddingModelChoice::OpenRouter)
            }
            _ => Err(anyhow!(
                "unknown embedding model '{}'. Valid options: bge-small, bge-base, nomic, gte-large, openai, openai-small, openai-ada, nvidia, gemini, mistral, openrouter",
                s
            )),
        }
    }
}

impl std::fmt::Display for EmbeddingModelChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl EmbeddingModelChoice {
    /// Infer the best embedding model from vector dimension stored in MV2 file.
    ///
    /// This enables auto-detection: users don't need to specify --query-embedding-model
    /// if the MV2 file has vectors. The dimension uniquely identifies the model family.
    ///
    /// # Dimension Mapping
    /// - 384  → BGE-small (default local model)
    /// - 768  → BGE-base (could also be Nomic, but same dimension works)
    /// - 1024 → GTE-large
    /// - 1536 → OpenAI small/ada
    /// - 3072 → OpenAI large
    pub fn from_dimension(dim: u32) -> Option<Self> {
        match dim {
            384 => Some(EmbeddingModelChoice::BgeSmall),
            768 => Some(EmbeddingModelChoice::BgeBase), // Could be Nomic, but same dim
            1024 => Some(EmbeddingModelChoice::GteLarge),
            1536 => Some(EmbeddingModelChoice::OpenAISmall), // Could be Ada, same dim
            3072 => Some(EmbeddingModelChoice::OpenAILarge),
            0 => None, // No vectors in file
            _ => {
                tracing::warn!("Unknown embedding dimension {}, using default model", dim);
                None
            }
        }
    }
}

/// CLI configuration loaded from environment variables and config file
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub api_key: Option<String>,
    pub api_url: String,
    /// Default memory ID for dashboard sync (from config file)
    pub memory_id: Option<String>,
    pub cache_dir: PathBuf,
    pub ticket_pubkey: Option<VerifyingKey>,
    pub models_dir: PathBuf,
    pub offline: bool,
    /// Embedding model for semantic search (can be overridden by CLI flag)
    pub embedding_model: EmbeddingModelChoice,
}

impl PartialEq for CliConfig {
    fn eq(&self, other: &Self) -> bool {
        self.api_key == other.api_key
            && self.api_url == other.api_url
            && self.memory_id == other.memory_id
            && self.cache_dir == other.cache_dir
            && self.models_dir == other.models_dir
            && self.offline == other.offline
            && self.embedding_model == other.embedding_model
    }
}

impl Eq for CliConfig {}

impl CliConfig {
    pub fn load() -> Result<Self> {
        // Load persistent config file (if exists) for fallback values
        let persistent_config = crate::commands::config::PersistentConfig::load().ok();

        // API Key: env var takes precedence, then config file
        let api_key = env::var("MEMVID_API_KEY")
            .ok()
            .and_then(|value| {
                let trimmed = value.trim().to_string();
                (!trimmed.is_empty()).then_some(trimmed)
            })
            .or_else(|| persistent_config.as_ref().and_then(|c| c.api_key.clone()));

        // API URL: env var takes precedence, then config file, then default
        let api_url = env::var("MEMVID_API_URL")
            .ok()
            .or_else(|| persistent_config.as_ref().and_then(|c| c.api_url.clone()))
            .unwrap_or_else(|| DEFAULT_API_URL.to_string());

        // Memory ID: env var takes precedence, then config file (memory.default or legacy memory_id)
        let memory_id = env::var("MEMVID_MEMORY_ID")
            .ok()
            .and_then(|value| {
                let trimmed = value.trim().to_string();
                (!trimmed.is_empty()).then_some(trimmed)
            })
            .or_else(|| {
                persistent_config
                    .as_ref()
                    .and_then(|c| c.default_memory_id())
            });

        let cache_dir_raw =
            env::var("MEMVID_CACHE_DIR").unwrap_or_else(|_| DEFAULT_CACHE_DIR.to_string());
        let cache_dir = expand_path(&cache_dir_raw)?;

        let models_dir_raw =
            env::var("MEMVID_MODELS_DIR").unwrap_or_else(|_| "~/.memvid/models".to_string());
        let models_dir = expand_path(&models_dir_raw)?;

        // Default public key for memvid.com dashboard ticket verification
        // This allows users to use --memory-id without setting MEMVID_TICKET_PUBKEY
        // Must match memvid-core's MEMVID_TICKET_PUBKEY constant
        const DEFAULT_TICKET_PUBKEY: &str = "DFKNhP/yO5i1b9aKL+aHeBaGunz9sMfOF736fzYws4Q=";

        let ticket_pubkey_str = env::var("MEMVID_TICKET_PUBKEY")
            .ok()
            .and_then(|value| {
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
            .unwrap_or_else(|| DEFAULT_TICKET_PUBKEY.to_string());

        let ticket_pubkey = Some(memvid_core::parse_ed25519_public_key_base64(
            &ticket_pubkey_str,
        )?);

        let offline = env::var("MEMVID_OFFLINE")
            .ok()
            .map(|value| match value.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" => true,
                _ => false,
            })
            .unwrap_or(false);

        // Load embedding model from env var, default to BGE-small
        let embedding_model = env::var("MEMVID_EMBEDDING_MODEL")
            .ok()
            .and_then(|value| {
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    EmbeddingModelChoice::from_str(trimmed).ok()
                }
            })
            .unwrap_or_default();

        Ok(Self {
            api_key,
            api_url,
            memory_id,
            cache_dir,
            ticket_pubkey,
            models_dir,
            offline,
            embedding_model,
        })
    }

    /// Create a new config with a different embedding model
    pub fn with_embedding_model(&self, model: EmbeddingModelChoice) -> Self {
        Self {
            embedding_model: model,
            ..self.clone()
        }
    }
}

fn expand_path(value: &str) -> Result<PathBuf> {
    if value.trim().is_empty() {
        return Err(anyhow!("cache directory cannot be empty"));
    }

    let expanded = if let Some(stripped) = value.strip_prefix("~/") {
        home_dir()?.join(stripped)
    } else if let Some(stripped) = value.strip_prefix("~\\") {
        // Support Windows-style "~\" prefix.
        home_dir()?.join(stripped)
    } else if value == "~" {
        home_dir()?
    } else {
        PathBuf::from(value)
    };

    if expanded.is_absolute() {
        Ok(expanded)
    } else {
        Ok(env::current_dir()?.join(expanded))
    }
}

fn home_dir() -> Result<PathBuf> {
    if let Some(path) = env::var_os("HOME") {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    #[cfg(windows)]
    {
        if let Some(path) = env::var_os("USERPROFILE") {
            if !path.is_empty() {
                return Ok(PathBuf::from(path));
            }
        }
        if let (Some(drive), Some(path)) = (env::var_os("HOMEDRIVE"), env::var_os("HOMEPATH")) {
            if !drive.is_empty() && !path.is_empty() {
                return Ok(PathBuf::from(format!(
                    "{}{}",
                    drive.to_string_lossy(),
                    path.to_string_lossy()
                )));
            }
        }
    }

    Err(anyhow!("unable to resolve home directory"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use base64::Engine;
    use ed25519_dalek::SigningKey;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn set_or_unset(var: &str, value: Option<String>) {
        match value {
            Some(v) => unsafe { env::set_var(var, v) },
            None => unsafe { env::remove_var(var) },
        }
    }

    #[test]
    fn defaults_expand_using_home_directory() {
        let _guard = env_lock();

        let previous_home = env::var("HOME").ok();
        #[cfg(windows)]
        let previous_userprofile = env::var("USERPROFILE").ok();

        for var in [
            "MEMVID_API_KEY",
            "MEMVID_API_URL",
            "MEMVID_CACHE_DIR",
            "MEMVID_TICKET_PUBKEY",
            "MEMVID_MODELS_DIR",
            "MEMVID_OFFLINE",
        ] {
            unsafe { env::remove_var(var) };
        }

        let tmp = tempfile::tempdir().expect("tmpdir");
        let tmp_path = tmp.path().to_path_buf();
        unsafe { env::set_var("HOME", &tmp_path) };
        #[cfg(windows)]
        unsafe {
            env::set_var("USERPROFILE", &tmp_path)
        };

        let config = CliConfig::load().expect("load");
        assert_eq!(config.api_key, None);
        assert_eq!(config.api_url, "https://memvid.com");
        assert_eq!(config.cache_dir, tmp_path.join(".cache/memvid"));
        // ticket_pubkey has a default value now, so it should be Some
        assert!(config.ticket_pubkey.is_some());
        assert_eq!(config.models_dir, tmp_path.join(".memvid/models"));
        assert!(!config.offline);

        set_or_unset("HOME", previous_home);
        #[cfg(windows)]
        {
            set_or_unset("USERPROFILE", previous_userprofile);
        }
    }

    #[test]
    fn env_overrides_are_respected() {
        let _guard = env_lock();

        let previous_env: Vec<(&'static str, Option<String>)> = [
            "MEMVID_API_KEY",
            "MEMVID_API_URL",
            "MEMVID_CACHE_DIR",
            "MEMVID_TICKET_PUBKEY",
            "MEMVID_MODELS_DIR",
            "MEMVID_OFFLINE",
        ]
        .into_iter()
        .map(|var| (var, env::var(var).ok()))
        .collect();

        unsafe { env::set_var("MEMVID_API_KEY", "abc123") };
        unsafe { env::set_var("MEMVID_API_URL", "https://staging.memvid.app") };
        unsafe { env::set_var("MEMVID_CACHE_DIR", "~/memvid-cache") };
        unsafe { env::set_var("MEMVID_MODELS_DIR", "~/models") };
        unsafe { env::set_var("MEMVID_OFFLINE", "true") };
        let signing = SigningKey::from_bytes(&[9u8; 32]);
        let encoded = BASE64_STANDARD.encode(signing.verifying_key().as_bytes());
        unsafe { env::set_var("MEMVID_TICKET_PUBKEY", encoded) };

        let tmp = tempfile::tempdir().expect("tmpdir");
        let tmp_path = tmp.path().to_path_buf();
        unsafe { env::set_var("HOME", &tmp_path) };
        #[cfg(windows)]
        unsafe {
            env::set_var("USERPROFILE", &tmp_path)
        };

        let config = CliConfig::load().expect("load");
        assert_eq!(config.api_key.as_deref(), Some("abc123"));
        assert_eq!(config.api_url, "https://staging.memvid.app");
        assert_eq!(config.cache_dir, tmp_path.join("memvid-cache"));
        assert_eq!(
            config.ticket_pubkey.expect("pubkey").as_bytes(),
            signing.verifying_key().as_bytes()
        );
        assert_eq!(config.models_dir, tmp_path.join("models"));
        assert!(config.offline);

        for (var, value) in previous_env {
            set_or_unset(var, value);
        }
    }

    #[test]
    fn rejects_empty_cache_dir() {
        let _guard = env_lock();

        let previous = env::var("MEMVID_CACHE_DIR").ok();
        unsafe { env::set_var("MEMVID_CACHE_DIR", " ") };
        let err = CliConfig::load().expect_err("should fail");
        assert!(err.to_string().contains("cache directory"));
        set_or_unset("MEMVID_CACHE_DIR", previous);
    }
}

/// Initialize tracing/logging based on verbosity level
pub fn init_tracing(verbosity: u8) -> Result<()> {
    use std::io::IsTerminal;
    use tracing_subscriber::{filter::Directive, fmt, EnvFilter};

    let level = match verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    let mut env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    for directive_str in ["llama_cpp=error", "llama_cpp_sys=error", "ggml=error"] {
        if let Ok(directive) = directive_str.parse::<Directive>() {
            env_filter = env_filter.add_directive(directive);
        }
    }

    // Disable ANSI color codes when stderr is not a terminal (e.g., piped or
    // combined with `2>&1`). This prevents control characters from polluting
    // JSON output when combined with stdout.
    let use_ansi = std::io::stderr().is_terminal();

    fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .without_time()
        .with_ansi(use_ansi)
        .try_init()
        .map_err(|err| anyhow!(err))?;
    Ok(())
}

/// Resolve LLM context budget override from CLI or environment
pub fn resolve_llm_context_budget_override(cli_value: Option<usize>) -> Result<Option<usize>> {
    use anyhow::bail;

    if let Some(value) = cli_value {
        if value == 0 {
            bail!("--llm-context-depth must be a positive integer");
        }
        return Ok(Some(value));
    }

    let raw_env = match env::var("MEMVID_LLM_CONTEXT_BUDGET") {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };

    let trimmed = raw_env.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let digits: String = trimmed
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace() && *ch != '_')
        .collect();

    if digits.is_empty() {
        bail!("MEMVID_LLM_CONTEXT_BUDGET must be a positive integer value");
    }

    let value: usize = digits.parse().map_err(|err| {
        anyhow!(
            "MEMVID_LLM_CONTEXT_BUDGET value '{}' is not a valid number: {}",
            trimmed,
            err
        )
    })?;

    if value == 0 {
        bail!("MEMVID_LLM_CONTEXT_BUDGET must be a positive integer");
    }

    Ok(Some(value))
}

use crate::gemini_embeddings::GeminiEmbeddingProvider;
use crate::mistral_embeddings::MistralEmbeddingProvider;
use crate::nvidia_embeddings::NvidiaEmbeddingProvider;
use crate::openai_embeddings::OpenAIEmbeddingProvider;
use crate::openrouter_embeddings::OpenRouterEmbeddingProvider;

/// Internal embedding backend - local fastembed or remote providers.
#[derive(Clone)]
enum EmbeddingBackend {
    #[cfg(feature = "local-embeddings")]
    FastEmbed(std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>),
    OpenAI(std::sync::Arc<OpenAIEmbeddingProvider>),
    Nvidia(std::sync::Arc<NvidiaEmbeddingProvider>),
    Gemini(std::sync::Arc<GeminiEmbeddingProvider>),
    Mistral(std::sync::Arc<MistralEmbeddingProvider>),
    OpenRouter(std::sync::Arc<OpenRouterEmbeddingProvider>),
}

/// Embedding runtime wrapper supporting local and remote embeddings
#[derive(Clone)]
pub struct EmbeddingRuntime {
    backend: EmbeddingBackend,
    model: EmbeddingModelChoice,
    dimension: std::sync::Arc<AtomicUsize>,
}

impl EmbeddingRuntime {
    #[cfg(feature = "local-embeddings")]
    fn new_fastembed(
        backend: fastembed::TextEmbedding,
        model: EmbeddingModelChoice,
        dimension: usize,
    ) -> Self {
        Self {
            backend: EmbeddingBackend::FastEmbed(std::sync::Arc::new(std::sync::Mutex::new(
                backend,
            ))),
            model,
            dimension: std::sync::Arc::new(AtomicUsize::new(dimension)),
        }
    }

    fn new_openai(
        provider: OpenAIEmbeddingProvider,
        model: EmbeddingModelChoice,
        dimension: usize,
    ) -> Self {
        Self {
            backend: EmbeddingBackend::OpenAI(std::sync::Arc::new(provider)),
            model,
            dimension: std::sync::Arc::new(AtomicUsize::new(dimension)),
        }
    }

    fn new_nvidia(provider: NvidiaEmbeddingProvider, model: EmbeddingModelChoice) -> Self {
        Self {
            backend: EmbeddingBackend::Nvidia(std::sync::Arc::new(provider)),
            model,
            dimension: std::sync::Arc::new(AtomicUsize::new(0)),
        }
    }

    fn new_gemini(
        provider: GeminiEmbeddingProvider,
        model: EmbeddingModelChoice,
        dimension: usize,
    ) -> Self {
        Self {
            backend: EmbeddingBackend::Gemini(std::sync::Arc::new(provider)),
            model,
            dimension: std::sync::Arc::new(AtomicUsize::new(dimension)),
        }
    }

    fn new_mistral(
        provider: MistralEmbeddingProvider,
        model: EmbeddingModelChoice,
        dimension: usize,
    ) -> Self {
        Self {
            backend: EmbeddingBackend::Mistral(std::sync::Arc::new(provider)),
            model,
            dimension: std::sync::Arc::new(AtomicUsize::new(dimension)),
        }
    }

    fn new_openrouter(
        provider: OpenRouterEmbeddingProvider,
        model: EmbeddingModelChoice,
        dimension: usize,
    ) -> Self {
        Self {
            backend: EmbeddingBackend::OpenRouter(std::sync::Arc::new(provider)),
            model,
            dimension: std::sync::Arc::new(AtomicUsize::new(dimension)),
        }
    }

    const MAX_OPENAI_EMBEDDING_TEXT_LEN: usize = 20_000;
    // NVIDIA Integrate embeddings enforce a 4096 token limit; use a tighter char cap as a guardrail.
    const MAX_NVIDIA_EMBEDDING_TEXT_LEN: usize = 12_000;

    // Gemini has an 8192 token limit, using conservative estimate
    const MAX_GEMINI_EMBEDDING_TEXT_LEN: usize = 20_000;
    // Mistral has an 8192 token limit, using conservative estimate
    const MAX_MISTRAL_EMBEDDING_TEXT_LEN: usize = 20_000;
    // OpenRouter uses variable limits depending on model, using conservative estimate
    const MAX_OPENROUTER_EMBEDDING_TEXT_LEN: usize = 20_000;

    fn max_remote_embedding_chars(&self) -> usize {
        match &self.backend {
            EmbeddingBackend::OpenAI(_) => Self::MAX_OPENAI_EMBEDDING_TEXT_LEN,
            EmbeddingBackend::Nvidia(_) => Self::MAX_NVIDIA_EMBEDDING_TEXT_LEN,
            EmbeddingBackend::Gemini(_) => Self::MAX_GEMINI_EMBEDDING_TEXT_LEN,
            EmbeddingBackend::Mistral(_) => Self::MAX_MISTRAL_EMBEDDING_TEXT_LEN,
            EmbeddingBackend::OpenRouter(_) => Self::MAX_OPENROUTER_EMBEDDING_TEXT_LEN,
            #[cfg(feature = "local-embeddings")]
            EmbeddingBackend::FastEmbed(_) => usize::MAX,
        }
    }

    /// Truncate text for embedding to reduce the risk of provider token-limit errors.
    fn truncate_for_embedding<'a>(text: &'a str, max_chars: usize) -> std::borrow::Cow<'a, str> {
        if text.len() <= max_chars {
            std::borrow::Cow::Borrowed(text)
        } else {
            // Find the last valid UTF-8 char boundary within the limit
            let truncated = &text[..max_chars];
            let end = truncated
                .char_indices()
                .rev()
                .next()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(max_chars);
            tracing::info!(
                "Truncated embedding text from {} to {} chars",
                text.len(),
                end
            );
            std::borrow::Cow::Owned(text[..end].to_string())
        }
    }

    fn note_dimension(&self, observed: usize) -> Result<()> {
        if observed == 0 {
            return Err(anyhow!("embedding provider returned zero-length embedding"));
        }

        let current = self.dimension.load(Ordering::Relaxed);
        if current == 0 {
            self.dimension.store(observed, Ordering::Relaxed);
            return Ok(());
        }

        if current != observed {
            return Err(anyhow!(
                "embedding provider returned {observed}D vectors but runtime expects {current}D"
            ));
        }

        Ok(())
    }

    fn truncate_if_remote<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match &self.backend {
            EmbeddingBackend::OpenAI(_)
            | EmbeddingBackend::Nvidia(_)
            | EmbeddingBackend::Gemini(_)
            | EmbeddingBackend::Mistral(_)
            | EmbeddingBackend::OpenRouter(_) => {
                Self::truncate_for_embedding(text, self.max_remote_embedding_chars())
            }
            #[cfg(feature = "local-embeddings")]
            EmbeddingBackend::FastEmbed(_) => std::borrow::Cow::Borrowed(text),
        }
    }

    pub fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        let text = self.truncate_if_remote(text);
        let embedding = match &self.backend {
            #[cfg(feature = "local-embeddings")]
            EmbeddingBackend::FastEmbed(model) => {
                let mut guard = model
                    .lock()
                    .map_err(|_| anyhow!("fastembed runtime poisoned"))?;
                let outputs = guard
                    .embed(vec![text.into_owned()], None)
                    .map_err(|err| anyhow!("failed to compute embedding with fastembed: {err}"))?;
                outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("fastembed returned no embedding output"))?
            }
            EmbeddingBackend::OpenAI(provider) => {
                use memvid_core::EmbeddingProvider;
                provider
                    .embed_text(&text)
                    .map_err(|err| anyhow!("failed to compute embedding with OpenAI: {err}"))?
            }
            EmbeddingBackend::Nvidia(provider) => provider
                .embed_passage(&text)
                .map_err(|err| anyhow!("failed to compute embedding with NVIDIA: {err}"))?,
            EmbeddingBackend::Gemini(provider) => provider
                .embed_text(&text)
                .map_err(|err| anyhow!("failed to compute embedding with Gemini: {err}"))?,
            EmbeddingBackend::Mistral(provider) => provider
                .embed_text(&text)
                .map_err(|err| anyhow!("failed to compute embedding with Mistral: {err}"))?,
            EmbeddingBackend::OpenRouter(provider) => provider
                .embed_text(&text)
                .map_err(|err| anyhow!("failed to compute embedding with OpenRouter: {}", err))?,
        };

        self.note_dimension(embedding.len())?;
        Ok(embedding)
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let text = self.truncate_if_remote(text);
        match &self.backend {
            EmbeddingBackend::Nvidia(provider) => {
                let embedding = provider
                    .embed_query(&text)
                    .map_err(|err| anyhow!("failed to compute embedding with NVIDIA: {err}"))?;
                self.note_dimension(embedding.len())?;
                Ok(embedding)
            }
            _ => self.embed_passage(&text),
        }
    }

    pub fn embed_batch_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let truncated: Vec<std::borrow::Cow<'_, str>> =
            texts.iter().map(|t| self.truncate_if_remote(t)).collect();
        let truncated_refs: Vec<&str> = truncated.iter().map(|c| c.as_ref()).collect();

        let embeddings = match &self.backend {
            #[cfg(feature = "local-embeddings")]
            EmbeddingBackend::FastEmbed(model) => {
                let mut guard = model
                    .lock()
                    .map_err(|_| anyhow!("fastembed runtime poisoned"))?;
                guard
                    .embed(
                        truncated_refs
                            .iter()
                            .map(|s| (*s).to_string())
                            .collect::<Vec<String>>(),
                        None,
                    )
                    .map_err(|err| anyhow!("failed to compute embeddings with fastembed: {err}"))?
            }
            EmbeddingBackend::OpenAI(provider) => {
                use memvid_core::EmbeddingProvider;
                provider
                    .embed_batch(&truncated_refs)
                    .map_err(|err| anyhow!("failed to compute embeddings with OpenAI: {err}"))?
            }
            EmbeddingBackend::Nvidia(provider) => provider
                .embed_passages(&truncated_refs)
                .map_err(|err| anyhow!("failed to compute embeddings with NVIDIA: {err}"))?,
            EmbeddingBackend::Gemini(provider) => provider
                .embed_batch(&truncated_refs)
                .map_err(|err| anyhow!("failed to compute embeddings with Gemini: {err}"))?,
            EmbeddingBackend::Mistral(provider) => provider
                .embed_batch(&truncated_refs)
                .map_err(|err| anyhow!("failed to compute embeddings with Mistral: {}", err))?,
            EmbeddingBackend::OpenRouter(provider) => provider
                .embed_batch(&truncated_refs)
                .map_err(|err| anyhow!("failed to compute embeddings with OpenRouter: {}", err))?,
        };

        if let Some(first) = embeddings.first() {
            self.note_dimension(first.len())?;
        }
        if let Some(expected) = embeddings.first().map(|e| e.len()) {
            if embeddings.iter().any(|e| e.len() != expected) {
                return Err(anyhow!(
                    "embedding provider returned mixed vector dimensions"
                ));
            }
        }

        Ok(embeddings)
    }

    pub fn embed_batch_queries(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let truncated: Vec<std::borrow::Cow<'_, str>> =
            texts.iter().map(|t| self.truncate_if_remote(t)).collect();
        let truncated_refs: Vec<&str> = truncated.iter().map(|c| c.as_ref()).collect();

        match &self.backend {
            EmbeddingBackend::Nvidia(provider) => {
                let embeddings = provider
                    .embed_queries(&truncated_refs)
                    .map_err(|err| anyhow!("failed to compute embeddings with NVIDIA: {err}"))?;

                if let Some(first) = embeddings.first() {
                    self.note_dimension(first.len())?;
                }
                if let Some(expected) = embeddings.first().map(|e| e.len()) {
                    if embeddings.iter().any(|e| e.len() != expected) {
                        return Err(anyhow!(
                            "embedding provider returned mixed vector dimensions"
                        ));
                    }
                }

                Ok(embeddings)
            }
            _ => self.embed_batch_passages(&truncated_refs),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension.load(Ordering::Relaxed)
    }

    pub fn model_choice(&self) -> EmbeddingModelChoice {
        self.model
    }

    pub fn provider_kind(&self) -> &'static str {
        match &self.backend {
            #[cfg(feature = "local-embeddings")]
            EmbeddingBackend::FastEmbed(_) => "fastembed",
            EmbeddingBackend::OpenAI(_) => "openai",
            EmbeddingBackend::Nvidia(_) => "nvidia",
            EmbeddingBackend::Gemini(_) => "gemini",
            EmbeddingBackend::Mistral(_) => "mistral",
            EmbeddingBackend::OpenRouter(_) => "openrouter",
        }
    }

    pub fn provider_model_id(&self) -> String {
        match &self.backend {
            #[cfg(feature = "local-embeddings")]
            EmbeddingBackend::FastEmbed(_) => self.model.canonical_model_id().to_string(),
            EmbeddingBackend::OpenAI(provider) => {
                use memvid_core::EmbeddingProvider;
                provider.model().to_string()
            }
            EmbeddingBackend::Nvidia(provider) => provider.model().to_string(),
            EmbeddingBackend::Gemini(provider) => provider.model().to_string(),
            EmbeddingBackend::Mistral(provider) => provider.model().to_string(),
            EmbeddingBackend::OpenRouter(provider) => provider.model().to_string(),
        }
    }
}

impl memvid_core::VecEmbedder for EmbeddingRuntime {
    fn embed_query(&self, text: &str) -> memvid_core::Result<Vec<f32>> {
        EmbeddingRuntime::embed_query(self, text).map_err(|err| {
            memvid_core::MemvidError::EmbeddingFailed {
                reason: err.to_string().into_boxed_str(),
            }
        })
    }

    fn embedding_dimension(&self) -> usize {
        self.dimension()
    }
}

/// Ensure fastembed cache directory exists
#[cfg(feature = "local-embeddings")]
fn ensure_fastembed_cache(config: &CliConfig) -> Result<PathBuf> {
    use std::fs;

    let cache_dir = config.models_dir.clone();
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Get approximate model size in MB for user-friendly error messages
fn model_size_mb(model: EmbeddingModelChoice) -> usize {
    match model {
        EmbeddingModelChoice::BgeSmall => 33,
        EmbeddingModelChoice::BgeBase => 110,
        EmbeddingModelChoice::Nomic => 137,
        EmbeddingModelChoice::GteLarge => 327,
        // Remote/cloud models don't require local download
        EmbeddingModelChoice::OpenAILarge
        | EmbeddingModelChoice::OpenAISmall
        | EmbeddingModelChoice::OpenAIAda
        | EmbeddingModelChoice::Nvidia
        | EmbeddingModelChoice::Gemini
        | EmbeddingModelChoice::Mistral
        | EmbeddingModelChoice::OpenRouter => 0,
    }
}

/// Instantiate an embedding runtime with the configured model
fn instantiate_embedding_runtime(config: &CliConfig) -> Result<EmbeddingRuntime> {
    use tracing::info;

    let embedding_model = config.embedding_model;

    if embedding_model.dimensions() > 0 {
        info!(
            "Loading embedding model: {} ({}D)",
            embedding_model.name(),
            embedding_model.dimensions()
        );
    } else {
        info!("Loading embedding model: {}", embedding_model.name());
    }

    if config.offline && embedding_model.is_remote() {
        anyhow::bail!(
            "remote embeddings are unavailable while offline; set MEMVID_OFFLINE=0 or use a local embedding model"
        );
    }

    // Check if OpenAI model
    if embedding_model.is_openai() {
        return instantiate_openai_runtime(embedding_model);
    }

    if embedding_model == EmbeddingModelChoice::Nvidia {
        return instantiate_nvidia_runtime(None);
    }

    if embedding_model == EmbeddingModelChoice::Gemini {
        return instantiate_gemini_runtime();
    }

    if embedding_model == EmbeddingModelChoice::Mistral {
        return instantiate_mistral_runtime();
    }

    if embedding_model == EmbeddingModelChoice::OpenRouter {
        return instantiate_openrouter_runtime();
    }

    // Local fastembed model
    #[cfg(feature = "local-embeddings")]
    {
        return instantiate_fastembed_runtime(config, embedding_model);
    }

    #[cfg(not(feature = "local-embeddings"))]
    {
        anyhow::bail!(
            "Local embeddings are not available on this platform. \
            Please use a remote embedding provider:\n\
            - Set OPENAI_API_KEY and use --embedding-model openai-large\n\
            - Set GEMINI_API_KEY and use --embedding-model gemini\n\
            - Set MISTRAL_API_KEY and use --embedding-model mistral\n\
            - Set NVIDIA_API_KEY and use --embedding-model nvidia\n\
            - Set OPENROUTER_API_KEY and use --embedding-model openrouter"
        );
    }
}

/// Instantiate OpenAI embedding runtime
fn instantiate_openai_runtime(embedding_model: EmbeddingModelChoice) -> Result<EmbeddingRuntime> {
    use anyhow::bail;
    use memvid_core::EmbeddingConfig;
    use tracing::info;

    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
        anyhow!("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
    })?;

    if api_key.is_empty() {
        bail!("OPENAI_API_KEY cannot be empty");
    }

    let config = match embedding_model {
        EmbeddingModelChoice::OpenAILarge => EmbeddingConfig::openai_large(),
        EmbeddingModelChoice::OpenAISmall => EmbeddingConfig::openai_small(),
        EmbeddingModelChoice::OpenAIAda => EmbeddingConfig::openai_ada(),
        _ => unreachable!("is_openai() should have been false"),
    };

    let provider = OpenAIEmbeddingProvider::new(api_key, config.clone())
        .map_err(|err| anyhow!("failed to create OpenAI embedding provider: {err}"))?;

    info!(
        "OpenAI embedding provider ready: model={}, dimension={}",
        config.model, config.dimension
    );

    Ok(EmbeddingRuntime::new_openai(
        provider,
        embedding_model,
        config.dimension,
    ))
}

fn normalize_nvidia_embedding_model_override(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lowered = trimmed.to_ascii_lowercase();
    if lowered == "nvidia" || lowered == "nv" {
        return None;
    }

    let without_prefix = trimmed
        .strip_prefix("nvidia:")
        .or_else(|| trimmed.strip_prefix("nv:"))
        .unwrap_or(trimmed)
        .trim();

    if without_prefix.is_empty() {
        return None;
    }

    if without_prefix.eq_ignore_ascii_case("nv-embed-v1") {
        return Some("nvidia/nv-embed-v1".to_string());
    }

    if without_prefix.contains('/') {
        return Some(without_prefix.to_string());
    }

    Some(format!("nvidia/{without_prefix}"))
}

/// Instantiate NVIDIA embedding runtime
fn instantiate_nvidia_runtime(model_override: Option<&str>) -> Result<EmbeddingRuntime> {
    use tracing::info;

    let normalized = model_override.and_then(normalize_nvidia_embedding_model_override);
    let provider = NvidiaEmbeddingProvider::from_env(normalized.as_deref())
        .map_err(|err| anyhow!("failed to create NVIDIA embedding provider: {err}"))?;

    info!(
        "NVIDIA embedding provider ready: model={}",
        provider.model()
    );

    Ok(EmbeddingRuntime::new_nvidia(
        provider,
        EmbeddingModelChoice::Nvidia,
    ))
}

/// Instantiate Gemini embedding runtime
fn instantiate_gemini_runtime() -> Result<EmbeddingRuntime> {
    use tracing::info;

    let provider = GeminiEmbeddingProvider::from_env()
        .map_err(|err| anyhow!("failed to create Gemini embedding provider: {err}"))?;

    let dimension = provider.dimension();
    info!(
        "Gemini embedding provider ready: model={}, dimension={}",
        provider.model(),
        dimension
    );

    Ok(EmbeddingRuntime::new_gemini(
        provider,
        EmbeddingModelChoice::Gemini,
        dimension,
    ))
}

/// Instantiate Mistral embedding runtime
fn instantiate_mistral_runtime() -> Result<EmbeddingRuntime> {
    use tracing::info;

    let provider = MistralEmbeddingProvider::from_env()
        .map_err(|err| anyhow!("failed to create Mistral embedding provider: {err}"))?;

    let dimension = provider.dimension();
    info!(
        "Mistral embedding provider ready: model={}, dimension={}",
        provider.model(),
        dimension
    );

    Ok(EmbeddingRuntime::new_mistral(
        provider,
        EmbeddingModelChoice::Mistral,
        dimension,
    ))
}

/// Instantiate OpenRouter embedding runtime
fn instantiate_openrouter_runtime() -> Result<EmbeddingRuntime> {
    use memvid_core::EmbeddingProvider;
    use tracing::info;

    let provider = OpenRouterEmbeddingProvider::from_env()
        .map_err(|err| anyhow!("failed to create OpenRouter embedding provider: {}", err))?;

    let dimension = provider.dimension();
    info!(
        "OpenRouter embedding provider ready: model={}, dimension={}",
        provider.model(),
        dimension
    );

    Ok(EmbeddingRuntime::new_openrouter(
        provider,
        EmbeddingModelChoice::OpenRouter,
        dimension,
    ))
}

/// Instantiate fastembed (local) embedding runtime
#[cfg(feature = "local-embeddings")]
fn instantiate_fastembed_runtime(
    config: &CliConfig,
    embedding_model: EmbeddingModelChoice,
) -> Result<EmbeddingRuntime> {
    use anyhow::bail;
    use fastembed::{InitOptions, TextEmbedding};
    use std::fs;

    let cache_dir = ensure_fastembed_cache(config)?;

    if config.offline {
        let mut entries = fs::read_dir(&cache_dir)?;
        if entries.next().is_none() {
            bail!(
                "semantic embeddings unavailable while offline; allow one connected run so fastembed can cache model weights"
            );
        }
    }

    let options = InitOptions::new(embedding_model.to_fastembed_model())
        .with_cache_dir(cache_dir)
        .with_show_download_progress(true);
    let mut model = TextEmbedding::try_new(options).map_err(|err| {
        // Provide platform-specific guidance for model download issues
        let platform_hint = if cfg!(target_os = "windows") {
            "\n\nWindows users: If model downloads fail, try:\n\
            1. Run as Administrator\n\
            2. Check your antivirus isn't blocking downloads\n\
            3. Use OpenAI embeddings instead: set OPENAI_API_KEY and use --embedding-model openai"
        } else if cfg!(target_os = "linux") {
            "\n\nLinux users: If model downloads fail, try:\n\
            1. Check disk space in ~/.memvid/models\n\
            2. Ensure you have network access to huggingface.co\n\
            3. Use OpenAI embeddings instead: export OPENAI_API_KEY=... and use --embedding-model openai"
        } else {
            "\n\nIf model downloads fail, try using OpenAI embeddings:\n\
            export OPENAI_API_KEY=your-key && memvid ... --embedding-model openai"
        };

        anyhow!(
            "Failed to initialize embedding model '{}': {err}\n\n\
            This typically means the model couldn't be downloaded or loaded.\n\
            Model size: ~{} MB{}\n\n\
            See: https://docs.memvid.com/embedding-models",
            embedding_model.name(),
            model_size_mb(embedding_model),
            platform_hint
        )
    })?;

    let probe = model
        .embed(vec!["memvid probe".to_string()], None)
        .map_err(|err| anyhow!("failed to determine embedding dimension: {err}"))?;
    let dimension = probe.first().map(|vec| vec.len()).unwrap_or(0);

    if dimension == 0 {
        bail!("fastembed reported zero-length embeddings");
    }

    // Verify dimension matches expected
    if dimension != embedding_model.dimensions() {
        tracing::warn!(
            "Embedding dimension mismatch: expected {}, got {}",
            embedding_model.dimensions(),
            dimension
        );
    }

    Ok(EmbeddingRuntime::new_fastembed(
        model,
        embedding_model,
        dimension,
    ))
}

/// Load embedding runtime (fails if unavailable)
pub fn load_embedding_runtime(config: &CliConfig) -> Result<EmbeddingRuntime> {
    use anyhow::bail;

    match instantiate_embedding_runtime(config) {
        Ok(runtime) => Ok(runtime),
        Err(err) => {
            if config.offline {
                bail!(
                    "semantic embeddings unavailable while offline; allow one connected run so fastembed can cache model weights ({err})"
                );
            }
            Err(err)
        }
    }
}

/// Try to load embedding runtime (returns None if unavailable)
pub fn try_load_embedding_runtime(config: &CliConfig) -> Option<EmbeddingRuntime> {
    use tracing::warn;

    match instantiate_embedding_runtime(config) {
        Ok(runtime) => Some(runtime),
        Err(err) => {
            warn!("semantic embeddings unavailable: {err}");
            None
        }
    }
}

/// Load embedding runtime with an optional model override.
/// If `model_override` is provided, it will be used instead of the config's embedding_model.
pub fn load_embedding_runtime_with_model(
    config: &CliConfig,
    model_override: Option<&str>,
) -> Result<EmbeddingRuntime> {
    use tracing::info;

    let mut raw_override: Option<&str> = None;
    let embedding_model = match model_override {
        Some(model_str) => {
            raw_override = Some(model_str);
            let parsed = model_str.parse::<EmbeddingModelChoice>()?;
            if parsed.dimensions() > 0 {
                info!(
                    "Using embedding model override: {} ({}D)",
                    parsed.name(),
                    parsed.dimensions()
                );
            } else {
                info!("Using embedding model override: {}", parsed.name());
            }
            parsed
        }
        None => config.embedding_model,
    };

    if embedding_model.dimensions() > 0 {
        info!(
            "Loading embedding model: {} ({}D)",
            embedding_model.name(),
            embedding_model.dimensions()
        );
    } else {
        info!("Loading embedding model: {}", embedding_model.name());
    }

    if config.offline && embedding_model.is_remote() {
        anyhow::bail!(
            "remote embeddings are unavailable while offline; set MEMVID_OFFLINE=0 or use a local embedding model"
        );
    }

    if embedding_model.is_openai() {
        return instantiate_openai_runtime(embedding_model);
    }

    if embedding_model == EmbeddingModelChoice::Nvidia {
        return instantiate_nvidia_runtime(raw_override);
    }

    if embedding_model == EmbeddingModelChoice::Gemini {
        return instantiate_gemini_runtime();
    }

    if embedding_model == EmbeddingModelChoice::Mistral {
        return instantiate_mistral_runtime();
    }

    #[cfg(feature = "local-embeddings")]
    {
        return instantiate_fastembed_runtime(config, embedding_model);
    }

    #[cfg(not(feature = "local-embeddings"))]
    {
        anyhow::bail!(
            "Local embeddings are not available on this platform. \
            Please use a remote embedding provider."
        );
    }
}

/// Try to load embedding runtime with model override (returns None if unavailable)
pub fn try_load_embedding_runtime_with_model(
    config: &CliConfig,
    model_override: Option<&str>,
) -> Option<EmbeddingRuntime> {
    use tracing::warn;

    match load_embedding_runtime_with_model(config, model_override) {
        Ok(runtime) => Some(runtime),
        Err(err) => {
            warn!("semantic embeddings unavailable: {err}");
            None
        }
    }
}

/// Load embedding runtime by auto-detecting from MV2 vector dimension.
///
/// Priority:
/// 1. Explicit model override (--query-embedding-model flag)
/// 2. Auto-detect from MV2 file's stored dimension
/// 3. Fall back to config default
///
/// This allows users to omit --query-embedding-model when querying files
/// created with non-default embedding models (like OpenAI).
pub fn load_embedding_runtime_for_mv2(
    config: &CliConfig,
    model_override: Option<&str>,
    mv2_dimension: Option<u32>,
) -> Result<EmbeddingRuntime> {
    use tracing::info;

    // Priority 1: Explicit override
    if let Some(model_str) = model_override {
        return load_embedding_runtime_with_model(config, Some(model_str));
    }

    // Priority 2: Auto-detect from MV2 dimension
    if let Some(dim) = mv2_dimension {
        if let Some(detected_model) = EmbeddingModelChoice::from_dimension(dim) {
            info!(
                "Auto-detected embedding model from MV2: {} ({}D)",
                detected_model.name(),
                dim
            );

            // For OpenAI models, check if API key is available
            if detected_model.is_openai() {
                if std::env::var("OPENAI_API_KEY").is_ok() {
                    return load_embedding_runtime_with_model(config, Some(detected_model.name()));
                } else {
                    // OpenAI detected but no API key - provide helpful error
                    return Err(anyhow!(
                        "MV2 file uses OpenAI embeddings ({}D) but OPENAI_API_KEY is not set.\n\n\
                        Options:\n\
                        1. Set OPENAI_API_KEY environment variable\n\
                        2. Use --query-embedding-model to specify a different model\n\
                        3. Use lexical-only search with --mode lex\n\n\
                        See: https://docs.memvid.com/embedding-models",
                        dim
                    ));
                }
            }

            return load_embedding_runtime_with_model(config, Some(detected_model.name()));
        }
    }

    // Priority 3: Fall back to config default
    load_embedding_runtime(config)
}

/// Try to load embedding runtime for MV2 with auto-detection (returns None if unavailable)
pub fn try_load_embedding_runtime_for_mv2(
    config: &CliConfig,
    model_override: Option<&str>,
    mv2_dimension: Option<u32>,
) -> Option<EmbeddingRuntime> {
    use tracing::warn;

    match load_embedding_runtime_for_mv2(config, model_override, mv2_dimension) {
        Ok(runtime) => Some(runtime),
        Err(err) => {
            warn!("semantic embeddings unavailable: {err}");
            None
        }
    }
}
