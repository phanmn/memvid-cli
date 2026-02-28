//! Contextual retrieval module for improving chunk embeddings.
//!
//! Based on Anthropic's contextual retrieval technique: before embedding each chunk,
//! we prepend a context summary that places the chunk within the larger document.
//! This helps semantic search find chunks that would otherwise miss due to lack of context.
//!
//! For example, a chunk saying "I've been using basil and mint in my cooking lately"
//! might get a context prefix like:
//! "This is a conversation where the user discusses their cooking preferences
//! and mentions growing herbs in their garden."
//!
//! This allows semantic queries like "dinner with homegrown ingredients" to find
//! the chunk even though it doesn't explicitly mention "dinner" or "homegrown".

use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
#[cfg(feature = "llama-cpp")]
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, info, warn};

/// The contextual prompt for generating chunk context.
const CONTEXTUAL_PROMPT: &str = r#"You are a document analysis assistant. Given a document and a chunk from that document, provide a brief context that situates the chunk within the document.

<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Provide a short context (2-3 sentences max) that:
1. Summarizes the document's topic and purpose
2. Notes any user preferences, personal information, or key facts mentioned in the document
3. Explains what this specific chunk is about within that context

Focus especially on first-person statements, preferences, and personal context that might be important for later retrieval.

Respond with ONLY the context, no preamble or explanation."#;

/// OpenAI API request message
#[derive(Debug, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

/// OpenAI API request
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
}

/// OpenAI API response
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

/// Contextual retrieval engine that can use either OpenAI or local models.
pub enum ContextualEngine {
    /// OpenAI API-based context generation
    OpenAI { api_key: String, model: String },
    /// Local LLM-based context generation (llama.cpp)
    #[cfg(feature = "llama-cpp")]
    Local { model_path: PathBuf },
}

impl ContextualEngine {
    /// Create a new OpenAI-based contextual engine.
    pub fn openai() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow!("OPENAI_API_KEY environment variable not set"))?;
        Ok(Self::OpenAI {
            api_key,
            model: "gpt-4o-mini".to_string(),
        })
    }

    /// Create a new OpenAI-based contextual engine with a specific model.
    pub fn openai_with_model(model: &str) -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow!("OPENAI_API_KEY environment variable not set"))?;
        Ok(Self::OpenAI {
            api_key,
            model: model.to_string(),
        })
    }

    /// Create a new local LLM-based contextual engine.
    #[cfg(feature = "llama-cpp")]
    pub fn local(model_path: PathBuf) -> Self {
        Self::Local { model_path }
    }

    /// Generate context for a chunk within a document.
    /// Returns the context string to prepend to the chunk before embedding.
    pub fn generate_context(&self, document: &str, chunk: &str) -> Result<String> {
        match self {
            Self::OpenAI { api_key, model } => {
                let client = crate::http::blocking_client(Duration::from_secs(60))?;
                Self::generate_context_openai(&client, api_key, model, document, chunk)
            }
            #[cfg(feature = "llama-cpp")]
            Self::Local { model_path } => Self::generate_context_local(model_path, document, chunk),
        }
    }

    /// Generate contextual prefixes for multiple chunks in parallel (OpenAI only).
    /// Returns a vector of context strings in the same order as the input chunks.
    pub fn generate_contexts_batch(
        &self,
        document: &str,
        chunks: &[String],
    ) -> Result<Vec<String>> {
        match self {
            Self::OpenAI { api_key, model } => {
                Self::generate_contexts_batch_openai(api_key, model, document, chunks)
            }
            #[cfg(feature = "llama-cpp")]
            Self::Local { model_path } => {
                // Local models don't support batching efficiently, fall back to sequential
                let mut contexts = Vec::with_capacity(chunks.len());
                for chunk in chunks {
                    let ctx = Self::generate_context_local(model_path, document, chunk)?;
                    contexts.push(ctx);
                }
                Ok(contexts)
            }
        }
    }

    /// Generate context using OpenAI API.
    fn generate_context_openai(
        client: &Client,
        api_key: &str,
        model: &str,
        document: &str,
        chunk: &str,
    ) -> Result<String> {
        // Truncate document if too long (keep first ~6000 chars to fit in context)
        let truncated_doc = if document.len() > 6000 {
            format!("{}...[truncated]", &document[..6000])
        } else {
            document.to_string()
        };

        let prompt = CONTEXTUAL_PROMPT
            .replace("{document}", &truncated_doc)
            .replace("{chunk}", chunk);

        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            max_tokens: 200,
            temperature: 0.0,
        };

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| anyhow!("OpenAI API request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow!("OpenAI API error {}: {}", status, body));
        }

        let chat_response: ChatResponse = response
            .json()
            .map_err(|e| anyhow!("Failed to parse OpenAI response: {}", e))?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow!("No response from OpenAI"))
    }

    /// Generate contexts for chunks sequentially using OpenAI.
    /// Uses sequential processing to avoid issues with blocking HTTP in rayon threads.
    fn generate_contexts_batch_openai(
        api_key: &str,
        model: &str,
        document: &str,
        chunks: &[String],
    ) -> Result<Vec<String>> {
        let client = crate::http::blocking_client(Duration::from_secs(60))?;

        eprintln!(
            "  Generating contextual prefixes for {} chunks...",
            chunks.len()
        );
        info!(
            "Generating contextual prefixes for {} chunks sequentially",
            chunks.len()
        );

        let mut contexts = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 && i % 5 == 0 {
                eprintln!("    Context progress: {}/{}", i, chunks.len());
            }

            match Self::generate_context_openai(&client, api_key, model, document, chunk) {
                Ok(ctx) => {
                    debug!(
                        "Generated context for chunk {}: {}...",
                        i,
                        &ctx[..ctx.len().min(50)]
                    );
                    contexts.push(ctx);
                }
                Err(e) => {
                    warn!("Failed to generate context for chunk {}: {}", i, e);
                    contexts.push(String::new()); // Empty context on failure
                }
            }
        }

        eprintln!(
            "  Contextual prefix generation complete ({} contexts)",
            contexts.len()
        );
        info!("Contextual prefix generation complete");
        Ok(contexts)
    }

    /// Generate context using local LLM.
    #[cfg(feature = "llama-cpp")]
    fn generate_context_local(model_path: &PathBuf, document: &str, chunk: &str) -> Result<String> {
        use llama_cpp::standard_sampler::StandardSampler;
        use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
        use tokio::runtime::Runtime;

        if !model_path.exists() {
            return Err(anyhow!(
                "Model file not found: {}. Run 'memvid models install phi-3.5-mini' first.",
                model_path.display()
            ));
        }

        // Load model
        debug!("Loading local model from {}", model_path.display());
        let model = LlamaModel::load_from_file(model_path, LlamaParams::default())
            .map_err(|e| anyhow!("Failed to load model: {}", e))?;

        // Truncate document if too long
        let truncated_doc = if document.len() > 4000 {
            format!("{}...[truncated]", &document[..4000])
        } else {
            document.to_string()
        };

        // Build prompt in Phi-3.5 format
        let prompt = format!(
            r#"<|system|>
You are a document analysis assistant. Given a document and a chunk, provide brief context.
<|end|>
<|user|>
Document:
{truncated_doc}

Chunk:
{chunk}

Provide a short context (2-3 sentences) that summarizes what this document is about and what user preferences or key facts are mentioned. Focus on first-person statements.
<|end|>
<|assistant|>
"#
        );

        // Create session
        let mut session_params = SessionParams::default();
        session_params.n_ctx = 4096;
        session_params.n_batch = 512;
        if session_params.n_ubatch == 0 {
            session_params.n_ubatch = 512;
        }

        let mut session = model
            .create_session(session_params)
            .map_err(|e| anyhow!("Failed to create session: {}", e))?;

        // Tokenize and prime context
        let tokens = model
            .tokenize_bytes(prompt.as_bytes(), true, true)
            .map_err(|e| anyhow!("Failed to tokenize: {}", e))?;

        session
            .advance_context_with_tokens(&tokens)
            .map_err(|e| anyhow!("Failed to prime context: {}", e))?;

        // Generate
        let handle = session
            .start_completing_with(StandardSampler::default(), 200)
            .map_err(|e| anyhow!("Failed to start completion: {}", e))?;

        let runtime = Runtime::new().map_err(|e| anyhow!("Failed to create runtime: {}", e))?;
        let generated = runtime.block_on(async { handle.into_string_async().await });

        Ok(generated.trim().to_string())
    }
}

/// Apply contextual prefixes to chunks for embedding.
/// Returns new chunk texts with context prepended.
pub fn apply_contextual_prefixes(
    _document: &str,
    chunks: &[String],
    contexts: &[String],
) -> Vec<String> {
    chunks
        .iter()
        .zip(contexts.iter())
        .map(|(chunk, context)| {
            if context.is_empty() {
                chunk.clone()
            } else {
                format!("[Context: {}]\n\n{}", context, chunk)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_contextual_prefixes() {
        let document = "A conversation about cooking";
        let chunks = vec!["I like basil".to_string(), "I grow tomatoes".to_string()];
        let contexts = vec![
            "User discusses their herb preferences".to_string(),
            "User mentions their garden".to_string(),
        ];

        let result = apply_contextual_prefixes(document, &chunks, &contexts);

        assert_eq!(result.len(), 2);
        assert!(result[0].contains("[Context:"));
        assert!(result[0].contains("I like basil"));
        assert!(result[1].contains("User mentions their garden"));
    }

    #[test]
    fn test_apply_contextual_prefixes_empty_context() {
        let document = "A document";
        let chunks = vec!["Some text".to_string()];
        let contexts = vec![String::new()];

        let result = apply_contextual_prefixes(document, &chunks, &contexts);

        assert_eq!(result[0], "Some text");
    }
}
