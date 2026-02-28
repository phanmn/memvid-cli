//! Enrichment engines for the CLI.
//!
//! This module provides CLI-specific enrichment engines, including the
//! LLM-based engine that uses local Phi models and cloud-based engines
//! for OpenAI, Claude, Gemini, xAI, Groq, and Mistral.

pub mod claude;
pub mod gemini;
pub mod groq;
#[cfg(feature = "llama-cpp")]
pub mod llm;
pub mod mistral;
pub mod openai;
pub mod xai;

#[cfg(feature = "candle-llm")]
pub mod candle_phi;

pub use claude::ClaudeEngine;
pub use gemini::GeminiEngine;
pub use groq::GroqEngine;
#[cfg(feature = "llama-cpp")]
pub use llm::LlmEngine;
pub use mistral::MistralEngine;
pub use openai::OpenAiEngine;
pub use xai::XaiEngine;

#[cfg(feature = "candle-llm")]
pub use candle_phi::CandlePhiEngine;
