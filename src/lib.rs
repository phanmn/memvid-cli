#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
//! Memvid CLI library

pub mod analytics;
pub mod api;
pub mod api_fetch;
pub mod commands;
pub mod config;
pub mod contextual;
pub mod enrich;
pub mod error;
pub mod gemini_embeddings;
mod http;
pub mod mistral_embeddings;
pub mod nvidia_embeddings;
pub mod openai_embeddings;
pub mod openai_reranker;
pub mod org_ticket_cache;
pub mod ticket_cache;
pub mod utils;

// Note: WhisperTranscriber is now in memvid-core for SDK binding support
