//! CLI-specific error types and rendering helpers.

use std::fmt;

#[cfg(feature = "encryption")]
use memvid_core::encryption::EncryptionError;
use memvid_core::MemvidError;

use crate::utils::format_bytes;

/// Error indicating that the memory has reached its capacity limit
#[derive(Debug)]
pub struct CapacityExceededMessage {
    pub current: u64,
    pub limit: u64,
    pub required: u64,
}

impl fmt::Display for CapacityExceededMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Storage capacity exceeded\n\n\
             Current usage: {}\n\
             Capacity limit: {}\n\
             Required: {}\n\n\
             To continue, either:\n\
             1. Upgrade your plan: https://app.memvid.com/plan\n\
             2. Sync tickets: memvid tickets sync <file> --memory-id <UUID>",
            format_bytes(self.current),
            format_bytes(self.limit),
            format_bytes(self.required)
        )
    }
}

impl std::error::Error for CapacityExceededMessage {}

/// Error indicating that an API key is required for large files
#[derive(Debug)]
pub struct ApiKeyRequiredMessage {
    pub file_size: u64,
    pub limit: u64,
}

impl fmt::Display for ApiKeyRequiredMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "API key required for files larger than {}\n\n\
             File size: {}\n\
             Free tier limit: {}\n\n\
             To use this file:\n\
             1. Get your API key from https://app.memvid.com/api\n\
             2. Sync tickets: memvid tickets sync <file> --memory-id <UUID>",
            format_bytes(self.limit),
            format_bytes(self.file_size),
            format_bytes(self.limit)
        )
    }
}

impl std::error::Error for ApiKeyRequiredMessage {}

/// Error indicating that a memory is already bound to a different dashboard memory
#[derive(Debug)]
pub struct MemoryAlreadyBoundMessage {
    pub existing_memory_id: String,
    pub existing_memory_name: String,
    pub bound_at: String,
}

impl fmt::Display for MemoryAlreadyBoundMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "This file is already bound to memory '{}' ({})\n\
             Bound at: {}\n\n\
             Each memory can only be bound to one file.\n\
             To use more memories, upgrade your plan at https://memvid.com/dashboard/plan",
            self.existing_memory_name, self.existing_memory_id, self.bound_at
        )
    }
}

impl std::error::Error for MemoryAlreadyBoundMessage {}

/// Error indicating that a frame with the same URI already exists
#[derive(Debug)]
pub struct DuplicateUriError {
    uri: String,
}

impl DuplicateUriError {
    pub fn new<S: Into<String>>(uri: S) -> Self {
        Self { uri: uri.into() }
    }
}

impl fmt::Display for DuplicateUriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "frame with URI '{}' already exists. Use --update-existing to replace it or --allow-duplicate to keep both entries.",
            self.uri
        )
    }
}

impl std::error::Error for DuplicateUriError {}

/// Render a fallible CLI error into a stable exit code + human message.
pub fn render_error(err: &anyhow::Error) -> (i32, String) {
    // Prefer richer, user-friendly wrappers when present.
    if let Some(cap) = err.downcast_ref::<CapacityExceededMessage>() {
        return (2, cap.to_string());
    }

    #[cfg(feature = "encryption")]
    {
        let enc = err
            .chain()
            .find_map(|cause| cause.downcast_ref::<EncryptionError>());
        if let Some(enc_err) = enc {
            let message = match enc_err {
                EncryptionError::InvalidMagic { .. } => format!(
                    "{enc_err}\nHint: is this an encrypted .mv2e capsule and not a plain .mv2 file?"
                ),
                _ => enc_err.to_string(),
            };
            return (5, message);
        }
    }

    // Bubble up core errors even when wrapped in anyhow context.
    let core = err
        .chain()
        .find_map(|cause| cause.downcast_ref::<MemvidError>());
    if let Some(core_err) = core {
        match core_err {
            MemvidError::CapacityExceeded {
                current,
                limit,
                required,
            } => {
                let msg = CapacityExceededMessage {
                    current: *current,
                    limit: *limit,
                    required: *required,
                }
                .to_string();
                return (2, msg);
            }
            MemvidError::ApiKeyRequired { file_size, limit } => {
                let msg = ApiKeyRequiredMessage {
                    file_size: *file_size,
                    limit: *limit,
                }
                .to_string();
                return (2, msg);
            }
            MemvidError::Lock(reason) => {
                return (3, format!("File lock error: {reason}\nHint: check the active writer with `memvid who <file>` or request release with `memvid nudge <file>`"));
            }
            MemvidError::Locked(locked_err) => {
                return (3, format!("File lock error: {}\nHint: check the active writer with `memvid who <file>` or request release with `memvid nudge <file>`", locked_err.message));
            }
            MemvidError::InvalidHeader { reason } => {
                return (4, format!("{core_err}\nHint: run `memvid doctor <file>` to rebuild indexes and repair the footer.\nDetails: {reason}"));
            }
            MemvidError::EncryptedFile { .. } => {
                return (5, core_err.to_string());
            }
            MemvidError::InvalidToc { reason } => {
                return (4, format!("{core_err}\nHint: run `memvid doctor <file>` to rebuild indexes and repair the footer.\nDetails: {reason}"));
            }
            MemvidError::WalCorruption { reason, .. } => {
                return (4, format!("{core_err}\nHint: run `memvid doctor <file>` to rebuild indexes and repair the footer.\nDetails: {reason}"));
            }
            MemvidError::ManifestWalCorrupted { reason, .. } => {
                return (4, format!("{core_err}\nHint: run `memvid doctor <file>` to rebuild indexes and repair the footer.\nDetails: {reason}"));
            }
            MemvidError::TicketRequired { tier } => {
                return (2, format!("ticket required for tier {tier:?}. Apply a ticket before mutating this memory."));
            }
            _ => {
                return (1, core_err.to_string());
            }
        }
    }

    // Fallback: generic error text, non-zero exit.
    (1, err.to_string())
}
