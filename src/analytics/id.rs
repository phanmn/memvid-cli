//! Anonymous ID generation for analytics
//!
//! Creates SHA256-based anonymous identifiers that cannot be reversed
//! but are consistent for the same user/file combination.

use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Cached machine ID
static MACHINE_ID: OnceLock<String> = OnceLock::new();

/// Get or generate a stable machine identifier
/// Uses hostname + username hash for privacy
fn get_machine_id() -> &'static str {
    MACHINE_ID.get_or_init(|| {
        let mut hasher = Sha256::new();

        // Add hostname
        if let Ok(hostname) = hostname::get() {
            hasher.update(hostname.to_string_lossy().as_bytes());
        }

        // Add username
        hasher.update(whoami::username().as_bytes());

        // Add home directory for uniqueness
        if let Some(home) = dirs::home_dir() {
            hasher.update(home.to_string_lossy().as_bytes());
        }

        // Add a static salt
        hasher.update(b"memvid_telemetry_v1");

        let result = hasher.finalize();
        format!("{:x}", result)[..16].to_string()
    })
}

/// Generate an anonymous ID for a user
/// Format: `anon_` + SHA256(machine_id + optional_file)[:16]
///
/// For paid users with API key, use the API key prefix instead
pub fn generate_anon_id(file_path: Option<&str>) -> String {
    // Check for API key (paid user)
    if let Ok(api_key) = std::env::var("MEMVID_API_KEY") {
        if api_key.len() >= 8 {
            let prefix = &api_key[..8];
            let mut hasher = Sha256::new();
            hasher.update(prefix.as_bytes());
            hasher.update(b"memvid_paid_v1");
            let result = hasher.finalize();
            return format!("paid_{:x}", result)[..21].to_string(); // paid_ + 16 chars
        }
    }

    // Free user - use machine ID
    let machine_id = get_machine_id();
    let mut hasher = Sha256::new();
    hasher.update(machine_id.as_bytes());

    // Add file path if provided for more granularity
    if let Some(path) = file_path {
        hasher.update(path.as_bytes());
    }

    hasher.update(b"memvid_anon_v1");
    let result = hasher.finalize();
    format!("anon_{:x}", result)[..21].to_string() // anon_ + 16 chars
}

/// Generate a hash for a file path
/// Used to track file activity without revealing the actual path
pub fn generate_file_hash(file_path: &str) -> String {
    let mut hasher = Sha256::new();

    // Normalize the path
    let normalized = std::path::Path::new(file_path)
        .canonicalize()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| file_path.to_string());

    hasher.update(normalized.as_bytes());
    hasher.update(b"memvid_file_v1");
    let result = hasher.finalize();
    format!("{:x}", result)[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anon_id_format() {
        let id = generate_anon_id(None);
        assert!(id.starts_with("anon_") || id.starts_with("paid_"));
        assert_eq!(id.len(), 21);
    }

    #[test]
    fn test_anon_id_consistency() {
        let id1 = generate_anon_id(Some("/test/file.mv2"));
        let id2 = generate_anon_id(Some("/test/file.mv2"));
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_file_hash_format() {
        let hash = generate_file_hash("/test/file.mv2");
        assert_eq!(hash.len(), 16);
    }
}
