use std::env;
use std::time::Duration;

use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use reqwest::{NoProxy, Proxy};

fn env_first(names: &[&str]) -> Option<String> {
    for name in names {
        if let Ok(value) = env::var(name) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn env_no_proxy() -> Option<NoProxy> {
    env_first(&["NO_PROXY", "no_proxy"]).and_then(|value| NoProxy::from_string(&value))
}

/// Build a blocking reqwest `Client` without using OS "system proxy" discovery.
///
/// On macOS, reqwest's system proxy support relies on `system-configuration`, which can
/// panic in restricted/sandboxed environments. To keep `memvid` robust, we disable system
/// proxy auto-detection and re-apply proxy settings from standard environment variables:
/// `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, and `NO_PROXY`.
pub(crate) fn blocking_client(timeout: Duration) -> Result<Client> {
    let mut builder = Client::builder().timeout(timeout).no_proxy();

    let no_proxy = env_no_proxy();

    // Order matters: scheme-specific proxies should be checked before ALL_PROXY.
    if let Some(https_proxy) = env_first(&["HTTPS_PROXY", "https_proxy"]) {
        let mut proxy =
            Proxy::https(https_proxy).map_err(|err| anyhow!("invalid HTTPS proxy URL: {err}"))?;
        proxy = proxy.no_proxy(no_proxy.clone());
        builder = builder.proxy(proxy);
    }

    if let Some(http_proxy) = env_first(&["HTTP_PROXY", "http_proxy"]) {
        let mut proxy =
            Proxy::http(http_proxy).map_err(|err| anyhow!("invalid HTTP proxy URL: {err}"))?;
        proxy = proxy.no_proxy(no_proxy.clone());
        builder = builder.proxy(proxy);
    }

    if let Some(all_proxy) = env_first(&["ALL_PROXY", "all_proxy"]) {
        let mut proxy =
            Proxy::all(all_proxy).map_err(|err| anyhow!("invalid ALL proxy URL: {err}"))?;
        proxy = proxy.no_proxy(no_proxy.clone());
        builder = builder.proxy(proxy);
    }

    builder
        .build()
        .map_err(|err| anyhow!("failed to construct HTTP client: {err}"))
}
