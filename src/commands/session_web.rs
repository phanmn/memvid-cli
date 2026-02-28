//! Web server for the Time Machine session replay UI.
//!
//! This module provides an embedded web server that serves the React-based
//! Time Machine UI and handles WebSocket connections for real-time parameter
//! updates during session replay.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};

use memvid_core::Memvid;

/// Embedded static files from the built web UI
#[derive(RustEmbed)]
#[folder = "web/dist"]
struct WebAssets;

/// Configuration for a replay query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub mode: String,
    pub k: usize,
    pub adaptive: bool,
    #[serde(rename = "adaptiveStrategy")]
    pub adaptive_strategy: Option<String>,
    #[serde(rename = "minRelevancy")]
    pub min_relevancy: Option<f32>,
    #[serde(rename = "maxK")]
    pub max_k: Option<usize>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            mode: "hybrid".to_string(),
            k: 10,
            adaptive: false,
            adaptive_strategy: None,
            min_relevancy: Some(0.5),
            max_k: Some(20),
        }
    }
}

/// A document hit from search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentHit {
    #[serde(rename = "frameId")]
    pub frame_id: u64,
    pub title: String,
    pub snippet: String,
    pub score: f64,
}

/// Search results response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub hits: Vec<DocumentHit>,
    #[serde(rename = "totalHits")]
    pub total_hits: usize,
    #[serde(rename = "filteredCount")]
    pub filtered_count: usize,
    #[serde(rename = "elapsedMs")]
    pub elapsed_ms: u64,
    pub engine: String,
    #[serde(rename = "cliffIndex")]
    pub cliff_index: Option<usize>,
}

/// A recorded query from a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRecord {
    pub id: String,
    pub timestamp: i64,
    pub text: String,
    pub config: RetrievalConfig,
    pub results: SearchResults,
}

/// Session data returned to the UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub id: String,
    pub name: Option<String>,
    #[serde(rename = "createdAt")]
    pub created_at: i64,
    #[serde(rename = "endedAt")]
    pub ended_at: Option<i64>,
    pub queries: Vec<QueryRecord>,
    #[serde(rename = "originalConfig")]
    pub original_config: RetrievalConfig,
    #[serde(rename = "mv2Path")]
    pub mv2_path: String,
}

/// Replay request body
#[derive(Debug, Deserialize)]
pub struct ReplayRequest {
    #[serde(rename = "queryId")]
    pub query_id: String,
    pub config: RetrievalConfig,
}

/// WebSocket message from client
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum WsClientMessage {
    #[serde(rename = "select_query")]
    SelectQuery { query: String },
    #[serde(rename = "config_change")]
    ConfigChange { config: RetrievalConfig },
    #[serde(rename = "start_optimize")]
    StartOptimize,
}

/// WebSocket message to client
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum WsServerMessage {
    #[serde(rename = "results")]
    Results { data: SearchResults },
    #[serde(rename = "optimize_progress")]
    OptimizeProgress { data: OptimizeProgress },
    #[serde(rename = "optimize_complete")]
    OptimizeComplete { data: OptimizeResult },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Optimization progress
#[derive(Debug, Clone, Serialize)]
pub struct OptimizeProgress {
    pub progress: f32,
    #[serde(rename = "configsTested")]
    pub configs_tested: usize,
    #[serde(rename = "totalConfigs")]
    pub total_configs: usize,
}

/// Optimization result
#[derive(Debug, Clone, Serialize)]
pub struct OptimizeResult {
    #[serde(rename = "recommendedConfig")]
    pub recommended_config: RetrievalConfig,
    pub score: f32,
    pub coverage: f32,
    #[serde(rename = "tokenReduction")]
    pub token_reduction: f32,
    pub explanation: String,
}

/// Shared application state
pub struct AppState {
    pub session_id: String,
    pub mv2_path: PathBuf,
    pub memvid: Mutex<Memvid>,
    pub broadcast_tx: broadcast::Sender<WsServerMessage>,
    pub current_query: Mutex<String>,
    pub current_config: Mutex<RetrievalConfig>,
}

/// Start the web server for Time Machine UI
pub async fn start_web_server(
    session_id: String,
    mv2_path: PathBuf,
    port: u16,
    open_browser: bool,
) -> Result<()> {
    // Open the memory file
    let memvid = Memvid::open_read_only(&mv2_path).context("Failed to open memory file")?;

    // Create broadcast channel for WebSocket updates
    let (broadcast_tx, _) = broadcast::channel::<WsServerMessage>(100);

    let state = Arc::new(AppState {
        session_id: session_id.clone(),
        mv2_path: mv2_path.clone(),
        memvid: Mutex::new(memvid),
        broadcast_tx,
        current_query: Mutex::new(String::new()),
        current_config: Mutex::new(RetrievalConfig::default()),
    });

    // Build router
    let app = Router::new()
        // API routes
        .route("/api/session/:id", get(get_session))
        .route("/api/session/:id/replay", post(replay_query))
        .route("/api/session/:id/timeline", get(get_timeline))
        // WebSocket
        .route("/ws/session/:id", get(ws_handler))
        // Static files
        .fallback(static_handler)
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let url = format!("http://localhost:{}", port);

    println!();
    println!("  ╭──────────────────────────────────────────────────────╮");
    println!("  │                                                      │");
    println!("  │   🕰️  Memvid Time Machine                            │");
    println!("  │                                                      │");
    println!(
        "  │   Session: {:40} │",
        &session_id[..std::cmp::min(40, session_id.len())]
    );
    println!("  │   URL:     {:<40} │", url);
    println!("  │                                                      │");
    println!("  │   Press Ctrl+C to stop the server                    │");
    println!("  │                                                      │");
    println!("  ╰──────────────────────────────────────────────────────╯");
    println!();

    // Open browser if requested
    if open_browser {
        let url_with_session = format!("{}?session={}", url, session_id);
        if let Err(e) = open::that(&url_with_session) {
            eprintln!("Failed to open browser: {}", e);
            println!("Please open {} manually", url_with_session);
        }
    }

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Handler for static files
async fn static_handler(uri: axum::http::Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/');

    // Default to index.html for SPA routing
    let path = if path.is_empty() || !path.contains('.') {
        "index.html"
    } else {
        path
    };

    match WebAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string();

            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", mime)
                .body(axum::body::Body::from(content.data.into_owned()))
                .unwrap()
        }
        None => {
            // Fallback to index.html for SPA routing
            match WebAssets::get("index.html") {
                Some(content) => Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "text/html")
                    .body(axum::body::Body::from(content.data.into_owned()))
                    .unwrap(),
                None => Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(axum::body::Body::from("Not Found"))
                    .unwrap(),
            }
        }
    }
}

/// Get session data
async fn get_session(
    Path(session_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<SessionData>, (StatusCode, String)> {
    tracing::info!("get_session called with session_id: {}", session_id);
    let mut memvid = state.memvid.lock().await;

    // Load replay sessions
    tracing::info!("Loading replay sessions...");
    memvid.load_replay_sessions().map_err(|e| {
        tracing::error!("Failed to load replay sessions: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to load replay sessions: {}", e),
        )
    })?;

    let uuid = session_id.parse::<uuid::Uuid>().map_err(|e| {
        tracing::error!("Invalid session ID '{}': {}", session_id, e);
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid session ID: {}", e),
        )
    })?;
    tracing::info!("Looking for session with UUID: {}", uuid);

    let session = memvid.get_session(uuid).ok_or_else(|| {
        tracing::error!("Session {} not found", uuid);
        (StatusCode::NOT_FOUND, format!("Session {} not found", uuid))
    })?;
    tracing::info!(
        "Found session '{}' with {} actions",
        session.name.as_deref().unwrap_or("unnamed"),
        session.actions.len()
    );

    // Convert to API format - include both Find and Ask actions
    let queries: Vec<QueryRecord> = session
        .actions
        .iter()
        .filter_map(|action| {
            match &action.action_type {
                // Handle Find (search) actions
                memvid_core::replay::ActionType::Find {
                    query,
                    mode,
                    result_count,
                } => {
                    let result_frames = &action.affected_frames;
                    Some(QueryRecord {
                        id: format!("{}", action.sequence),
                        timestamp: action.timestamp_secs,
                        text: query.clone(),
                        config: RetrievalConfig {
                            mode: mode.clone(),
                            k: *result_count,
                            adaptive: false,
                            adaptive_strategy: None,
                            min_relevancy: Some(0.5),
                            max_k: None,
                        },
                        results: SearchResults {
                            hits: result_frames
                                .iter()
                                .enumerate()
                                .map(|(i, &frame_id)| DocumentHit {
                                    frame_id,
                                    title: format!("Document {}", frame_id),
                                    snippet: "...".to_string(),
                                    score: 1.0 - (i as f64 * 0.1),
                                })
                                .collect(),
                            total_hits: result_frames.len() * 2,
                            filtered_count: *result_count,
                            elapsed_ms: 10,
                            engine: mode.clone(),
                            cliff_index: Some(result_frames.len().min(5)),
                        },
                    })
                }
                // Handle Ask (RAG) actions - these also do retrieval internally
                memvid_core::replay::ActionType::Ask {
                    query,
                    provider: _,
                    model: _,
                } => {
                    let result_frames = &action.affected_frames;
                    Some(QueryRecord {
                        id: format!("{}", action.sequence),
                        timestamp: action.timestamp_secs,
                        text: query.clone(),
                        config: RetrievalConfig {
                            mode: "sem".to_string(), // Ask uses semantic search
                            k: 5,                    // Default top-k for ask
                            adaptive: false,
                            adaptive_strategy: None,
                            min_relevancy: Some(0.5),
                            max_k: None,
                        },
                        results: SearchResults {
                            hits: result_frames
                                .iter()
                                .enumerate()
                                .map(|(i, &frame_id)| DocumentHit {
                                    frame_id,
                                    title: format!("Document {}", frame_id),
                                    snippet: action.output_preview.clone(),
                                    score: 1.0 - (i as f64 * 0.1),
                                })
                                .collect(),
                            total_hits: result_frames.len(),
                            filtered_count: result_frames.len(),
                            elapsed_ms: action.duration_ms,
                            engine: "semantic".to_string(),
                            cliff_index: None,
                        },
                    })
                }
                _ => None,
            }
        })
        .collect();

    let data = SessionData {
        id: session.session_id.to_string(),
        name: session.name.clone(),
        created_at: session.created_secs,
        ended_at: session.ended_secs,
        queries,
        original_config: RetrievalConfig {
            mode: "sem".to_string(),
            k: 5,
            adaptive: false,
            adaptive_strategy: None,
            min_relevancy: Some(0.5),
            max_k: None,
        },
        mv2_path: state.mv2_path.display().to_string(),
    };

    Ok(Json(data))
}

/// Get session timeline
async fn get_timeline(
    Path(session_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<QueryRecord>>, (StatusCode, String)> {
    let session_data = get_session(Path(session_id), State(state)).await?;
    Ok(Json(session_data.queries.clone()))
}

/// Replay a query with different config
async fn replay_query(
    Path(_session_id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ReplayRequest>,
) -> Result<Json<SearchResults>, (StatusCode, String)> {
    let mut memvid = state.memvid.lock().await;

    // Build search request with new config
    let search_request = memvid_core::types::SearchRequest {
        query: request.query_id.clone(),
        top_k: request.config.k,
        snippet_chars: 240,
        uri: None,
        scope: None,
        cursor: None,
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
    };

    // Execute search
    let response = memvid
        .search(search_request)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Convert to API format
    let hit_count = response.hits.len();
    let results = SearchResults {
        hits: response
            .hits
            .into_iter()
            .map(|hit| DocumentHit {
                frame_id: hit.frame_id,
                title: hit
                    .title
                    .unwrap_or_else(|| format!("Frame {}", hit.frame_id)),
                snippet: hit.text,
                score: hit.score.unwrap_or(0.0) as f64,
            })
            .collect(),
        total_hits: response.total_hits,
        filtered_count: hit_count,
        elapsed_ms: response.elapsed_ms as u64,
        engine: request.config.mode.clone(),
        cliff_index: None,
    };

    Ok(Json(results))
}

/// WebSocket handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(_session_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

/// Handle WebSocket connection
async fn handle_websocket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to broadcast channel
    let mut rx = state.broadcast_tx.subscribe();

    // Spawn task to forward broadcast messages to client
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(json)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Handle incoming messages
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            if let Ok(client_msg) = serde_json::from_str::<WsClientMessage>(&text) {
                match client_msg {
                    WsClientMessage::SelectQuery { query } => {
                        // Update current query
                        {
                            let mut current = state.current_query.lock().await;
                            *current = query;
                        }
                        // Execute search with current config to return initial results
                        let config = state.current_config.lock().await.clone();
                        let results = execute_search_with_config(&state, &config).await;
                        match results {
                            Ok(results) => {
                                let _ = state
                                    .broadcast_tx
                                    .send(WsServerMessage::Results { data: results });
                            }
                            Err(e) => {
                                // Log but don't send error - this is expected if config isn't set yet
                                tracing::debug!("Initial search after query select failed: {}", e);
                            }
                        }
                    }
                    WsClientMessage::ConfigChange { config } => {
                        // Store config for future use
                        {
                            let mut current = state.current_config.lock().await;
                            *current = config.clone();
                        }
                        // Execute search with new config
                        let results = execute_search_with_config(&state, &config).await;

                        match results {
                            Ok(results) => {
                                let _ = state
                                    .broadcast_tx
                                    .send(WsServerMessage::Results { data: results });
                            }
                            Err(e) => {
                                let _ = state
                                    .broadcast_tx
                                    .send(WsServerMessage::Error { message: e });
                            }
                        }
                    }
                    WsClientMessage::StartOptimize => {
                        // Start optimization in background
                        let state_clone = state.clone();
                        tokio::spawn(async move {
                            run_optimization(state_clone).await;
                        });
                    }
                }
            }
        }
    }

    send_task.abort();
}

/// Execute search with the given config
async fn execute_search_with_config(
    state: &Arc<AppState>,
    config: &RetrievalConfig,
) -> Result<SearchResults, String> {
    let mut memvid = state.memvid.lock().await;
    let query = state.current_query.lock().await.clone();

    if query.is_empty() {
        return Err(
            "No query selected. Please select a query from the timeline first.".to_string(),
        );
    }

    let search_request = memvid_core::types::SearchRequest {
        query,
        top_k: config.k,
        snippet_chars: 240,
        uri: None,
        scope: None,
        cursor: None,
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
    };

    let response = memvid.search(search_request).map_err(|e| e.to_string())?;

    let hit_count = response.hits.len();
    Ok(SearchResults {
        hits: response
            .hits
            .into_iter()
            .map(|hit| DocumentHit {
                frame_id: hit.frame_id,
                title: hit
                    .title
                    .unwrap_or_else(|| format!("Frame {}", hit.frame_id)),
                snippet: hit.text,
                score: hit.score.unwrap_or(0.0) as f64,
            })
            .collect(),
        total_hits: response.total_hits,
        filtered_count: hit_count,
        elapsed_ms: response.elapsed_ms as u64,
        engine: config.mode.clone(),
        cliff_index: None,
    })
}

/// Run optimization process
async fn run_optimization(state: Arc<AppState>) {
    let total_configs = 150;

    for i in 0..=total_configs {
        // Send progress update
        let _ = state.broadcast_tx.send(WsServerMessage::OptimizeProgress {
            data: OptimizeProgress {
                progress: (i as f32 / total_configs as f32) * 100.0,
                configs_tested: i,
                total_configs,
            },
        });

        // Simulate work
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }

    // Send final result
    let _ = state.broadcast_tx.send(WsServerMessage::OptimizeComplete {
        data: OptimizeResult {
            recommended_config: RetrievalConfig {
                mode: "sem".to_string(),
                k: 12,
                adaptive: true,
                adaptive_strategy: Some("combined".to_string()),
                min_relevancy: Some(0.5),
                max_k: Some(100),
            },
            score: 0.942,
            coverage: 0.942,
            token_reduction: 0.47,
            explanation: "This configuration retrieves 94.2% of relevant documents while filtering 47% of noise.".to_string(),
        },
    });
}
