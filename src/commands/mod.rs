pub mod binding;
#[cfg(feature = "encryption")]
pub mod capsule;
pub mod config;
pub mod creation;
pub mod data;
pub mod debug;
pub mod enrich;
pub mod follow;
pub mod inspection;
pub mod maintenance;
pub mod models;
pub mod plan;
pub mod search;
#[cfg(feature = "replay")]
pub mod session;
#[cfg(feature = "web")]
pub mod session_web;
pub mod sketch;
pub mod status;
pub mod tables;
pub mod tickets;
pub mod version;

pub use binding::{handle_binding, BindingArgs};
#[cfg(feature = "encryption")]
pub use capsule::{handle_lock, handle_unlock, LockArgs, UnlockArgs};
pub use config::{
    handle_config, ConfigArgs, ConfigCheckArgs, ConfigCommand, ConfigGetArgs, ConfigListArgs,
    ConfigSetArgs, ConfigUnsetArgs, PersistentConfig,
};
pub use creation::*;
pub use data::{
    handle_api_fetch, handle_correct, handle_delete, handle_put, handle_update, ApiFetchArgs,
    CorrectArgs, DeleteArgs, LockCliArgs, PutArgs, UpdateArgs,
};
#[cfg(feature = "parallel_segments")]
pub use data::{handle_put_many, PutManyArgs};
pub use debug::*;
pub use enrich::{
    handle_enrich, handle_export, handle_facts, handle_memories, handle_schema, handle_state,
    EnrichArgs, EnrichEngine, ExportArgs, ExportFormat, FactsArgs, MemoriesArgs, SchemaArgs,
    SchemaCommand, SchemaInferArgs, SchemaListArgs, StateArgs,
};
pub use follow::{
    handle_follow, FollowArgs, FollowCommand, FollowEntitiesArgs, FollowStatsArgs,
    FollowTraverseArgs,
};
pub use inspection::{
    extension_from_mime, frame_to_json, handle_stats, handle_view, handle_who,
    parse_preview_bounds, print_frame_summary, PreviewBounds, StatsArgs, ViewArgs, WhoArgs,
};
pub use maintenance::{
    handle_doctor, handle_nudge, handle_process_queue, handle_verify, handle_verify_single_file,
    DoctorArgs, NudgeArgs, ProcessQueueArgs, VerifyArgs, VerifySingleFileArgs,
};
pub use models::{
    default_enrichment_model, get_installed_model_path, handle_models, LlmModel, ModelsArgs,
    ModelsCommand, ModelsInstallArgs, ModelsListArgs, ModelsRemoveArgs, ModelsVerifyArgs,
};
pub use plan::{handle_plan, PlanArgs, PlanClearArgs, PlanCommand, PlanShowArgs, PlanSyncArgs};
pub use search::{
    handle_ask, handle_audit, handle_find, handle_timeline, handle_vec_search, AskArgs, AskModeArg,
    AuditArgs, AuditFormat, FindArgs, SearchMode, TimelineArgs, VecSearchArgs,
};
#[cfg(feature = "temporal_track")]
pub use search::{handle_when, WhenArgs};
#[cfg(feature = "replay")]
pub use session::{handle_session, SessionArgs};
pub use sketch::{
    handle_sketch, SketchArgs, SketchBuildArgs, SketchCommand, SketchInfoArgs, SketchVariantArg,
};
pub use status::{handle_status, StatusArgs};
pub use tables::{
    handle_tables, ExportFormatArg, ExtractionModeArg, QualityArg, TablesArgs, TablesCommand,
    TablesExportArgs, TablesImportArgs, TablesListArgs, TablesViewArgs,
};
pub use tickets::{
    handle_ticket_apply, handle_ticket_issue, handle_ticket_revoke, handle_ticket_status,
    handle_ticket_sync, handle_tickets, TicketsApplyArgs, TicketsArgs, TicketsCommand,
    TicketsIssueArgs, TicketsRevokeArgs, TicketsStatusArgs, TicketsSyncArgs,
};
pub use version::*;
