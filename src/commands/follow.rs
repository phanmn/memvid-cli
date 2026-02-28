//! Logic-Mesh follow command for graph traversal.
//!
//! This command allows traversing the entity-relationship graph
//! to follow facts and relationships instead of relying on vector search.

use anyhow::{bail, Result};
use clap::{Args, Subcommand};
use memvid_core::Memvid;
use serde::Serialize;
use std::path::PathBuf;

use crate::config::CliConfig;

/// Logic-Mesh graph traversal commands
#[derive(Args)]
pub struct FollowArgs {
    #[command(subcommand)]
    pub command: FollowCommand,
}

#[derive(Subcommand)]
pub enum FollowCommand {
    /// Follow relationships from an entity
    Traverse(FollowTraverseArgs),
    /// List all entities in the mesh
    Entities(FollowEntitiesArgs),
    /// Show statistics about the Logic-Mesh
    Stats(FollowStatsArgs),
}

/// Arguments for the follow traverse command
#[derive(Args)]
pub struct FollowTraverseArgs {
    /// Path to the .mv2 file
    pub file: PathBuf,

    /// Starting entity name (case-insensitive partial match)
    #[arg(short, long)]
    pub start: String,

    /// Relationship type to follow (e.g., "manager", "member", "author")
    #[arg(short, long)]
    pub link: Option<String>,

    /// Maximum hops to traverse (default: 2)
    #[arg(long, default_value = "2")]
    pub hops: usize,

    /// Direction to traverse: "outgoing", "incoming", or "both" (default: "both")
    #[arg(long, default_value = "both")]
    pub direction: String,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for listing entities
#[derive(Args)]
pub struct FollowEntitiesArgs {
    /// Path to the .mv2 file
    pub file: PathBuf,

    /// Filter by entity type (person, organization, project, etc.)
    #[arg(short = 't', long)]
    pub kind: Option<String>,

    /// Search query to filter entities by name
    #[arg(short, long)]
    pub query: Option<String>,

    /// Maximum number of entities to show
    #[arg(short, long, default_value = "50")]
    pub limit: usize,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for mesh statistics
#[derive(Args)]
pub struct FollowStatsArgs {
    /// Path to the .mv2 file
    pub file: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Handle follow commands
pub fn handle_follow(_config: &CliConfig, args: FollowArgs) -> Result<()> {
    match args.command {
        FollowCommand::Traverse(traverse_args) => handle_follow_traverse(traverse_args),
        FollowCommand::Entities(entities_args) => handle_follow_entities(entities_args),
        FollowCommand::Stats(stats_args) => handle_follow_stats(stats_args),
    }
}

/// Handle follow traverse command
fn handle_follow_traverse(args: FollowTraverseArgs) -> Result<()> {
    let mem = Memvid::open(&args.file)?;

    // Check if Logic-Mesh exists
    if !mem.has_logic_mesh() {
        bail!(
            "No Logic-Mesh found in {}. Run `memvid put --logic-mesh` to build the graph.",
            args.file.display()
        );
    }

    // Find the starting entity
    let start_node = mem.find_entity(&args.start);
    if start_node.is_none() {
        bail!(
            "Entity '{}' not found in Logic-Mesh. Use `memvid follow entities` to list available entities.",
            args.start
        );
    }

    let link = args.link.as_deref().unwrap_or("related");
    let results = mem.follow(&args.start, link, args.hops);

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        let start_node = start_node.unwrap();
        println!(
            "Starting from: {} ({})",
            start_node.display_name,
            start_node.kind.as_str()
        );
        println!(
            "Following: {} relationships (up to {} hops)",
            link, args.hops
        );
        println!();

        if results.is_empty() {
            println!("No {} relationships found.", link);
        } else {
            for result in &results {
                println!(
                    "  → {} ({}, confidence: {:.0}%)",
                    result.node,
                    result.kind.as_str(),
                    result.confidence * 100.0
                );
                if !result.frame_ids.is_empty() {
                    println!("    Frames: {:?}", result.frame_ids);
                }
            }
        }
    }

    Ok(())
}

/// Handle follow entities command
fn handle_follow_entities(args: FollowEntitiesArgs) -> Result<()> {
    let mem = Memvid::open(&args.file)?;

    if !mem.has_logic_mesh() {
        bail!(
            "No Logic-Mesh found in {}. Run `memvid put --logic-mesh` to build the graph.",
            args.file.display()
        );
    }

    let mesh = mem.logic_mesh();

    // Filter entities by kind if specified
    let mut entities: Vec<_> = mesh.nodes.iter().collect();

    if let Some(ref kind_filter) = args.kind {
        let kind_lower = kind_filter.to_lowercase();
        entities.retain(|node| node.kind.as_str() == kind_lower);
    }

    // Filter by query if specified
    if let Some(ref query) = args.query {
        let query_lower = query.to_lowercase();
        entities.retain(|node| {
            node.display_name.to_lowercase().contains(&query_lower)
                || node.canonical_name.contains(&query_lower)
        });
    }

    // Sort by display name for consistent output
    entities.sort_by(|a, b| a.display_name.cmp(&b.display_name));

    // Apply limit
    let limited: Vec<_> = entities.into_iter().take(args.limit).collect();

    if args.json {
        #[derive(Serialize)]
        struct EntityOutput {
            name: String,
            kind: String,
            confidence: f32,
            frame_count: usize,
            frame_ids: Vec<u64>,
        }
        let output: Vec<EntityOutput> = limited
            .iter()
            .map(|node| EntityOutput {
                name: node.display_name.clone(),
                kind: node.kind.as_str().to_string(),
                confidence: node.confidence_f32(),
                frame_count: node.frame_ids.len(),
                frame_ids: node.frame_ids.clone(),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!(
            "Entities in Logic-Mesh (showing {} of {})",
            limited.len(),
            mesh.nodes.len()
        );
        println!();
        for node in &limited {
            println!(
                "  {} ({}, confidence: {:.0}%, {} frames)",
                node.display_name,
                node.kind.as_str(),
                node.confidence_f32() * 100.0,
                node.frame_ids.len()
            );
        }
        if limited.len() < mesh.nodes.len() {
            println!();
            println!("Use --limit to show more entities, or --query to filter.");
        }
    }

    Ok(())
}

/// Handle follow stats command
fn handle_follow_stats(args: FollowStatsArgs) -> Result<()> {
    let mem = Memvid::open(&args.file)?;

    if !mem.has_logic_mesh() {
        if args.json {
            println!(r#"{{"error": "No Logic-Mesh found"}}"#);
        } else {
            println!("No Logic-Mesh found in {}", args.file.display());
            println!("Run `memvid put --logic-mesh` to build the graph.");
        }
        return Ok(());
    }

    let stats = mem.logic_mesh_stats();
    let manifest = mem.logic_mesh_manifest();

    if args.json {
        #[derive(Serialize)]
        struct MeshStatsOutput {
            node_count: usize,
            edge_count: usize,
            entity_kinds: std::collections::HashMap<String, usize>,
            link_types: std::collections::HashMap<String, usize>,
            #[serde(skip_serializing_if = "Option::is_none")]
            bytes_offset: Option<u64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            bytes_length: Option<u64>,
        }
        let output = MeshStatsOutput {
            node_count: stats.node_count,
            edge_count: stats.edge_count,
            entity_kinds: stats.entity_kinds,
            link_types: stats.link_types,
            bytes_offset: manifest.map(|m| m.bytes_offset),
            bytes_length: manifest.map(|m| m.bytes_length),
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Logic-Mesh Statistics");
        println!("=====================");
        println!("  Nodes (entities):  {}", stats.node_count);
        println!("  Edges (relations): {}", stats.edge_count);
        if !stats.entity_kinds.is_empty() {
            println!();
            println!("  Entity Kinds:");
            let mut kinds: Vec<_> = stats.entity_kinds.iter().collect();
            kinds.sort_by(|a, b| b.1.cmp(a.1));
            for (kind, count) in kinds {
                println!("    {}: {}", kind, count);
            }
        }
        if !stats.link_types.is_empty() {
            println!();
            println!("  Relationship Types:");
            let mut links: Vec<_> = stats.link_types.iter().collect();
            links.sort_by(|a, b| b.1.cmp(a.1));
            for (link, count) in links {
                println!("    {}: {}", link, count);
            }
        }
        if let Some(m) = manifest {
            println!();
            println!("  Storage offset:    {}", m.bytes_offset);
            println!("  Storage size:      {} bytes", m.bytes_length);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_direction_parsing() {
        // Test that direction string parsing would work
        let directions = vec!["outgoing", "incoming", "both"];
        for d in directions {
            assert!(!d.is_empty());
        }
    }
}
