# memvid-cli

Command-line interface for Memvid v2 - AI memory with crash-safe, single-file storage.

## Installation

### Standard Installation (Most Users)

```bash
cargo install memvid-cli
```

This works on:
- **Windows x86-64** (Intel/AMD) - most Windows PCs
- **macOS** (Intel and Apple Silicon)
- **Linux x86-64**

### Platform Support

| Platform | Default Install | Commands Available |
|----------|-----------------|-------------------|
| Windows x86-64 | ✅ Works | All commands |
| Windows ARM | ❌ Use lite install | Read-only mode |
| macOS (Intel/M1/M2) | ✅ Works | All commands |
| Linux x86-64 | ✅ Works | All commands |
| Linux ARM / Docker ARM | ❌ Use lite install | Read-only mode |

### Lite Installation (Windows ARM / Linux ARM)

For Windows ARM devices (Surface Pro X, Parallels on Mac M1/M2) or Linux ARM:

```bash
cargo install memvid-cli --no-default-features --features temporal_track,temporal_enrich,parallel_segments,candle-llm
```

This enables **read-only mode**:
- ✅ `memvid find` - search existing .mv2 files
- ✅ `memvid stats` - view statistics
- ✅ `memvid view` - view frame content
- ✅ `memvid timeline` - timeline queries
- ✅ `memvid ask` - Q&A with retrieval
- ❌ `memvid put` - cannot ingest PDFs/DOCX (plain text still works)

### Optional Features

```bash
# With Metal acceleration (macOS Apple Silicon)
cargo install memvid-cli --features metal

# With audio playback
cargo install memvid-cli --features audio-playback
```

### Build Requirements

The default installation includes:
- `llama.cpp` - requires LLVM/libclang for compilation
- `extractous` - requires GraalVM native binaries (only available for x86-64 and native macOS ARM)

## Quick Start

```bash
# Create a new memory file
memvid create journal.mv2

# Add content
memvid put journal.mv2 --input document.pdf --embeddings
memvid put journal.mv2 --input notes/ --embeddings

# Search
memvid find journal.mv2 --query "meeting notes" --json

# Q&A with retrieval
memvid ask journal.mv2 --question "What was discussed in the Q3 review?"

# Timeline queries
memvid timeline journal.mv2 --since 1706745600 --limit 10

# View a specific frame
memvid view journal.mv2 --frame-id 3 --json
```

## Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new .mv2 memory file |
| `put` | Ingest content (files, directories, URLs) |
| `find` | Search with lexical or semantic queries |
| `ask` | Q&A with retrieval-augmented generation |
| `timeline` | Query frames by time range |
| `view` | Display frame content and metadata |
| `update` | Modify frame metadata |
| `delete` | Remove frames by URI or ID |
| `verify` | Check file integrity |
| `doctor` | Repair and rebuild indices |
| `stats` | Show memory statistics |
| `tickets` | Manage capacity tickets |

## Features

- **Single File**: Everything in one portable `.mv2` file
- **Crash Safe**: Embedded WAL ensures data durability
- **Hybrid Search**: Lexical (Tantivy) + Semantic (HNSW) search
- **Multi-Format**: PDF, DOCX, XLSX, images, audio, video
- **Local LLM**: Built-in Q&A with Phi-3 or llama.cpp
- **Timeline**: Time-based document organization
- **Offline**: Works completely offline after model download

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MEMVID_MODELS_DIR` | Model cache directory (default: `~/.memvid/models`) |
| `MEMVID_OFFLINE` | Skip model downloads (use cached) |
| `MEMVID_API_URL` | Control plane endpoint |
| `MEMVID_API_KEY` | API authentication key |

## License

Apache-2.0

## Links

- [Documentation](https://docs.memvid.com)
- [GitHub](https://github.com/memvid/memvid)
