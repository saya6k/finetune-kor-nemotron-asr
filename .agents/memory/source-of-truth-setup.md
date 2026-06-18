---
name: source-of-truth-setup
description: "Project memory and CLAUDE.md Source of Truth locations"
metadata:
  type: project
---

## Source of Truth Locations

This project uses the repository itself as the Source of Truth (SoT) for both project memory and `CLAUDE.md`, with symlinks from `~/.claude/`.

### Memory SoT
- **SoT**: `~/Projects/finetune-kor-nemotron-asr/.agents/memory/`
- **Symlink**: `~/.claude/projects/-Users-saya6k-Projects-finetune-kor-nemotron-asr/memory` → `~/Projects/finetune-kor-nemotron-asr/.agents/memory`
- New memories should be written to `.agents/memory/` in the project directory.
- The `MEMORY.md` index file lives alongside the memory files.

### CLAUDE.md SoT
- **SoT**: `~/Projects/finetune-kor-nemotron-asr/AGENTS.md`
- **Symlink**: `~/Projects/finetune-kor-nemotron-asr/CLAUDE.md` → `AGENTS.md`
- The file is named `AGENTS.md` to follow the `.agents/` convention, with `CLAUDE.md` as a compatibility symlink.
- Both names are recognized by Claude Code.

### Rationale
- Keeps all project configuration self-contained within the repository.
- The `.agents/` directory is the standard Claude Code project-level config location.
- Symlinks in `~/.claude/` maintain backward compatibility with the global Claude configuration.
