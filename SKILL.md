---
name: nanobanana-mcp
description: Generate social media visuals through Kie AI Nano Banana jobs API using an explicit style string passed by the caller plus platform context, then poll task status and save resulting images to disk. Use when asked to create or regenerate post visuals, pass platform-specific requirements (LinkedIn/X/Threads/Instagram/Facebook/Telegram/Reddit), or set up/operate an MCP image generation bridge. Do not read AGENTS.md inside this skill; style must be provided as an input parameter.
---

# NanoBanana MCP

## Overview

Use this skill to generate social visuals via the Kie AI Nano Banana API.
Run the bundled MCP server, submit image jobs, poll completion, and save images.

## Quick Start

1. Set `NANOBANANA_API_KEY` in `.env` (or `skills/nanobanana-mcp/.env`).
2. Run this one command to install deps (if needed) and start the MCP server:
   `cd <PROJECT_FOLDER> && python3 -m pip install -r scripts/requirements.txt && python3 scripts/server.py`
3. For later runs (faster start), use: `cd <PROJECT_FOLDER> && python3 scripts/server.py`.
4. Call tool `generate_visual` with `style`, `prompt`, and `platform`.

## Core Workflow

1. Use `describe_imager_interface` to confirm endpoints and expected inputs.
2. Use `generate_visual` with an explicit `style` string for standard behavior.
3. Use `create_visual_task` + `get_visual_task` for async/manual polling flows.

## Latency Expectations

- `generate_visual` is intentionally blocking by default and waits for remote job completion.
- Typical latency comes from upstream queue time + polling cycles + final image download/save.
- Default polling behavior is every ~3s with up to ~300s total wait unless overridden by tool inputs/env.
- For faster interaction, prefer async flow: submit with `create_visual_task` and poll with `get_visual_task`.

## Execution Stability

- For real generation runs, use one direct `run_in_terminal` execution path as the default.
- Do not use notebook/snippet execution tools for production generation runs.
- Use a single command per run and wait for completion before sending any new command.
- If a run is cancelled/skipped by the host, report it and rerun the same command unchanged.

## Tool Usage

- `generate_visual`
  Pass explicit `style` and `prompt`, plus optional `platform`, `aspect_ratio`, `resolution`, and `output_path`.
- `create_visual_task`
  Submit task only and return `task_id`.
- `get_visual_task`
  Query job status/result by `task_id`.

## Inputs To Prefer

- `platform`: use lowercase values like `linkedin`, `x`, `instagram`, `facebook`.
- `aspect_ratio`: choose by target feed format.
- `resolution`: default `1K`; increase only when needed.
- `output_format`: default `png`.

## Resources

- API and env details: `references/kie-api.md`
- MCP server implementation: `scripts/server.py`

## Constraints

- Do not create temporary/local scripts or helper files to run generation.
- Use only existing skills and MCP tools already available in the workspace.
- Do not change, add, or rewrite files unless the user explicitly asks for it.

## Failure Policy

- Do not use alternative execution paths when a run fails.
- Do not switch tools, command variants, or fallback modes after failure.
- On failure, stop and return the exact error output for user decision.
- When applicable, report this exact status: `nanobanana-mcp was invoked, but the generation run failed.`
- When applicable, preserve this exact escalation status message: `The run failed due to restricted network/DNS in sandbox, so I’m rerunning the same generation command with escalated permissions to allow outbound API access.`
