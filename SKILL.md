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
2. On the first run, install dependencies:
   `python3 -m pip install -r skills/nanobanana-mcp/scripts/requirements.txt`
3. Start server with system `python3` (ignore `.venv`): `python3 skills/nanobanana-mcp/scripts/server.py`.
4. Call tool `generate_visual` with `style`, `prompt`, and `platform`.

## Core Workflow
1. Use `describe_imager_interface` to confirm endpoints and expected inputs.
2. Use `generate_visual` with an explicit `style` string for standard behavior.
3. Use `create_visual_task` + `get_visual_task` for async/manual polling flows.

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
