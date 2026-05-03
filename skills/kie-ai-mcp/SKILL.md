---
name: kie-ai-mcp
description: Generate social media visuals through the Kie AI jobs API using an explicit style string passed by the caller plus platform context, then poll task status and save resulting images to disk. Use when asked to create or regenerate post visuals, pass platform-specific requirements (LinkedIn/X/Threads/Instagram/Facebook/Telegram/Reddit), or set up/operate an MCP image generation bridge. Do not read AGENTS.md inside this skill; style must be provided as an input parameter.
---

# Kie AI MCP

## Overview

Use this skill to generate social visuals via the Kie AI image generation API.
Run the bundled MCP server, inspect the supported model list, submit image jobs, poll completion, and save images.

## Quick Start

1. Copy `.env.example` to `.env` next to the binary (or in the project root) and set `KIE_AI_API_KEY`.
2. Build the server:
   ```bash
   make build
   ```
3. Run the MCP server:
   ```bash
   ./bin/kie-ai-mcp
   ```
   Or for development: `go run ./src`
4. Call `list_models` to inspect the hardcoded supported models when model choice matters.
5. Call tool `generate_visual` with `style` and `prompt`.

## Core Workflow

1. Use `describe_imager_interface` to confirm endpoints and expected inputs.
2. Use `list_models` to discover the supported hardcoded model IDs.
3. Use `generate_visual` with an explicit `style` string for the standard end-to-end flow.
4. Use `create_visual_task` + `get_visual_task` for async/manual polling flows.
5. Use `generate_visual_batch` when multiple prompt/style pairs should run in parallel.

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

- `list_models`
  Return the current hardcoded model catalog and use the returned `id` as the `model` parameter in generation tools.
- `generate_visual`
  Pass explicit `style` and `prompt`, plus optional `platform`, `model`, `aspect_ratio`, `resolution`, `output_format`, and `output_path`.
- `create_visual_task`
  Submit task only and return `task_id`; supports optional `model`, `negative_prompt`, and `callback_url`.
- `get_visual_task`
  Query job status/result by `task_id`.
- `generate_visual_batch`
  Submit multiple generation items in one call with `items` JSON and optional defaults such as `default_style`, `default_platform`, and polling settings.

## Inputs To Prefer

- `platform`: use lowercase values like `linkedin`, `x`, `instagram`, `facebook`.
- `model`: prefer one of the IDs returned by `list_models`.
- `aspect_ratio`: choose by target feed format.
- `resolution`: default `1K`; increase only when needed.
- `output_format`: default `png`.

## Supported Models

The server currently hardcodes these text-to-image models:

- `gpt-image-2-text-to-image`
- `gpt-image/1.5-text-to-image`
- `nano-banana-2`
- `nano-banana-pro`
- `qwen/text-to-image`
- `qwen2/text-to-image`
- `grok-imagine/text-to-image`
- `z-image`

## Resources

- API and env details: `references/kie-api.md`
- MCP server implementation: Go binary at `bin/kie-ai-mcp` (source: `src/main.go`)

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
