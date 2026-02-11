# NanoBanana MCP Server

Minimal MCP server that bridges to the Kie AI Nano Banana image API. It creates tasks, polls for completion, and saves the resulting image to disk. Designed for AppSec/social visual workflows, but works with any prompt and style.

**Author:** [roxl.net](https://roxl.net)

## Example
![NanoBanana MCP example](images/nanobanana-mcp.png)

## What It Does
- Creates image generation tasks via Kie AI
- Polls task status until completion (or returns a task id for async flows)
- Downloads the resulting image and writes it to disk
- Supports an AppSec-specific helper that loads style from `AGENTS.md`

## Quick Start
1. Add your API key to `.env` (or use `.env.example` as a template).
2. On the first run, install dependencies from `skills/nanobanana-mcp/scripts/requirements.txt`:

```bash
python3 -m pip install -r skills/nanobanana-mcp/scripts/requirements.txt
```

3. Start the server with system `python3` (ignore `.venv`):

```bash
python3 skills/nanobanana-mcp/scripts/server.py
```

## Environment Variables
Required:
- `NANOBANANA_API_KEY`

Optional:
- `NANOBANANA_BASE_URL` (default `https://api.kie.ai/api/v1`)
- `NANOBANANA_MODEL` (default `nano-banana-pro`)
- `NANOBANANA_CREATE_TASK_PATH` (default `/jobs/createTask`)
- `NANOBANANA_GET_TASK_PATH` (default `/jobs/recordInfo`)
- `NANOBANANA_TIMEOUT_SECONDS` (default `90`)
- `NANOBANANA_POLL_INTERVAL_SECONDS` (default `3`)
- `NANOBANANA_POLL_TIMEOUT_SECONDS` (default `300`)
- `NANOBANANA_OUTPUT_DIR` (default `skills/nanobanana-mcp/output`)
- `NANOBANANA_AGENTS_FILE` (default `AGENTS.md`)

See `.env.example` for a full template.

## MCP Tools
- `describe_imager_interface` — Returns tool contracts and environment info.
- `create_visual_task` — Creates a task and returns `task_id`.
- `get_visual_task` — Fetches task status by `task_id`.
- `generate_visual` — Creates a task and waits for the resulting image.
- `generate_visual_from_appsec_imager` — Loads style from `AGENTS.md` role `appsec-imager`.

## Notes
- The server reads `.env` from the project root or from `skills/nanobanana-mcp/.env`.
- Images are saved to `NANOBANANA_OUTPUT_DIR` unless `output_path` is passed.
- No secrets are stored in this repository. Do not commit your `.env`.

## References
- API details: `skills/nanobanana-mcp/references/kie-api.md`
- Server implementation: `skills/nanobanana-mcp/scripts/server.py`
