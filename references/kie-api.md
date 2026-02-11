# Kie API Reference

## Endpoints
- Create task: `POST https://api.kie.ai/api/v1/jobs/createTask`
- Get task detail: `GET https://api.kie.ai/api/v1/jobs/recordInfo?taskId=...`

## Create Task Body
```json
{
  "model": "nano-banana-pro",
  "callBackUrl": "https://your-domain.com/api/callback",
  "input": {
    "prompt": "...",
    "image_input": [],
    "aspect_ratio": "1:1",
    "resolution": "1K",
    "output_format": "png",
    "platform": "linkedin"
  }
}
```

## Environment Variables
- `NANOBANANA_API_KEY` required.
- `NANOBANANA_BASE_URL` default `https://api.kie.ai/api/v1`.
- `NANOBANANA_MODEL` default `nano-banana-pro`.
- `NANOBANANA_CREATE_TASK_PATH` default `/jobs/createTask`.
- `NANOBANANA_GET_TASK_PATH` default `/jobs/recordInfo`.
- `NANOBANANA_TIMEOUT_SECONDS` default `90`.
- `NANOBANANA_POLL_INTERVAL_SECONDS` default `3`.
- `NANOBANANA_POLL_TIMEOUT_SECONDS` default `300`.
- `NANOBANANA_OUTPUT_DIR` default `skills/nanobanana-mcp/output`.
- `NANOBANANA_AGENTS_FILE` default `AGENTS.md`.

## Notes
- `generate_visual_from_appsec_imager` auto-loads style from role `appsec-imager` in `AGENTS.md`.
- If API response shape changes, update field extractors in `scripts/server.py`.
