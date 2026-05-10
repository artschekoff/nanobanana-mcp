# Remove Background Feature Design

**Date:** 2026-05-10  
**API:** `recraft/remove-background` via kie.ai

---

## Overview

Add three MCP tools for background removal using the Recraft AI model on kie.ai. The tools follow the same async submit → poll → download pattern as the existing visual generation tools.

---

## API Contract

- **Endpoint:** `POST https://api.kie.ai/api/v1/jobs/createTask` (same as image generation)
- **Model:** `recraft/remove-background`
- **Input fields:** `image: string` (URL or base64 data URL), `nsfw_checker: bool`
- **Status polling:** `GET /jobs/recordInfo?taskId=...` (same endpoint as existing tools)
- **Output:** transparent PNG returned as a URL in `resultJson.resultUrls`

---

## New Types (`src/internal/kieai/types.go`)

```go
type RemoveBackgroundRequest struct {
    Model       string                `json:"model"`
    CallBackURL string                `json:"callBackUrl,omitempty"`
    Input       RemoveBackgroundInput `json:"input"`
}

type RemoveBackgroundInput struct {
    Image       string `json:"image"`
    NSFWChecker bool   `json:"nsfw_checker"`
}
```

---

## New Client Function (`src/internal/kieai/client.go`)

`CreateRemoveBackgroundTask(cfg Config, req RemoveBackgroundRequest) (string, error)`

Same implementation shape as `CreateTask`: marshal body → POST → parse `APIResponse` → return `taskId`. No polling logic here.

---

## New Tool File (`src/internal/tools/remove_background.go`)

### `createRemoveBgTaskHandler(cfg)`

Submits a remove-background job. Returns `task_id` immediately.

Parameters:
| Name | Type | Required | Default | Notes |
|---|---|---|---|---|
| `image` | string | yes | — | URL or local file path |
| `nsfw_checker` | bool | no | `true` | |
| `callback_url` | string | no | `""` | |

Image resolution:
- If value starts with `http://` or `https://` → pass directly
- Otherwise → treat as local file path: `os.ReadFile` → check size ≤ 5MB → base64 encode → `data:<mime>;base64,<data>`
- MIME detection: call `http.DetectContentType` on first 512 bytes of file data

Returns:
```json
{"ok": true, "task_id": "...", "model": "recraft/remove-background"}
```

### `getRemoveBgTaskHandler(cfg)`

Checks task status. Identical logic to `getVisualTaskHandler` (same polling endpoint, same response shape).

Parameters: `task_id` (required)

Returns:
```json
{"ok": true, "task_id": "...", "status": "success", "image_urls": ["https://..."]}
```

### `removeBackgroundHandler(cfg)`

End-to-end: submit → poll → download → save file.

Parameters:
| Name | Type | Required | Default |
|---|---|---|---|
| `image` | string | yes | — |
| `nsfw_checker` | bool | no | `true` |
| `output_path` | string | no | `cfg.OutputDir` |
| `wait_for_result` | bool | no | `true` |
| `poll_interval_seconds` | float | no | `cfg.PollIntervalSec` |
| `poll_timeout_seconds` | int | no | `cfg.PollTimeoutSec` |
| `callback_url` | string | no | `""` |

If `wait_for_result=false`: submit only, return `task_id` with hint to use `get_remove_bg_task`.

If `wait_for_result=true`: submit → `PollUntilDone` → `DownloadImage` → write file → return path.

Output filename: `<timestamp>-remove-bg.png` using existing `defaultFileName` helper (prompt `"remove-bg"`, ext always `"png"` — output is always a transparent PNG).

Returns:
```json
{"ok": true, "task_id": "...", "status": "success", "output_path": "...", "image_url": "...", "bytes": 12345}
```

---

## Tool Registration (`src/internal/tools/register.go`)

Add three `s.AddTool(...)` calls:

```
create_remove_bg_task  — "Submit a background removal job to Kie AI. Returns task_id immediately."
get_remove_bg_task     — "Check status of a background removal task. Returns image_url when done."
remove_background      — "Remove image background end-to-end: submit, poll, download, save file."
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `image` param missing | `mcp.NewToolResultError("'image' is required")` |
| Local file not found | `mcp.NewToolResultError("read image file: ...")` |
| File >5MB | `mcp.NewToolResultError("image file exceeds 5MB limit")` |
| API error | Surface API error message |
| Poll timeout | Surface `"task did not complete within Xs"` |
| Download failure | Surface download error |

---

## Files Changed

| File | Change |
|---|---|
| `src/internal/kieai/types.go` | Add `RemoveBackgroundRequest`, `RemoveBackgroundInput` |
| `src/internal/kieai/client.go` | Add `CreateRemoveBackgroundTask()` |
| `src/internal/tools/remove_background.go` | New file — 3 handlers |
| `src/internal/tools/register.go` | Register 3 new tools |

No changes to existing types, handlers, or tests.

---

## Out of Scope

- Batch remove-background (can be added later following `generate_visual_batch` pattern)
- New environment variables (reuses existing `KIE_AI_*` config)
