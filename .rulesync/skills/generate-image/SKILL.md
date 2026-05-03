---
name: generate-image
description: Generate images via the Kie AI MCP server. Use when the user asks to create, generate, or make images. Handles both single and batch generation.
---

Use the `generate_visual` MCP tool. It submits the job, polls the execution queue, downloads, and saves the image to disk.

## Async execution model

Image generation is never instant. The tools expose two modes:

**Blocking (default):** `generate_visual` with `wait_for_result: true` — polls the queue internally, returns `output_path` when done. Use this for single images.

**Non-blocking:** `generate_visual` with `wait_for_result: false` — returns `task_id` immediately. Then call `get_visual_task` in a loop until `status == "success"`. Use this when the user wants to submit and check later, or when managing multiple jobs manually.

## Required args

- `style` — visual style, be specific: "photorealistic, 8K, studio lighting, golden hour"
- `prompt` — subject, composition, mood, colors: "a serene mountain lake at dusk with reflections"

## Optional args

| Arg | Values | Default |
|-----|--------|---------|
| `platform` | instagram, linkedin, facebook | — |
| `aspect_ratio` | 1:1, 16:9, 9:16, 4:3 | 1:1 |
| `output_format` | png, jpeg, webp | png |
| `output_path` | file or directory path | output/ |
| `negative_prompt` | elements to exclude | — |
| `wait_for_result` | true / false | true |
| `poll_timeout_seconds` | integer | 300 |

## Platform → aspect ratio guide

| Platform | Recommended |
|----------|-------------|
| Instagram feed | 1:1 |
| Instagram Story / Reels | 9:16 |
| LinkedIn | 4:3 or 16:9 |
| Facebook | 16:9 |

## Steps

1. Confirm `style` and `prompt` with the user if not provided.
2. Call `generate_visual` — it blocks until the image is saved (up to 5 min).
3. Report `output_path` and `image_url` to the user.
4. For multiple images: use `generate_visual_batch` with an `items` array (runs in parallel).

## Single image example

User: "generate a sunset mountain landscape for Instagram"

```json
{
  "style": "photorealistic, golden hour lighting, cinematic depth of field, 8K detail",
  "prompt": "sunset over snow-capped mountain range, warm orange and pink sky, silhouetted pine trees in foreground, reflection in alpine lake",
  "platform": "instagram",
  "aspect_ratio": "1:1",
  "output_format": "png"
}
```

## Batch example

User: "generate 3 product photos in different styles"

```json
{
  "items": [
    {"prompt": "minimalist ceramic mug on white background", "style": "studio product photography, soft shadows, 8K"},
    {"prompt": "same mug in rustic kitchen setting", "style": "warm lifestyle photography, natural light"},
    {"prompt": "mug with steam rising, macro lens", "style": "artistic macro photography, bokeh background"}
  ],
  "default_platform": "instagram",
  "max_workers": 3
}
```
