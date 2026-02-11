#!/usr/bin/env python3
"""NanoBanana MCP bridge for Kie AI API.

API flow:
1. POST /api/v1/jobs/createTask
2. Optional polling via POST /api/v1/jobs/getTask
3. Save resulting image to disk when available
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise SystemExit("Missing dependency 'python-dotenv'. Install with: pip install python-dotenv") from exc

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Missing dependency 'mcp'. Install with: pip install mcp") from exc


def _load_dotenv() -> None:
    """Load .env located next to this file without overriding existing env vars."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists() or not env_path.is_file():
        return
    load_dotenv(dotenv_path=env_path, override=False)


_load_dotenv()

SERVER_NAME = "nanobanana-mcp"
DEFAULT_BASE_URL = os.getenv("NANOBANANA_BASE_URL", "https://api.kie.ai/api/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("NANOBANANA_MODEL", "nano-banana-pro")
DEFAULT_TIMEOUT = int(os.getenv("NANOBANANA_TIMEOUT_SECONDS", "90"))
DEFAULT_POLL_INTERVAL = float(os.getenv("NANOBANANA_POLL_INTERVAL_SECONDS", "3"))
DEFAULT_POLL_TIMEOUT = int(os.getenv("NANOBANANA_POLL_TIMEOUT_SECONDS", "300"))
DEFAULT_CREATE_TASK_PATH = os.getenv("NANOBANANA_CREATE_TASK_PATH", "/jobs/createTask")
DEFAULT_GET_TASK_PATH = os.getenv("NANOBANANA_GET_TASK_PATH", "/jobs/recordInfo")
DEFAULT_OUTPUT_DIR = Path(os.getenv("NANOBANANA_OUTPUT_DIR", "skills/nanobanana-mcp/output"))
DEFAULT_AGENTS_FILE = os.getenv("NANOBANANA_AGENTS_FILE", "AGENTS.md")

mcp = FastMCP(SERVER_NAME)


def _env_or_fail(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable '{name}' is required")
    return value


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:max_len] if slug else "visual"


def _default_file_name(prompt: str, fmt: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{_slugify(prompt)}.{fmt}"


def _resolve_output_path(output_path: Optional[str], prompt: str, image_format: str) -> Path:
    if output_path:
        path = Path(output_path)
        if path.is_dir() or str(output_path).endswith(("/", "\\")):
            path = path / _default_file_name(prompt, image_format)
        elif path.suffix.lower() != f".{image_format}":
            path = path.with_suffix(f".{image_format}")
    else:
        path = DEFAULT_OUTPUT_DIR / _default_file_name(prompt, image_format)

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _compose_prompt(
    style: str,
    prompt: str,
    negative_prompt: Optional[str],
    platform: Optional[str] = None,
) -> str:
    sections = [
        "Follow this visual style strictly:",
        style.strip(),
        "",
    ]
    if platform and platform.strip():
        sections.extend(
            [
                "Target social platform:",
                platform.strip(),
                "",
            ]
        )
    sections.extend(
        [
        "Generate this scene:",
        prompt.strip(),
        ]
    )
    if negative_prompt:
        sections.extend(["", "Avoid:", negative_prompt.strip()])
    return "\n".join(sections).strip()


def _extract_role_section(md_text: str, role_name: str) -> Optional[str]:
    # Match headings like: ### `appsec-imager` or ### appsec-imager
    heading_re = re.compile(rf"^###\s+`?{re.escape(role_name)}`?\s*$", re.IGNORECASE)
    next_heading_re = re.compile(r"^###\s+")
    lines = md_text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if heading_re.match(line.strip()):
            start = i + 1
            break
    if start is None:
        return None

    end = len(lines)
    for i in range(start, len(lines)):
        if next_heading_re.match(lines[i].strip()):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def _extract_behavior_bullets(role_section: str) -> List[str]:
    lines = role_section.splitlines()
    behavior_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("behavior:"):
            behavior_idx = i + 1
            break

    scan_from = behavior_idx if behavior_idx is not None else 0
    bullets: List[str] = []
    for line in lines[scan_from:]:
        stripped = line.strip()
        if stripped.startswith("### "):
            break
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
        elif behavior_idx is not None and stripped and not stripped.startswith("#"):
            # Stop behavior scan when next non-bullet content starts.
            break
    return bullets


def _load_role_style(role_name: str = "appsec-imager", agents_file: Optional[str] = None) -> str:
    path = Path(agents_file or os.getenv("NANOBANANA_AGENTS_FILE", DEFAULT_AGENTS_FILE))
    if not path.exists():
        raise FileNotFoundError(
            f"AGENTS file not found: {path.as_posix()} (set NANOBANANA_AGENTS_FILE or pass agents_file)"
        )

    raw = path.read_text(encoding="utf-8")
    section = _extract_role_section(raw, role_name=role_name)
    if not section:
        raise ValueError(f"Role '{role_name}' not found in {path.as_posix()}")

    bullets = _extract_behavior_bullets(section)
    if not bullets:
        raise ValueError(
            f"Role '{role_name}' found but no behavior bullet list detected in {path.as_posix()}"
        )

    return "\n".join(f"- {item}" for item in bullets)


def _build_url(base_url: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return f"{base_url}{path}"


def _http_post_json(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


def _http_get_json(url: str, headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


def _http_get_bytes(url: str, headers: Dict[str, str], timeout: int) -> bytes:
    def _fetch(current_headers: Dict[str, str]) -> bytes:
        req = urllib.request.Request(url, headers=current_headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    try:
        return _fetch(headers)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403:
            # Retry without auth header and with a basic UA for CDN links.
            retry_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            retry_headers.setdefault("User-Agent", "Mozilla/5.0")
            try:
                return _fetch(retry_headers)
            except urllib.error.HTTPError as exc_retry:
                details = exc_retry.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc_retry.code} downloading image {url}: {details}") from exc_retry
        raise RuntimeError(f"HTTP {exc.code} downloading image {url}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error downloading image {url}: {exc}") from exc


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _find_key_recursive(payload: Any, names: Tuple[str, ...]) -> Optional[Any]:
    if isinstance(payload, dict):
        for name in names:
            if name in payload:
                return payload[name]
        for value in payload.values():
            found = _find_key_recursive(value, names)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_key_recursive(item, names)
            if found is not None:
                return found
    return None


def _extract_task_id(payload: Dict[str, Any]) -> Optional[str]:
    value = _find_key_recursive(payload, ("taskId", "task_id", "jobId", "job_id", "id"))
    if value is None:
        return None
    return str(value)


def _extract_status(payload: Dict[str, Any]) -> Optional[str]:
    value = _find_key_recursive(payload, ("status", "state", "taskStatus", "jobStatus"))
    if value is None:
        return None
    return str(value).strip().lower()


def _extract_image_url(payload: Dict[str, Any]) -> Optional[str]:
    for node in _walk(payload):
        if isinstance(node, str) and node.startswith(("http://", "https://")):
            lower = node.lower()
            if any(ext in lower for ext in (".png", ".jpg", ".jpeg", ".webp")):
                return node
    value = _find_key_recursive(payload, ("imageUrl", "image_url", "url", "outputUrl", "output_url"))
    return str(value) if isinstance(value, str) and value.startswith(("http://", "https://")) else None


def _extract_result_json_urls(payload: Dict[str, Any]) -> List[str]:
    value = _find_key_recursive(payload, ("resultJson", "result_json"))
    if not value:
        return []
    if isinstance(value, dict):
        urls = value.get("resultUrls") or value.get("result_urls") or []
        return [u for u in urls if isinstance(u, str)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        urls = parsed.get("resultUrls") or parsed.get("result_urls") or []
        return [u for u in urls if isinstance(u, str)]
    return []


def _extract_b64(payload: Dict[str, Any]) -> Optional[str]:
    value = _find_key_recursive(payload, ("b64_json", "b64", "base64", "imageBase64", "image_base64"))
    return str(value) if isinstance(value, str) and value.strip() else None


def _is_success(status: Optional[str]) -> bool:
    if not status:
        return False
    return status in {"success", "succeeded", "completed", "done", "finish", "finished"}


def _is_failure(status: Optional[str]) -> bool:
    if not status:
        return False
    return status in {"failed", "error", "cancelled", "canceled", "timeout"}


def _task_payload(task_id: str) -> Dict[str, Any]:
    # Different providers use different keys; send several common forms.
    return {
        "taskId": task_id,
        "task_id": task_id,
        "id": task_id,
    }


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
    }


def _create_task(
    *,
    api_key: str,
    base_url: str,
    path: str,
    body: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    return _http_post_json(_build_url(base_url, path), body, headers=_headers(api_key), timeout=timeout)


def _get_task(
    *,
    api_key: str,
    base_url: str,
    path: str,
    task_id: str,
    timeout: int,
) -> Dict[str, Any]:
    url = _build_url(base_url, path)
    if "?" in url:
        url = f"{url}&taskId={task_id}"
    else:
        url = f"{url}?taskId={task_id}"
    return _http_get_json(url, headers=_headers(api_key), timeout=timeout)


@mcp.tool()
def describe_imager_interface() -> Dict[str, Any]:
    """Return tool contracts and environment configuration for NanoBanana via Kie AI."""
    return {
        "server": SERVER_NAME,
        "api": {
            "create_task": _build_url(DEFAULT_BASE_URL, DEFAULT_CREATE_TASK_PATH),
            "get_task": _build_url(DEFAULT_BASE_URL, DEFAULT_GET_TASK_PATH),
        },
        "tools": {
            "create_visual_task": {
                "required": ["style", "prompt"],
                "optional": [
                    "platform",
                    "callback_url",
                    "aspect_ratio",
                    "resolution",
                    "output_format",
                    "image_input",
                    "model",
                    "negative_prompt",
                ],
            },
            "get_visual_task": {
                "required": ["task_id"],
                "optional": [],
            },
            "generate_visual": {
                "required": ["style", "prompt"],
                "optional": [
                    "platform",
                    "output_path",
                    "callback_url",
                    "aspect_ratio",
                    "resolution",
                    "output_format",
                    "image_input",
                    "model",
                    "negative_prompt",
                    "wait_for_result",
                    "poll_interval_seconds",
                    "poll_timeout_seconds",
                ],
            },
            "generate_visual_from_appsec_imager": {
                "required": ["prompt"],
                "optional": [
                    "platform",
                    "agents_file",
                    "output_path",
                    "callback_url",
                    "aspect_ratio",
                    "resolution",
                    "output_format",
                    "image_input",
                    "model",
                    "negative_prompt",
                    "wait_for_result",
                    "poll_interval_seconds",
                    "poll_timeout_seconds",
                ],
            },
        },
        "environment": {
            "NANOBANANA_API_KEY": "required",
            "NANOBANANA_BASE_URL": f"optional (default: {DEFAULT_BASE_URL})",
            "NANOBANANA_MODEL": f"optional (default: {DEFAULT_MODEL})",
            "NANOBANANA_CREATE_TASK_PATH": f"optional (default: {DEFAULT_CREATE_TASK_PATH})",
            "NANOBANANA_GET_TASK_PATH": f"optional (default: {DEFAULT_GET_TASK_PATH})",
            "NANOBANANA_TIMEOUT_SECONDS": f"optional (default: {DEFAULT_TIMEOUT})",
            "NANOBANANA_POLL_INTERVAL_SECONDS": f"optional (default: {DEFAULT_POLL_INTERVAL})",
            "NANOBANANA_POLL_TIMEOUT_SECONDS": f"optional (default: {DEFAULT_POLL_TIMEOUT})",
            "NANOBANANA_OUTPUT_DIR": f"optional (default: {DEFAULT_OUTPUT_DIR.as_posix()})",
            "NANOBANANA_AGENTS_FILE": f"optional (default: {DEFAULT_AGENTS_FILE})",
        },
    }


@mcp.tool()
def create_visual_task(
    style: str,
    prompt: str,
    platform: Optional[str] = None,
    callback_url: Optional[str] = None,
    aspect_ratio: str = "1:1",
    resolution: str = "1K",
    output_format: str = "png",
    image_input: Optional[List[str]] = None,
    model: Optional[str] = None,
    negative_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Create NanoBanana job via Kie AI createTask API."""
    if not style.strip():
        raise ValueError("'style' cannot be empty")
    if not prompt.strip():
        raise ValueError("'prompt' cannot be empty")

    output_format = output_format.lower().strip()
    if output_format == "jpg":
        output_format = "jpeg"
    if output_format not in {"png", "jpeg", "webp"}:
        raise ValueError("'output_format' must be one of: png, jpeg, webp")

    api_key = _env_or_fail("NANOBANANA_API_KEY")
    base_url = os.getenv("NANOBANANA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    create_task_path = os.getenv("NANOBANANA_CREATE_TASK_PATH", DEFAULT_CREATE_TASK_PATH)
    selected_model = model or os.getenv("NANOBANANA_MODEL", DEFAULT_MODEL)
    timeout_seconds = int(os.getenv("NANOBANANA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT)))

    final_prompt = _compose_prompt(
        style=style,
        prompt=prompt,
        negative_prompt=negative_prompt,
        platform=platform,
    )

    body: Dict[str, Any] = {
        "model": selected_model,
        "input": {
            "prompt": final_prompt,
            "image_input": image_input or [],
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
        },
    }
    if platform and platform.strip():
        body["input"]["platform"] = platform.strip()
    if callback_url:
        body["callBackUrl"] = callback_url

    raw = _create_task(
        api_key=api_key,
        base_url=base_url,
        path=create_task_path,
        body=body,
        timeout=timeout_seconds,
    )
    task_id = _extract_task_id(raw)

    return {
        "ok": True,
        "task_id": task_id,
        "model": selected_model,
        "platform": platform,
        "output_format": output_format,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "raw": raw,
        "prompt_preview": final_prompt[:300],
    }


@mcp.tool()
def get_visual_task(task_id: str) -> Dict[str, Any]:
    """Fetch NanoBanana job status via Kie AI getTask API."""
    task_id = task_id.strip()
    if not task_id:
        raise ValueError("'task_id' cannot be empty")

    api_key = _env_or_fail("NANOBANANA_API_KEY")
    base_url = os.getenv("NANOBANANA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    get_task_path = os.getenv("NANOBANANA_GET_TASK_PATH", DEFAULT_GET_TASK_PATH)
    timeout_seconds = int(os.getenv("NANOBANANA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT)))

    raw = _get_task(
        api_key=api_key,
        base_url=base_url,
        path=get_task_path,
        task_id=task_id,
        timeout=timeout_seconds,
    )
    status = _extract_status(raw)
    image_url = _extract_image_url(raw)
    has_b64 = _extract_b64(raw) is not None

    return {
        "ok": True,
        "task_id": task_id,
        "status": status,
        "image_url": image_url,
        "has_b64": has_b64,
        "raw": raw,
    }


@mcp.tool()
def generate_visual(
    style: str,
    prompt: str,
    platform: Optional[str] = None,
    output_path: Optional[str] = None,
    callback_url: Optional[str] = None,
    aspect_ratio: str = "1:1",
    resolution: str = "1K",
    output_format: str = "png",
    image_input: Optional[List[str]] = None,
    model: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    wait_for_result: bool = True,
    poll_interval_seconds: Optional[float] = None,
    poll_timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a NanoBanana task and optionally wait for final image file."""
    created = create_visual_task(
        style=style,
        prompt=prompt,
        platform=platform,
        callback_url=callback_url,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        output_format=output_format,
        image_input=image_input,
        model=model,
        negative_prompt=negative_prompt,
    )

    task_id = created.get("task_id")
    if not task_id:
        return {
            "ok": False,
            "error": "Task created but task_id was not found in response",
            "create_response": created,
        }

    if not wait_for_result:
        return {
            "ok": True,
            "task_id": task_id,
            "status": "submitted",
            "platform": platform,
            "create_response": created,
        }

    api_key = _env_or_fail("NANOBANANA_API_KEY")
    base_url = os.getenv("NANOBANANA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    get_task_path = os.getenv("NANOBANANA_GET_TASK_PATH", DEFAULT_GET_TASK_PATH)
    timeout_seconds = int(os.getenv("NANOBANANA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT)))
    interval = poll_interval_seconds if poll_interval_seconds is not None else DEFAULT_POLL_INTERVAL
    max_wait = poll_timeout_seconds if poll_timeout_seconds is not None else DEFAULT_POLL_TIMEOUT

    start = time.time()
    last: Dict[str, Any] = {}

    while time.time() - start <= max_wait:
        last = _get_task(
            api_key=api_key,
            base_url=base_url,
            path=get_task_path,
            task_id=task_id,
            timeout=timeout_seconds,
        )
        status = _extract_status(last)

        if _is_failure(status):
            return {
                "ok": False,
                "task_id": task_id,
                "status": status,
                "create_response": created,
                "task_response": last,
            }

        if _is_success(status):
            break

        time.sleep(max(0.2, float(interval)))
    else:
        raise TimeoutError(f"Task {task_id} did not finish in {max_wait}s")

    image_b64 = _extract_b64(last)
    image_url = _extract_image_url(last)
    if not image_url:
        urls = _extract_result_json_urls(last)
        if urls:
            image_url = urls[0]

    if image_b64:
        image_bytes = base64.b64decode(image_b64)
    elif image_url:
        image_bytes = _http_get_bytes(image_url, headers=_headers(api_key), timeout=timeout_seconds)
    else:
        return {
            "ok": True,
            "task_id": task_id,
            "status": _extract_status(last),
            "warning": "Task completed but no image payload found. Use callback payload or inspect task_response.",
            "create_response": created,
            "task_response": last,
        }

    normalized_format = output_format.lower().strip()
    if normalized_format == "jpg":
        normalized_format = "jpeg"
    if normalized_format not in {"png", "jpeg", "webp"}:
        normalized_format = "png"

    file_path = _resolve_output_path(output_path=output_path, prompt=prompt, image_format=normalized_format)
    file_path.write_bytes(image_bytes)

    mime, _ = mimetypes.guess_type(file_path.name)
    return {
        "ok": True,
        "task_id": task_id,
        "status": _extract_status(last),
        "platform": platform,
        "output_path": file_path.as_posix(),
        "mime_type": mime or "application/octet-stream",
        "bytes": file_path.stat().st_size,
        "image_url": image_url,
        "create_response": created,
        "task_response": last,
    }


@mcp.tool()
def generate_visual_from_appsec_imager(
    prompt: str,
    platform: Optional[str] = None,
    agents_file: Optional[str] = None,
    output_path: Optional[str] = None,
    callback_url: Optional[str] = None,
    aspect_ratio: str = "1:1",
    resolution: str = "1K",
    output_format: str = "png",
    image_input: Optional[List[str]] = None,
    model: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    wait_for_result: bool = True,
    poll_interval_seconds: Optional[float] = None,
    poll_timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate visual using style auto-loaded from role `appsec-imager` in AGENTS.md."""
    if not prompt.strip():
        raise ValueError("'prompt' cannot be empty")

    style = _load_role_style(role_name="appsec-imager", agents_file=agents_file)
    result = generate_visual(
        style=style,
        prompt=prompt,
        platform=platform,
        output_path=output_path,
        callback_url=callback_url,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        output_format=output_format,
        image_input=image_input,
        model=model,
        negative_prompt=negative_prompt,
        wait_for_result=wait_for_result,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
    )
    result["style_source"] = (agents_file or os.getenv("NANOBANANA_AGENTS_FILE", DEFAULT_AGENTS_FILE))
    result["style_role"] = "appsec-imager"
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
