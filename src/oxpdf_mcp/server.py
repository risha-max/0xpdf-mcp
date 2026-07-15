import atexit
import asyncio
import json
import os
import random
import stat
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from threading import BoundedSemaphore, Lock
from uuid import uuid4
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import urljoin, urlparse

import httpx

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class PdfParsingApiClient:
    def __init__(
        self,
        base_url: str,
        api_prefix: str,
        api_key: str | None = None,
        bearer_token: str | None = None,
        timeout_seconds: float = 120.0,
        allowed_file_roots: list[Path] | None = None,
        allowed_api_hosts: list[str] | None = None,
        require_https: bool = False,
        max_batch_files: int = 25,
        max_batch_total_mb: int = 100,
        max_concurrent_heavy_tools: int = 4,
        max_retries: int = 3,
        retry_backoff_base_seconds: float = 0.4,
        response_char_limit: int = 12000,
        max_string_chars: int = 800,
        max_list_items: int = 40,
        max_object_depth: int = 5,
        max_estimated_tokens: int = 3000,
        response_budget_policy: str = "truncate",
        disallow_full_response_mode: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.api_prefix = api_prefix.strip("/")
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self.allowed_file_roots = allowed_file_roots or [Path.cwd().resolve()]
        self.allowed_api_hosts = set(allowed_api_hosts or [])
        self.require_https = require_https
        self.max_batch_files = max_batch_files
        self.max_batch_total_bytes = max_batch_total_mb * 1024 * 1024
        self._heavy_tool_semaphore = BoundedSemaphore(max_concurrent_heavy_tools)
        self.max_retries = max_retries
        self.retry_backoff_base_seconds = retry_backoff_base_seconds
        self.response_char_limit = response_char_limit
        self.max_string_chars = max_string_chars
        self.max_list_items = max_list_items
        self.max_object_depth = max_object_depth
        self.max_estimated_tokens = max_estimated_tokens
        self.response_budget_policy = response_budget_policy
        self.disallow_full_response_mode = disallow_full_response_mode
        self._metrics_lock = Lock()
        self._metrics: dict[str, Any] = {
            "requests_total": 0,
            "requests_failed": 0,
            "retries_total": 0,
            "latency_ms_total": 0.0,
            "latency_ms_max": 0.0,
            "by_path": {},
        }
        self._validate_base_url()
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        self._http_async = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        atexit.register(self.close)

    def close(self) -> None:
        self._http.close()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._http_async.aclose())
        except RuntimeError:
            asyncio.run(self._http_async.aclose())

    async def _to_thread(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, BinaryIO, str]]] | None = None,
    ) -> dict[str, Any]:
        if response_mode not in {"compact", "full"}:
            raise ValueError("response_mode must be 'compact' or 'full'")
        self._ensure_response_mode_allowed(response_mode)
        request_id = str(uuid4())
        started = time.monotonic()
        response: httpx.Response | None = None
        retries_taken = 0
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http_async.request(
                    method=method,
                    url=self._api_url(path),
                    headers={**self._build_headers(), "X-Request-ID": request_id},
                    params=params,
                    json=json_body,
                    files=files,
                )
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    self._record_metrics(
                        path=path,
                        latency_ms=(time.monotonic() - started) * 1000.0,
                        success=False,
                        retries=retries_taken,
                    )
                    raise RuntimeError(
                        f"API request failed for {path}: network/transport error ({exc.__class__.__name__})"
                    ) from exc
                retries_taken += 1
                await self._asleep_backoff(attempt)
                continue
            if response.status_code in {429, 502, 503, 504} and attempt < self.max_retries:
                retries_taken += 1
                await self._asleep_backoff(attempt)
                continue
            break
        assert response is not None
        if response.status_code >= 400:
            self._record_metrics(
                path=path,
                latency_ms=(time.monotonic() - started) * 1000.0,
                success=False,
                retries=retries_taken,
            )
            raise RuntimeError(self._safe_error_message(response, path))
        self._record_metrics(
            path=path,
            latency_ms=(time.monotonic() - started) * 1000.0,
            success=True,
            retries=retries_taken,
        )
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = response.json()
            if isinstance(payload, dict):
                if response_mode == "compact":
                    payload = self._apply_token_budget(
                        payload,
                        request_id=request_id,
                        max_estimated_tokens_override=max_estimated_tokens_override,
                    )
                else:
                    payload.setdefault("_mcp_meta", {})
                    payload["_mcp_meta"]["request_id"] = request_id
                    payload["_mcp_meta"]["token_budget_mode"] = "full"
            return payload
        raw_payload = {"raw_response": response.text}
        if response_mode == "compact":
            return self._apply_token_budget(
                raw_payload,
                request_id=request_id,
                max_estimated_tokens_override=max_estimated_tokens_override,
            )
        raw_payload["_mcp_meta"] = {"request_id": request_id, "token_budget_mode": "full"}
        return raw_payload

    def _validate_base_url(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("PDF_PARSING_API_BASE_URL must use http or https")
        if self.require_https and parsed.scheme != "https":
            raise ValueError("PDF_PARSING_API_BASE_URL must use https when HTTPS is required")
        if self.allowed_api_hosts and (parsed.hostname or "") not in self.allowed_api_hosts:
            raise ValueError("PDF_PARSING_API_BASE_URL host is not in allowed API hosts")

    def _build_headers(self, *, authenticated: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if not authenticated:
            return headers
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _ensure_response_mode_allowed(self, response_mode: str) -> None:
        if response_mode == "full" and self.disallow_full_response_mode:
            raise ValueError(
                "response_mode='full' is disabled for this MCP server "
                "(set PDF_PARSING_MCP_DISALLOW_FULL_RESPONSE_MODE=false to allow)"
            )

    def _record_metrics(self, path: str, latency_ms: float, success: bool, retries: int) -> None:
        with self._metrics_lock:
            self._metrics["requests_total"] += 1
            if not success:
                self._metrics["requests_failed"] += 1
            self._metrics["retries_total"] += retries
            self._metrics["latency_ms_total"] += latency_ms
            self._metrics["latency_ms_max"] = max(self._metrics["latency_ms_max"], latency_ms)
            by_path: dict[str, Any] = self._metrics["by_path"]
            stat = by_path.setdefault(
                path,
                {
                    "count": 0,
                    "failed": 0,
                    "retries": 0,
                    "latency_ms_total": 0.0,
                    "latency_ms_max": 0.0,
                },
            )
            stat["count"] += 1
            if not success:
                stat["failed"] += 1
            stat["retries"] += retries
            stat["latency_ms_total"] += latency_ms
            stat["latency_ms_max"] = max(stat["latency_ms_max"], latency_ms)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def _truncate_for_model(self, value: Any, depth: int = 0) -> Any:
        if depth > self.max_object_depth:
            return {
                "_truncated": True,
                "reason": "max_depth",
                "message": f"Nested object truncated at depth {self.max_object_depth}.",
            }
        if isinstance(value, str):
            if len(value) <= self.max_string_chars:
                return value
            return (
                f"{value[: self.max_string_chars]}...[{len(value) - self.max_string_chars} chars omitted]"
            )
        if isinstance(value, list):
            items = value[: self.max_list_items]
            out = [self._truncate_for_model(v, depth + 1) for v in items]
            if len(value) > self.max_list_items:
                out.append(
                    {
                        "_truncated": True,
                        "reason": "max_list_items",
                        "omitted_items": len(value) - self.max_list_items,
                    }
                )
            return out
        if isinstance(value, dict):
            return {k: self._truncate_for_model(v, depth + 1) for k, v in value.items()}
        return value

    def _apply_token_budget(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        max_estimated_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        if self.response_budget_policy not in {"truncate", "error"}:
            raise ValueError("response_budget_policy must be 'truncate' or 'error'")
        compact = self._truncate_for_model(payload)
        encoded = json.dumps(compact, ensure_ascii=False, default=str)
        truncated_by_chars = False
        budget_tokens = (
            max_estimated_tokens_override
            if max_estimated_tokens_override is not None
            else self.max_estimated_tokens
        )
        estimated_tokens = self._estimate_tokens(encoded)
        if self.response_budget_policy == "error" and (
            len(encoded) > self.response_char_limit or estimated_tokens > budget_tokens
        ):
            raise RuntimeError(
                "MCP response budget exceeded in strict mode. "
                "Use tighter filters/pagination or switch response_mode to compact with truncation."
            )
        if len(encoded) > self.response_char_limit or estimated_tokens > budget_tokens:
            truncated_by_chars = True
            old_string_chars = self.max_string_chars
            old_list_items = self.max_list_items
            try:
                self.max_string_chars = max(200, old_string_chars // 2)
                self.max_list_items = max(10, old_list_items // 2)
                compact = self._truncate_for_model(payload)
                encoded = json.dumps(compact, ensure_ascii=False, default=str)
                estimated_tokens = self._estimate_tokens(encoded)
            finally:
                self.max_string_chars = old_string_chars
                self.max_list_items = old_list_items
        compact.setdefault("_mcp_meta", {})
        compact["_mcp_meta"].update(
            {
                "request_id": request_id,
                "response_characters": len(encoded),
                "estimated_tokens": estimated_tokens,
                "token_budget_mode": "compact",
                "response_truncated": truncated_by_chars,
                "response_budget_policy": self.response_budget_policy,
                "max_estimated_tokens_applied": budget_tokens,
            }
        )
        return compact

    def get_runtime_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            total = self._metrics["requests_total"]
            avg = (self._metrics["latency_ms_total"] / total) if total else 0.0
            by_path_summary: dict[str, Any] = {}
            for path, stat in self._metrics["by_path"].items():
                count = stat["count"]
                by_path_summary[path] = {
                    "count": count,
                    "failed": stat["failed"],
                    "retries": stat["retries"],
                    "latency_ms_avg": round((stat["latency_ms_total"] / count) if count else 0.0, 3),
                    "latency_ms_max": round(stat["latency_ms_max"], 3),
                }
            return {
                "requests_total": total,
                "requests_failed": self._metrics["requests_failed"],
                "retries_total": self._metrics["retries_total"],
                "latency_ms_avg": round(avg, 3),
                "latency_ms_max": round(self._metrics["latency_ms_max"], 3),
                "by_path": by_path_summary,
            }

    def reset_runtime_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            self._metrics = {
                "requests_total": 0,
                "requests_failed": 0,
                "retries_total": 0,
                "latency_ms_total": 0.0,
                "latency_ms_max": 0.0,
                "by_path": {},
            }
        return {"ok": True}

    def _api_url(self, path: str) -> str:
        clean_path = path.lstrip("/")
        return urljoin(self.base_url, f"{self.api_prefix}/{clean_path}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, BinaryIO, str]]] | None = None,
    ) -> dict[str, Any]:
        if response_mode not in {"compact", "full"}:
            raise ValueError("response_mode must be 'compact' or 'full'")
        self._ensure_response_mode_allowed(response_mode)
        request_id = str(uuid4())
        started = time.monotonic()
        response: httpx.Response | None = None
        retries_taken = 0
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.request(
                    method=method,
                    url=self._api_url(path),
                    headers={**self._build_headers(), "X-Request-ID": request_id},
                    params=params,
                    json=json_body,
                    files=files,
                )
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    self._record_metrics(
                        path=path,
                        latency_ms=(time.monotonic() - started) * 1000.0,
                        success=False,
                        retries=retries_taken,
                    )
                    raise RuntimeError(
                        f"API request failed for {path}: network/transport error ({exc.__class__.__name__})"
                    ) from exc
                retries_taken += 1
                self._sleep_backoff(attempt)
                continue
            if response.status_code in {429, 502, 503, 504} and attempt < self.max_retries:
                retries_taken += 1
                self._sleep_backoff(attempt)
                continue
            break
        assert response is not None
        if response.status_code >= 400:
            self._record_metrics(
                path=path,
                latency_ms=(time.monotonic() - started) * 1000.0,
                success=False,
                retries=retries_taken,
            )
            raise RuntimeError(self._safe_error_message(response, path))
        self._record_metrics(
            path=path,
            latency_ms=(time.monotonic() - started) * 1000.0,
            success=True,
            retries=retries_taken,
        )
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = response.json()
            if isinstance(payload, dict):
                if response_mode == "compact":
                    payload = self._apply_token_budget(
                        payload,
                        request_id=request_id,
                        max_estimated_tokens_override=max_estimated_tokens_override,
                    )
                else:
                    payload.setdefault("_mcp_meta", {})
                    payload["_mcp_meta"]["request_id"] = request_id
                    payload["_mcp_meta"]["token_budget_mode"] = "full"
            return payload
        raw_payload = {"raw_response": response.text}
        if response_mode == "compact":
            return self._apply_token_budget(
                raw_payload,
                request_id=request_id,
                max_estimated_tokens_override=max_estimated_tokens_override,
            )
        raw_payload["_mcp_meta"] = {"request_id": request_id, "token_budget_mode": "full"}
        return raw_payload

    def _sleep_backoff(self, attempt: int) -> None:
        jitter = random.uniform(0.85, 1.15)
        delay = self.retry_backoff_base_seconds * (2**attempt) * jitter
        time.sleep(max(0.05, delay))

    async def _asleep_backoff(self, attempt: int) -> None:
        jitter = random.uniform(0.85, 1.15)
        delay = self.retry_backoff_base_seconds * (2**attempt) * jitter
        await asyncio.sleep(max(0.05, delay))

    def _safe_error_message(self, response: httpx.Response, path: str) -> str:
        detail = "Request failed"
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = response.json()
                detail = str(body.get("detail") or body.get("message") or detail)
            except Exception:
                detail = detail
        if len(detail) > 220:
            detail = f"{detail[:217]}..."
        return f"API request failed ({response.status_code}) for {path}: {detail}"

    def _resolve_pdf_path(self, pdf_path: str) -> Path:
        path = Path(pdf_path).expanduser()
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed for PDF paths: {path}")
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"Path is not a file: {resolved}")
        if resolved.suffix.lower() != ".pdf":
            raise ValueError(f"Only .pdf files are allowed: {resolved}")
        if not any(root == resolved or root in resolved.parents for root in self.allowed_file_roots):
            raise ValueError("Path is outside allowed PDF roots")
        return resolved

    def _open_pdf_readonly(self, path: Path) -> BinaryIO:
        """Open a PDF without following symlinks; re-check path is inside allowed roots."""
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise ValueError(f"Cannot open PDF safely: {path}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ValueError(f"Path is not a regular file: {path}")
            opened_path = path
            proc_fd = f"/proc/self/fd/{fd}"
            if os.path.exists(proc_fd):
                opened_path = Path(os.readlink(proc_fd)).resolve()
            elif path.exists():
                opened_path = path.resolve()
            if not any(
                root == opened_path or root in opened_path.parents
                for root in self.allowed_file_roots
            ):
                raise ValueError("Path is outside allowed PDF roots")
            handle = os.fdopen(fd, "rb")
            fd = -1
            return handle
        finally:
            if fd >= 0:
                os.close(fd)

    def _pdf_file_tuple(self, pdf_path: str, stack: ExitStack) -> tuple[str, BinaryIO, str]:
        path = self._resolve_pdf_path(pdf_path)
        handle = stack.enter_context(self._open_pdf_readonly(path))
        magic = handle.read(5)
        handle.seek(0)
        if magic != b"%PDF-":
            raise ValueError(f"File does not appear to be a valid PDF: {path}")
        return path.name, handle, "application/pdf"

    def parse_pdf_sync(
        self,
        pdf_path: str,
        *,
        response_mode: str = "compact",
        schema_id: str | None = None,
        schema_name: str | None = None,
        schema_template: str | None = None,
        schema_json: dict[str, Any] | None = None,
        pages: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
    ) -> dict[str, Any]:
        if not self._heavy_tool_semaphore.acquire(blocking=False):
            raise RuntimeError("MCP is busy processing other heavy parse tasks. Please retry.")
        try:
            params: dict[str, Any] = {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "schema_template": schema_template,
                "pages": pages,
            }
            params = {k: v for k, v in params.items() if v is not None}
            form_fields: list[tuple[str, tuple[str | None, str, str | None]]] = [
                ("use_ocr", (None, str(use_ocr).lower(), None)),
                ("ocr_engine", (None, ocr_engine, None)),
            ]
            if schema_json is not None:
                form_fields.append(("schema_json", (None, json.dumps(schema_json), None)))
            with ExitStack() as stack:
                filename, file_handle, content_type = self._pdf_file_tuple(pdf_path, stack)
                files: list[tuple[str, tuple[str, BinaryIO, str]]] = [
                    ("file", (filename, file_handle, content_type))
                ]
                return self._request(
                    "POST",
                    "pdf/parse",
                    response_mode=response_mode,
                    params=params,
                    files=files + form_fields,
                )  # type: ignore[arg-type]
        finally:
            self._heavy_tool_semaphore.release()

    def parse_pdf_stream(
        self,
        pdf_path: str,
        *,
        response_mode: str = "compact",
        schema_id: str | None = None,
        schema_name: str | None = None,
        schema_template: str | None = None,
        schema_json: dict[str, Any] | None = None,
        pages: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        max_events: int = 200,
        event_preview_chars: int = 300,
        include_events: bool = False,
    ) -> dict[str, Any]:
        if not self._heavy_tool_semaphore.acquire(blocking=False):
            raise RuntimeError("MCP is busy processing other heavy parse tasks. Please retry.")
        try:
            if max_events < 1:
                raise ValueError("max_events must be >= 1")
            max_events = min(max_events, 1000)
            params: dict[str, Any] = {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "schema_template": schema_template,
                "pages": pages,
            }
            params = {k: v for k, v in params.items() if v is not None}
            form_fields: list[tuple[str, tuple[str | None, str, str | None]]] = [
                ("use_ocr", (None, str(use_ocr).lower(), None)),
                ("ocr_engine", (None, ocr_engine, None)),
            ]
            if schema_json is not None:
                form_fields.append(("schema_json", (None, json.dumps(schema_json), None)))

            with ExitStack() as stack:
                filename, file_handle, content_type = self._pdf_file_tuple(pdf_path, stack)
                files: list[tuple[str, tuple[str, BinaryIO, str]]] = [
                    ("file", (filename, file_handle, content_type))
                ]
                with self._http.stream(
                    "POST",
                    self._api_url("pdf/parse-stream"),
                    headers=self._build_headers(),
                    params=params,
                    files=files + form_fields,  # type: ignore[arg-type]
                ) as response:
                    if response.status_code >= 400:
                        raise RuntimeError(self._safe_error_message(response, "pdf/parse-stream"))
                    summary = _consume_sse_stream(
                        response.iter_lines(),
                        max_events=max_events,
                        include_events=include_events,
                    )
            preview: list[dict[str, Any]] = []
            for evt in summary["tail_events"]:
                data = evt.get("data")
                if isinstance(data, str) and len(data) > event_preview_chars:
                    data = f"{data[:event_preview_chars]}..."
                preview.append({"event": evt.get("event"), "data": data})
            result: dict[str, Any] = {
                "truncated": summary["truncated"],
                "event_count": summary["event_count"],
                "events_by_type": summary["events_by_type"],
                "tail_preview": preview,
            }
            if summary.get("final_event") is not None:
                result["final_event"] = summary["final_event"]
            if include_events:
                result["events"] = summary["events"]
            if response_mode == "compact":
                return self._apply_token_budget(result, request_id=str(uuid4()))
            result.setdefault("_mcp_meta", {})
            result["_mcp_meta"]["token_budget_mode"] = "full"
            return result
        finally:
            self._heavy_tool_semaphore.release()

    def submit_parse_job(
        self,
        pdf_path: str,
        *,
        response_mode: str = "compact",
        schema_id: str | None = None,
        schema_name: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        if not self._heavy_tool_semaphore.acquire(blocking=False):
            raise RuntimeError("MCP is busy processing other heavy parse tasks. Please retry.")
        try:
            params: dict[str, Any] = {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "use_ocr": use_ocr,
                "ocr_engine": ocr_engine,
                "webhook_url": webhook_url,
            }
            params = {k: v for k, v in params.items() if v is not None}
            with ExitStack() as stack:
                filename, file_handle, content_type = self._pdf_file_tuple(pdf_path, stack)
                files: list[tuple[str, tuple[str, BinaryIO, str]]] = [
                    ("file", (filename, file_handle, content_type))
                ]
                return self._request(
                    "POST", "pdf/upload", response_mode=response_mode, params=params, files=files
                )
        finally:
            self._heavy_tool_semaphore.release()

    def get_parse_job_status(
        self, job_id: str, *, response_mode: str = "compact", include_result: bool = False
    ) -> dict[str, Any]:
        response = self._request("GET", f"pdf/status/{job_id}", response_mode=response_mode)
        if include_result:
            return response
        if isinstance(response, dict):
            response.pop("result", None)
            response.setdefault("_mcp_meta", {})
            response["_mcp_meta"]["result_omitted"] = True
        return response

    def submit_parse_batch(
        self,
        pdf_paths: list[str],
        *,
        response_mode: str = "compact",
        schema_id: str | None = None,
        schema_name: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        if not self._heavy_tool_semaphore.acquire(blocking=False):
            raise RuntimeError("MCP is busy processing other heavy parse tasks. Please retry.")
        try:
            if not pdf_paths:
                raise ValueError("pdf_paths cannot be empty")
            if len(pdf_paths) > self.max_batch_files:
                raise ValueError(f"Too many files; maximum is {self.max_batch_files}")
            total_size = 0
            params: dict[str, Any] = {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "use_ocr": use_ocr,
                "ocr_engine": ocr_engine,
                "webhook_url": webhook_url,
            }
            params = {k: v for k, v in params.items() if v is not None}
            with ExitStack() as stack:
                files: list[tuple[str, tuple[str, BinaryIO, str]]] = []
                for pdf_path in pdf_paths:
                    path = self._resolve_pdf_path(pdf_path)
                    total_size += path.stat().st_size
                    if total_size > self.max_batch_total_bytes:
                        raise ValueError("Total batch size exceeds configured MCP safety limit")
                    filename, file_handle, content_type = self._pdf_file_tuple(pdf_path, stack)
                    files.append(("files", (filename, file_handle, content_type)))
                return self._request(
                    "POST", "pdf/batch", response_mode=response_mode, params=params, files=files
                )
        finally:
            self._heavy_tool_semaphore.release()

    def get_parse_batch_status(
        self,
        batch_id: str,
        *,
        response_mode: str = "compact",
        include_items: bool = True,
        item_limit: int = 20,
        include_item_results: bool = False,
    ) -> dict[str, Any]:
        response = self._request("GET", f"pdf/batch/{batch_id}", response_mode=response_mode)
        if not isinstance(response, dict):
            return response
        items = response.get("items")
        if not isinstance(items, list):
            return response
        if not include_items:
            response.pop("items", None)
            response.setdefault("_mcp_meta", {})
            response["_mcp_meta"]["items_omitted"] = True
            return response
        safe_limit = max(1, min(item_limit, 100))
        clipped = items[:safe_limit]
        if not include_item_results:
            for item in clipped:
                if isinstance(item, dict):
                    item.pop("result", None)
        response["items"] = clipped
        response["pagination"] = {
            "total_count": len(items),
            "count": len(clipped),
            "limit": safe_limit,
            "has_more": len(items) > safe_limit,
            "next_offset": safe_limit if len(items) > safe_limit else None,
        }
        return response

    def wait_for_batch_completion(
        self,
        batch_id: str,
        *,
        timeout_seconds: int = 600,
        initial_interval_seconds: float = 1.5,
        max_interval_seconds: float = 15.0,
        jitter_factor: float = 0.2,
    ) -> dict[str, Any]:
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        if initial_interval_seconds <= 0 or max_interval_seconds <= 0:
            raise ValueError("Polling intervals must be > 0")
        started = time.monotonic()
        interval = initial_interval_seconds
        polls = 0
        while True:
            polls += 1
            status = self.get_parse_batch_status(batch_id)
            current = str(status.get("status", "")).lower()
            if current in {"completed", "failed", "partial"}:
                status["polls"] = polls
                status["elapsed_seconds"] = round(time.monotonic() - started, 3)
                return status
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                return {
                    "batch_id": batch_id,
                    "status": "timeout",
                    "elapsed_seconds": round(elapsed, 3),
                    "polls": polls,
                    "last_status": status,
                }
            sleep_for = interval * (1.0 + random.uniform(-jitter_factor, jitter_factor))
            sleep_for = max(0.1, sleep_for)
            time.sleep(sleep_for)
            interval = min(max_interval_seconds, interval * 1.8)

    def wait_for_job_completion(
        self,
        job_id: str,
        *,
        timeout_seconds: int = 300,
        initial_interval_seconds: float = 1.0,
        max_interval_seconds: float = 10.0,
        jitter_factor: float = 0.2,
    ) -> dict[str, Any]:
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        if initial_interval_seconds <= 0 or max_interval_seconds <= 0:
            raise ValueError("Polling intervals must be > 0")
        started = time.monotonic()
        interval = initial_interval_seconds
        polls = 0
        while True:
            polls += 1
            status = self.get_parse_job_status(job_id)
            current = str(status.get("status", "")).lower()
            if current in {"completed", "failed"}:
                status["polls"] = polls
                status["elapsed_seconds"] = round(time.monotonic() - started, 3)
                return status
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                return {
                    "id": job_id,
                    "status": "timeout",
                    "elapsed_seconds": round(elapsed, 3),
                    "polls": polls,
                    "last_status": status,
                }
            sleep_for = interval * (1.0 + random.uniform(-jitter_factor, jitter_factor))
            sleep_for = max(0.1, sleep_for)
            time.sleep(sleep_for)
            interval = min(max_interval_seconds, interval * 1.8)

    async def health_check_async(self) -> dict[str, Any]:
        response = await self._http_async.get(
            urljoin(self.base_url, "health"),
            headers={**self._build_headers(authenticated=False), "X-Request-ID": str(uuid4())},
        )
        if response.status_code >= 400:
            raise RuntimeError(self._safe_error_message(response, "health"))
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return {"raw_response": response.text}

    async def get_pricing_current_async(
        self,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        return await self._arequest(
            "GET",
            "pricing/current",
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
        )

    async def get_runtime_metrics_async(self) -> dict[str, Any]:
        return await self._to_thread(self.get_runtime_metrics)

    async def reset_runtime_metrics_async(self) -> dict[str, Any]:
        return await self._to_thread(self.reset_runtime_metrics)

    async def parse_pdf_sync_async(
        self,
        pdf_path: str,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        schema_id: str | None = None,
        schema_name: str | None = None,
        schema_template: str | None = None,
        schema_json: dict[str, Any] | None = None,
        pages: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
    ) -> dict[str, Any]:
        if not self._heavy_tool_semaphore.acquire(blocking=False):
            raise RuntimeError("MCP is busy processing other heavy parse tasks. Please retry.")
        try:
            params: dict[str, Any] = {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "schema_template": schema_template,
                "pages": pages,
            }
            params = {k: v for k, v in params.items() if v is not None}
            form_fields: list[tuple[str, tuple[str | None, str, str | None]]] = [
                ("use_ocr", (None, str(use_ocr).lower(), None)),
                ("ocr_engine", (None, ocr_engine, None)),
            ]
            if schema_json is not None:
                form_fields.append(("schema_json", (None, json.dumps(schema_json), None)))
            with ExitStack() as stack:
                filename, file_handle, content_type = self._pdf_file_tuple(pdf_path, stack)
                files: list[tuple[str, tuple[str, BinaryIO, str]]] = [
                    ("file", (filename, file_handle, content_type))
                ]
                return await self._arequest(
                    "POST",
                    "pdf/parse",
                    response_mode=response_mode,
                    max_estimated_tokens_override=max_estimated_tokens_override,
                    params=params,
                    files=files + form_fields,
                )  # type: ignore[arg-type]
        finally:
            self._heavy_tool_semaphore.release()

    async def parse_pdf_stream_async(
        self,
        pdf_path: str,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        schema_id: str | None = None,
        schema_name: str | None = None,
        schema_template: str | None = None,
        schema_json: dict[str, Any] | None = None,
        pages: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        max_events: int = 200,
        event_preview_chars: int = 300,
        include_events: bool = False,
    ) -> dict[str, Any]:
        if not self._heavy_tool_semaphore.acquire(blocking=False):
            raise RuntimeError("MCP is busy processing other heavy parse tasks. Please retry.")
        try:
            if max_events < 1:
                raise ValueError("max_events must be >= 1")
            max_events = min(max_events, 1000)
            params: dict[str, Any] = {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "schema_template": schema_template,
                "pages": pages,
            }
            params = {k: v for k, v in params.items() if v is not None}
            form_fields: list[tuple[str, tuple[str | None, str, str | None]]] = [
                ("use_ocr", (None, str(use_ocr).lower(), None)),
                ("ocr_engine", (None, ocr_engine, None)),
            ]
            if schema_json is not None:
                form_fields.append(("schema_json", (None, json.dumps(schema_json), None)))
            request_id = str(uuid4())
            with ExitStack() as stack:
                filename, file_handle, content_type = self._pdf_file_tuple(pdf_path, stack)
                files: list[tuple[str, tuple[str, BinaryIO, str]]] = [
                    ("file", (filename, file_handle, content_type))
                ]
                async with self._http_async.stream(
                    "POST",
                    self._api_url("pdf/parse-stream"),
                    headers={**self._build_headers(), "X-Request-ID": request_id},
                    params=params,
                    files=files + form_fields,  # type: ignore[arg-type]
                ) as response:
                    if response.status_code >= 400:
                        raise RuntimeError(self._safe_error_message(response, "pdf/parse-stream"))
                    summary = await _consume_sse_stream_async(
                        response.aiter_lines(),
                        max_events=max_events,
                        include_events=include_events,
                    )
            preview: list[dict[str, Any]] = []
            for evt in summary["tail_events"]:
                data = evt.get("data")
                if isinstance(data, str) and len(data) > event_preview_chars:
                    data = f"{data[:event_preview_chars]}..."
                preview.append({"event": evt.get("event"), "data": data})
            result: dict[str, Any] = {
                "truncated": summary["truncated"],
                "event_count": summary["event_count"],
                "events_by_type": summary["events_by_type"],
                "tail_preview": preview,
                "_mcp_meta": {"request_id": request_id},
            }
            if summary.get("final_event") is not None:
                result["final_event"] = summary["final_event"]
            if include_events:
                result["events"] = summary["events"]
            if response_mode == "compact":
                return self._apply_token_budget(
                    result,
                    request_id=str(uuid4()),
                    max_estimated_tokens_override=max_estimated_tokens_override,
                )
            result.setdefault("_mcp_meta", {})
            result["_mcp_meta"]["token_budget_mode"] = "full"
            return result
        finally:
            self._heavy_tool_semaphore.release()

    async def submit_parse_job_async(
        self,
        pdf_path: str,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        schema_id: str | None = None,
        schema_name: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        if not self._heavy_tool_semaphore.acquire(blocking=False):
            raise RuntimeError("MCP is busy processing other heavy parse tasks. Please retry.")
        try:
            params: dict[str, Any] = {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "use_ocr": use_ocr,
                "ocr_engine": ocr_engine,
                "webhook_url": webhook_url,
            }
            params = {k: v for k, v in params.items() if v is not None}
            with ExitStack() as stack:
                filename, file_handle, content_type = self._pdf_file_tuple(pdf_path, stack)
                files: list[tuple[str, tuple[str, BinaryIO, str]]] = [
                    ("file", (filename, file_handle, content_type))
                ]
                return await self._arequest(
                    "POST",
                    "pdf/upload",
                    response_mode=response_mode,
                    max_estimated_tokens_override=max_estimated_tokens_override,
                    params=params,
                    files=files,
                )
        finally:
            self._heavy_tool_semaphore.release()

    async def get_parse_job_status_async(
        self,
        job_id: str,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        include_result: bool = False,
    ) -> dict[str, Any]:
        response = await self._arequest(
            "GET",
            f"pdf/status/{job_id}",
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
        )
        if include_result:
            return response
        if isinstance(response, dict):
            response.pop("result", None)
            response.setdefault("_mcp_meta", {})
            response["_mcp_meta"]["result_omitted"] = True
        return response

    async def submit_parse_batch_async(
        self,
        pdf_paths: list[str],
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        schema_id: str | None = None,
        schema_name: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        if not self._heavy_tool_semaphore.acquire(blocking=False):
            raise RuntimeError("MCP is busy processing other heavy parse tasks. Please retry.")
        try:
            if not pdf_paths:
                raise ValueError("pdf_paths cannot be empty")
            if len(pdf_paths) > self.max_batch_files:
                raise ValueError(f"Too many files; maximum is {self.max_batch_files}")
            total_size = 0
            params: dict[str, Any] = {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "use_ocr": use_ocr,
                "ocr_engine": ocr_engine,
                "webhook_url": webhook_url,
            }
            params = {k: v for k, v in params.items() if v is not None}
            with ExitStack() as stack:
                files: list[tuple[str, tuple[str, BinaryIO, str]]] = []
                for pdf_path in pdf_paths:
                    path = self._resolve_pdf_path(pdf_path)
                    total_size += path.stat().st_size
                    if total_size > self.max_batch_total_bytes:
                        raise ValueError("Total batch size exceeds configured MCP safety limit")
                    filename, file_handle, content_type = self._pdf_file_tuple(pdf_path, stack)
                    files.append(("files", (filename, file_handle, content_type)))
                return await self._arequest(
                    "POST",
                    "pdf/batch",
                    response_mode=response_mode,
                    max_estimated_tokens_override=max_estimated_tokens_override,
                    params=params,
                    files=files,
                )
        finally:
            self._heavy_tool_semaphore.release()

    async def get_parse_batch_status_async(
        self,
        batch_id: str,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        include_items: bool = True,
        item_limit: int = 20,
        include_item_results: bool = False,
    ) -> dict[str, Any]:
        response = await self._arequest(
            "GET",
            f"pdf/batch/{batch_id}",
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
        )
        if not isinstance(response, dict):
            return response
        items = response.get("items")
        if not isinstance(items, list):
            return response
        if not include_items:
            response.pop("items", None)
            response.setdefault("_mcp_meta", {})
            response["_mcp_meta"]["items_omitted"] = True
            return response
        safe_limit = max(1, min(item_limit, 100))
        clipped = items[:safe_limit]
        if not include_item_results:
            for item in clipped:
                if isinstance(item, dict):
                    item.pop("result", None)
        response["items"] = clipped
        response["pagination"] = {
            "total_count": len(items),
            "count": len(clipped),
            "limit": safe_limit,
            "has_more": len(items) > safe_limit,
            "next_offset": safe_limit if len(items) > safe_limit else None,
        }
        return response

    async def list_schemas_async(
        self,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        response = await self._arequest(
            "GET",
            "schemas",
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
        )
        if not isinstance(response, dict):
            return response
        schemas = response.get("schemas")
        if not isinstance(schemas, list):
            return response
        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)
        sliced = schemas[safe_offset : safe_offset + safe_limit]
        has_more = safe_offset + safe_limit < len(schemas)
        response["schemas"] = sliced
        response["pagination"] = {
            "total_count": len(schemas),
            "count": len(sliced),
            "offset": safe_offset,
            "limit": safe_limit,
            "has_more": has_more,
            "next_offset": (safe_offset + safe_limit) if has_more else None,
        }
        return response

    async def create_schema_async(
        self,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        name: str,
        schema_def: dict[str, Any],
        is_default: bool = False,
    ) -> dict[str, Any]:
        payload = {"name": name, "schema_def": schema_def, "is_default": is_default}
        return await self._arequest(
            "POST",
            "schemas",
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
            json_body=payload,
        )

    async def generate_schema_from_description_async(
        self,
        *,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        description: str,
        refinement: str | None = None,
        current_schema: dict[str, Any] | None = None,
        selected_text: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"description": description}
        if refinement is not None:
            payload["refinement"] = refinement
        if current_schema is not None:
            payload["current_schema"] = current_schema
        if selected_text is not None:
            payload["selected_text"] = selected_text
        return await self._arequest(
            "POST",
            "schemas/generate",
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
            json_body=payload,
        )

    async def wait_for_job_completion_async(
        self,
        job_id: str,
        *,
        timeout_seconds: int = 300,
        initial_interval_seconds: float = 1.0,
        max_interval_seconds: float = 10.0,
        jitter_factor: float = 0.2,
    ) -> dict[str, Any]:
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        if initial_interval_seconds <= 0 or max_interval_seconds <= 0:
            raise ValueError("Polling intervals must be > 0")
        started = time.monotonic()
        interval = initial_interval_seconds
        polls = 0
        while True:
            polls += 1
            status = await self.get_parse_job_status_async(job_id)
            current = str(status.get("status", "")).lower()
            if current in {"completed", "failed"}:
                status["polls"] = polls
                status["elapsed_seconds"] = round(time.monotonic() - started, 3)
                return status
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                return {
                    "id": job_id,
                    "status": "timeout",
                    "elapsed_seconds": round(elapsed, 3),
                    "polls": polls,
                    "last_status": status,
                }
            sleep_for = interval * (1.0 + random.uniform(-jitter_factor, jitter_factor))
            await asyncio.sleep(max(0.1, sleep_for))
            interval = min(max_interval_seconds, interval * 1.8)

    async def wait_for_batch_completion_async(
        self,
        batch_id: str,
        *,
        timeout_seconds: int = 600,
        initial_interval_seconds: float = 1.5,
        max_interval_seconds: float = 15.0,
        jitter_factor: float = 0.2,
    ) -> dict[str, Any]:
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        if initial_interval_seconds <= 0 or max_interval_seconds <= 0:
            raise ValueError("Polling intervals must be > 0")
        started = time.monotonic()
        interval = initial_interval_seconds
        polls = 0
        while True:
            polls += 1
            status = await self.get_parse_batch_status_async(batch_id)
            current = str(status.get("status", "")).lower()
            if current in {"completed", "failed", "partial"}:
                status["polls"] = polls
                status["elapsed_seconds"] = round(time.monotonic() - started, 3)
                return status
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                return {
                    "batch_id": batch_id,
                    "status": "timeout",
                    "elapsed_seconds": round(elapsed, 3),
                    "polls": polls,
                    "last_status": status,
                }
            sleep_for = interval * (1.0 + random.uniform(-jitter_factor, jitter_factor))
            await asyncio.sleep(max(0.1, sleep_for))
            interval = min(max_interval_seconds, interval * 1.8)

    def list_schemas(
        self, *, response_mode: str = "compact", limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        response = self._request("GET", "schemas", response_mode=response_mode)
        if not isinstance(response, dict):
            return response
        schemas = response.get("schemas")
        if not isinstance(schemas, list):
            return response
        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)
        sliced = schemas[safe_offset : safe_offset + safe_limit]
        has_more = safe_offset + safe_limit < len(schemas)
        response["schemas"] = sliced
        response["pagination"] = {
            "total_count": len(schemas),
            "count": len(sliced),
            "offset": safe_offset,
            "limit": safe_limit,
            "has_more": has_more,
            "next_offset": (safe_offset + safe_limit) if has_more else None,
        }
        return response

    def create_schema(
        self,
        *,
        response_mode: str = "compact",
        name: str,
        schema_def: dict[str, Any],
        is_default: bool = False,
    ) -> dict[str, Any]:
        payload = {"name": name, "schema_def": schema_def, "is_default": is_default}
        return self._request("POST", "schemas", response_mode=response_mode, json_body=payload)

    def generate_schema_from_description(
        self,
        *,
        response_mode: str = "compact",
        description: str,
        refinement: str | None = None,
        current_schema: dict[str, Any] | None = None,
        selected_text: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"description": description}
        if refinement is not None:
            payload["refinement"] = refinement
        if current_schema is not None:
            payload["current_schema"] = current_schema
        if selected_text is not None:
            payload["selected_text"] = selected_text
        return self._request(
            "POST", "schemas/generate", response_mode=response_mode, json_body=payload
        )

    def get_pricing_current(self, *, response_mode: str = "compact") -> dict[str, Any]:
        return self._request("GET", "pricing/current", response_mode=response_mode)

    def health_check(self) -> dict[str, Any]:
        response = self._http.get(
            urljoin(self.base_url, "health"),
            headers=self._build_headers(authenticated=False),
        )
        if response.status_code >= 400:
            raise RuntimeError(self._safe_error_message(response, "health"))
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return {"raw_response": response.text}


def _consume_sse_stream(
    lines: Any, *, max_events: int, include_events: bool
) -> dict[str, Any]:
    events: list[dict[str, Any]] = [] if include_events else []
    tail_events: list[dict[str, Any]] = []
    events_by_type: dict[str, int] = {}
    final_event: dict[str, Any] | None = None
    event_name = "message"
    data_chunks: list[str] = []
    event_count = 0
    truncated = False

    def _record_event(evt: dict[str, Any]) -> None:
        nonlocal event_count, truncated, final_event, tail_events, events, events_by_type
        event_count += 1
        evt_name = str(evt.get("event") or "message")
        events_by_type[evt_name] = events_by_type.get(evt_name, 0) + 1
        if include_events:
            if len(events) < max_events:
                events.append(evt)
            else:
                truncated = True
        tail_events.append(evt)
        if len(tail_events) > 5:
            tail_events = tail_events[-5:]
        if evt_name in {"complete", "completed", "error"}:
            final_event = evt

    def _flush_event() -> None:
        nonlocal data_chunks, event_name
        if not data_chunks:
            event_name = "message"
            return
        joined_data = "\n".join(data_chunks)
        try:
            payload: Any = json.loads(joined_data)
        except json.JSONDecodeError:
            payload = joined_data
        _record_event({"event": event_name, "data": payload})
        event_name = "message"
        data_chunks = []

    for line in lines:
        if line is None:
            continue
        if line == "":
            _flush_event()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_chunks.append(line.split(":", 1)[1].lstrip())
    _flush_event()
    return {
        "events": events,
        "tail_events": tail_events,
        "events_by_type": events_by_type,
        "event_count": event_count,
        "truncated": truncated,
        "final_event": final_event,
    }


async def _consume_sse_stream_async(
    lines: Any, *, max_events: int, include_events: bool
) -> dict[str, Any]:
    events: list[dict[str, Any]] = [] if include_events else []
    tail_events: list[dict[str, Any]] = []
    events_by_type: dict[str, int] = {}
    final_event: dict[str, Any] | None = None
    event_name = "message"
    data_chunks: list[str] = []
    event_count = 0
    truncated = False

    def _record_event(evt: dict[str, Any]) -> None:
        nonlocal event_count, truncated, final_event, tail_events, events, events_by_type
        event_count += 1
        evt_name = str(evt.get("event") or "message")
        events_by_type[evt_name] = events_by_type.get(evt_name, 0) + 1
        if include_events:
            if len(events) < max_events:
                events.append(evt)
            else:
                truncated = True
        tail_events.append(evt)
        if len(tail_events) > 5:
            tail_events = tail_events[-5:]
        if evt_name in {"complete", "completed", "error"}:
            final_event = evt

    def _flush_event() -> None:
        nonlocal data_chunks, event_name
        if not data_chunks:
            event_name = "message"
            return
        joined_data = "\n".join(data_chunks)
        try:
            payload: Any = json.loads(joined_data)
        except json.JSONDecodeError:
            payload = joined_data
        _record_event({"event": event_name, "data": payload})
        event_name = "message"
        data_chunks = []

    async for line in lines:
        if line is None:
            continue
        if line == "":
            _flush_event()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_chunks.append(line.split(":", 1)[1].lstrip())
    _flush_event()
    return {
        "events": events,
        "tail_events": tail_events,
        "events_by_type": events_by_type,
        "event_count": event_count,
        "truncated": truncated,
        "final_event": final_event,
    }


def validate_mcp_startup_configuration(
    *,
    environment: str,
    allowed_api_hosts: list[str],
    require_https: bool,
    allowed_file_roots: list[Path],
) -> None:
    """Fail fast when production MCP is launched with unsafe defaults."""
    env = environment.strip().lower()
    if env in ("production", "prod"):
        if not allowed_api_hosts:
            raise ValueError(
                "PDF_PARSING_ALLOWED_API_HOSTS must be set in production "
                "(comma-separated API hostnames)"
            )
        if not require_https:
            raise ValueError("PDF_PARSING_REQUIRE_HTTPS must be true in production")
        if len(allowed_file_roots) == 1 and allowed_file_roots[0] == Path.cwd().resolve():
            raise ValueError(
                "PDF_PARSING_ALLOWED_FILE_ROOTS must not default to CWD in production; "
                "use a dedicated ingest directory"
            )


def create_mcp_server() -> "FastMCP":
    from mcp.server.fastmcp import FastMCP

    environment = os.getenv("ENVIRONMENT", os.getenv("ENV", ""))
    roots = [
        Path(p).expanduser().resolve()
        for p in os.getenv("PDF_PARSING_ALLOWED_FILE_ROOTS", str(Path.cwd())).split(":")
        if p.strip()
    ]
    base_url = os.getenv("PDF_PARSING_API_BASE_URL", "https://api.0xpdf.io")
    allowed_hosts = [
        host.strip()
        for host in os.getenv("PDF_PARSING_ALLOWED_API_HOSTS", "api.0xpdf.io").split(",")
        if host.strip()
    ]
    require_https_env = os.getenv("PDF_PARSING_REQUIRE_HTTPS")
    if require_https_env is None:
        require_https = base_url.startswith("https://")
    else:
        require_https = require_https_env.lower() == "true"
    try:
        validate_mcp_startup_configuration(
            environment=environment,
            allowed_api_hosts=allowed_hosts,
            require_https=require_https,
            allowed_file_roots=roots,
        )
    except ValueError as exc:
        print(f"MCP startup configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    client = PdfParsingApiClient(
        base_url=base_url,
        api_prefix=os.getenv("PDF_PARSING_API_PREFIX", "api/v1"),
        api_key=os.getenv("PDF_PARSING_API_KEY"),
        bearer_token=os.getenv("PDF_PARSING_BEARER_TOKEN"),
        timeout_seconds=float(os.getenv("PDF_PARSING_API_TIMEOUT_SECONDS", "120")),
        allowed_file_roots=roots,
        allowed_api_hosts=allowed_hosts,
        require_https=require_https,
        disallow_full_response_mode=os.getenv(
            "PDF_PARSING_MCP_DISALLOW_FULL_RESPONSE_MODE", "true"
        ).lower()
        == "true",
        max_batch_files=int(os.getenv("PDF_PARSING_MCP_MAX_BATCH_FILES", "25")),
        max_batch_total_mb=int(os.getenv("PDF_PARSING_MCP_MAX_BATCH_TOTAL_MB", "100")),
        max_concurrent_heavy_tools=int(
            os.getenv("PDF_PARSING_MCP_MAX_CONCURRENT_HEAVY_TOOLS", "4")
        ),
        max_retries=int(os.getenv("PDF_PARSING_MCP_MAX_RETRIES", "3")),
        retry_backoff_base_seconds=float(
            os.getenv("PDF_PARSING_MCP_RETRY_BACKOFF_BASE_SECONDS", "0.4")
        ),
        response_char_limit=int(os.getenv("PDF_PARSING_MCP_RESPONSE_CHAR_LIMIT", "12000")),
        max_string_chars=int(os.getenv("PDF_PARSING_MCP_MAX_STRING_CHARS", "800")),
        max_list_items=int(os.getenv("PDF_PARSING_MCP_MAX_LIST_ITEMS", "40")),
        max_object_depth=int(os.getenv("PDF_PARSING_MCP_MAX_OBJECT_DEPTH", "5")),
        max_estimated_tokens=int(os.getenv("PDF_PARSING_MCP_MAX_ESTIMATED_TOKENS", "3000")),
        response_budget_policy=os.getenv("PDF_PARSING_MCP_RESPONSE_BUDGET_POLICY", "truncate"),
    )
    mcp = FastMCP("0xpdf")

    @mcp.tool(
        description="Health check for backend connectivity. Useful before running parse tools."
    )
    async def health_check() -> dict[str, Any]:
        return await client.health_check_async()

    @mcp.tool(description="Get current billing/pricing status for the authenticated user.")
    async def get_pricing_current(
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        return await client.get_pricing_current_async(
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
        )

    @mcp.tool(description="Get MCP runtime transport metrics for optimization/debugging.")
    async def get_mcp_runtime_metrics() -> dict[str, Any]:
        return await client.get_runtime_metrics_async()

    @mcp.tool(description="Reset MCP runtime transport metrics counters.")
    async def reset_mcp_runtime_metrics() -> dict[str, Any]:
        return await client.reset_runtime_metrics_async()

    @mcp.tool(description="Parse one PDF synchronously through /api/v1/pdf/parse.")
    async def parse_pdf_sync(
        pdf_path: str,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        schema_id: str | None = None,
        schema_name: str | None = None,
        schema_template: str | None = None,
        schema_json: dict[str, Any] | None = None,
        pages: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
    ) -> dict[str, Any]:
        return await client.parse_pdf_sync_async(
            pdf_path=pdf_path,
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
            schema_id=schema_id,
            schema_name=schema_name,
            schema_template=schema_template,
            schema_json=schema_json,
            pages=pages,
            use_ocr=use_ocr,
            ocr_engine=ocr_engine,
        )

    @mcp.tool(description="Parse one PDF as SSE stream through /api/v1/pdf/parse-stream.")
    async def parse_pdf_stream(
        pdf_path: str,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        schema_id: str | None = None,
        schema_name: str | None = None,
        schema_template: str | None = None,
        schema_json: dict[str, Any] | None = None,
        pages: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        max_events: int = 200,
        event_preview_chars: int = 300,
        include_events: bool = False,
    ) -> dict[str, Any]:
        return await client.parse_pdf_stream_async(
            pdf_path=pdf_path,
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
            schema_id=schema_id,
            schema_name=schema_name,
            schema_template=schema_template,
            schema_json=schema_json,
            pages=pages,
            use_ocr=use_ocr,
            ocr_engine=ocr_engine,
            max_events=max_events,
            event_preview_chars=event_preview_chars,
            include_events=include_events,
        )

    @mcp.tool(description="Submit one PDF for async processing via /api/v1/pdf/upload.")
    async def submit_parse_job(
        pdf_path: str,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        schema_id: str | None = None,
        schema_name: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        return await client.submit_parse_job_async(
            pdf_path=pdf_path,
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
            schema_id=schema_id,
            schema_name=schema_name,
            use_ocr=use_ocr,
            ocr_engine=ocr_engine,
            webhook_url=webhook_url,
        )

    @mcp.tool(description="Get status for an async parse job created by submit_parse_job.")
    async def get_parse_job_status(
        job_id: str,
        response_mode: str = "compact",
        include_result: bool = False,
        max_estimated_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        return await client.get_parse_job_status_async(
            job_id=job_id,
            response_mode=response_mode,
            include_result=include_result,
            max_estimated_tokens_override=max_estimated_tokens_override,
        )

    @mcp.tool(
        description=(
            "Poll job status with exponential backoff and jitter until it is completed/failed or times out."
        )
    )
    async def wait_for_job_completion(
        job_id: str,
        timeout_seconds: int = 300,
        initial_interval_seconds: float = 1.0,
        max_interval_seconds: float = 10.0,
        jitter_factor: float = 0.2,
    ) -> dict[str, Any]:
        return await client.wait_for_job_completion_async(
            job_id=job_id,
            timeout_seconds=timeout_seconds,
            initial_interval_seconds=initial_interval_seconds,
            max_interval_seconds=max_interval_seconds,
            jitter_factor=jitter_factor,
        )

    @mcp.tool(description="Submit multiple PDFs for async batch parsing via /api/v1/pdf/batch.")
    async def submit_parse_batch(
        pdf_paths: list[str],
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
        schema_id: str | None = None,
        schema_name: str | None = None,
        use_ocr: bool = False,
        ocr_engine: str = "surya",
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        return await client.submit_parse_batch_async(
            pdf_paths=pdf_paths,
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
            schema_id=schema_id,
            schema_name=schema_name,
            use_ocr=use_ocr,
            ocr_engine=ocr_engine,
            webhook_url=webhook_url,
        )

    @mcp.tool(description="Get status for an async parse batch created by submit_parse_batch.")
    async def get_parse_batch_status(
        batch_id: str,
        response_mode: str = "compact",
        include_items: bool = True,
        item_limit: int = 20,
        include_item_results: bool = False,
        max_estimated_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        return await client.get_parse_batch_status_async(
            batch_id=batch_id,
            response_mode=response_mode,
            include_items=include_items,
            item_limit=item_limit,
            include_item_results=include_item_results,
            max_estimated_tokens_override=max_estimated_tokens_override,
        )

    @mcp.tool(
        description=(
            "Poll batch status with exponential backoff and jitter until it is completed/failed/partial or times out."
        )
    )
    async def wait_for_batch_completion(
        batch_id: str,
        timeout_seconds: int = 600,
        initial_interval_seconds: float = 1.5,
        max_interval_seconds: float = 15.0,
        jitter_factor: float = 0.2,
    ) -> dict[str, Any]:
        return await client.wait_for_batch_completion_async(
            batch_id=batch_id,
            timeout_seconds=timeout_seconds,
            initial_interval_seconds=initial_interval_seconds,
            max_interval_seconds=max_interval_seconds,
            jitter_factor=jitter_factor,
        )

    @mcp.tool(description="List user schemas from /api/v1/schemas.")
    async def list_schemas(
        response_mode: str = "compact",
        limit: int = 20,
        offset: int = 0,
        max_estimated_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        return await client.list_schemas_async(
            response_mode=response_mode,
            limit=limit,
            offset=offset,
            max_estimated_tokens_override=max_estimated_tokens_override,
        )

    @mcp.tool(
        description=(
            "Save a JSON schema to your account (POST /api/v1/schemas). "
            "Requires API key. No wallet charge per save; subject to max saved schemas limit. "
            "Use after generate_schema_from_description or for hand-written schemas."
        )
    )
    async def create_schema(
        name: str,
        schema_def: dict[str, Any],
        is_default: bool = False,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        return await client.create_schema_async(
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
            name=name,
            schema_def=schema_def,
            is_default=is_default,
        )

    @mcp.tool(
        description=(
            "AI-generate a JSON schema from a description (POST /api/v1/schemas/generate). "
            "Requires API key, payment method, and prepaid wallet balance. "
            "Debits the same wallet ledger as PDF/OCR parsing on each successful generation. "
            "Pair with create_schema to persist."
        )
    )
    async def generate_schema_from_description(
        description: str,
        refinement: str | None = None,
        current_schema: dict[str, Any] | None = None,
        selected_text: str | None = None,
        response_mode: str = "compact",
        max_estimated_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        return await client.generate_schema_from_description_async(
            response_mode=response_mode,
            max_estimated_tokens_override=max_estimated_tokens_override,
            description=description,
            refinement=refinement,
            current_schema=current_schema,
            selected_text=selected_text,
        )

    return mcp


if __name__ == "__main__":
    create_mcp_server().run()
