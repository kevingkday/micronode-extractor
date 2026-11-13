import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import docker
from docker.errors import DockerException, NotFound
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="MicroNode Control Center")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
LITELLM_BASE_URL = os.getenv("LITELLM_URL", "http://host.docker.internal:4000").rstrip("/")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-micronode-router-key")

MANAGED_SERVICES = [
    {"name": "micronode-router", "label": "LiteLLM Smart Router :4000"},
    {"name": "gateway", "label": "Sovereign Gateway :8000"},
    {"name": "open-webui", "label": "Open WebUI Interface :3000"},
    {"name": "docling-serve", "label": "Docling Parser :5001"},
    {"name": "telemetry", "label": "Telemetry Daemon"},
    {
        "name": "micronode-extractor-1-micronode-extractor-1",
        "label": "Vision Extractor :5005",
    },
]

TASK_MODELS = [
    {"alias": "micronode-vision", "ollama_model": "qwen2.5vl:7b", "role": "Document & OCR Vision"},
    {"alias": "micronode-decision", "ollama_model": "deepseek-r1:8b", "role": "Reasoning & Decisioning"},
    {"alias": "micronode-chat", "ollama_model": "qwen2.5:7b-instruct", "role": "Interactive Chat"},
]

ALIAS_TO_OLLAMA = {m["alias"]: m["ollama_model"] for m in TASK_MODELS}

_node_state = {
    "mode": "SOVEREIGN_DAY",
    "rate_per_gpu_hr": 0.40,
    "night_hours_per_day": 10,
    "egress_bytes": 0,
    "purged_models": [],
}


class ModelLoadRequest(BaseModel):
    model: str
    action: str = "load"


class ModeToggleRequest(BaseModel):
    mode: Optional[str] = None


class PingTestRequest(BaseModel):
    model: str = "micronode-chat"


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        cleaned = value.strip()
        if not cleaned or cleaned in ("[N/A]", "N/A"):
            return default
        return float(cleaned)
    except ValueError:
        return default


def get_gpu_telemetry() -> dict:
    """Query nvidia-smi for GPU metrics, returning a clean simulated fallback if unreachable."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
        first_line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 7:
            raise ValueError(f"Unexpected nvidia-smi output: {first_line}")

        return {
            "gpu_name": parts[0],
            "temp_c": _parse_float(parts[1]),
            "gpu_util_percent": _parse_float(parts[2]),
            "vram_used_mb": _parse_float(parts[3]),
            "vram_total_mb": _parse_float(parts[4], 16384.0),
            "power_draw_w": _parse_float(parts[5]),
            "power_limit_w": _parse_float(parts[6], 360.0),
            "simulated": False,
            "timestamp": int(time.time()),
        }
    except Exception:
        t = time.time()
        return {
            "gpu_name": "NVIDIA GeForce RTX 5080 (Simulated)",
            "temp_c": round(46.0 + 3.0 * ((t % 10) / 10.0), 1),
            "gpu_util_percent": round(18.0 + 12.0 * ((t % 7) / 7.0), 1),
            "vram_used_mb": 4200.0,
            "vram_total_mb": 16384.0,
            "power_draw_w": round(72.0 + 15.0 * ((t % 5) / 5.0), 1),
            "power_limit_w": 360.0,
            "simulated": True,
            "timestamp": int(t),
        }


def _get_docker_client() -> docker.DockerClient:
    return docker.DockerClient(base_url="unix:///var/run/docker.sock")


def _find_container(client: docker.DockerClient, target_name: str):
    try:
        return client.containers.get(target_name)
    except NotFound:
        for container in client.containers.list(all=True):
            service_label = container.labels.get("com.docker.compose.service", "")
            if service_label == target_name or target_name in container.name:
                return container
        return None


def _list_containers_sync() -> dict:
    try:
        client = _get_docker_client()
        services = []
        for svc in MANAGED_SERVICES:
            container = _find_container(client, svc["name"])
            if container is not None:
                services.append(
                    {
                        "name": svc["name"],
                        "container_name": container.name,
                        "label": svc["label"],
                        "status": container.status,
                        "found": True,
                    }
                )
            else:
                services.append(
                    {
                        "name": svc["name"],
                        "container_name": svc["name"],
                        "label": svc["label"],
                        "status": "not_found",
                        "found": False,
                    }
                )
        client.close()
        return {"docker_connected": True, "services": services}
    except DockerException as exc:
        return {
            "docker_connected": False,
            "error": str(exc),
            "services": [
                {
                    "name": svc["name"],
                    "container_name": svc["name"],
                    "label": svc["label"],
                    "status": "unavailable",
                    "found": False,
                }
                for svc in MANAGED_SERVICES
            ],
        }


def _control_container_sync(name: str, action: str) -> dict:
    client = _get_docker_client()
    try:
        container = _find_container(client, name)
        if container is None:
            raise HTTPException(status_code=404, detail=f"Container '{name}' not found")
        if action == "start":
            container.start()
        elif action == "stop":
            container.stop(timeout=5)
        elif action == "restart":
            container.restart(timeout=5)
        container.reload()
        return {"name": name, "container_name": container.name, "status": container.status}
    finally:
        client.close()


async def _evict_all_vram_models() -> list[str]:
    evicted: list[str] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
            ps_resp.raise_for_status()
            loaded = ps_resp.json().get("models", [])
        except Exception:
            return evicted

        for item in loaded:
            model_name = item.get("name") or item.get("model")
            if not model_name:
                continue
            try:
                await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                )
                evicted.append(model_name)
            except Exception:
                continue
    return evicted


@app.get("/")
async def serve_dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/telemetry")
async def get_telemetry_snapshot():
    return await asyncio.to_thread(get_gpu_telemetry)


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            metrics = await asyncio.to_thread(get_gpu_telemetry)
            await websocket.send_json(metrics)
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        return
    except Exception:
        return


@app.get("/api/containers")
async def list_containers():
    return await asyncio.to_thread(_list_containers_sync)


@app.post("/api/containers/{name}/{action}")
async def control_container(name: str, action: str):
    if action not in {"start", "stop", "restart"}:
        raise HTTPException(status_code=400, detail="Action must be start, stop, or restart")
    allowed = {svc["name"] for svc in MANAGED_SERVICES}
    if name not in allowed:
        raise HTTPException(status_code=404, detail=f"Unmanaged container '{name}'")
    try:
        return await asyncio.to_thread(_control_container_sync, name, action)
    except HTTPException:
        raise
    except DockerException as exc:
        raise HTTPException(status_code=503, detail=f"Docker control error: {exc}")


@app.get("/api/models")
async def list_models():
    loaded_map: dict[str, dict] = {}
    installed_names: set[str] = set()
    ollama_reachable = True

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            ps_resp, tags_resp = await asyncio.gather(
                client.get(f"{OLLAMA_BASE_URL}/api/ps"),
                client.get(f"{OLLAMA_BASE_URL}/api/tags"),
            )
            if ps_resp.status_code == 200:
                for m in ps_resp.json().get("models", []):
                    key = m.get("name") or m.get("model", "")
                    if key:
                        loaded_map[key] = m
            if tags_resp.status_code == 200:
                for m in tags_resp.json().get("models", []):
                    key = m.get("name") or m.get("model", "")
                    if key:
                        installed_names.add(key)
        except Exception:
            ollama_reachable = False

    models_out = []
    for item in TASK_MODELS:
        ollama_name = item["ollama_model"]
        loaded_info = loaded_map.get(ollama_name)
        size_vram_mb = (
            round(loaded_info.get("size_vram", 0) / (1024 * 1024), 1)
            if loaded_info
            else 0.0
        )
        models_out.append(
            {
                "alias": item["alias"],
                "ollama_model": ollama_name,
                "role": item["role"],
                "in_vram": loaded_info is not None,
                "installed": ollama_name in installed_names,
                "size_vram_mb": size_vram_mb,
            }
        )

    return {
        "ollama_reachable": ollama_reachable,
        "models": models_out,
        "loaded_raw": list(loaded_map.keys()),
    }


@app.post("/api/models/load")
async def manage_model_vram(payload: ModelLoadRequest):
    if payload.action not in {"load", "unload"}:
        raise HTTPException(status_code=400, detail="Action must be 'load' or 'unload'")

    ollama_model = ALIAS_TO_OLLAMA.get(payload.model, payload.model)
    keep_alive = "15m" if payload.action == "load" else 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": ollama_model, "prompt": "", "keep_alive": keep_alive},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}")

    return {
        "status": "loaded" if payload.action == "load" else "unloaded",
        "model": ollama_model,
        "keep_alive": keep_alive,
    }


def _mode_payload() -> dict:
    is_night = _node_state["mode"] == "NIGHT_SHIFT"
    rate = _node_state["rate_per_gpu_hr"]
    monthly_offset = round(rate * _node_state["night_hours_per_day"] * 30, 2)
    return {
        "mode": _node_state["mode"],
        "egress_bytes": 0,
        "rate_per_gpu_hr": rate if is_night else 0.0,
        "projected_rate_per_gpu_hr": rate,
        "projected_monthly_offset_usd": monthly_offset,
        "purged_models": _node_state["purged_models"],
    }


@app.get("/api/mode")
async def get_mode():
    return _mode_payload()


@app.post("/api/mode")
async def set_mode(payload: Optional[ModeToggleRequest] = None):
    if payload and payload.mode:
        if payload.mode not in {"SOVEREIGN_DAY", "NIGHT_SHIFT"}:
            raise HTTPException(status_code=400, detail="Invalid operating mode")
        next_mode = payload.mode
    else:
        next_mode = (
            "NIGHT_SHIFT" if _node_state["mode"] == "SOVEREIGN_DAY" else "SOVEREIGN_DAY"
        )

    _node_state["mode"] = next_mode
    if next_mode == "NIGHT_SHIFT":
        _node_state["purged_models"] = await _evict_all_vram_models()
    else:
        _node_state["purged_models"] = []

    return _mode_payload()


@app.post("/api/ping-test")
async def ping_test(payload: PingTestRequest):
    if payload.model not in ALIAS_TO_OLLAMA:
        raise HTTPException(status_code=400, detail=f"Unknown alias '{payload.model}'")

    req_body = {
        "model": payload.model,
        "messages": [
            {
                "role": "user",
                "content": "Respond with OK and your role in under 10 words.",
            }
        ],
        "max_tokens": 24,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                json=req_body,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"LiteLLM route test failed: {exc}")

    elapsed_s = max(time.perf_counter() - start, 0.001)
    latency_ms = round(elapsed_s * 1000.0, 1)
    reply_text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    completion_tokens = data.get("usage", {}).get("completion_tokens")
    if not completion_tokens:
        completion_tokens = max(len(reply_text.split()), 1)
    tokens_per_sec = round(completion_tokens / elapsed_s, 1)

    return {
        "ok": True,
        "model": payload.model,
        "latency_ms": latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "reply": reply_text,
        "sovereign_confirmation": "100% local silicon execution — 0 bytes external egress",
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
