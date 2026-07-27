from __future__ import annotations

from collections.abc import Callable
from typing import Any


def status(ready: bool = True) -> dict[str, str]:
    return {"status": "ok" if ready else "unavailable"}


def create_app(status_provider: Callable[[], dict[str, str | bool | None]] | None = None) -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("install the api extra for health endpoints") from exc
    provider = status_provider or status
    app = FastAPI()
    app.get("/health/live")(provider)
    app.get("/health/ready")(provider)
    app.get("/metrics")(lambda: {"bot_ready": int(provider().get("status") == "ok")})
    return app
