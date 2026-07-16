from __future__ import annotations

from typing import Any


def status(ready: bool = True) -> dict[str, str]:
    return {"status": "ok" if ready else "unavailable"}


def create_app() -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("install the api extra for health endpoints") from exc
    app = FastAPI()
    app.get("/health/live")(lambda: status())
    app.get("/health/ready")(lambda: status())
    app.get("/metrics")(lambda: {"bot_ready": 1})
    return app
