"""API HTTP locale de Develop (FastAPI) : voir ``docs/API.md``."""
from develop.api.app import create_app  # noqa: F401
from develop.api.state import AppState  # noqa: F401

__all__ = ["AppState", "create_app"]
