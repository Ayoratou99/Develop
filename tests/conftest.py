"""Fixtures partagées pour les tests de Develop."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_project() -> Path:
    """Chemin du mini-projet d'exemple (frontend + backend Laravel + service Node + docs + SQL)."""
    return FIXTURES / "sample_project"
