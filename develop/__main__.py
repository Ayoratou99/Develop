"""Point d'entrée ``python -m develop`` : délègue à :func:`develop.cli.main`."""
from __future__ import annotations

import sys

from develop.cli import main

if __name__ == "__main__":
    sys.exit(main())
