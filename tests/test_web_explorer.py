"""Tests de non-régression de l'explorateur web (``develop/web/static/views/explorer.js``).

L'interface est en JavaScript vanille sans étape de build : on vérifie d'abord statiquement que le
tableau « Champs » d'un modèle passe par ``formatFieldValue`` (plus de ``JSON.stringify`` direct qui
affichait « null » et « {} »), puis, lorsque ``node`` est présent sur la machine, on exécute la
fonction extraite du fichier sur les valeurs exactes renvoyées par ``GET /api/models/{id}``.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

EXPLORER_JS = Path(__file__).resolve().parent.parent / "develop" / "web" / "static" / "views" / "explorer.js"
_FUNC_RE = re.compile(r"^export function formatFieldValue\(value\) \{\n.*?\n\}\n", re.M | re.S)


def _source() -> str:
    return EXPLORER_JS.read_text(encoding="utf-8")


def _helper_source() -> str:
    m = _FUNC_RE.search(_source())
    assert m, "formatFieldValue introuvable dans explorer.js"
    return m.group(0).replace("export function", "function", 1)


def test_fields_table_uses_format_field_value():
    """Le tableau « Champs » ne doit plus sérialiser les cellules via typeof/JSON.stringify."""
    src = _source()
    assert "typeof f[c] === 'object' ? JSON.stringify(f[c])" not in src
    assert "cols.map((c) => fieldCell(f[c]))" in src
    assert "const text = formatFieldValue(value);" in src
    # la cellule absente est grisée (classe .muted existante), jamais « null »
    assert "text === '—' ? { class: 'muted' } : null" in src


def test_format_field_value_renders_null_and_empty_as_dash(tmp_path):
    """Régression : nullable/default null et meta {} (ex. Eloquent Order) → « — », pas « null »/« {} »."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node absent : la fonction JS n'est vérifiée que statiquement")
    # Ligne « reference » et « total_amount » telles que renvoyées par l'API pour le modèle Order.
    cases = [
        {"name": "reference", "type": "", "nullable": None, "default": None, "meta": {}},
        {"name": "total_amount", "type": "integer", "nullable": None, "default": None, "meta": {"cast": "integer"}},
        {"name": "email", "type": "string", "nullable": False, "default": "''", "meta": {"unique": True, "length": 120}},
        {"name": "status", "type": "enum", "nullable": True, "default": 0, "meta": {"enum_values": ["new", "paid"]}},
        {"name": "tags", "type": "json", "nullable": True, "default": None, "meta": {"nested": {"a": 1}, "empty": []}},
    ]
    script = tmp_path / "format_field_value.js"
    script.write_text(
        _helper_source()
        + "\nconst rows = JSON.parse(process.argv[2]);\n"
        + "const cols = ['name', 'type', 'nullable', 'default', 'meta'];\n"
        + "process.stdout.write(JSON.stringify(rows.map((f) => cols.map((c) => formatFieldValue(f[c])))));\n"
        + "process.stdout.write('\\n' + JSON.stringify([formatFieldValue(undefined), formatFieldValue([]), formatFieldValue(['a', 'b'])]));\n",
        encoding="utf-8",
    )
    out = subprocess.run([node, str(script), json.dumps(cases)], capture_output=True, text=True, check=True).stdout
    rows_line, extra_line = out.split("\n")
    rows = json.loads(rows_line)
    assert rows[0] == ["reference", "—", "—", "—", "—"]
    assert rows[1] == ["total_amount", "integer", "—", "—", "cast: integer"]
    assert rows[2] == ["email", "string", "non", "''", "unique: oui, length: 120"]
    assert rows[3] == ["status", "enum", "oui", "0", "enum_values: new, paid"]
    assert rows[4] == ["tags", "json", "oui", "—", "nested: a: 1, empty: —"]
    for row in rows:
        assert "null" not in row and "{}" not in row and "undefined" not in row
    assert json.loads(extra_line) == ["—", "—", "a, b"]
