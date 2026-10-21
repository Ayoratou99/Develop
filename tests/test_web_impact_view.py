"""Tests de la vue web « Analyse d'impact » (develop/web/static/views/impact.js).

Non-régression : la carte « Changements » affichait le statut brut renvoyé par ImpactAnalyzer
(« selected » pour un fichier ou un symbole choisi, « modified »… pour Git) au lieu d'un libellé
français. Deux niveaux de vérification :

1. un test statique (toujours exécuté) : la table de traduction de la vue couvre tous les statuts
   que l'analyseur peut produire et la pastille n'affiche plus le statut brut ;
2. un test fonctionnel (si Node ≥ 22 est installé) : la vraie vue est montée dans un DOM minimal
   via tests/web_harness/impact_view_harness.mjs, « Analyser » est simulé avec un rapport donné et
   l'on vérifie le texte des pastilles ainsi que le texte envoyé au chat par « Expliquer avec l'IA ».
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from develop.impact_analysis import analyzer as impact_analyzer

ROOT = Path(__file__).resolve().parent.parent
VIEW = ROOT / "develop" / "web" / "static" / "views" / "impact.js"
HARNESS = ROOT / "tests" / "web_harness" / "impact_view_harness.mjs"

# Libellés attendus pour chaque statut que l'analyseur peut renvoyer (valeurs de _STATUS_LABELS + « selected »).
EXPECTED_LABELS = {
    "selected": "sélectionné", "modified": "modifié", "added": "ajouté",
    "deleted": "supprimé", "renamed": "renommé", "untracked": "non suivi",
}


def _node() -> Optional[str]:
    """Chemin de Node si une version avec module.registerHooks (≥ 22.15) est disponible, sinon None."""
    exe = shutil.which("node")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v(\d+)\.(\d+)", out)
    if not m:
        return None
    major, minor = int(m.group(1)), int(m.group(2))
    return exe if (major, minor) >= (22, 15) else None


def _render(report: Dict[str, Any], source: str = "file") -> Dict[str, Any]:
    """Monte la vue dans le banc Node, simule « Analyser » avec `report` et renvoie ce qui est affiché."""
    node = _node()
    if node is None:
        pytest.skip("Node ≥ 22.15 absent")
    proc = subprocess.run([node, str(HARNESS), json.dumps(report), source],
                          capture_output=True, text=True, timeout=60, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stderr, proc.stderr
    return json.loads(proc.stdout)


def _report(changed: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"changed": changed, "affected": [], "endpoints": [], "models": [], "frontend_calls": [],
            "documents": [], "notes": [], "stats": {"changed_files": len(changed)}, "warnings": []}


# --------------------------------------------------------------------------- 1. statique

def test_status_labels_cover_every_analyzer_status():
    """La table STATUS_LABELS de la vue traduit tous les statuts possibles du rapport, et la pastille l'utilise."""
    src = VIEW.read_text(encoding="utf-8")
    m = re.search(r"const STATUS_LABELS = \{(.*?)\};", src, re.S)
    assert m, "table STATUS_LABELS absente de impact.js"
    labels = dict(re.findall(r"(\w+):\s*'([^']*)'", m.group(1)))
    analyzer_statuses = set(impact_analyzer._STATUS_LABELS.values()) | {"selected"}
    assert analyzer_statuses <= set(labels), "statuts non traduits : %s" % sorted(analyzer_statuses - set(labels))
    for status, label in EXPECTED_LABELS.items():
        assert labels[status] == label
    # Le statut brut n'est plus passé tel quel à badge().
    assert "badge(c.status" not in src
    assert "statusBadge(c.status)" in src


# --------------------------------------------------------------------------- 2. fonctionnel (Node)

def test_file_source_shows_selected_in_french():
    """Source « Un fichier » : la carte Changements affiche « sélectionné », jamais le statut brut."""
    out = _render(_report([{"path": "app/Services/OrderService.php", "status": "selected", "repo_id": "back",
                            "file_id": 7, "symbols_changed": ["OrderService", "OrderService.confirm"]}]), "file")
    assert out["errors"] == []
    assert out["requests"][-1][1] == "/api/impact" and out["requests"][-1][2]["source"] == "file"
    assert [c["badge"] for c in out["changes"]] == ["sélectionné"]
    assert out["changes"][0]["badgeClass"] == "badge badge-neutral"
    assert "selected" not in out["changes"][0]["text"]
    # Le texte envoyé au chat par « Expliquer avec l'IA » est traduit lui aussi.
    assert "- app/Services/OrderService.php (sélectionné) : OrderService, OrderService.confirm" in out["prefill"]
    assert "selected" not in out["prefill"] and out["navigated"] == ["#/chat"]


def test_symbol_source_shows_selected_in_french():
    """Source « Un symbole » : même libellé « sélectionné » (statut renvoyé par ImpactAnalyzer.from_symbol)."""
    out = _render(_report([{"path": "app/Services/OrderService.php", "status": "selected", "repo_id": "back",
                            "file_id": 7, "symbols_changed": ["OrderService.confirm"]}]), "symbol")
    assert out["requests"][-1][2] == {"source": "symbol", "max_hops": 3, "symbol_id": 42}
    assert [c["badge"] for c in out["changes"]] == ["sélectionné"]
    assert "(sélectionné) : OrderService.confirm" in out["prefill"]


def test_git_statuses_are_translated_with_their_style():
    """Source Git : chaque statut est traduit et stylé ; un statut inconnu reste lisible, un statut absent vaut « modifié »."""
    changed = [{"path": "%s.php" % s, "status": s} for s in ("modified", "added", "deleted", "renamed", "untracked", "weird")]
    changed.append({"path": "sans_statut.php"})
    out = _render(_report(changed), "git")
    assert [c["badge"] for c in out["changes"]] == ["modifié", "ajouté", "supprimé", "renommé", "non suivi", "weird", "modifié"]
    kinds = [c["badgeClass"].replace("badge ", "") for c in out["changes"]]
    assert kinds == ["badge-accent", "badge-ok", "badge-error", "badge-accent", "badge-warn", "badge-neutral", "badge-accent"]
    assert "- deleted.php (supprimé)" in out["prefill"] and "- sans_statut.php (modifié)" in out["prefill"]
