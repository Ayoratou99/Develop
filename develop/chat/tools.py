"""Outils locaux exposés au modèle (recherche, lecture bornée, symboles, routes, modèles, notes, impact).

Le modèle peut demander *plus* de contexte, mais jamais *tout* : chaque outil renvoie un résultat
borné (fenêtre de 120 lignes, listes limitées) et tronqué au budget de tokens restant. Un outil ne
lève jamais : toute erreur est renvoyée dans ``ToolResult.error`` avec un message en français.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from develop.chat.budget import estimate_tokens
from develop.chat.context_builder import LocalHit
from develop.llm.base import ToolCall, ToolSpec
from develop.project_scanner import path_is_within
from develop.storage import Database, node_key

logger = logging.getLogger("develop.chat.tools")

READ_WINDOW_MAX_LINES = 120
MAX_FILE_BYTES = 2_000_000
SUMMARY_MAX_CHARS = 200
TRUNCATED_NOTE = "\n… tronqué (budget de contexte atteint)"
SEARCH_LIMIT = 8
LIST_LIMIT = 200

TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="search",
        description="Recherche dans le projet indexé (code, documents, notes) et renvoie les passages les plus pertinents avec leurs sources.",
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "Texte ou identifiants à chercher (nom de fonction, table, route, mots-clés)."},
            "source_types": {"type": "array", "items": {"type": "string"},
                             "description": "Optionnel : limiter à 'file', 'document' et/ou 'note'."},
        }, "required": ["query"]},
    ),
    ToolSpec(
        name="read_file",
        description="Lit une fenêtre d'un fichier d'un dépôt (au plus 120 lignes). Le chemin est relatif à la racine du dépôt ; un suffixe de chemin suffit.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "Chemin (ou fin de chemin) du fichier, ex. app/Services/OrderService.php."},
            "start": {"type": "integer", "description": "Première ligne (1 par défaut)."},
            "end": {"type": "integer", "description": "Dernière ligne (au plus start + 119)."},
        }, "required": ["path"]},
    ),
    ToolSpec(
        name="symbol",
        description="Trouve une fonction, classe, méthode ou composant par son nom et renvoie sa définition (fichier, lignes, signature) avec le code.",
        parameters={"type": "object", "properties": {
            "name": {"type": "string", "description": "Nom du symbole, ex. OrderService ou confirm."},
        }, "required": ["name"]},
    ),
    ToolSpec(
        name="callers",
        description="Liste qui appelle, importe ou utilise un symbole (relations entrantes du graphe du projet).",
        parameters={"type": "object", "properties": {
            "name": {"type": "string", "description": "Nom du symbole dont on veut les appelants."},
        }, "required": ["name"]},
    ),
    ToolSpec(
        name="endpoint",
        description="Retrouve une route serveur et/ou les appels client correspondants à partir d'un chemin d'URL (et éventuellement d'une méthode HTTP).",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "Chemin de la route, ex. /orders/{id}/confirm ou /api/orders."},
            "method": {"type": "string", "description": "Méthode HTTP optionnelle : GET, POST, PUT, PATCH, DELETE."},
        }, "required": ["path"]},
    ),
    ToolSpec(
        name="model",
        description="Décrit un modèle de données (classe, table SQL, migration, interface TypeScript) : champs, types et relations.",
        parameters={"type": "object", "properties": {
            "name": {"type": "string", "description": "Nom du modèle ou de la table, ex. Order ou orders."},
        }, "required": ["name"]},
    ),
    ToolSpec(
        name="notes",
        description="Cherche dans les notes internes datées (décisions, réunions), avec filtre de dates optionnel (AAAA-MM-JJ).",
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "Mots-clés (optionnel : vide = notes les plus récentes)."},
            "date_from": {"type": "string", "description": "Date de début incluse, AAAA-MM-JJ."},
            "date_to": {"type": "string", "description": "Date de fin incluse, AAAA-MM-JJ."},
        }},
    ),
    ToolSpec(
        name="impact",
        description="Analyse d'impact : éléments potentiellement affectés par un changement d'un fichier (path) ou d'un symbole (symbol).",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "Chemin (ou fin de chemin) du fichier modifié."},
            "symbol": {"type": "string", "description": "Nom du symbole modifié (si pas de fichier)."},
        }},
    ),
    ToolSpec(
        name="list_dir",
        description="Liste le contenu indexé d'un dossier d'un dépôt (sous-dossiers et fichiers), pour s'orienter avant de lire.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "Dossier relatif au dépôt ('' ou '.' = racines des dépôts)."},
        }},
    ),
]

TOOL_NAMES = [t.name for t in TOOL_SPECS]


@dataclass
class ToolResult:
    """Résultat d'un outil : résumé court pour l'UI, texte pour le modèle, sources à ajouter."""

    name: str
    summary: str = ""  # français, ≤ 200 caractères (événement tool_result)
    text: str = ""  # contenu renvoyé au modèle, tronqué au budget
    hits: List[Any] = field(default_factory=list)  # hits au format SearchHit (sources ajoutées)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "summary": self.summary, "text": self.text, "error": self.error,
                "hits": len(self.hits)}


# --------------------------------------------------------------------------- utilitaires

def _short(text: Any, limit: int = SUMMARY_MAX_CHARS) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + "…"


def fit_text(text: str, budget_tokens: int, is_code: bool = False) -> str:
    """Tronque ``text`` pour tenir dans ``budget_tokens`` : coupe entre deux lignes, mention « tronqué ».

    Seules des lignes entières sont conservées, quelle que soit la position du dernier retour à la
    ligne dans la fenêtre autorisée (jamais de ligne coupée au milieu, comme
    :func:`develop.chat.context_builder.truncate_block`). Si même la première ligne ne tient pas,
    seule la mention est renvoyée.
    """
    if not text:
        return ""
    budget = max(0, int(budget_tokens or 0))
    if estimate_tokens(text, is_code) <= budget:
        return text
    per_token = 3.5 if is_code else 4.0
    allowed = int(budget * per_token) - len(TRUNCATED_NOTE)
    if allowed <= 0:
        return TRUNCATED_NOTE.strip()
    # dernier retour à la ligne dont le préfixe tient dans la fenêtre (un retour à la ligne situé
    # exactement à la limite convient : le texte qui le précède fait ``allowed`` caractères)
    newline = text.rfind("\n", 0, allowed + 1)
    if newline < 0:
        return TRUNCATED_NOTE.strip()
    cut = text[:newline].rstrip()
    if not cut:
        return TRUNCATED_NOTE.strip()
    return cut + TRUNCATED_NOTE


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def parse_arguments(raw: Any) -> Dict[str, Any]:
    """Arguments d'un appel d'outil : dict, chaîne JSON, ou rien."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except ValueError:
            return {"query": text}
        return data if isinstance(data, dict) else {"value": data}
    return {}


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+", str(value))
        return int(m.group(0)) if m else default


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value).strip()


def _normalize_rel_path(path: str) -> str:
    p = str(path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.strip("/")
    p = re.sub(r"/{2,}", "/", p)
    return p


def _normalize_route(path: str) -> str:
    """Normalisation d'une route, via ``develop.code_analysis`` si disponible."""
    try:
        from develop.code_analysis import normalize_route_path  # import paresseux
        return normalize_route_path(path)
    except Exception:  # noqa: BLE001 - extracteurs en cours d'écriture ailleurs : repli local
        p = re.sub(r"^[a-z][a-z0-9+.-]*://[^/]*", "", str(path or ""), flags=re.IGNORECASE)
        p = p.split("?", 1)[0].split("#", 1)[0]
        p = re.sub(r"\$\{[^}]*\}|\{[^}/]*\}|<[^>/]*>|:[A-Za-z_][A-Za-z0-9_]*\??|\$[A-Za-z_][A-Za-z0-9_]*|\[[^\]/]*\]", ":param", p)
        p = "/".join(":param" if seg.isdigit() else seg for seg in p.split("/"))
        p = re.sub(r"/{2,}", "/", p).rstrip("/").lower()
        return p if p.startswith("/") else "/" + p


def _routes_match(a: str, b: str) -> bool:
    try:
        from develop.code_analysis import route_paths_match  # type: ignore
        return bool(route_paths_match(a, b))
    except Exception:  # noqa: BLE001
        if a == b:
            return True
        return a.endswith(b) or b.endswith(a)


def _row_dict(row: Any) -> Dict[str, Any]:
    try:
        return {k: row[k] for k in row.keys()}
    except AttributeError:
        return dict(row)


# --------------------------------------------------------------------------- exécuteur

class ToolExecutor:
    """Exécute les appels d'outils contre la mémoire Develop (lecture seule des dépôts)."""

    def __init__(self, db: Database, workspace: Any, search_engine: Any = None, impact_analyzer: Any = None,
                 git: Any = None):
        self.db = db
        self.workspace = workspace
        self.search_engine = search_engine
        self.impact_analyzer = impact_analyzer
        self.git = git

    # ----- point d'entrée -----
    async def execute(self, call: ToolCall, budget_tokens: int) -> ToolResult:
        """Exécute ``call`` et renvoie toujours un :class:`ToolResult` (jamais d'exception)."""
        name = str(getattr(call, "name", "") or "").strip()
        args = parse_arguments(getattr(call, "arguments", None))
        budget = max(64, int(budget_tokens or 0))
        handler = getattr(self, "_tool_" + name, None) if name in TOOL_NAMES else None
        if handler is None:
            return ToolResult(name=name or "?", summary="Outil inconnu « %s »." % _short(name, 40),
                              text="Outil inconnu « %s ». Outils disponibles : %s." % (name, ", ".join(TOOL_NAMES)),
                              error="Outil inconnu : %s" % name)
        try:
            result: ToolResult = await handler(args, budget)
        except Exception as exc:  # noqa: BLE001 - un outil ne fait jamais échouer la réponse
            logger.warning("outil %s en erreur: %s", name, exc, exc_info=True)
            message = "Erreur de l'outil %s : %s" % (name, _short(str(exc), 150))
            return ToolResult(name=name, summary=_short(message), text=message, error=str(exc))
        result.name = name
        is_code = name in ("read_file", "symbol")
        result.text = fit_text(result.text or "", budget, is_code=is_code)
        result.summary = _short(result.summary or result.text or "Aucun résultat.")
        return result

    # ----- aides : fichiers -----
    def _repo_root(self, repo_id: Optional[str]) -> Optional[Path]:
        repo_cfg = None
        if self.workspace is not None and hasattr(self.workspace, "repo_by_id"):
            repo_cfg = self.workspace.repo_by_id(repo_id)
        path = getattr(repo_cfg, "path", None)
        if not path:
            row = self.db.get_repo(repo_id) if repo_id else None
            path = getattr(row, "path", None)
        if not path:
            return None
        try:
            return Path(path).expanduser().resolve()
        except OSError:
            return None

    def _repo_name(self, repo_id: Optional[str]) -> str:
        repo_cfg = self.workspace.repo_by_id(repo_id) if (self.workspace is not None and hasattr(self.workspace, "repo_by_id")) else None
        name = getattr(repo_cfg, "name", None)
        if name:
            return str(name)
        row = self.db.get_repo(repo_id) if repo_id else None
        return str(getattr(row, "name", "") or repo_id or "")

    def _repo_names(self) -> Dict[str, str]:
        names: Dict[str, str] = {}
        for repo in getattr(self.workspace, "repos", []) or []:
            names[str(getattr(repo, "name", "")).lower()] = getattr(repo, "id", "")
        for row in self.db.list_repos():
            names.setdefault(str(row.name or "").lower(), row.id)
        return names

    def find_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Fichier indexé correspondant à ``path`` : exact, puis suffixe, puis nom de base."""
        rel = _normalize_rel_path(path)
        if not rel:
            return None
        candidates = [rel]
        # « backend/app/x.php » où backend est le nom d'un dépôt → « app/x.php »
        first, _, rest = rel.partition("/")
        if rest and first.lower() in self._repo_names():
            candidates.append(rest)
        for cand in candidates:
            rows = self.db.query("SELECT * FROM files WHERE path = ? ORDER BY repo_id LIMIT 1", (cand,))
            if rows:
                return _row_dict(rows[0])
        for cand in candidates:
            esc = cand.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = self.db.query("SELECT * FROM files WHERE path LIKE ? ESCAPE '\\' ORDER BY length(path), repo_id, path LIMIT 1",
                                 ("%/" + esc,))
            if rows:
                return _row_dict(rows[0])
        base = os.path.basename(rel)
        if base and base != rel:
            esc = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = self.db.query("SELECT * FROM files WHERE path = ? OR path LIKE ? ESCAPE '\\' ORDER BY length(path), repo_id, path LIMIT 1",
                                 (base, "%/" + esc))
            if rows:
                return _row_dict(rows[0])
        return None

    def read_window(self, file_row: Dict[str, Any], start: Optional[int], end: Optional[int],
                    max_lines: int = READ_WINDOW_MAX_LINES) -> Optional[Dict[str, Any]]:
        """``{path, start, end, text, total_lines}`` d'une fenêtre de fichier (moteur de recherche ou disque)."""
        file_id = _as_int(file_row.get("id"))
        s = max(1, _as_int(start, 1) or 1)
        e = _as_int(end, None)
        if e is None or e < s:
            e = s + max_lines - 1
        if e - s + 1 > max_lines:
            e = s + max_lines - 1
        engine = self.search_engine
        if engine is not None and hasattr(engine, "file_snippet") and file_id is not None:
            try:
                snippet = engine.file_snippet(file_id, s, e, max_lines)
                if inspect.isawaitable(snippet):  # pragma: no cover - tolérance
                    snippet = None
                if isinstance(snippet, dict) and snippet.get("text") is not None:
                    return {"path": snippet.get("path") or file_row.get("path"), "start": int(snippet.get("start", s)),
                            "end": int(snippet.get("end", e)), "text": str(snippet.get("text") or ""),
                            "total_lines": int(snippet.get("total_lines") or 0)}
            except Exception as exc:  # noqa: BLE001
                logger.debug("file_snippet indisponible: %s", exc)
        lines = self._read_lines(file_row)
        if lines is None:
            return None
        total = len(lines)
        if total == 0:
            return {"path": file_row.get("path"), "start": 0, "end": 0, "text": "", "total_lines": 0}
        s = min(s, total)
        e = min(e, total)
        return {"path": file_row.get("path"), "start": s, "end": e, "text": "\n".join(lines[s - 1:e]), "total_lines": total}

    def _read_lines(self, file_row: Dict[str, Any]) -> Optional[List[str]]:
        root = self._repo_root(file_row.get("repo_id"))
        if root is None:
            return None
        rel = _normalize_rel_path(str(file_row.get("path") or ""))
        try:
            full = (root / rel).resolve()
        except OSError:
            return None
        if not path_is_within(str(root), str(full)) or not full.is_file():
            return None  # hors du dépôt (lien, « .. ») : refusé ; comparaison insensible à la casse sous Windows
        try:
            if full.stat().st_size > MAX_FILE_BYTES:
                return None
            data = full.read_bytes()
        except OSError:
            return None
        return data.decode("utf-8", "replace").splitlines()

    def _file_hit(self, file_row: Dict[str, Any], window: Dict[str, Any], reason: str, title: str = "",
                  source_type: str = "file", source_id: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> LocalHit:
        file_id = _as_int(file_row.get("id"))
        ex: Dict[str, Any] = {"language": file_row.get("language") or "", "repo_name": self._repo_name(file_row.get("repo_id"))}
        if extra:
            ex.update(extra)
        return LocalHit(source_type=source_type, source_id=source_id if source_id is not None else file_id,
                        title=title or os.path.basename(str(file_row.get("path") or "")), path=file_row.get("path"),
                        repo_id=file_row.get("repo_id"), file_id=file_id, line_start=window.get("start"),
                        line_end=window.get("end"), text=window.get("text") or "", score=0.0, channels=["tool"],
                        reason=reason, extra=ex)

    def _file_row_by_id(self, file_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if file_id is None:
            return None
        row = self.db.get_file_by_id(int(file_id))
        return row.to_dict() if row is not None else None

    def _node_label(self, key: str) -> str:
        """Libellé lisible d'un noeud du graphe (``symbol:3`` → ``OrderController.confirm (chemin:L30)``)."""
        prefix, _, ident = key.partition(":")
        try:
            if prefix == "symbol":
                sym = self.db.get_symbol(int(ident))
                if sym is None:
                    return key
                f = self._file_row_by_id(sym.file_id)
                name = "%s.%s" % (sym.parent, sym.name) if sym.parent else sym.name
                where = "%s:L%s" % (f.get("path"), sym.line_start) if f else ""
                return "%s (%s)" % (name, where) if where else name
            if prefix == "file":
                f = self._file_row_by_id(int(ident))
                return f.get("path", key) if f else key
            if prefix == "endpoint":
                ep = self.db.get_endpoint(int(ident))
                if ep is None:
                    return key
                f = self._file_row_by_id(ep.file_id)
                return "%s %s [%s] (%s:L%s)" % (ep.method, ep.path, ep.side, f.get("path") if f else "?", ep.line)
            if prefix == "model":
                m = self.db.get_model(int(ident))
                return "modèle %s" % m.name if m else key
            if prefix == "document":
                d = self.db.get_document(int(ident))
                return "document « %s »" % d.title if d else key
            if prefix == "note":
                n = self.db.get_note(int(ident))
                return "note du %s « %s »" % (n.note_date, n.title) if n else key
            if prefix == "table":
                return "table %s" % ident
        except (ValueError, TypeError):
            return key
        return key

    # ----- outils -----
    async def _tool_search(self, args: Dict[str, Any], budget: int) -> ToolResult:
        query = _as_str(args.get("query") or args.get("q") or args.get("text"))
        if not query:
            return ToolResult(name="search", summary="Requête vide.", text="Aucune requête fournie à search.", error="argument query manquant")
        engine = self.search_engine
        hits: List[Any] = []
        if engine is None:
            # repli : recherche lexicale directe dans la mémoire
            for fh in self.db.fts_search(query, limit=SEARCH_LIMIT):
                chunk = self.db.get_chunk(fh.chunk_id)
                if chunk is None:
                    continue
                hits.append(self._hit_from_chunk(chunk, fh.score, "correspondance lexicale (FTS)"))
        else:
            kwargs: Dict[str, Any] = {"limit": SEARCH_LIMIT}
            types = args.get("source_types")
            if isinstance(types, str):
                types = [t.strip() for t in types.split(",") if t.strip()]
            if isinstance(types, (list, tuple)) and types:
                kwargs["source_types"] = list(types)
            embed = getattr(engine, "embed_query", None)
            if callable(embed):
                try:
                    vec = await _maybe_await(embed(query))
                    if vec is not None:
                        kwargs["query_vector"] = vec
                except Exception as exc:  # noqa: BLE001
                    logger.debug("embed_query a échoué: %s", exc)
            result = await _maybe_await(engine.search(query, **kwargs))
            hits = list(getattr(result, "hits", result if isinstance(result, list) else []) or [])
        hits = hits[:SEARCH_LIMIT]
        if not hits:
            return ToolResult(name="search", summary="Aucun résultat pour « %s »." % _short(query, 60),
                              text="Aucun résultat dans le projet pour « %s ». Essayez un autre identifiant ou mot-clé." % query)
        lines = ["%d résultat(s) pour « %s » :" % (len(hits), query)]
        for i, h in enumerate(hits):
            where = _hit_where(h)
            snippet = _short(getattr(h, "snippet", "") or getattr(h, "text", "") or "", 160)
            reason = getattr(h, "reason", "") or ""
            lines.append("%d. %s%s%s" % (i + 1, where, " — " + reason if reason else "", (" : " + snippet) if snippet else ""))
        return ToolResult(name="search", summary="%d résultat(s) pour « %s »." % (len(hits), _short(query, 60)),
                          text="\n".join(lines), hits=hits)

    def _hit_from_chunk(self, chunk: Any, score: float, reason: str) -> LocalHit:
        extra: Dict[str, Any] = {}
        title = ""
        path = None
        repo_id = None
        file_id = None
        if chunk.source_type == "file":
            f = self._file_row_by_id(chunk.source_id)
            if f:
                path, repo_id, file_id = f.get("path"), f.get("repo_id"), f.get("id")
                extra["language"] = f.get("language") or ""
                extra["repo_name"] = self._repo_name(repo_id)
        elif chunk.source_type == "document":
            d = self.db.get_document(int(chunk.source_id))
            title = d.title if d else "document"
            extra["section"] = (chunk.meta or {}).get("section") or (chunk.meta or {}).get("heading")
        elif chunk.source_type == "note":
            n = self.db.get_note(int(chunk.source_id))
            if n:
                title = n.title
                extra["note_date"] = n.note_date
                extra["tags"] = list(n.tags or [])
        page = (chunk.meta or {}).get("page")
        return LocalHit(source_type=chunk.source_type or "file", source_id=chunk.source_id, chunk_id=chunk.id, title=title,
                        path=path, repo_id=repo_id, file_id=file_id, line_start=chunk.line_start, line_end=chunk.line_end,
                        page=_as_int(page), snippet="", text=chunk.text or "", score=float(score), channels=["lexical"],
                        reason=reason, extra=extra)

    async def _tool_read_file(self, args: Dict[str, Any], budget: int) -> ToolResult:
        path = _as_str(args.get("path") or args.get("file") or args.get("filename"))
        if not path:
            return ToolResult(name="read_file", summary="Chemin manquant.", text="Aucun chemin fourni à read_file.", error="argument path manquant")
        file_row = self.find_file(path)
        if file_row is None:
            return ToolResult(name="read_file", summary="Fichier introuvable : %s." % _short(path, 80),
                              text="Fichier introuvable dans l'index : « %s ». Utilisez list_dir ou search pour trouver le bon chemin." % path,
                              error="fichier introuvable")
        start = _as_int(args.get("start") or args.get("line_start"), 1)
        end = _as_int(args.get("end") or args.get("line_end"), None)
        window = self.read_window(file_row, start, end)
        if window is None:
            return ToolResult(name="read_file", summary="Lecture impossible : %s." % _short(file_row.get("path"), 80),
                              text="Impossible de lire « %s » (dépôt introuvable, fichier absent ou trop volumineux)." % file_row.get("path"),
                              error="lecture impossible")
        label = "%s:L%d-%d" % (window["path"], window["start"], window["end"])
        if window["total_lines"]:
            label += " (sur %d lignes)" % window["total_lines"]
        text = "%s\n%s" % (label, window["text"])
        hit = self._file_hit(file_row, window, reason="lu via l'outil read_file")
        return ToolResult(name="read_file", summary="Lecture de %s." % label, text=text, hits=[hit])

    async def _tool_symbol(self, args: Dict[str, Any], budget: int) -> ToolResult:
        name = _as_str(args.get("name") or args.get("symbol") or args.get("query"))
        if not name:
            return ToolResult(name="symbol", summary="Nom manquant.", text="Aucun nom fourni à symbol.", error="argument name manquant")
        rows = self.db.search_symbols(name, limit=10)
        if not rows:
            return ToolResult(name="symbol", summary="Symbole introuvable : %s." % _short(name, 60),
                              text="Aucun symbole nommé « %s » dans l'index. Essayez search ou un nom plus court." % name)
        lines = ["%d symbole(s) pour « %s » :" % (len(rows), name)]
        files: Dict[int, Optional[Dict[str, Any]]] = {}
        for sym in rows:
            if sym.file_id not in files:
                files[sym.file_id] = self._file_row_by_id(sym.file_id)
            f = files[sym.file_id]
            qual = "%s.%s" % (sym.parent, sym.name) if sym.parent else sym.name
            where = "%s:L%s-%s" % (f.get("path") if f else "?", sym.line_start, sym.line_end)
            sig = (" — " + sym.signature) if sym.signature else ""
            lines.append("- %s [%s] %s%s" % (qual, sym.kind, where, sig))
        hits: List[Any] = []
        engine_lookup = getattr(self.search_engine, "lookup_symbol", None) if self.search_engine is not None else None
        if callable(engine_lookup):
            try:
                hits = list(await _maybe_await(engine_lookup(name)) or [])[:3]
            except Exception as exc:  # noqa: BLE001
                logger.debug("lookup_symbol a échoué: %s", exc)
                hits = []
        if not hits:
            for sym in rows[:3]:
                f = files.get(sym.file_id)
                if not f:
                    continue
                window = self.read_window(f, sym.line_start or 1, sym.line_end or sym.line_start)
                if window and window["text"]:
                    qual = "%s.%s" % (sym.parent, sym.name) if sym.parent else sym.name
                    hits.append(self._file_hit(f, window, reason="définition de %s" % qual, title=qual,
                                               source_type="symbol", source_id=sym.id, extra={"symbol": qual, "symbol_id": sym.id}))
        return ToolResult(name="symbol", summary="%d symbole(s) pour « %s »." % (len(rows), _short(name, 60)),
                          text="\n".join(lines), hits=hits)

    async def _tool_callers(self, args: Dict[str, Any], budget: int) -> ToolResult:
        name = _as_str(args.get("name") or args.get("symbol") or args.get("query"))
        if not name:
            return ToolResult(name="callers", summary="Nom manquant.", text="Aucun nom fourni à callers.", error="argument name manquant")
        rows = self.db.search_symbols(name, limit=5)
        if not rows:
            return ToolResult(name="callers", summary="Symbole introuvable : %s." % _short(name, 60),
                              text="Aucun symbole nommé « %s » dans l'index." % name)
        lines: List[str] = []
        caller_keys: List[str] = []
        total = 0
        for sym in rows:
            qual = "%s.%s" % (sym.parent, sym.name) if sym.parent else sym.name
            rels = self.db.relations_to(node_key("symbol", sym.id))
            f = self._file_row_by_id(sym.file_id)
            lines.append("Appelants de %s (%s:L%s) : %d relation(s) entrante(s)" % (qual, f.get("path") if f else "?", sym.line_start, len(rels)))
            for r in rels[:40]:
                lines.append("- %s [%s, confiance %.2f]" % (self._node_label(r.src), r.kind, float(r.confidence or 1.0)))
                if r.src.startswith("symbol:") and r.src not in caller_keys:
                    caller_keys.append(r.src)
            total += len(rels)
        hits: List[Any] = []
        for key in caller_keys[:3]:
            sym = self.db.get_symbol(int(key.split(":", 1)[1]))
            if sym is None:
                continue
            f = self._file_row_by_id(sym.file_id)
            if not f:
                continue
            end = sym.line_end or sym.line_start
            window = self.read_window(f, sym.line_start or 1, end, max_lines=40)
            if window and window["text"]:
                qual = "%s.%s" % (sym.parent, sym.name) if sym.parent else sym.name
                hits.append(self._file_hit(f, window, reason="appelant de %s" % name, title=qual, source_type="symbol",
                                           source_id=sym.id, extra={"symbol": qual, "symbol_id": sym.id}))
        if total == 0:
            lines.append("Aucune relation entrante connue : le symbole n'est peut-être appelé que dynamiquement, ou l'index est incomplet.")
        return ToolResult(name="callers", summary="%d appelant(s) connu(s) pour « %s »." % (total, _short(name, 60)),
                          text="\n".join(lines), hits=hits)

    async def _tool_endpoint(self, args: Dict[str, Any], budget: int) -> ToolResult:
        path = _as_str(args.get("path") or args.get("route") or args.get("url"))
        method = _as_str(args.get("method")).upper()
        if not path:
            return ToolResult(name="endpoint", summary="Chemin manquant.", text="Aucun chemin fourni à endpoint.", error="argument path manquant")
        normalized = _normalize_route(path)
        rows = self.db.find_endpoints(normalized, method or None)
        if not rows:
            rows = [ep for ep in self.db.list_endpoints()
                    if ep.normalized_path and _routes_match(ep.normalized_path, normalized)
                    and (not method or (ep.method or "").upper() in (method, "ANY"))]
        if not rows:
            return ToolResult(name="endpoint", summary="Aucune route pour %s %s." % (method or "", _short(path, 60)),
                              text="Aucune route ni appel client ne correspond à « %s %s » (normalisé : %s)." % (method, path, normalized))
        lines = ["%d correspondance(s) pour %s %s (normalisé : %s) :" % (len(rows), method or "*", path, normalized)]
        hits: List[Any] = []
        for ep in rows[:20]:
            f = self._file_row_by_id(ep.file_id)
            where = "%s:L%s" % (f.get("path") if f else "?", ep.line)
            lines.append("- [%s] %s %s → %s (%s) %s" % (ep.side, ep.method, ep.path, ep.handler or "?", ep.framework or "?", where))
            if f and len(hits) < 4 and ep.line:
                window = self.read_window(f, max(1, int(ep.line) - 8), int(ep.line) + 8, max_lines=20)
                if window and window["text"]:
                    title = "%s %s" % (ep.method, ep.path)
                    hits.append(self._file_hit(f, window, reason="route %s" % ep.side, title=title, source_type="endpoint",
                                               source_id=ep.id, extra={"endpoint_id": ep.id}))
        return ToolResult(name="endpoint", summary="%d route(s)/appel(s) pour %s %s." % (len(rows), method or "*", _short(path, 50)),
                          text="\n".join(lines), hits=hits)

    async def _tool_model(self, args: Dict[str, Any], budget: int) -> ToolResult:
        name = _as_str(args.get("name") or args.get("model") or args.get("table"))
        if not name:
            return ToolResult(name="model", summary="Nom manquant.", text="Aucun nom fourni à model.", error="argument name manquant")
        rows = self.db.find_models(name=name)
        seen = {m.id for m in rows}
        rows += [m for m in self.db.find_models(table_name=name) if m.id not in seen]
        if not rows:
            esc = name.lower().replace("%", "\\%").replace("_", "\\_")
            found = self.db.query("SELECT * FROM data_models WHERE lower(name) LIKE ? ESCAPE '\\' OR lower(table_name) LIKE ? ESCAPE '\\' ORDER BY name LIMIT 5",
                                  ("%" + esc + "%", "%" + esc + "%"))
            ids = [int(r["id"]) for r in found]
            rows = [m for m in (self.db.get_model(i) for i in ids) if m is not None]
        if not rows:
            return ToolResult(name="model", summary="Modèle introuvable : %s." % _short(name, 60),
                              text="Aucun modèle de données nommé « %s » dans l'index." % name)
        texts: List[str] = []
        hits: List[Any] = []
        for m in rows[:5]:
            f = self._file_row_by_id(m.file_id)
            desc = [
                "Modèle %s (%s%s) — %s:L%s" % (m.name, m.kind or "?", (", table " + m.table_name) if m.table_name else "",
                                             f.get("path") if f else "?", m.line),
            ]
            fields = m.fields or []
            if fields:
                desc.append("Champs (%d) :" % len(fields))
                for fld in fields[:60]:
                    if isinstance(fld, dict):
                        extra = []
                        if fld.get("nullable"):
                            extra.append("nullable")
                        if fld.get("default") not in (None, ""):
                            extra.append("défaut %s" % fld.get("default"))
                        meta = fld.get("meta") or {}
                        if isinstance(meta, dict) and meta.get("enum_values"):
                            extra.append("valeurs : %s" % ", ".join(str(v) for v in meta["enum_values"]))
                        desc.append("- %s : %s%s" % (fld.get("name"), fld.get("type") or "?", (" (" + ", ".join(extra) + ")") if extra else ""))
                    else:
                        desc.append("- %s" % fld)
            else:
                desc.append("Champs : non extraits.")
            rels = m.relations or []
            if rels:
                desc.append("Relations (%d) :" % len(rels))
                for rel in rels[:30]:
                    if isinstance(rel, dict):
                        desc.append("- %s → %s%s" % (rel.get("kind"), rel.get("target"), (" (" + str(rel.get("field")) + ")") if rel.get("field") else ""))
                    else:
                        desc.append("- %s" % rel)
            text = "\n".join(desc)
            texts.append(text)
            hits.append(LocalHit(source_type="model", source_id=m.id, title=m.name, path=f.get("path") if f else None,
                                 repo_id=f.get("repo_id") if f else None, file_id=m.file_id, line_start=m.line, line_end=m.line,
                                 text=text, channels=["tool"], reason="description du modèle %s" % m.name,
                                 extra={"model_id": m.id, "language": "text", "repo_name": self._repo_name(f.get("repo_id")) if f else ""}))
        return ToolResult(name="model", summary="%d modèle(s) pour « %s »." % (len(rows), _short(name, 60)),
                          text="\n\n".join(texts), hits=hits)

    async def _tool_notes(self, args: Dict[str, Any], budget: int) -> ToolResult:
        query = _as_str(args.get("query") or args.get("q"))
        date_from = _as_str(args.get("date_from") or args.get("from")) or None
        date_to = _as_str(args.get("date_to") or args.get("to")) or None
        notes: List[Any] = []
        if query:
            seen: List[int] = []
            for fh in self.db.fts_search(query, limit=30, source_types=["note"]):
                if fh.source_id is not None and fh.source_id not in seen:
                    seen.append(int(fh.source_id))
            for nid in seen:
                n = self.db.get_note(nid)
                if n is None:
                    continue
                if date_from and (n.note_date or "") < date_from:
                    continue
                if date_to and (n.note_date or "") > date_to:
                    continue
                notes.append(n)
            if not notes:
                low = query.lower()
                for n in self.db.list_notes(date_from=date_from, date_to=date_to, limit=200):
                    hay = ("%s %s %s" % (n.title, n.body, " ".join(str(t) for t in (n.tags or [])))).lower()
                    if all(term in hay for term in low.split()):
                        notes.append(n)
        else:
            notes = self.db.list_notes(date_from=date_from, date_to=date_to, limit=20)
        notes = notes[:20]
        if not notes:
            span = " entre %s et %s" % (date_from or "…", date_to or "…") if (date_from or date_to) else ""
            return ToolResult(name="notes", summary="Aucune note trouvée%s." % span,
                              text="Aucune note ne correspond à « %s »%s." % (query or "*", span))
        lines = ["%d note(s)%s :" % (len(notes), (" pour « %s »" % query) if query else "")]
        hits: List[Any] = []
        for n in notes:
            tags = " [%s]" % ", ".join(str(t) for t in n.tags) if n.tags else ""
            lines.append("- Note du %s — %s%s : %s" % (n.note_date, n.title, tags, _short(n.body, 300)))
            if len(hits) < 5:
                hits.append(LocalHit(source_type="note", source_id=n.id, title=n.title, text=(n.body or "")[:2000],
                                     channels=["tool"], reason="note trouvée via l'outil notes",
                                     extra={"note_id": n.id, "note_date": n.note_date, "tags": list(n.tags or [])}))
        return ToolResult(name="notes", summary="%d note(s) trouvée(s)." % len(notes), text="\n".join(lines), hits=hits)

    async def _tool_impact(self, args: Dict[str, Any], budget: int) -> ToolResult:
        if self.impact_analyzer is None:
            return ToolResult(name="impact", summary="Analyse d'impact indisponible.",
                              text="L'analyse d'impact n'est pas disponible dans cette session.", error="analyseur absent")
        path = _as_str(args.get("path") or args.get("file"))
        symbol = _as_str(args.get("symbol") or args.get("name"))
        max_hops = _as_int(args.get("max_hops"), 2) or 2
        report: Any = None
        label = ""
        if path:
            file_row = self.find_file(path)
            if file_row is None:
                return ToolResult(name="impact", summary="Fichier introuvable : %s." % _short(path, 80),
                                  text="Fichier introuvable dans l'index : « %s »." % path, error="fichier introuvable")
            label = str(file_row.get("path"))
            report = await _maybe_await(self.impact_analyzer.from_file(file_row.get("repo_id"), label, max_hops=max_hops))
        elif symbol:
            rows = self.db.search_symbols(symbol, limit=1)
            if not rows:
                return ToolResult(name="impact", summary="Symbole introuvable : %s." % _short(symbol, 60),
                                  text="Aucun symbole nommé « %s » dans l'index." % symbol, error="symbole introuvable")
            label = rows[0].name
            report = await _maybe_await(self.impact_analyzer.from_symbol(rows[0].id, max_hops=max_hops))
        else:
            return ToolResult(name="impact", summary="Argument manquant.", text="Indiquez path ou symbol pour l'analyse d'impact.",
                              error="argument path/symbol manquant")
        data = report.to_dict() if hasattr(report, "to_dict") else (report if isinstance(report, dict) else {})
        text, hits = self._format_impact(label, data)
        affected = len(data.get("affected") or [])
        return ToolResult(name="impact", summary="Impact de %s : %d élément(s) potentiellement affecté(s)." % (_short(label, 60), affected),
                          text=text, hits=hits)

    def _format_impact(self, label: str, data: Dict[str, Any]) -> Tuple[str, List[Any]]:
        lines = ["Analyse d'impact pour %s :" % label]
        changed = data.get("changed") or []
        if changed:
            lines.append("Changé : " + ", ".join(str(c.get("path", c)) if isinstance(c, dict) else str(c) for c in changed[:10]))
        affected = list(data.get("affected") or [])
        affected.sort(key=lambda a: -float((a or {}).get("confidence", 0) or 0) if isinstance(a, dict) else 0)
        hits: List[Any] = []
        if affected:
            lines.append("Éléments potentiellement affectés (%d) :" % len(affected))
            for a in affected[:30]:
                if not isinstance(a, dict):
                    lines.append("- %s" % a)
                    continue
                where = a.get("path") or ""
                if a.get("line"):
                    where += ":L%s" % a.get("line")
                lines.append("- %s %s (%s) — %s [confiance %.2f, %s saut(s)]" % (
                    a.get("type") or "?", a.get("name") or a.get("node") or "?", where, a.get("reason") or "?",
                    float(a.get("confidence") or 0), a.get("hops", "?")))
                if len(hits) < 5 and a.get("path"):
                    hits.append(LocalHit(source_type="file", source_id=a.get("file_id"), title=str(a.get("name") or ""),
                                         path=a.get("path"), repo_id=a.get("repo_id"), file_id=a.get("file_id"),
                                         line_start=a.get("line"), line_end=a.get("line"),
                                         text="%s %s — %s" % (a.get("type") or "", a.get("name") or "", a.get("reason") or ""),
                                         channels=["graph"], reason="impact : %s" % (a.get("reason") or ""),
                                         extra={"language": "text"}))
        else:
            lines.append("Aucun élément affecté trouvé dans le graphe de relations.")
        for key, title in (("endpoints", "Routes"), ("models", "Modèles"), ("frontend_calls", "Appels frontend"),
                           ("documents", "Documents"), ("notes", "Notes")):
            items = data.get(key) or []
            if items:
                lines.append("%s (%d) : %s" % (title, len(items), ", ".join(_impact_item_label(i) for i in items[:15])))
        for w in data.get("warnings") or []:
            lines.append("Avertissement : %s" % w)
        return "\n".join(lines), hits

    async def _tool_list_dir(self, args: Dict[str, Any], budget: int) -> ToolResult:
        raw = _as_str(args.get("path") or args.get("dir") or "")
        rel = _normalize_rel_path(raw)
        if rel == ".":
            rel = ""
        repo_names = self._repo_names()
        repo_filter: Optional[str] = None
        if rel:
            first, _, rest = rel.partition("/")
            if first.lower() in repo_names and (rest or not self.db.query(
                    "SELECT 1 FROM files WHERE path LIKE ? LIMIT 1", (rel + "/%",))):
                repo_filter = repo_names[first.lower()]
                rel = rest
        if rel:
            rows = self.db.query("SELECT repo_id, path, language, size FROM files WHERE path LIKE ? ORDER BY repo_id, path",
                                 (rel + "/%",))
        else:
            rows = self.db.query("SELECT repo_id, path, language, size FROM files ORDER BY repo_id, path")
        if repo_filter:
            rows = [r for r in rows if r["repo_id"] == repo_filter]
        if not rows:
            return ToolResult(name="list_dir", summary="Dossier vide ou inconnu : %s." % (raw or "/"),
                              text="Aucun fichier indexé sous « %s »." % (raw or "/"))
        dirs: Dict[Tuple[str, str], int] = {}
        files: List[str] = []
        prefix_len = len(rel) + 1 if rel else 0
        for r in rows:
            remainder = r["path"][prefix_len:]
            head, sep, _ = remainder.partition("/")
            key = (r["repo_id"], head)
            if sep:
                dirs[key] = dirs.get(key, 0) + 1
            else:
                files.append("%s/%s (%s, %s o)" % (self._repo_name(r["repo_id"]), r["path"], r["language"] or "?", r["size"] or 0))
        lines = ["Contenu de « %s » (%d fichier(s) indexé(s)) :" % (raw or "/", len(rows))]
        for (repo_id, head), count in sorted(dirs.items(), key=lambda kv: (kv[0][0], kv[0][1]))[:LIST_LIMIT]:
            lines.append("- %s/%s%s/ (%d fichier(s))" % (self._repo_name(repo_id), (rel + "/") if rel else "", head, count))
        for f in files[:LIST_LIMIT]:
            lines.append("- " + f)
        if len(dirs) + len(files) > LIST_LIMIT:
            lines.append("… liste limitée à %d entrées." % LIST_LIMIT)
        return ToolResult(name="list_dir", summary="%d dossier(s), %d fichier(s) sous « %s »." % (len(dirs), len(files), raw or "/"),
                          text="\n".join(lines))


def _impact_item_label(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("path") or item.get("title") or item.get("node") or "?")
    return str(item)


def _hit_where(h: Any) -> str:
    source_type = getattr(h, "source_type", "file") or "file"
    if source_type == "document":
        where = "Document « %s »" % (getattr(h, "title", "") or getattr(h, "path", "") or "?")
        page = getattr(h, "page", None)
        return where + (" p.%s" % page if page else "")
    if source_type == "note":
        return "Note « %s »" % (getattr(h, "title", "") or "?")
    where = str(getattr(h, "path", "") or getattr(h, "title", "") or "?")
    start, end = getattr(h, "line_start", None), getattr(h, "line_end", None)
    if start is not None:
        where += ":L%s" % start
        if end is not None and end != start:
            where += "-%s" % end
    return where
