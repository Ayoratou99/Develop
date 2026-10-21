"""Explorateur : dépôts, fichiers, symboles, endpoints, modèles de données, voisinage du graphe.

Conventions de forme (celles attendues par ``develop/web/static/views/explorer.js``) :

* un **symbole** porte ``file: {id, path, repo_id, language}`` ;
* un **endpoint** ou un **modèle** porte ``file`` (chemin), ``file_id`` et ``repo_id`` ;
* une **relation** porte ``source`` et ``target`` (``{key, type, label, id, path, repo_id, line}``), sa
  ``description`` en français (``knowledge_graph.linker.describe_relation`` si disponible, sinon une
  description générique) et sa ``confidence``.

Toutes les lectures de contenu se font sur disque, en lecture seule, bornées à
:data:`MAX_CONTENT_LINES` lignes par appel.

Secrets (README « détectés mais jamais lus ni indexés ») : un fichier reconnu par
:func:`develop.project_scanner.is_secret_file` (``.env`` réel, clé privée, certificat…) peut figurer
dans la table ``files`` — l'indexeur l'enregistre sans contenu — mais aucune route ne renvoie jamais
son texte : ``GET /api/files/{id}/content`` répond 404 ``file_unreadable`` et :func:`_read_lines`
(aussi utilisée pour les extraits de symboles) renvoie ``None``. Un gabarit (``.env.example``) reste
lisible.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from fastapi import APIRouter, Query, Request

from develop.api.routes import api_error, file_ref, get_state, repo_root, require_db
from develop.api.state import AppState
from develop.git_tools import GitError, GitRepo, find_git_root, git_available
from develop.project_scanner import is_secret_file, read_text
from develop.storage import Database, EndpointRow, FileRow, ModelRow, RelationRow, SymbolRow, node_key

router = APIRouter(prefix="/api", tags=["explorer"])

MAX_CONTENT_LINES = 2000
SNIPPET_MAX_LINES = 80
MAX_FILE_BYTES = 2_000_000
_LIKE_ESCAPE = {"\\": "\\\\", "%": "\\%", "_": "\\_"}


def _like(value: str) -> str:
    return "%" + "".join(_LIKE_ESCAPE.get(ch, ch) for ch in value) + "%"


# --------------------------------------------------------------------------- aides

def _files_by_ids(db: Database, ids: Iterable[Optional[int]]) -> Dict[int, FileRow]:
    wanted = sorted({int(i) for i in ids if i is not None})
    out: Dict[int, FileRow] = {}
    for i in range(0, len(wanted), 400):
        batch = wanted[i:i + 400]
        for r in db.query("SELECT * FROM files WHERE id IN (%s)" % ",".join("?" * len(batch)), batch):
            f = FileRow.from_row(r)
            out[int(f.id)] = f  # type: ignore[arg-type]
    return out


def _symbol_item(s: SymbolRow, f: Optional[FileRow]) -> Dict[str, Any]:
    return {"id": s.id, "name": s.name, "kind": s.kind, "parent": s.parent, "signature": s.signature,
            "file": file_ref(f), "file_id": s.file_id, "line_start": s.line_start, "line_end": s.line_end,
            "meta": dict(s.meta or {})}


def _endpoint_item(e: EndpointRow, f: Optional[FileRow]) -> Dict[str, Any]:
    return {"id": e.id, "side": e.side, "method": e.method, "path": e.path, "normalized_path": e.normalized_path,
            "handler": e.handler, "framework": e.framework, "file": f.path if f else None, "file_id": e.file_id,
            "repo_id": f.repo_id if f else None, "line": e.line, "meta": dict(e.meta or {})}


def _model_item(m: ModelRow, f: Optional[FileRow]) -> Dict[str, Any]:
    return {"id": m.id, "name": m.name, "table_name": m.table_name, "kind": m.kind, "file": f.path if f else None,
            "file_id": m.file_id, "repo_id": f.repo_id if f else None, "line": m.line,
            "field_count": len(m.fields or [])}


def node_info(db: Database, key: str) -> Dict[str, Any]:
    """``{key, type, label, id, file_id, path, repo_id, line, name}`` d'un noeud du graphe (jamais None)."""
    info: Dict[str, Any] = {"key": key, "type": "unknown", "label": key, "id": None, "file_id": None, "path": None,
                            "repo_id": None, "line": None, "name": key}
    if ":" not in key:
        return info
    prefix, ident = key.split(":", 1)
    info["type"] = prefix
    if prefix in ("table", "module"):
        info["label"] = ("table %s" % ident) if prefix == "table" else ident
        info["name"] = ident
        return info
    try:
        num = int(ident)
    except ValueError:
        return info
    info["id"] = num
    try:
        if prefix == "file":
            f = db.get_file_by_id(num)
            if f is not None:
                info.update(label=f.path, name=f.path, file_id=f.id, path=f.path, repo_id=f.repo_id)
        elif prefix == "symbol":
            s = db.get_symbol(num)
            if s is not None:
                f = db.get_file_by_id(s.file_id) if s.file_id else None
                parent = (s.parent or "").split("\\")[-1]
                label = "%s.%s" % (parent, s.name) if parent else s.name
                info.update(label=label, name=s.name, kind=s.kind, file_id=s.file_id, path=f.path if f else None,
                            repo_id=f.repo_id if f else None, line=s.line_start)
        elif prefix == "endpoint":
            e = db.get_endpoint(num)
            if e is not None:
                f = db.get_file_by_id(e.file_id) if e.file_id else None
                label = ("%s %s" % (e.method or "", e.path or "")).strip()
                info.update(label=("appel " if e.side == "client" else "") + label, name=label, side=e.side,
                            method=e.method, handler=e.handler, file_id=e.file_id, path=f.path if f else None,
                            repo_id=f.repo_id if f else None, line=e.line)
        elif prefix == "model":
            m = db.get_model(num)
            if m is not None:
                f = db.get_file_by_id(m.file_id) if m.file_id else None
                info.update(label=m.name, name=m.name, kind=m.kind, table_name=m.table_name, file_id=m.file_id,
                            path=f.path if f else None, repo_id=f.repo_id if f else None, line=m.line)
        elif prefix == "document":
            d = db.get_document(num)
            if d is not None:
                info.update(label=d.title or d.source_path or key, name=d.title or key, path=d.source_path)
        elif prefix == "note":
            n = db.get_note(num)
            if n is not None:
                label = n.title or key
                if n.note_date:
                    label = "%s (%s)" % (label, n.note_date)
                info.update(label=label, name=n.title or key, note_date=n.note_date)
    except Exception:  # noqa: BLE001 - un noeud orphelin ne casse pas la réponse
        pass
    return info


def describe_relation(rel: RelationRow, db: Database, dst_info: Optional[Dict[str, Any]] = None) -> str:
    """Description française d'une relation (``knowledge_graph.linker`` si présent, sinon générique)."""
    try:
        from develop.knowledge_graph.linker import describe_relation as _describe
    except Exception:  # noqa: BLE001 - module optionnel
        _describe = None  # type: ignore[assignment]
    if _describe is not None:
        try:
            return _describe(rel, db)
        except Exception:  # noqa: BLE001
            pass
    label = (dst_info or node_info(db, rel.dst))["label"]
    conf = float(rel.confidence) if rel.confidence is not None else 1.0
    return "%s → %s%s" % (rel.kind or "relation", label, " (%.2g)" % conf if conf < 0.999 else "")


def _node_file(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not info.get("file_id"):
        return None
    return {"id": info["file_id"], "path": info.get("path"), "repo_id": info.get("repo_id")}


def relation_item(rel: RelationRow, db: Database) -> Dict[str, Any]:
    src, dst = node_info(db, rel.src), node_info(db, rel.dst)
    return {"id": rel.id, "src": rel.src, "dst": rel.dst, "kind": rel.kind, "confidence": rel.confidence,
            "meta": dict(rel.meta or {}), "source": src, "target": dst, "src_file": _node_file(src),
            "dst_file": _node_file(dst), "line": dst.get("line"), "description": describe_relation(rel, db, dst)}


def _entity_neighbours(db: Database, key: str) -> List[Tuple[str, float]]:
    """Noeuds de la même entité qu'un modèle, avec la confiance du chemin, triés par confiance décroissante.

    Un modèle (``model:<id>``) désigne la même entité que la table qu'il définit (``defines_table`` →
    ``table:<nom>``), que les modèles reliés par ``same_entity`` (migration ↔ Eloquent ↔ table SQL ↔
    interface TypeScript, dans les deux sens) et que les tables définies par ces derniers. La confiance
    d'un chemin à deux sauts est le minimum des deux relations.
    """
    hops: Dict[str, float] = {}

    def offer(node: str, conf: Optional[float]) -> float:
        value = float(conf) if conf is not None else 1.0
        if node != key and (node not in hops or value > hops[node]):
            hops[node] = value
        return value

    for rel in db.relations_from(key, "defines_table"):
        if rel.dst.startswith("table:"):
            offer(rel.dst, rel.confidence)
    for rel in db.relations_from(key, "same_entity") + db.relations_to(key, "same_entity"):
        other = rel.dst if rel.src == key else rel.src
        if other == key or not other.startswith("model:"):
            continue
        conf = offer(other, rel.confidence)
        for sub in db.relations_from(other, "defines_table"):
            if sub.dst.startswith("table:"):
                offer(sub.dst, min(conf, float(sub.confidence) if sub.confidence is not None else 1.0))
    return sorted(hops.items(), key=lambda item: (-item[1], item[0]))


def _documents_mentioning(db: Database, key: str) -> List[Dict[str, Any]]:
    """Documents (et notes) liés à un noeud par ``mentions`` / ``documented_by``.

    Pour un modèle (``model:<id>``), les documents reliés aux autres noeuds de la **même entité**
    (:func:`_entity_neighbours` : table définie, modèles ``same_entity`` et leurs tables) sont ajoutés :
    le dictionnaire de données qui documente la table ``orders`` apparaît ainsi sur la migration, le
    modèle Eloquent ``Order`` et l'interface TypeScript ``Order``, comme dans « Éléments liés ». Chaque
    élément porte ``via`` : ``None`` quand le document cite le noeud lui-même, sinon ``{node, type, label}``
    du noeud intermédiaire ; la confiance d'un document indirect est le minimum entre celle de la relation
    documentaire et celle du chemin, et une relation directe l'emporte toujours sur un détour.
    """
    found: Dict[str, Dict[str, Any]] = {}

    def collect(node: str, hop: Optional[float], via: Optional[str]) -> None:
        for rel in db.relations_to(node):
            if rel.src.startswith(("document:", "note:")):
                _offer_document(found, rel.src, rel, hop, via)
        for rel in db.relations_from(node):
            if rel.dst.startswith(("document:", "note:")):
                _offer_document(found, rel.dst, rel, hop, via)

    collect(key, None, None)
    if key.startswith("model:"):
        for node, hop in _entity_neighbours(db, key):
            collect(node, hop, node)
    out: List[Dict[str, Any]] = []
    for node, extra in found.items():
        info = node_info(db, node)
        item = {"id": info["id"], "kind": info["type"], "title": info["label"], "node": node}
        if info["type"] == "document":
            item["document_id"] = info["id"]
        else:
            item["note_id"] = info["id"]
        via = extra.pop("via", None)
        item.update(extra)
        if via is None:
            item["via"] = None
        else:
            via_info = node_info(db, via)
            item["via"] = {"node": via, "type": via_info["type"], "label": via_info["label"]}
        out.append(item)
    out.sort(key=lambda d: (d["kind"], str(d["title"]).lower()))
    return out


def _offer_document(found: Dict[str, Dict[str, Any]], node: str, rel: RelationRow, hop: Optional[float],
                    via: Optional[str]) -> None:
    """Retient un document pour un noeud : une relation directe (``via`` None) l'emporte sur un détour,
    et entre deux détours le chemin le plus confiant (``min(relation, chemin)``) est conservé."""
    conf = float(rel.confidence) if rel.confidence is not None else 1.0
    if hop is not None:
        conf = min(conf, hop)
    current = found.get(node)
    if current is not None:
        if current["via"] is None:
            return  # le document cite le noeud lui-même : rien ne remplace cette relation
        if via is not None and conf <= current["confidence"]:
            return
    found[node] = {"relation": rel.kind, "confidence": conf, "via": via}


def _read_lines(state: AppState, f: FileRow) -> Optional[List[str]]:
    """Lignes du fichier sur disque ; ``None`` s'il est illisible, hors dépôt… ou s'il contient des secrets."""
    if is_secret_file(f.path):
        return None  # .env réel, clé, certificat : jamais lu, quel que soit l'appelant
    root = repo_root(state, f.repo_id)
    if root is None:
        return None
    abs_path = os.path.join(root, *f.path.split("/"))
    try:
        real_root = os.path.realpath(root)
        real_path = os.path.realpath(abs_path)
    except OSError:
        return None
    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        return None  # lien symbolique sortant du dépôt : refusé
    if not os.path.isfile(abs_path):
        return None
    text = read_text(abs_path, max_bytes=MAX_FILE_BYTES)
    if text is None:
        return None
    return text.splitlines()


def _git_info(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isdir(path) or not git_available():
        return None
    root = find_git_root(path)
    if root is None:
        return None
    try:
        repo = GitRepo(str(root))
        rel = os.path.relpath(os.path.realpath(path), os.path.realpath(str(root))).replace(os.sep, "/")
        changes = repo.status()
        if rel in (".", ""):
            dirty = bool(changes)
        else:
            dirty = any(c.path == rel or c.path.startswith(rel + "/") for c in changes)
        return {"branch": repo.current_branch(), "dirty": dirty, "root": str(root), "head": repo.head()}
    except (GitError, OSError):
        return None


# --------------------------------------------------------------------------- dépôts

@router.get("/repos")
def list_repos(request: Request) -> List[Dict[str, Any]]:
    """``[{...RepoConfig, counts:{files,symbols,endpoints,models}, head_sha, last_indexed_at, git:{branch,dirty}}]``."""
    state = get_state(request)
    db = require_db(state)
    ws = state.workspace
    assert ws is not None
    rows = {r.id: r for r in db.list_repos()}
    counts: Dict[str, Dict[str, int]] = {}

    def fill(key: str, sql: str) -> None:
        for r in db.query(sql):
            counts.setdefault(r["repo_id"], {})[key] = int(r["n"])

    fill("files", "SELECT repo_id, count(*) AS n FROM files GROUP BY repo_id")
    fill("symbols", "SELECT f.repo_id AS repo_id, count(*) AS n FROM symbols s JOIN files f ON f.id = s.file_id GROUP BY f.repo_id")
    fill("endpoints", "SELECT f.repo_id AS repo_id, count(*) AS n FROM endpoints e JOIN files f ON f.id = e.file_id GROUP BY f.repo_id")
    fill("models", "SELECT f.repo_id AS repo_id, count(*) AS n FROM data_models m JOIN files f ON f.id = m.file_id GROUP BY f.repo_id")
    out: List[Dict[str, Any]] = []
    for repo in ws.repos:
        row = rows.get(repo.id)
        d = repo.model_dump()
        c = counts.get(repo.id, {})
        d["counts"] = {k: c.get(k, 0) for k in ("files", "symbols", "endpoints", "models")}
        d["head_sha"] = row.head_sha if row else None
        d["last_indexed_at"] = row.last_indexed_at if row else None
        d["exists"] = os.path.isdir(repo.path)
        d["git"] = _git_info(repo.path)
        out.append(d)
    return out


# --------------------------------------------------------------------------- fichiers

@router.get("/files")
def list_files(request: Request, repo_id: Optional[str] = None, q: Optional[str] = None,
               language: Optional[str] = None, kind: Optional[str] = None,
               limit: int = Query(default=5000, ge=1, le=100000), offset: int = Query(default=0, ge=0)) -> List[Dict[str, Any]]:
    """Liste plate ``[{id, repo_id, path, language, kind, size}]`` (l'interface construit l'arbre)."""
    db = require_db(get_state(request))
    where: List[str] = []
    params: List[Any] = []
    for col, val in (("repo_id", repo_id), ("language", language), ("kind", kind)):
        if val:
            where.append("%s = ?" % col)
            params.append(val)
    if q and q.strip():
        where.append("path LIKE ? ESCAPE '\\'")
        params.append(_like(q.strip()))
    sql = "SELECT id, repo_id, path, language, kind, size FROM files"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY repo_id, path LIMIT ? OFFSET ?"
    params += [limit, offset]
    return [dict(r) for r in db.query(sql, params)]


def _file_or_404(db: Database, file_id: int) -> FileRow:
    f = db.get_file_by_id(file_id)
    if f is None:
        raise api_error(404, "file_not_found", "Fichier %s introuvable dans l'index." % file_id)
    return f


@router.get("/files/{file_id}")
def get_file(file_id: int, request: Request) -> Dict[str, Any]:
    """Fichier + ``symbols[]``, ``endpoints[]``, ``models[]``, ``imports[]``, ``relations{out[],in[]}``, ``documents_mentioning[]``."""
    db = require_db(get_state(request))
    f = _file_or_404(db, file_id)
    key = node_key("file", file_id)
    data = f.to_dict()
    data["symbols"] = [_symbol_item(s, f) for s in db.list_symbols(file_id=file_id, limit=10000)]
    data["endpoints"] = [_endpoint_item(EndpointRow.from_row(r), f)
                         for r in db.query("SELECT * FROM endpoints WHERE file_id = ? ORDER BY line, id", (file_id,))]
    data["models"] = [_model_item(ModelRow.from_row(r), f)
                      for r in db.query("SELECT * FROM data_models WHERE file_id = ? ORDER BY line, id", (file_id,))]
    rels_out = db.relations_from(key)
    rels_in = db.relations_to(key)
    imports: List[Dict[str, Any]] = []
    resolved_modules: Set[str] = set()
    for rel in rels_out:
        if rel.kind != "imports":
            continue
        target = node_info(db, rel.dst)
        modules = [str(m) for m in (rel.meta or {}).get("modules") or []]
        resolved_modules.update(modules)
        imports.append({"module": ", ".join(modules) or target["label"], "modules": modules, "resolved": True,
                        "resolved_file_id": target["file_id"], "resolved_path": target["path"],
                        "repo_id": target["repo_id"], "confidence": rel.confidence,
                        "label": "%s → %s" % (", ".join(modules) or "?", target["label"])})
    for rel in rels_out:
        if rel.kind != "imports_raw":
            continue
        module = rel.dst[len("module:"):] if rel.dst.startswith("module:") else rel.dst
        if module in resolved_modules:
            continue
        imports.append({"module": module, "modules": [module], "resolved": False, "resolved_file_id": None,
                        "resolved_path": None, "repo_id": None, "confidence": rel.confidence, "label": module})
    data["imports"] = imports
    data["relations"] = {
        "out": [relation_item(r, db) for r in rels_out if r.kind != "imports_raw"],
        "in": [relation_item(r, db) for r in rels_in],
    }
    data["documents_mentioning"] = _documents_mentioning(db, key)
    return data


@router.get("/files/{file_id}/content")
def get_file_content(file_id: int, request: Request, start: int = Query(default=1, ge=1),
                     end: Optional[int] = Query(default=None, ge=1)) -> Dict[str, Any]:
    """``{path, language, start, end, total_lines, text}`` : lecture disque, 1-based inclusif, ≤ 2000 lignes.

    404 ``file_unreadable`` si le fichier est illisible **ou** s'il contient des secrets (``.env`` réel, clé,
    certificat) : son contenu n'est jamais renvoyé, même s'il est présent sur le disque.
    """
    state = get_state(request)
    db = require_db(state)
    f = _file_or_404(db, file_id)
    if is_secret_file(f.path):
        raise api_error(404, "file_unreadable",
                        "Le fichier %s contient des secrets (variables d'environnement, clé ou certificat) : "
                        "Develop le détecte mais ne lit jamais son contenu." % f.path)
    lines = _read_lines(state, f)
    if lines is None:
        raise api_error(404, "file_unreadable",
                        "Le fichier %s n'est pas lisible sur le disque (déplacé, binaire ou dépôt absent) : relancez l'indexation." % f.path)
    total = len(lines)
    s = max(1, start)
    e = end if end is not None else s + MAX_CONTENT_LINES - 1
    e = min(e, s + MAX_CONTENT_LINES - 1, total)
    if total == 0 or s > total:
        return {"id": f.id, "repo_id": f.repo_id, "path": f.path, "language": f.language, "start": s, "end": s - 1,
                "total_lines": total, "text": ""}
    return {"id": f.id, "repo_id": f.repo_id, "path": f.path, "language": f.language, "start": s, "end": e,
            "total_lines": total, "text": "\n".join(lines[s - 1:e])}


# --------------------------------------------------------------------------- symboles

@router.get("/symbols")
def list_symbols(request: Request, q: Optional[str] = None, kind: Optional[str] = None, repo_id: Optional[str] = None,
                 limit: int = Query(default=50, ge=1, le=5000)) -> List[Dict[str, Any]]:
    """``[{id, name, kind, parent, signature, file:{id,path,repo_id}, line_start, line_end}]``."""
    db = require_db(get_state(request))
    if q and q.strip():
        rows = db.search_symbols(q.strip(), limit=limit * 4 if (kind or repo_id) else limit)
    else:
        rows = db.list_symbols(kind=kind or None, limit=limit * 4 if repo_id else limit)
    files = _files_by_ids(db, (s.file_id for s in rows))
    out: List[Dict[str, Any]] = []
    for s in rows:
        f = files.get(int(s.file_id)) if s.file_id is not None else None
        if kind and s.kind != kind:
            continue
        if repo_id and (f is None or f.repo_id != repo_id):
            continue
        out.append(_symbol_item(s, f))
        if len(out) >= limit:
            break
    return out


@router.get("/symbols/{symbol_id}")
def get_symbol(symbol_id: int, request: Request) -> Dict[str, Any]:
    """Symbole + fichier + ``callers[]`` (relations entrantes) + ``uses[]`` (sortantes) + ``snippet``."""
    state = get_state(request)
    db = require_db(state)
    s = db.get_symbol(symbol_id)
    if s is None:
        raise api_error(404, "symbol_not_found", "Symbole %s introuvable dans l'index." % symbol_id)
    f = db.get_file_by_id(s.file_id) if s.file_id else None
    key = node_key("symbol", symbol_id)

    def related(rel: RelationRow, other: str) -> Dict[str, Any]:
        info = node_info(db, other)
        fid = info.get("file_id")
        fr = db.get_file_by_id(fid) if fid else None
        item: Dict[str, Any] = {"node": other, "type": info["type"], "name": info["label"], "kind": info.get("kind") or info["type"],
                                "file": file_ref(fr), "line": info.get("line"), "relation": rel.kind,
                                "confidence": rel.confidence, "description": describe_relation(rel, db)}
        if info["type"] == "symbol":
            item["symbol_id"] = info["id"]
        elif info["type"] == "endpoint":
            item["endpoint_id"] = info["id"]
        elif info["type"] == "model":
            item["model_id"] = info["id"]
        return item

    data = _symbol_item(s, f)
    data["file"] = file_ref(f)
    data["repo_id"] = f.repo_id if f else None
    data["path"] = f.path if f else None
    data["callers"] = [related(r, r.src) for r in db.relations_to(key)]
    data["uses"] = [related(r, r.dst) for r in db.relations_from(key)]
    data["endpoints"] = [node_info(db, r.src) for r in db.relations_to(key, "handles")]
    snippet = None
    if f is not None and s.line_start:
        lines = _read_lines(state, f)
        if lines:
            start = max(1, int(s.line_start))
            end = min(int(s.line_end or start), start + SNIPPET_MAX_LINES - 1, len(lines))
            snippet = "\n".join(lines[start - 1:end])
            data["snippet_start"] = start
            data["snippet_end"] = end
    data["snippet"] = snippet
    return data


# --------------------------------------------------------------------------- endpoints

@router.get("/endpoints")
def list_endpoints(request: Request, side: Optional[str] = None, q: Optional[str] = None,
                   limit: int = Query(default=1000, ge=1, le=20000)) -> List[Dict[str, Any]]:
    """Routes serveur et appels client, avec ``matches[]`` (relations ``calls_endpoint`` dans les deux sens)."""
    db = require_db(get_state(request))
    if side and side not in ("server", "client"):
        raise api_error(422, "validation_error", "« side » doit valoir server ou client.")
    all_eps = db.list_endpoints()
    by_id = {int(e.id): e for e in all_eps if e.id is not None}
    files = _files_by_ids(db, (e.file_id for e in all_eps))
    matches: Dict[int, List[RelationRow]] = {}
    for r in db.query("SELECT * FROM relations WHERE kind = 'calls_endpoint' ORDER BY confidence DESC, id"):
        rel = RelationRow.from_row(r)
        for end in (rel.src, rel.dst):
            if end.startswith("endpoint:"):
                try:
                    matches.setdefault(int(end.split(":", 1)[1]), []).append(rel)
                except ValueError:
                    continue
    needle = (q or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for e in all_eps:
        if side and e.side != side:
            continue
        f = files.get(int(e.file_id)) if e.file_id is not None else None
        if needle:
            hay = " ".join(str(x or "") for x in (e.method, e.path, e.normalized_path, e.handler, e.framework,
                                                  f.path if f else "")).lower()
            if needle not in hay:
                continue
        item = _endpoint_item(e, f)
        found: List[Dict[str, Any]] = []
        for rel in matches.get(int(e.id), []):  # type: ignore[arg-type]
            other_key = rel.dst if rel.src == node_key("endpoint", e.id) else rel.src
            try:
                other = by_id.get(int(other_key.split(":", 1)[1]))
            except ValueError:
                other = None
            if other is None:
                continue
            of = files.get(int(other.file_id)) if other.file_id is not None else None
            m = _endpoint_item(other, of)
            m["endpoint_id"] = other.id
            m["confidence"] = rel.confidence
            found.append(m)
        item["matches"] = found
        out.append(item)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- modèles

@router.get("/models")
def list_models(request: Request, q: Optional[str] = None,
                limit: int = Query(default=1000, ge=1, le=20000)) -> List[Dict[str, Any]]:
    """``[{id, name, table_name, kind, file, line, field_count}]``."""
    db = require_db(get_state(request))
    rows = db.list_models()
    files = _files_by_ids(db, (m.file_id for m in rows))
    needle = (q or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for m in rows:
        if needle and needle not in ("%s %s %s" % (m.name, m.table_name or "", m.kind or "")).lower():
            continue
        out.append(_model_item(m, files.get(int(m.file_id)) if m.file_id is not None else None))
        if len(out) >= limit:
            break
    return out


@router.get("/models/{model_id}")
def get_model(model_id: int, request: Request) -> Dict[str, Any]:
    """Modèle + ``fields[]``, ``relations[]``, ``linked[]`` (same_entity), ``endpoints_using[]``, ``documents_mentioning[]``.

    ``documents_mentioning`` couvre toute l'entité (table définie, modèles ``same_entity`` et leurs tables) :
    chaque élément porte ``via`` (``None`` si le document cite ce modèle, sinon le noeud intermédiaire).
    """
    db = require_db(get_state(request))
    m = db.get_model(model_id)
    if m is None:
        raise api_error(404, "model_not_found", "Modèle de données %s introuvable dans l'index." % model_id)
    f = db.get_file_by_id(m.file_id) if m.file_id else None
    key = node_key("model", model_id)
    data = _model_item(m, f)
    data["fields"] = list(m.fields or [])
    data["relations"] = list(m.relations or [])
    linked: Dict[str, Dict[str, Any]] = {}
    for rel in db.relations_from(key, "same_entity") + db.relations_to(key, "same_entity"):
        other = rel.dst if rel.src == key else rel.src
        if other == key or not other.startswith("model:"):
            continue
        info = node_info(db, other)
        linked.setdefault(other, {"model_id": info["id"], "kind": info.get("kind"), "name": info["label"],
                                  "table_name": info.get("table_name"), "file": info["path"], "file_id": info["file_id"],
                                  "repo_id": info["repo_id"], "line": info["line"], "confidence": rel.confidence})
    for rel in db.relations_from(key, "defines_table"):
        info = node_info(db, rel.dst)
        linked.setdefault(rel.dst, {"model_id": None, "kind": "table", "name": info["label"], "table_name": info["name"],
                                    "file": None, "file_id": None, "repo_id": None, "line": None,
                                    "confidence": rel.confidence})
    data["linked"] = list(linked.values())
    # routes reliées directement (``endpoint:E`` → modèle, via leur contrôleur), puis routes serveur des fichiers
    # qui utilisent le modèle (gestionnaires définis dans le fichier des routes) ; jamais deux fois la même
    uses = db.relations_to(key, "uses_model")
    using_endpoints = {int(r.src.split(":", 1)[1]) for r in uses
                       if r.src.startswith("endpoint:") and r.src.split(":", 1)[1].isdigit()}
    using_files = {int(r.src.split(":", 1)[1]) for r in uses
                   if r.src.startswith("file:") and r.src.split(":", 1)[1].isdigit()}
    endpoints_using: List[Dict[str, Any]] = []
    seen_endpoints: Set[int] = set()
    if using_endpoints:
        ph = ",".join("?" * len(using_endpoints))
        rows = db.query("SELECT * FROM endpoints WHERE id IN (%s) AND side = 'server' ORDER BY file_id, line, id" % ph,
                        sorted(using_endpoints))
        files = _files_by_ids(db, {r["file_id"] for r in rows})
        for r in rows:
            seen_endpoints.add(int(r["id"]))
            endpoints_using.append(_endpoint_item(EndpointRow.from_row(r), files.get(r["file_id"])))
    if using_files:
        files = _files_by_ids(db, using_files)
        for fid in sorted(using_files):
            for r in db.query("SELECT * FROM endpoints WHERE file_id = ? AND side = 'server' ORDER BY line, id", (fid,)):
                if int(r["id"]) in seen_endpoints:
                    continue
                seen_endpoints.add(int(r["id"]))
                endpoints_using.append(_endpoint_item(EndpointRow.from_row(r), files.get(fid)))
    data["endpoints_using"] = endpoints_using
    data["documents_mentioning"] = _documents_mentioning(db, key)
    return data


# --------------------------------------------------------------------------- graphe

@router.get("/graph/neighbors")
def graph_neighbors(request: Request, node: str = Query(..., min_length=3),
                    depth: int = Query(default=1, ge=0, le=4),
                    limit: int = Query(default=200, ge=1, le=5000)) -> Dict[str, Any]:
    """Sous-graphe autour d'un noeud (``knowledge_graph.neighbors_graph``) ; 501 si le module est absent."""
    db = require_db(get_state(request))
    try:
        from develop.knowledge_graph.linker import neighbors_graph
    except Exception:  # noqa: BLE001
        raise api_error(501, "not_implemented", "Le module knowledge_graph n'est pas disponible : voisinage indisponible.")
    if node.split(":", 1)[0] not in ("file", "symbol", "endpoint", "model", "document", "note", "table"):
        raise api_error(422, "validation_error", "« node » doit être de la forme type:id (ex. file:12).")
    return neighbors_graph(db, node, depth=depth, limit=limit)
