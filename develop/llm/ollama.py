"""Adaptateur Ollama (http://127.0.0.1:11434) : /api/tags, /api/show, /api/chat, /api/embed, /api/ps, /api/generate (chargement)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

import httpx

from develop.llm import offline_guard
from develop.llm.base import (ChatChunk, ChatMessage, EngineStatus, LLMClient, LLMError, LLMUnavailable,
                              ModelInfo, ModelNotFound, ToolCall, ToolSpec, context_length_from_model_info)


KEEP_ALIVE = "30m"  # garde les modèles en mémoire entre deux questions (Ollama décharge après 5 min par défaut)
EMBED_NUM_CTX = 4096  # fenêtre suffisante pour un morceau (≤ 8 000 caractères) : évite 4,8 Go de cache KV inutile


def _msg_to_ollama(m: ChatMessage) -> Dict[str, Any]:
    d: Dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls]
    if m.role == "tool" and m.name:
        d["tool_name"] = m.name
    return d


class OllamaClient(LLMClient):
    engine = "ollama"
    reports_loaded_models = True  # /api/ps dit ce qui est en mémoire ; /api/generate keep_alive 0 décharge

    def __init__(self, base_url: str = "http://127.0.0.1:11434", guard: Optional[offline_guard.OfflineGuard] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self.base_url = base_url.rstrip("/")
        self._client = offline_guard.make_async_client(guard=guard, inner=transport, base_url=self.base_url)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ----- utilitaires -----
    async def _get_json(self, path: str, timeout: float = 5.0) -> Any:
        try:
            r = await self._client.get(path, timeout=timeout)
        except offline_guard.OfflineViolation:
            raise
        except httpx.HTTPError as exc:
            raise LLMUnavailable("Ollama injoignable sur %s (%s). Lancez « ollama serve » ou l'application Ollama."
                                 % (self.base_url, exc.__class__.__name__)) from exc
        if r.status_code >= 400:
            raise LLMError("Ollama a répondu %d sur %s" % (r.status_code, path))
        try:
            return r.json()
        except ValueError as exc:
            raise LLMError("Réponse Ollama invalide sur %s" % path) from exc

    async def _post_json(self, path: str, payload: Dict[str, Any], timeout: float = 30.0) -> Any:
        try:
            r = await self._client.post(path, json=payload, timeout=timeout)
        except offline_guard.OfflineViolation:
            raise
        except httpx.HTTPError as exc:
            raise LLMUnavailable("Ollama injoignable sur %s (%s)." % (self.base_url, exc.__class__.__name__)) from exc
        if r.status_code == 404:
            raise ModelNotFound("Modèle « %s » introuvable sur Ollama. Téléchargez-le avec : ollama pull %s"
                                % (payload.get("model") or payload.get("name"), payload.get("model") or payload.get("name")))
        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.json().get("error", "")
            except ValueError:
                detail = r.text[:200]
            raise LLMError("Ollama a répondu %d sur %s : %s" % (r.status_code, path, detail))
        try:
            return r.json()
        except ValueError as exc:
            raise LLMError("Réponse Ollama invalide sur %s" % path) from exc

    # ----- interface -----
    async def health(self) -> EngineStatus:
        try:
            version = await self._get_json("/api/version")
            models = await self.list_models()
        except (LLMError, offline_guard.OfflineViolation) as exc:
            return EngineStatus(ok=False, engine=self.engine, base_url=self.base_url, error=str(exc))
        return EngineStatus(ok=True, engine=self.engine, base_url=self.base_url,
                            version=str((version or {}).get("version", "")), models=models)

    async def list_models(self) -> List[ModelInfo]:
        data = await self._get_json("/api/tags")
        out: List[ModelInfo] = []
        for m in (data or {}).get("models", []):
            details = m.get("details") or {}
            name = m.get("name") or m.get("model") or ""
            caps: List[str] = []
            fam = str(details.get("family", ""))
            if any(k in name.lower() for k in ("embed", "minilm", "bge", "e5-", "arctic-embed")) or fam in ("nomic-bert", "bert"):
                caps.append("embedding")
            else:
                caps.append("completion")
            out.append(ModelInfo(name=name, size_bytes=m.get("size"), family=fam,
                                 parameter_size=str(details.get("parameter_size", "")),
                                 quantization=str(details.get("quantization_level", "")), capabilities=caps))
        return out

    async def model_info(self, name: str) -> ModelInfo:
        data = await self._post_json("/api/show", {"model": name, "name": name}, timeout=20.0)
        details = (data or {}).get("details") or {}
        caps = list((data or {}).get("capabilities") or [])
        info = ModelInfo(name=name, family=str(details.get("family", "")),
                         parameter_size=str(details.get("parameter_size", "")),
                         quantization=str(details.get("quantization_level", "")),
                         context_length=context_length_from_model_info((data or {}).get("model_info") or {}),
                         capabilities=caps or ["completion"])
        mi = (data or {}).get("model_info") or {}
        for key, value in mi.items():
            if key.endswith(".embedding_length") and isinstance(value, int):
                info.embedding_dim = value
        if "embedding" in caps and "completion" not in caps:
            info.capabilities = caps
        return info

    async def chat_stream(self, messages: Sequence[ChatMessage], model: str, *, temperature: float = 0.2,
                          num_ctx: Optional[int] = None, tools: Optional[Sequence[ToolSpec]] = None,
                          max_tokens: Optional[int] = None, think: Optional[bool] = None) -> AsyncIterator[ChatChunk]:
        options: Dict[str, Any] = {"temperature": temperature}
        if num_ctx:
            options["num_ctx"] = int(num_ctx)
        if max_tokens:
            options["num_predict"] = int(max_tokens)
        payload: Dict[str, Any] = {"model": model, "messages": [_msg_to_ollama(m) for m in messages],
                                   "stream": True, "options": options, "keep_alive": KEEP_ALIVE}
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
        if think is not None:  # uniquement pour les modèles qui supportent la réflexion (Ollama refuse sinon)
            payload["think"] = bool(think)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload, timeout=httpx.Timeout(600.0, connect=3.0)) as r:
                if r.status_code == 404:
                    raise ModelNotFound("Modèle « %s » introuvable sur Ollama. Téléchargez-le avec : ollama pull %s" % (model, model))
                if r.status_code >= 400:
                    body = await r.aread()
                    raise LLMError("Ollama a répondu %d : %s" % (r.status_code, body.decode("utf-8", "replace")[:300]))
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    if data.get("error"):
                        raise LLMError("Ollama : %s" % data["error"])
                    msg = data.get("message") or {}
                    calls = [ToolCall(name=(c.get("function") or {}).get("name", ""),
                                      arguments=(c.get("function") or {}).get("arguments") or {},
                                      id="call-%d" % (i + 1))
                             for i, c in enumerate(msg.get("tool_calls") or [])]
                    done = bool(data.get("done"))
                    usage: Dict[str, int] = {}
                    if done:
                        if "prompt_eval_count" in data:
                            usage["prompt_tokens"] = int(data["prompt_eval_count"])
                        if "eval_count" in data:
                            usage["completion_tokens"] = int(data["eval_count"])
                        if isinstance(data.get("load_duration"), (int, float)):
                            # temps passé à (re)charger le modèle pour cet appel (ns → ms) : ≈ 0 s'il était déjà en
                            # mémoire, des secondes sinon (keep_alive expiré, autre num_ctx) ; l'orchestrateur écarte
                            # alors le délai de cet appel de sa mesure de vitesse de lecture
                            usage["load_duration_ms"] = int(data["load_duration"] // 1_000_000)
                        if isinstance(data.get("prompt_eval_duration"), (int, float)):
                            # temps de lecture du prompt (ns → ms) : les tokens *non cachés* seulement, alors que
                            # ``prompt_eval_count`` (→ ``prompt_tokens``) compte tout le prompt, cache de préfixe
                            # compris — d'où ``prompt_tokens ÷ prompt_eval_duration`` ≫ vitesse réelle pour un prompt
                            # déjà vu ; conservé pour le diagnostic de la mesure de lecture
                            usage["prompt_eval_duration_ms"] = int(data["prompt_eval_duration"] // 1_000_000)
                    yield ChatChunk(text=msg.get("content") or "", tool_calls=calls, done=done, usage=usage)
                    if done:
                        return
        except offline_guard.OfflineViolation:
            raise
        except httpx.HTTPError as exc:
            raise LLMUnavailable("Ollama injoignable sur %s (%s)." % (self.base_url, exc.__class__.__name__)) from exc

    async def loaded_models(self) -> List[str]:
        """Modèles présents en mémoire (GET /api/ps) : ``[]`` quand Ollama n'a rien chargé — ce n'est pas « inconnu ».
        Ollama injoignable ou ``/api/ps`` en erreur → ``LLMError`` (de :meth:`_get_json`, pas :meth:`_ps` qui
        confondrait l'erreur avec une mémoire vide) : l'appelant ne conclut rien (ni annonce de chargement, ni temps
        de chargement pris pour une mesure de lecture)."""
        data = await self._get_json("/api/ps", timeout=3.0)
        return [str(m.get("name") or m.get("model") or "")
                for m in (data or {}).get("models", []) if isinstance(m, dict)]

    async def loaded_contexts(self) -> Dict[str, Optional[int]]:
        """``{modèle en mémoire: context_length}`` d'après GET /api/ps : la fenêtre (``num_ctx``) avec laquelle
        Ollama a chargé le modèle ; ``None`` quand le serveur ne la donne pas (version ancienne). Même contrat que
        :meth:`loaded_models` : ``{}`` = rien en mémoire, ``LLMError`` = Ollama injoignable (pas « vide »)."""
        data = await self._get_json("/api/ps", timeout=3.0)
        out: Dict[str, Optional[int]] = {}
        for m in (data or {}).get("models", []):
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or m.get("model") or "")
            if not name:
                continue
            ctx = m.get("context_length")
            out[name] = int(ctx) if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0 else None
        return out

    async def warm(self, model: str, num_ctx: Optional[int] = None) -> bool:
        """Charge le modèle en mémoire (POST /api/generate sans prompt) et le garde ``KEEP_ALIVE``.

        ``num_ctx`` est transmis dans ``options`` : Ollama recharge le modèle dès que les options du runner
        diffèrent (``server/sched.go``, ``needsReload``), et sans ``num_ctx`` il choisit une fenêtre « auto »
        (4k / 32k / 256k selon la mémoire) presque toujours différente de celle que le chat demande ensuite,
        d'où un second chargement silencieux (10 à 40 s) au premier message malgré le préchauffage."""
        if not model:
            return False
        payload: Dict[str, Any] = {"model": model, "keep_alive": KEEP_ALIVE}
        if num_ctx:
            payload["options"] = {"num_ctx": int(num_ctx)}
        try:
            await self._post_json("/api/generate", payload, timeout=300.0)
        except LLMError:
            return False
        return True

    async def unload(self, model: str) -> bool:
        """Décharge le modèle (POST /api/generate sans prompt avec ``keep_alive: 0``) : la RAM est libérée
        aussitôt ; le prochain message devra recharger le modèle (≈ 10 à 40 s)."""
        if not model:
            return False
        try:
            await self._post_json("/api/generate", {"model": model, "keep_alive": 0}, timeout=60.0)
        except LLMError:
            return False
        return True

    async def embed(self, texts: Sequence[str], model: str) -> List[List[float]]:
        if not texts:
            return []
        # API moderne (/api/embed, entrée multiple) puis repli sur l'ancienne (/api/embeddings, une par une).
        try:
            data = await self._post_json("/api/embed", {"model": model, "input": list(texts), "keep_alive": KEEP_ALIVE,
                                                        "options": {"num_ctx": EMBED_NUM_CTX}}, timeout=300.0)
            vectors = data.get("embeddings")
            if isinstance(vectors, list) and len(vectors) == len(texts):
                return [[float(x) for x in v] for v in vectors]
        except ModelNotFound:
            raise
        except LLMError:
            pass
        out: List[List[float]] = []
        for t in texts:
            data = await self._post_json("/api/embeddings", {"model": model, "prompt": t, "keep_alive": KEEP_ALIVE,
                                                             "options": {"num_ctx": EMBED_NUM_CTX}}, timeout=300.0)
            vec = data.get("embedding")
            if not isinstance(vec, list):
                raise LLMError("Réponse d'embedding invalide d'Ollama pour le modèle %s" % model)
            out.append([float(x) for x in vec])
        return out
