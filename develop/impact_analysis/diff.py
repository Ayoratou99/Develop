"""Lecture d'un diff unifié Git et mise en correspondance avec les symboles indexés.

Le parseur accepte aussi bien un diff standard (contexte de 3 lignes) qu'un diff produit
avec ``--unified=0``. Les chemins renvoyés sont ceux du **côté nouveau** (après le
changement) ; un fichier supprimé est rapporté sous son ancien chemin, puisqu'il n'en
a plus de nouveau. Les renommages (``rename from`` / ``rename to``, ou ``--- a/x`` /
``+++ b/y`` différents) sont rapportés sous le nouveau chemin.

Toutes les positions sont des numéros de ligne **côté nouveau**, à partir de 1, ce qui
permet de les croiser directement avec ``line_start`` / ``line_end`` des symboles du
fichier tel qu'il est maintenant sur le disque.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from develop.storage import Database, SymbolRow

__all__ = [
    "Hunk",
    "parse_unified_diff",
    "changed_line_ranges",
    "merge_ranges",
    "ranges_intersect",
    "symbols_touched",
    "strip_diff_prefix",
]

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_GIT_RE = re.compile(r'^diff --git "?a/(.*?)"? "?b/(.*?)"?$')
_DEV_NULL = "/dev/null"


@dataclass
class Hunk:
    """Un bloc ``@@ -a,b +c,d @@`` d'un diff unifié.

    ``added_lines`` : numéros (côté nouveau) des lignes ajoutées ou modifiées.
    ``removed_count`` : nombre de lignes supprimées dans ce bloc.
    ``removed_positions`` : pour chaque ligne supprimée, le numéro (côté nouveau) de la
    ligne qui suit immédiatement la suppression ; sert à situer une suppression pure, qui
    n'a par définition aucune ligne ajoutée.
    """

    path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    added_lines: List[int] = field(default_factory=list)
    removed_count: int = 0
    removed_positions: List[int] = field(default_factory=list)

    @property
    def is_pure_removal(self) -> bool:
        return self.removed_count > 0 and not self.added_lines


def strip_diff_prefix(raw: str) -> Optional[str]:
    """Chemin d'une ligne ``--- a/x`` / ``+++ b/y`` sans préfixe ni tabulation ; None pour ``/dev/null``."""
    text = raw.strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    # un diff classique ajoute parfois "\t" + horodatage après le chemin
    text = text.split("\t", 1)[0]
    if not text or text == _DEV_NULL:
        return None
    if text.startswith("a/") or text.startswith("b/"):
        return text[2:]
    return text


class _ParseState:
    """État du parseur : fichier courant, bloc courant, curseur côté nouveau, lignes restantes."""

    def __init__(self) -> None:
        self.result: Dict[str, List[Hunk]] = {}
        self.current: Optional[str] = None
        self.old_path: Optional[str] = None
        self.provisional: Optional[str] = None  # chemin b/ lu sur "diff --git", en attente de "+++"
        self.hunk: Optional[Hunk] = None
        self.cursor = 0
        self.old_remaining = 0
        self.new_remaining = 0

    @property
    def in_hunk(self) -> bool:
        return self.hunk is not None and (self.old_remaining > 0 or self.new_remaining > 0)

    def open_file(self, path: Optional[str]) -> None:
        self.hunk = None
        self.current = path
        if path is not None and path not in self.result:
            self.result[path] = []

    def drop_provisional_if_empty(self, keep: Optional[str]) -> None:
        p = self.provisional
        if p is not None and p != keep and not self.result.get(p):
            self.result.pop(p, None)
        self.provisional = None

    def start_hunk(self, m: "re.Match[str]") -> None:
        if self.current is None:
            self.hunk = None
            return
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) is not None else 1
        self.hunk = Hunk(path=self.current, old_start=old_start, old_count=old_count,
                         new_start=new_start, new_count=new_count)
        self.result[self.current].append(self.hunk)
        self.old_remaining, self.new_remaining = old_count, new_count
        # suppression pure (new_count == 0) : new_start désigne la ligne AVANT le trou
        self.cursor = new_start if new_count > 0 else new_start + 1

    def consume(self, raw: str) -> None:
        """Ligne de contenu d'un bloc : ``+``, ``-``, `` `` (contexte) ou vide."""
        h = self.hunk
        if h is None:
            return
        if raw.startswith("+"):
            h.added_lines.append(self.cursor)
            self.cursor += 1
            self.new_remaining -= 1
        elif raw.startswith("-"):
            h.removed_count += 1
            h.removed_positions.append(self.cursor)
            self.old_remaining -= 1
        else:
            self.cursor += 1
            self.old_remaining -= 1
            self.new_remaining -= 1


def parse_unified_diff(text: str) -> Dict[str, List[Hunk]]:
    """Découpe un diff unifié en blocs, groupés par chemin **côté nouveau**.

    - gère ``diff --git``, ``rename from/to``, ``--- /dev/null`` (ajout) et ``+++ /dev/null``
      (suppression : rapportée sous l'ancien chemin) ;
    - un fichier changé sans bloc (binaire, mode seul, renommage pur) est tout de même
      présent dans le résultat avec une liste vide ;
    - les lignes ``\\ No newline at end of file`` sont ignorées ;
    - tolérant : une ligne inattendue n'interrompt jamais l'analyse.
    """
    st = _ParseState()
    if not text:
        return st.result

    for raw in text.splitlines():
        if st.in_hunk:
            if raw.startswith("\\"):
                continue  # "\ No newline at end of file"
            st.consume(raw)
            continue
        # hors bloc : en-têtes de fichier ou nouveau bloc
        if raw.startswith("diff --git "):
            m = _DIFF_GIT_RE.match(raw)
            st.hunk = None
            st.old_path = None
            st.provisional = None
            if m:
                st.provisional = m.group(2)
                st.open_file(st.provisional)
            else:
                st.current = None
            continue
        if raw.startswith("rename from ") or raw.startswith("copy from "):
            st.old_path = raw.split(" from ", 1)[1].strip()
            continue
        if raw.startswith("rename to ") or raw.startswith("copy to "):
            new_path = raw.split(" to ", 1)[1].strip()
            st.drop_provisional_if_empty(new_path)
            st.open_file(new_path)
            continue
        if raw.startswith("--- "):
            st.hunk = None
            st.old_path = strip_diff_prefix(raw[4:])
            continue
        if raw.startswith("+++ "):
            new_path = strip_diff_prefix(raw[4:])
            if new_path is None:
                new_path = st.old_path or st.provisional  # suppression : ancien chemin
            st.drop_provisional_if_empty(new_path)
            st.open_file(new_path)
            continue
        m = _HUNK_RE.match(raw)
        if m:
            st.start_hunk(m)
            continue
        # "index …", "similarity index", "new file mode", "Binary files … differ", etc. : ignorés
    return st.result


def changed_line_ranges(hunks: Sequence[Hunk]) -> List[Tuple[int, int]]:
    """Plages ``(début, fin)`` inclusives, côté nouveau, fusionnées et triées.

    Les lignes ajoutées ou modifiées consécutives forment une plage ; une suppression
    pure (aucune ligne ajoutée au même endroit) est représentée par les deux lignes qui
    encadrent le trou (la ligne précédente et la suivante), pour que le symbole qui
    contenait les lignes supprimées soit retrouvé.
    """
    raw_ranges: List[Tuple[int, int]] = []
    for h in hunks:
        lines = sorted({int(n) for n in h.added_lines if n is not None and int(n) >= 1})
        start: Optional[int] = None
        prev = 0
        for n in lines:
            if start is None:
                start, prev = n, n
            elif n == prev + 1:
                prev = n
            else:
                raw_ranges.append((start, prev))
                start, prev = n, n
        if start is not None:
            raw_ranges.append((start, prev))
        if h.removed_count > 0:
            added = set(lines)
            positions = h.removed_positions or [h.new_start + 1 if h.new_count == 0 else h.new_start]
            for p in positions:
                if p in added:
                    continue  # ligne remplacée : la ligne ajoutée au même endroit couvre déjà la position
                raw_ranges.append((max(1, p - 1), max(1, p)))
    return merge_ranges(raw_ranges)


def merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Fusionne des plages inclusives qui se chevauchent ou se touchent, et les trie."""
    cleaned = sorted((min(a, b), max(a, b)) for a, b in ranges)
    merged: List[Tuple[int, int]] = []
    for lo, hi in cleaned:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def ranges_intersect(line_start: Optional[int], line_end: Optional[int],
                     ranges: Sequence[Tuple[int, int]]) -> bool:
    """Vrai si l'intervalle ``[line_start, line_end]`` croise au moins une plage.

    Un symbole sans position connue (``line_start`` None) est considéré comme touché :
    on préfère signaler trop que rater un impact.
    """
    if line_start is None:
        return True
    lo = int(line_start)
    hi = int(line_end) if line_end is not None else lo
    if hi < lo:
        lo, hi = hi, lo
    for a, b in ranges:
        if lo <= b and hi >= a:
            return True
    return False


def symbols_touched(db: Database, file_id: int, ranges: Sequence[Tuple[int, int]]) -> List[SymbolRow]:
    """Symboles du fichier dont ``[line_start, line_end]`` croise une des plages.

    Si ``ranges`` est vide (fichier entier : ajout, suppression, fichier non suivi),
    tous les symboles du fichier sont renvoyés. Ordre : position dans le fichier.
    """
    symbols = db.list_symbols(file_id=file_id, limit=100_000)
    if not ranges:
        return symbols
    return [s for s in symbols if ranges_intersect(s.line_start, s.line_end, ranges)]
