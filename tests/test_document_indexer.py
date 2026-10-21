"""Tests du module ``develop.document_indexer`` (chargement de documents et découpage en morceaux)."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import List

import pytest

from develop.document_indexer import (
    PAGE_SEPARATOR,
    SUPPORTED_FORMATS,
    Chunk,
    LoadedDocument,
    chunk_for_file,
    chunk_mode_for,
    chunk_text,
    decode_bytes,
    estimate_tokens,
    extract_markdown_tables,
    format_for_path,
    html_to_text,
    load_document,
    looks_like_data_dictionary,
    markdown_headings,
    page_for_offset,
)


# --------------------------------------------------------------------------- aides

def _norm(s: str) -> str:
    return " ".join(s.split())


def _check_invariants(chunks: List[Chunk], text: str, max_chars: int) -> None:
    """Invariants communs : ordinaux, lignes exactes, aucun morceau vide, taille bornée."""
    lines = text.splitlines()
    for i, c in enumerate(chunks):
        assert c.ordinal == i
        assert c.text.strip(), "morceau vide"
        assert 1 <= c.line_start <= c.line_end <= len(lines), (c.line_start, c.line_end, len(lines))
        window = "\n".join(lines[c.line_start - 1:c.line_end])
        assert _norm(c.text) in _norm(window), "texte du morceau absent de ses lignes"
        if len(c.text) > max_chars:
            single_line = c.line_start == c.line_end
            whole_table = bool(c.meta.get("has_table")) and len(c.text) < 2 * max_chars
            assert single_line or whole_table, "max_chars dépassé: %d > %d" % (len(c.text), max_chars)


def _make_pdf(path: Path, texts: List[str]) -> None:
    """Écrit un PDF minimal valide avec une page de texte par élément de ``texts`` (via pypdf)."""
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    writer = PdfWriter()
    add_object = getattr(writer, "add_object", None) or getattr(writer, "_add_object")
    for txt in texts:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): add_object(font)}),
        })
        stream = StreamObject()
        stream.set_data(("BT /F1 12 Tf 20 150 Td (%s) Tj ET" % txt).encode("latin-1"))
        page[NameObject("/Contents")] = add_object(stream)
    with open(path, "wb") as fh:
        writer.write(fh)


_DOCX_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Titre du document</w:t></w:r></w:p>"
    "<w:p><w:r><w:t xml:space=\"preserve\">Premier </w:t></w:r><w:r><w:t>paragraphe.</w:t></w:r></w:p>"
    "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Colonne</w:t></w:r></w:p></w:tc>"
    "<w:tc><w:p><w:r><w:t>Type</w:t></w:r></w:p></w:tc></w:tr>"
    "<w:tr><w:tc><w:p><w:r><w:t>id</w:t></w:r></w:p></w:tc>"
    "<w:tc><w:p><w:r><w:t>bigint</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
    "<w:p><w:r><w:t>Fin.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)
_DOCX_CORE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Mon DOCX</dc:title></cp:coreProperties>'
)


def _make_docx(path: Path, with_core: bool = True) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", _DOCX_XML)
        if with_core:
            z.writestr("docProps/core.xml", _DOCX_CORE)


# =========================================================================== loaders

def test_supported_formats_and_format_for_path():
    assert SUPPORTED_FORMATS["md"] == "markdown" and SUPPORTED_FORMATS["yml"] == "yaml"
    assert SUPPORTED_FORMATS["tsv"] == "csv" and SUPPORTED_FORMATS["htm"] == "html"
    assert "log" not in SUPPORTED_FORMATS
    assert format_for_path("/x/notes.MD") == "markdown"
    assert format_for_path("/x/app.log") is None
    assert format_for_path("/x/sans_extension") is None


def test_decode_bytes_fallbacks():
    assert decode_bytes("été".encode("utf-8")) == ("été", "utf-8")
    assert decode_bytes(b"\xef\xbb\xbfabc") == ("abc", "utf-8-sig")
    assert decode_bytes("été".encode("utf-16")) == ("été", "utf-16")
    text, enc = decode_bytes("Café crème".encode("latin-1"))
    assert (text, enc) == ("Café crème", "latin-1")


def test_load_markdown_data_dictionary(sample_project: Path):
    doc = load_document(str(sample_project / "docs" / "dictionnaire-donnees.md"))
    assert isinstance(doc, LoadedDocument)
    assert doc.format == "markdown" and doc.warnings == []
    assert doc.title == "Dictionnaire de données"
    assert doc.meta["looks_like_data_dictionary"] is True
    assert "Table `orders` (Commandes)" in doc.meta["headings"]
    assert doc.meta["encoding"] == "utf-8" and doc.pages is None


def test_load_markdown_spec_is_not_a_dictionary(sample_project: Path):
    doc = load_document(str(sample_project / "docs" / "specification-commandes.md"))
    assert doc.title == "Spécification fonctionnelle — Commandes"
    assert doc.meta["looks_like_data_dictionary"] is False
    decision = load_document(str(sample_project / "docs" / "decisions" / "2026-09-01-reunion-statuts.md"))
    assert decision.title.startswith("Réunion du 2026-09-01")
    assert "paid_at" in decision.text


def test_markdown_headings_ignore_code_fences():
    text = "# Un\n\n```\n# pas un titre\n```\n\n## Deux ##\n"
    assert markdown_headings(text) == [(1, "Un", 1), (2, "Deux", 7)]


def test_extract_markdown_tables_fixture(sample_project: Path):
    text = (sample_project / "docs" / "dictionnaire-donnees.md").read_text(encoding="utf-8")
    tables = extract_markdown_tables(text)
    assert len(tables) == 3
    orders = tables[0]
    assert orders.headers == ["Colonne", "Type", "Description"]
    assert len(orders.rows) == 6
    assert orders.rows[1][0] == "reference" and orders.rows[1][1] == "varchar(20)"
    assert orders.heading_path == ["Dictionnaire de données", "Table `orders` (Commandes)"]
    assert orders.line_start == 5 and orders.line_end == 12
    assert tables[2].heading_path[-1] == "Table `customers` (Clients)"
    assert [r[0] for r in tables[2].rows] == ["email", "full_name", "phone"]


def test_extract_markdown_tables_edge_cases():
    text = (
        "# T\n\n"
        "| a | b |\n|---|:-:|\n| 1 \\| un | 2 |\n| 3 |\n\n"
        "```\n| x | y |\n|---|---|\n| 1 | 2 |\n```\n"
        "Pas | un tableau\n"
    )
    tables = extract_markdown_tables(text)
    assert len(tables) == 1
    assert tables[0].headers == ["a", "b"]
    assert tables[0].rows == [["1 | un", "2"], ["3", ""]]
    assert extract_markdown_tables("") == []
    assert extract_markdown_tables("rien à voir\n") == []


def test_looks_like_data_dictionary():
    assert looks_like_data_dictionary("| Field | Data type | Meaning |\n|--|--|--|\n| id | int | pk |\n")
    assert looks_like_data_dictionary("| CHAMP | Description |\n|--|--|\n| id | clé |\n")
    assert not looks_like_data_dictionary("| Nom | Prix |\n|--|--|\n| Stylo | 2 |\n")
    assert not looks_like_data_dictionary("| Colonne | Type |\n|--|--|\n")  # sans ligne
    assert not looks_like_data_dictionary("Aucun tableau ici.")


def test_load_sql(sample_project: Path):
    doc = load_document(str(sample_project / "database" / "schema.sql"))
    assert doc.format == "sql" and doc.title == "schema.sql"
    assert doc.meta["tables"] == ["customers", "orders", "order_lines"]
    assert "CREATE TABLE orders" in doc.text and doc.warnings == []


def test_load_json_valid_and_invalid(sample_project: Path, tmp_path: Path):
    doc = load_document(str(sample_project / "docs" / "glossaire.json"))
    assert doc.format == "json" and doc.meta["json_valid"] is True
    assert '"SKU": "Référence produit (Stock Keeping Unit)"' in doc.text
    assert doc.meta["keys"] == ["SKU", "CMD"] and doc.title == "glossaire.json"
    bad = tmp_path / "bad.json"
    bad.write_text("{'pas': du json}", encoding="utf-8")
    doc2 = load_document(str(bad))
    assert doc2.text == "{'pas': du json}" and doc2.meta["json_valid"] is False
    assert any("JSON invalide" in w for w in doc2.warnings)


def test_load_csv_semicolon_tsv_and_truncation(tmp_path: Path):
    p = tmp_path / "produits.csv"
    p.write_text("nom;prix;qte\nStylo;1,50;10\nCahier;3,20;4\n", encoding="utf-8")
    doc = load_document(str(p))
    assert doc.format == "csv" and doc.meta["delimiter"] == ";"
    lines = doc.text.splitlines()
    assert lines[0] == "nom | prix | qte"
    assert lines[1] == "nom: Stylo | prix: 1,50 | qte: 10"
    assert doc.meta["columns"] == ["nom", "prix", "qte"] and doc.meta["rows"] == 2
    assert doc.title == "produits.csv" and doc.warnings == []

    t = tmp_path / "t.tsv"
    t.write_text("a\tb\n1\t2\n", encoding="utf-8")
    assert load_document(str(t)).text.splitlines()[1] == "a: 1 | b: 2"

    big = tmp_path / "big.csv"
    big.write_text("id,val\n" + "".join("%d,x\n" % i for i in range(5010)), encoding="utf-8")
    doc3 = load_document(str(big))
    assert doc3.meta["rows"] == 5000 and doc3.meta["truncated"] is True
    assert any("tronqué" in w for w in doc3.warnings)

    empty = tmp_path / "vide.csv"
    empty.write_text("", encoding="utf-8")
    assert load_document(str(empty)).text == ""


def test_load_text_encodings(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_bytes("Café crème\nDeuxième ligne".encode("latin-1"))
    doc = load_document(str(p))
    assert doc.text == "Café crème\nDeuxième ligne" and doc.title == "Café crème"
    assert doc.meta["encoding"] == "latin-1" and any("Latin-1" in w for w in doc.warnings)

    u16 = tmp_path / "u16.txt"
    u16.write_bytes("Bonjour\r\nMonde".encode("utf-16"))
    doc2 = load_document(str(u16))
    assert doc2.text == "Bonjour\nMonde" and doc2.meta["encoding"] == "utf-16"

    rst = tmp_path / "doc.rst"
    rst.write_text("\n\nTitre RST\n=========\n", encoding="utf-8")
    assert load_document(str(rst)).title == "Titre RST"


def test_load_empty_binary_huge_missing(tmp_path: Path, sample_project: Path):
    empty = tmp_path / "vide.md"
    empty.write_text("", encoding="utf-8")
    doc = load_document(str(empty))
    assert doc.text == "" and doc.title is None and any("vide" in w for w in doc.warnings)

    binary = tmp_path / "bin.txt"
    binary.write_bytes(b"\x00\x01\x02binaire\x00")
    doc2 = load_document(str(binary))
    assert doc2.text == "" and any("binaire" in w for w in doc2.warnings)

    doc3 = load_document(str(sample_project / "README.md"), max_bytes=10)
    assert doc3.text == "" and any("volumineux" in w for w in doc3.warnings)

    doc4 = load_document(str(tmp_path / "absent.md"))
    assert doc4.text == "" and any("introuvable" in w for w in doc4.warnings)

    doc5 = load_document(str(tmp_path))  # un dossier
    assert doc5.text == "" and doc5.warnings

    doc6 = load_document(str(sample_project / "frontend" / "logo.png"))
    assert doc6.format == "unknown" and any("non pris en charge" in w for w in doc6.warnings)

    doc7 = load_document(str(tmp_path / "x.log"))
    assert doc7.format == "unknown" and doc7.text == ""

    forced = load_document(str(sample_project / "README.md"), format_hint="text")
    assert forced.format == "text" and forced.title == "# MonProjet (fixture de test pour Develop)"


def test_load_pdf_pages_and_offsets(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, ["Hello Develop", "Deuxieme page"])
    doc = load_document(str(pdf))
    assert doc.format == "pdf" and doc.pages == 2 and doc.warnings == []
    assert "Hello Develop" in doc.text and "Deuxieme page" in doc.text
    assert doc.text.count(PAGE_SEPARATOR) == 1
    offsets = doc.meta["page_offsets"]
    assert offsets[0] == 0 and offsets[1] == doc.text.index("Deuxieme")
    assert page_for_offset(offsets, 0) == 1
    assert page_for_offset(offsets, doc.text.index("Deuxieme") + 3) == 2
    assert page_for_offset([], 42) == 1
    assert doc.title == "Hello Develop"


def test_load_pdf_without_text_or_corrupt(tmp_path: Path):
    from pypdf import PdfWriter

    blank = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(width=100, height=100)
    with open(blank, "wb") as fh:
        w.write(fh)
    doc = load_document(str(blank))
    assert doc.pages == 1 and isinstance(doc.text, str)
    assert any("OCR" in w for w in doc.warnings)

    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\n%%garbage sans xref\n")
    doc2 = load_document(str(corrupt))
    assert doc2.text == "" and doc2.warnings

    hand = tmp_path / "hand.pdf"
    hand.write_bytes(
        b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R"
        b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 10 100 Td (Hello Develop) Tj ET\nendstream\nendobj\n"
        b"trailer<</Root 1 0 R>>\n"
    )
    doc3 = load_document(str(hand))
    assert isinstance(doc3.text, str)  # ne lève jamais, même sans table xref


def test_html_to_text_and_load_html(tmp_path: Path):
    html = ("<html><head><title> Accueil  du site </title><style>p{color:red}</style>"
            "<script>var x = 1;</script></head><body><h1>Bonjour</h1><p>Texte &amp; co</p>"
            "<table><tr><td>a</td><td>b</td></tr></table></body></html>")
    text, title = html_to_text(html)
    assert title == "Accueil du site"
    assert "Bonjour" in text and "Texte & co" in text and "a | b" in text
    assert "var x" not in text and "color:red" not in text
    p = tmp_path / "page.htm"
    p.write_text(html, encoding="utf-8")
    doc = load_document(str(p))
    assert doc.format == "html" and doc.title == "Accueil du site" and "Bonjour" in doc.text
    assert html_to_text("")[0] == ""


def test_load_docx_stdlib_fallback_and_invalid(tmp_path: Path):
    p = tmp_path / "doc.docx"
    _make_docx(p)
    doc = load_document(str(p))
    assert doc.format == "docx" and doc.warnings == []
    assert doc.meta["docx_backend"] in ("stdlib", "python-docx")
    assert doc.title == "Mon DOCX"
    assert "Premier paragraphe." in doc.text and "Colonne | Type" in doc.text and "id | bigint" in doc.text
    assert "Fin." in doc.text

    p2 = tmp_path / "sans-titre.docx"
    _make_docx(p2, with_core=False)
    assert load_document(str(p2)).title == "Titre du document"

    bad = tmp_path / "bad.docx"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("rien.txt", "x")
    doc3 = load_document(str(bad))
    assert doc3.text == "" and any("DOCX illisible" in w for w in doc3.warnings)
    notzip = tmp_path / "notzip.docx"
    notzip.write_bytes(b"pas une archive")
    assert load_document(str(notzip)).warnings


def test_load_yaml_and_xml(tmp_path: Path):
    y = tmp_path / "config.yml"
    y.write_text("name: develop\nport: 8765\n", encoding="utf-8")
    doc = load_document(str(y))
    assert doc.format == "yaml" and doc.text == "name: develop\nport: 8765\n" and doc.title == "config.yml"
    if doc.meta.get("yaml_valid") is not None:  # pyyaml présent
        assert doc.meta["keys"] == ["name", "port"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2\nb: : c\n", encoding="utf-8")
    assert load_document(str(bad)).text.startswith("a: [1, 2")

    x = tmp_path / "data.xml"
    x.write_text("<?xml version='1.0'?><root><item>1</item></root>", encoding="utf-8")
    doc2 = load_document(str(x))
    assert doc2.format == "xml" and doc2.meta["root_tag"] == "root" and "<item>1</item>" in doc2.text
    badx = tmp_path / "bad.xml"
    badx.write_text("<root><item>", encoding="utf-8")
    doc3 = load_document(str(badx))
    assert doc3.text == "<root><item>" and any("XML" in w for w in doc3.warnings)


# =========================================================================== chunking

def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("abcde") == 2


def test_chunk_text_empty_and_invalid_kind():
    assert chunk_text("", "prose") == []
    assert chunk_text("  \n\n\t", "code") == []
    with pytest.raises(ValueError):
        chunk_text("x", "inconnu")
    with pytest.raises(ValueError):
        chunk_text("x", "prose", max_chars=0)


def test_chunk_markdown_data_dictionary(sample_project: Path):
    text = (sample_project / "docs" / "dictionnaire-donnees.md").read_text(encoding="utf-8")
    chunks = chunk_text(text, "markdown")
    _check_invariants(chunks, text, 1500)
    assert len(chunks) == 3
    assert chunks[0].line_start == 1 and chunks[0].text.startswith("# Dictionnaire de données")
    orders = [c for c in chunks if "| reference" in c.text]
    assert len(orders) == 1
    assert orders[0].meta["heading_path"] == ["Dictionnaire de données", "Table `orders` (Commandes)"]
    assert orders[0].meta["has_table"] is True and orders[0].meta["kind"] == "markdown"
    assert "| id " in orders[0].text and "| confirmed_at" in orders[0].text  # tableau jamais coupé
    assert chunks[2].meta["heading_path"][-1] == "Table `customers` (Clients)"
    assert chunks[2].line_end == len(text.splitlines())


def test_chunk_markdown_tables_kept_whole_below_twice_max(sample_project: Path):
    text = (sample_project / "docs" / "dictionnaire-donnees.md").read_text(encoding="utf-8")
    chunks = chunk_text(text, "markdown", max_chars=500)
    _check_invariants(chunks, text, 500)
    orders = [c for c in chunks if "| reference" in c.text][0]
    assert len(orders.text) > 500 and "| confirmed_at" in orders.text
    # Trop grand pour l'exception : découpé par fenêtres de lignes, l'en-tête rappelé en meta.
    small = chunk_text(text, "markdown", max_chars=200, overlap_chars=0)
    _check_invariants(small, text, 200)
    parts = [c for c in small if "| reference" in c.text or "| confirmed_at" in c.text]
    assert len(parts) >= 2
    assert all(c.meta["table_header"].startswith("| Colonne") for c in parts)
    assert all(c.meta["heading_path"][-1] == "Table `orders` (Commandes)" for c in parts)


def test_chunk_markdown_heading_paths(sample_project: Path):
    text = (sample_project / "docs" / "specification-commandes.md").read_text(encoding="utf-8")
    chunks = chunk_text(text, "markdown", max_chars=300)
    _check_invariants(chunks, text, 300)
    rules = [c for c in chunks if "Une commande est créée" in c.text][0]
    assert rules.meta["heading_path"] == ["Spécification fonctionnelle — Commandes", "Règles métier"]
    screens = [c for c in chunks if "Liste des commandes" in c.text][0]
    assert screens.meta["heading_path"] == ["Spécification fonctionnelle — Commandes", "Écrans"]
    assert rules.text.startswith("# Spécification")  # titre parent sans corps rattaché à la section suivante


def test_chunk_markdown_without_heading_and_with_fences():
    text = "Un paragraphe.\n\nUn autre paragraphe.\n"
    chunks = chunk_text(text, "markdown")
    assert len(chunks) == 1 and chunks[0].meta["heading_path"] == []
    assert (chunks[0].line_start, chunks[0].line_end) == (1, 3)

    fenced = "Intro\n\n# Titre\n\n```python\n# pas un titre\n\nx = 1\n```\n\nSuite.\n"
    chunks = chunk_text(fenced, "markdown", max_chars=60, overlap_chars=0)
    _check_invariants(chunks, fenced, 60)
    paths = {tuple(c.meta["heading_path"]) for c in chunks}
    assert paths == {(), ("Titre",)}
    code = [c for c in chunks if "x = 1" in c.text][0]
    assert code.meta.get("has_code_block") is True and "# pas un titre" in code.text
    assert code.text.startswith("# Titre")  # le titre reste collé à son premier bloc
    # Bloc de code trop long : coupé par fenêtres de lignes, jamais sur la ligne vide interne d'abord.
    tiny = chunk_text(fenced, "markdown", max_chars=25, overlap_chars=0)
    _check_invariants(tiny, fenced, 25)
    assert all(c.meta["heading_path"] == ["Titre"] for c in tiny if "```" in c.text)


def test_chunk_markdown_deep_heading_path():
    text = "# A\n\n## B\n\ntexte b\n\n### C\n\ntexte c\n\n## D\n\ntexte d\n"
    chunks = chunk_text(text, "markdown", max_chars=40)
    by_text = {c.text.splitlines()[-1]: c.meta["heading_path"] for c in chunks}
    assert by_text["texte b"] == ["A", "B"]
    assert by_text["texte c"] == ["A", "B", "C"]
    assert by_text["texte d"] == ["A", "D"]


_PY_SAMPLE = '''import os
import sys

CONST = 1


@dataclass
class Customer:
    name: str

    def greet(self):
        return "hi"


@app.get("/x")
def handler():
    return 1


def helper():
    pass
'''


def test_chunk_code_python_definitions_and_symbol_hints():
    chunks = chunk_text(_PY_SAMPLE, "code", language="python")
    _check_invariants(chunks, _PY_SAMPLE, 1500)
    assert len(chunks) == 1 and chunks[0].meta["symbol_hint"] == "Customer"
    assert chunks[0].meta["language"] == "python" and chunks[0].meta["kind"] == "code"

    small = chunk_text(_PY_SAMPLE, "code", max_chars=60, overlap_chars=0, language="python")
    _check_invariants(small, _PY_SAMPLE, 60)
    hints = [c.meta.get("symbol_hint") for c in small]
    assert "Customer" in hints and "greet" in hints and "handler" in hints and "helper" in hints
    cust = [c for c in small if c.meta.get("symbol_hint") == "Customer"][0]
    assert cust.text.startswith("@dataclass\nclass Customer")
    handler = [c for c in small if c.meta.get("symbol_hint") == "handler"][0]
    assert handler.text.startswith('@app.get("/x")\ndef handler')
    assert small[0].text.startswith("import os") and "symbol_hint" not in small[0].meta


def test_chunk_code_fixtures(sample_project: Path):
    routes = (sample_project / "backend" / "routes" / "api.php").read_text(encoding="utf-8")
    chunks = chunk_text(routes, "code", max_chars=120, overlap_chars=0, language="php")
    _check_invariants(chunks, routes, 120)
    hints = {c.meta.get("symbol_hint") for c in chunks}
    assert "Route::get" in hints and "Route::post" in hints

    ctrl = (sample_project / "backend" / "app" / "Http" / "Controllers" / "OrderController.php").read_text(encoding="utf-8")
    chunks = chunk_text(ctrl, "code", max_chars=120, overlap_chars=0, language="php")
    _check_invariants(chunks, ctrl, 120)
    hints = [c.meta.get("symbol_hint") for c in chunks]
    assert "OrderController" in hints and "index" in hints and "confirm" in hints

    js = (sample_project / "services" / "notifier" / "src" / "server.js").read_text(encoding="utf-8")
    chunks = chunk_text(js, "code", max_chars=150, overlap_chars=0, language="javascript")
    _check_invariants(chunks, js, 150)
    hints = {c.meta.get("symbol_hint") for c in chunks}
    assert "app.get" in hints and "app.post" in hints and "app.delete" in hints

    tsx = (sample_project / "frontend" / "src" / "components" / "CustomerSearch.tsx").read_text(encoding="utf-8")
    chunks = chunk_text(tsx, "code", language="typescript")
    assert chunks[0].meta["symbol_hint"] == "CustomerSearch"

    py = (sample_project / "tools" / "report.py").read_text(encoding="utf-8")
    chunks = chunk_text(py, "code", max_chars=200, overlap_chars=0, language="python")
    _check_invariants(chunks, py, 200)
    hints = [c.meta.get("symbol_hint") for c in chunks]
    assert hints[0] is None and "OrderRow" in hints and "ReportBuilder" in hints
    # Les petites méthodes sont regroupées avec leur classe à 200 caractères ; à 80, chacune a son morceau.
    small = chunk_text(py, "code", max_chars=80, overlap_chars=0, language="python")
    _check_invariants(small, py, 80)
    hints = [c.meta.get("symbol_hint") for c in small]
    assert "total" in hints and "__init__" in hints and "load_orders" in hints


def test_chunk_code_long_definition_windows_with_overlap():
    text = "def big():\n" + "".join("    x%03d = %d\n" % (i, i) for i in range(200)) + "    return x\n"
    chunks = chunk_text(text, "code", max_chars=300, overlap_chars=60)
    _check_invariants(chunks, text, 300)
    assert len(chunks) > 5
    assert all(c.meta["symbol_hint"] == "big" for c in chunks)
    assert chunks[0].line_start == 1
    assert chunks[-1].line_end == len(text.splitlines())
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.line_start < cur.line_start <= prev.line_end + 1
    assert any(cur.line_start <= prev.line_end for prev, cur in zip(chunks, chunks[1:]))  # chevauchement


def test_chunk_code_without_definitions_uses_blank_lines():
    text = "[section]\nkey = 1\n\n[autre]\nkey = 2\n\n\n[fin]\nz = 3\n"
    chunks = chunk_text(text, "code", max_chars=25, overlap_chars=0)
    _check_invariants(chunks, text, 25)
    assert [c.text for c in chunks] == ["[section]\nkey = 1", "[autre]\nkey = 2", "[fin]\nz = 3"]
    assert all("symbol_hint" not in c.meta for c in chunks)
    merged = chunk_text(text, "code")
    assert len(merged) == 1 and merged[0].line_end == 9


def test_chunk_prose_paragraphs_and_sentence_overlap():
    paras = []
    for p in range(6):
        paras.append(" ".join("Phrase %d du paragraphe %d, assez longue pour compter." % (s, p) for s in range(4)))
    text = "\n\n".join(paras) + "\n"
    chunks = chunk_text(text, "prose", max_chars=300, overlap_chars=80)
    _check_invariants(chunks, text, 300)
    assert len(chunks) >= 3
    for prev, cur in zip(chunks, chunks[1:]):
        first_sentence = cur.text[:cur.text.index(".") + 1]
        assert first_sentence in prev.text, "pas de chevauchement de phrases"
        assert cur.line_start <= prev.line_end
    assert "".join(c.text for c in chunks).count("Phrase 0 du paragraphe 5") >= 1


def test_chunk_prose_long_single_line():
    line = " ".join("Phrase numéro %d." % i for i in range(600))  # ≈ 10 000 caractères, une seule ligne
    assert len(line) >= 10_000
    text = line + "\n"
    chunks = chunk_text(text, "prose", max_chars=1500, overlap_chars=150)
    _check_invariants(chunks, text, 1500)
    assert len(chunks) >= 7 and all(c.line_start == c.line_end == 1 for c in chunks)
    assert all(len(c.text) <= 1500 for c in chunks)
    assert chunks[0].text.startswith("Phrase numéro 0.") and chunks[-1].text.endswith("Phrase numéro 599.")

    blob = "a" * 10_000
    for kind in ("prose", "code", "data", "markdown"):
        chunks = chunk_text(blob, kind, max_chars=1500)
        assert len(chunks) == 1 and chunks[0].text == blob and chunks[0].line_start == 1

    code_line = "x = [" + ", ".join(str(i) for i in range(3000)) + "]\n"
    chunks = chunk_text(code_line, "code", max_chars=500)
    assert len(chunks) == 1 and chunks[0].text.rstrip("\n") == code_line.rstrip("\n")


def test_chunk_data_windows_header_and_overlap(sample_project: Path):
    sql = (sample_project / "database" / "schema.sql").read_text(encoding="utf-8")
    chunks = chunk_text(sql, "data", max_chars=300, overlap_chars=60, language="sql")
    _check_invariants(chunks, sql, 300)
    assert len(chunks) >= 3 and "header" not in chunks[0].meta
    assert any(cur.line_start <= prev.line_end for prev, cur in zip(chunks, chunks[1:]))
    assert chunks[-1].line_end == len(sql.splitlines())

    csv_text = "id | nom | email\n" + "\n".join("id: %d | nom: Client %d | email: c%d@x.fr" % (i, i, i) for i in range(100))
    chunks = chunk_text(csv_text, "data", max_chars=400, overlap_chars=40)
    _check_invariants(chunks, csv_text, 400)
    assert len(chunks) > 1
    assert all(c.meta["header"] == "id | nom | email" for c in chunks)

    many = "\n".join("l%d" % i for i in range(100))
    chunks = chunk_text(many, "data", max_chars=100_000, overlap_chars=0)
    assert len(chunks) == 3 and all(c.line_end - c.line_start + 1 <= 40 for c in chunks)
    assert "header" not in chunks[0].meta

    numeric = "1,2,3\n4,5,6\n"
    assert "header" not in chunk_text(numeric, "data")[0].meta


def test_chunk_for_file_and_mode_selection():
    assert chunk_mode_for("python", "code") == "code"
    assert chunk_mode_for("markdown", "docs") == "markdown"
    assert chunk_mode_for("md", None) == "markdown"
    assert chunk_mode_for("sql", "database") == "data"
    assert chunk_mode_for("json", "config") == "data"
    assert chunk_mode_for("yaml", None) == "data" and chunk_mode_for("csv", "document") == "data"
    assert chunk_mode_for("pdf", "document") == "prose"
    assert chunk_mode_for(None, None) == "prose"
    assert chunk_mode_for("php", None) == "code"

    assert chunk_for_file("def f():\n    pass\n", "python", "code")[0].meta == {
        "kind": "code", "language": "python", "symbol_hint": "f"}
    assert chunk_for_file("# T\n\ntexte\n", "markdown", "docs")[0].meta["heading_path"] == ["T"]
    assert chunk_for_file("a,b\n1,2\n", "csv", "data")[0].meta["header"] == "a,b"
    assert chunk_for_file("Bonjour.\n", "pdf", "document")[0].meta["kind"] == "prose"
    assert chunk_for_file("", "python", "code") == []


def test_chunk_invariants_across_kinds_and_sizes(sample_project: Path, tmp_path: Path):
    docs = sample_project / "docs"
    samples = [
        (docs / "dictionnaire-donnees.md").read_text(encoding="utf-8"),
        (docs / "specification-commandes.md").read_text(encoding="utf-8"),
        (sample_project / "database" / "schema.sql").read_text(encoding="utf-8"),
        (sample_project / "tools" / "report.py").read_text(encoding="utf-8"),
        (sample_project / "frontend" / "src" / "components" / "OrderList.tsx").read_text(encoding="utf-8"),
        "ligne1\r\nligne2\r\n\r\nligne4\r\n",
        "Page un.\nSuite page un.\fPage deux.\n\nFin.\f",
        "\n\n   \n",
        "x",
        _PY_SAMPLE,
    ]
    for text in samples:
        for kind in ("code", "markdown", "prose", "data"):
            for max_chars in (20, 80, 300, 1500):
                chunks = chunk_text(text, kind, max_chars=max_chars, overlap_chars=max_chars // 5)
                _check_invariants(chunks, text, max_chars)
                if text.strip():
                    assert chunks, (kind, max_chars)
                    covered = set()
                    for c in chunks:
                        covered.update(range(c.line_start, c.line_end + 1))
                    for i, line in enumerate(text.splitlines(), 1):
                        if line.strip():
                            assert i in covered, "ligne %d non couverte (%s, %d)" % (i, kind, max_chars)


def test_pdf_chunks_map_back_to_pages(tmp_path: Path):
    pdf = tmp_path / "pages.pdf"
    _make_pdf(pdf, ["Premiere page du document", "Deuxieme page du document", "Troisieme page"])
    doc = load_document(str(pdf))
    assert doc.pages == 3
    chunks = chunk_for_file(doc.text, "pdf", "document", max_chars=40, overlap_chars=0)
    _check_invariants(chunks, doc.text, 40)
    # offset de début de chaque ligne (splitlines coupe aussi sur \f, le séparateur de page)
    line_offsets = []
    pos = 0
    for raw in doc.text.splitlines(keepends=True):
        line_offsets.append(pos)
        pos += len(raw)
    pages = {page_for_offset(doc.meta["page_offsets"], line_offsets[c.line_start - 1]): c.text for c in chunks}
    assert "Premiere" in pages[1] and "Deuxieme" in pages[2] and "Troisieme" in pages[3]


def test_loaded_latin1_document_chunks(tmp_path: Path):
    p = tmp_path / "ancien.txt"
    p.write_bytes("Première partie. Suite.\n\nSeconde partie, avec accents : é à ç.\n".encode("latin-1"))
    doc = load_document(str(p))
    chunks = chunk_for_file(doc.text, "text", "document")
    assert len(chunks) == 1 and "é à ç" in chunks[0].text
    _check_invariants(chunks, doc.text, 1500)


def test_glossaire_json_loaded_then_chunked(sample_project: Path):
    doc = load_document(str(sample_project / "docs" / "glossaire.json"))
    chunks = chunk_for_file(doc.text, "json", "docs")
    assert len(chunks) == 1 and chunks[0].meta["kind"] == "data"
    assert chunks[0].line_start == 1 and chunks[0].line_end == len(doc.text.splitlines())
    assert "Stock Keeping Unit" in chunks[0].text


def test_prose_chunks_never_cross_page_breaks():
    """Un morceau de prose ne chevauche jamais un saut de page (``\\f``, séparateur des pages d'un PDF) :
    le libellé « page N » calculé par l'indexeur sur sa première ligne est exact, et le rappel des dernières
    phrases ne remonte jamais à la page précédente (sinon un morceau « page 9 » montre le début de la page 10)."""
    pages = [
        "Le paramétrage décrit les champs. Chaque champ porte une clé et un libellé.\n"
        "Le contexte de commande est transmis s'il est connu :\n",
        "// GET /form-templates/all?populate=blocks\nLes identifiants sont des UUID. Le serveur revalide la colonne.\n",
        "Fin.\n",  # page très courte : un morceau à elle seule, jamais réunie à la page précédente
    ]
    text = PAGE_SEPARATOR.join(pages)
    page_of_line = {}
    page, lineno = 1, 1
    for raw in text.splitlines(keepends=True):
        page_of_line[lineno] = page
        lineno += 1
        if raw.endswith(PAGE_SEPARATOR):
            page += 1
    for max_chars in (60, 200, 1500):
        chunks = chunk_text(text, "prose", max_chars=max_chars, overlap_chars=max_chars // 4)
        _check_invariants(chunks, text, max_chars)
        for c in chunks:
            assert page_of_line[c.line_start] == page_of_line[c.line_end], (max_chars, c.line_start, c.line_end, c.text)
        first_of_page2 = next(c for c in chunks if page_of_line[c.line_start] == 2)
        assert first_of_page2.text.startswith("// GET /form-templates/all"), (max_chars, first_of_page2.text)
        assert "Fin." in [c.text for c in chunks]
    # Un bloc qui enjambe un saut de page sans ligne vide (texte brut) est coupé au saut ;
    # sans saut de page, deux courts paragraphes restent réunis comme avant.
    assert [c.text for c in chunk_text("alpha\fbeta", "prose")] == ["alpha", "beta"]
    assert len(chunk_text("alpha\n\nbeta\n", "prose")) == 1


# --------------------------------------------------------------------------- PDF : en-tête courant, sections numérotées

def _make_pdf_pages(path: Path, pages: List[List[str]]) -> None:
    """PDF minimal valide avec plusieurs lignes de texte par page (une ligne par élément, de haut en bas)."""
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    writer = PdfWriter()
    add_object = getattr(writer, "add_object", None) or getattr(writer, "_add_object")
    for lines in pages:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): add_object(font)}),
        })
        stream = StreamObject()
        ops = "BT /F1 12 Tf 20 280 Td " + " ".join("(%s) Tj 0 -16 Td" % line for line in lines) + " ET"
        stream.set_data(ops.encode("latin-1"))
        page[NameObject("/Contents")] = add_object(stream)
    with open(path, "wb") as fh:
        writer.write(fh)


def test_load_pdf_strips_running_header_and_page_numbers(tmp_path: Path):
    """L'en-tête courant (« FUIP - Guide » en tête de chaque page) et les numéros de page sortent du texte :
    sinon chaque morceau « contient » le titre du document et la recherche lexicale ne les distingue plus."""
    from develop.document_indexer import PAGE_SEPARATOR, strip_page_furniture

    pdf = tmp_path / "guide.pdf"
    _make_pdf_pages(pdf, [["FUIP - Guide", "1", "Sommaire", "1. Intro", "2. Suite"],
                          ["FUIP - Guide", "2", "1. Intro", "Texte introductif."],
                          ["FUIP - Guide", "3", "2. Suite", "Texte de la suite."],
                          ["FUIP - Guide", "4", "Fin du guide."]])
    doc = load_document(str(pdf))
    assert doc.pages == 4 and doc.warnings == []
    assert "FUIP - Guide" not in doc.text and doc.meta["running_lines"] == ["FUIP - Guide", "1", "2", "3", "4"]
    pages = doc.text.split(PAGE_SEPARATOR)
    assert pages[0] == "Sommaire\n1. Intro\n2. Suite\n" and pages[1] == "1. Intro\nTexte introductif.\n" and pages[3] == "Fin du guide.\n"
    # les décalages de pages suivent le texte nettoyé : chaque morceau retrouve sa page ; les titres numérotés
    # ouvrent des sections (le sommaire, lignes numérotées qui se suivent, reste un bloc sans titre)
    chunks = chunk_for_file(doc.text, "pdf", "document", max_chars=60, overlap_chars=0)
    _check_invariants(chunks, doc.text, 60)
    line_offsets = []
    pos = 0
    for raw in doc.text.splitlines(keepends=True):
        line_offsets.append(pos)
        pos += len(raw)
    by_page = {page_for_offset(doc.meta["page_offsets"], line_offsets[c.line_start - 1]): c for c in chunks}
    assert by_page[2].text == "1. Intro\nTexte introductif." and by_page[2].meta["section"] == "1. Intro"
    assert by_page[3].text.startswith("2. Suite\nTexte de la suite.") and by_page[3].meta["heading_path"] == ["2. Suite"]
    assert by_page[1].text == "Sommaire\n1. Intro\n2. Suite" and "heading_path" not in by_page[1].meta
    # fonction pure : deux pages ne font pas une répétition ; une ligne répétée après du contenu n'est pas un
    # en-tête (le retrait s'arrête à la première ligne de contenu) ; « page 3 / 3 » est un numéro de page
    assert strip_page_furniture(["A\nx\n", "A\ny\n"]) == (["A\nx\n", "A\ny\n"], [])
    cleaned, removed = strip_page_furniture(["Titre\nun\nNote\nx\npage 1 / 3\n", "Titre\ndeux\nNote\ny\npage 2 / 3\n",
                                             "Titre\ntrois\nNote\nz\npage 3 / 3\n"])
    assert cleaned == ["un\nNote\nx\n", "deux\nNote\ny\n", "trois\nNote\nz\n"]
    assert removed == ["Titre", "page 1 / 3", "page 2 / 3", "page 3 / 3"]
    # un en-tête présent sur une page sur trois n'est pas « courant » ; une page vide ne compte pas
    assert strip_page_furniture(["H\na\n", "b\n", "c\n", ""])[1] == []


def test_chunk_prose_numbered_sections_carry_heading_path():
    """Un document en prose (PDF, texte) découpé par ses titres numérotés : « 4. Les champs » est un morceau
    qui commence par son titre et porte ``heading_path`` / ``section``, pas la fin de la section 3 suivie du
    début de la 4 ; un sommaire, une liste d'étapes ou une date ne sont pas des titres."""
    from develop.document_indexer import numbered_heading

    text = ("Guide\nSommaire\n1. Intro\n2. Modèle\n3. Fin\n\n1. Intro\nPremière phrase de l'intro. Deuxième phrase de l'intro.\n\n"
            "2. Modèle\n2.1 Tables\nLes tables sont décrites ici.\n\n27 juillet 2025\nUne date n'est pas un titre.\n\n"
            "3. Fin\nDernier paragraphe.\n")
    chunks = chunk_text(text, "prose", max_chars=1500, overlap_chars=150)
    _check_invariants(chunks, text, 1500)
    assert [c.text.splitlines()[0] for c in chunks] == ["Guide", "1. Intro", "2. Modèle", "3. Fin"]
    assert "heading_path" not in chunks[0].meta and chunks[0].text.endswith("3. Fin")  # sommaire : bloc sans titre
    assert chunks[1].meta["heading_path"] == ["1. Intro"] and chunks[1].meta["section"] == "1. Intro"
    assert chunks[2].meta["heading_path"] == ["2. Modèle", "2.1 Tables"] and chunks[2].meta["section"] == "2.1 Tables"
    assert "27 juillet 2025" in chunks[2].text  # une date n'ouvre pas de section
    assert chunks[3].text == "3. Fin\nDernier paragraphe." and chunks[3].meta["kind"] == "prose"
    # section longue : plusieurs morceaux, tous titrés ; une nouvelle section commence par son titre, sans
    # rappel des dernières phrases de la précédente
    long = "4. Détails\n" + " ".join("Phrase %d de la section quatre." % i for i in range(40)) + "\n\n5. Suite\nTexte court.\n"
    chunks = chunk_text(long, "prose", max_chars=200, overlap_chars=50)
    _check_invariants(chunks, long, 200)
    four = [c for c in chunks if c.meta.get("section") == "4. Détails"]
    assert len(four) >= 4 and four[0].text.startswith("4. Détails\nPhrase 0") and len(four) == len(chunks) - 1
    assert all(c.meta["heading_path"] == ["4. Détails"] for c in four)
    assert chunks[-1].text == "5. Suite\nTexte court." and chunks[-1].meta["section"] == "5. Suite"
    # les étapes d'une liste et les dates ne sont pas des titres ; un sous-titre a un niveau
    assert numbered_heading("1. Créer le template puis le téléverser.") is None
    assert numbered_heading("27 juillet 2025") is None and numbered_heading("10. 30 €") is None
    assert numbered_heading("2.1 Modèle") == (2, "2.1 Modèle") and numbered_heading("4. Les champs (custom)") == (1, "4. Les champs (custom)")
    # texte sans titre numéroté : découpage inchangé (un seul morceau, sans heading_path)
    plain = chunk_text("Un paragraphe.\n\nUn autre paragraphe.\n", "prose")
    assert len(plain) == 1 and "heading_path" not in plain[0].meta
