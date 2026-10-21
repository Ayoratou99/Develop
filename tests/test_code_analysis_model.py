from __future__ import annotations

from develop.code_analysis import analyze_file, normalize_route_path, supported_languages
from develop.code_analysis.model import route_paths_match


def test_normalize_route_path_variants():
    n = normalize_route_path
    assert n("/orders/{order}/confirm") == "/orders/:param/confirm"
    assert n("/orders/${id}/confirm") == "/orders/:param/confirm"
    assert n("`${API_BASE}/orders/${id}`".replace("`", "")) == "/orders/:param"  # ${API_BASE} devient :param puis segment
    assert n("http://localhost:8000/api/orders?q=1#x") == "/api/orders"
    assert n("/users/:userId/posts/:id?") == "/users/:param/posts/:param"
    assert n("/items/<int:pk>/") == "/items/:param"
    assert n("/pages/[slug]") == "/pages/:param"
    assert n("/orders/42") == "/orders/:param"
    assert n("orders//list/") == "/orders/list"
    assert n("") == "/"
    assert n("/Orders") == "/orders"


def test_route_paths_match_with_and_without_base_prefix():
    assert route_paths_match("/orders/{order}", "http://localhost:8000/api/orders/${id}")
    assert route_paths_match("/orders", "/orders")
    assert not route_paths_match("/orders", "/customers")
    assert not route_paths_match("/", "/anything")
    assert not route_paths_match("/orders", "/api/myorders")


def test_analyze_file_unknown_language_is_safe():
    fa = analyze_file("x", "cobol", "a.cbl")
    assert fa.symbols == [] and fa.warnings
    assert "python" in supported_languages()
