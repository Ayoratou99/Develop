"""Tests de l'extracteur Java / Kotlin (Spring, JPA, JAX-RS, Micronaut, Feign, Retrofit, Ktor, clients HTTP)."""
from __future__ import annotations

import time
from pathlib import Path

from develop.code_analysis import analyze_file
from develop.code_analysis.extractors.java import EXTRACTOR
from develop.code_analysis.model import FileAnalysis, normalize_route_path
from develop.code_analysis.registry import get_extractor


def _sym(fa: FileAnalysis, name: str, kind: str = None, parent: str = None):
    """Premier symbole correspondant (nom, et éventuellement kind/parent)."""
    for s in fa.symbols:
        if s.name == name and (kind is None or s.kind == kind) and (parent is None or s.parent == parent):
            return s
    raise AssertionError("symbole %s introuvable dans %s" % (name, [(s.kind, s.name, s.parent) for s in fa.symbols]))


def _eps(fa: FileAnalysis, **filters):
    out = []
    for e in fa.endpoints:
        if all(getattr(e, k) == v for k, v in filters.items()):
            out.append(e)
    return out


# --------------------------------------------------------------------------- fixture réelle

def test_fixture_has_no_java_but_extractor_is_registered(sample_project: Path):
    """Le mini-projet ne contient pas de Java/Kotlin : on vérifie l'enregistrement et la robustesse
    de l'extracteur sur tous les fichiers texte de la fixture (jamais d'exception)."""
    files = [p for p in sample_project.rglob("*") if p.is_file()]
    assert files, "fixture vide ?"
    assert not [p for p in files if p.suffix in (".java", ".kt", ".kts")]
    assert get_extractor("java") is EXTRACTOR
    assert get_extractor("kotlin") is EXTRACTOR
    assert EXTRACTOR.languages == ["java", "kotlin"]
    for p in files:
        if p.suffix in (".png",):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        fa = EXTRACTOR.analyze(text, str(p))
        assert isinstance(fa, FileAnalysis)
        # Le langage vient de l'extension ; sans .java/.kt il est deviné d'après le contenu.
        assert fa.language in ("java", "kotlin")
        fa2 = analyze_file(text, "kotlin", str(p))
        assert isinstance(fa2, FileAnalysis) and fa2.language in ("java", "kotlin")
        assert EXTRACTOR.analyze(text, str(p) + ".kt").language == "kotlin"
        assert EXTRACTOR.analyze(text, str(p) + ".java").language == "java"


def test_language_detection_by_extension_then_content():
    assert EXTRACTOR.analyze("class A", "A.java").language == "java"
    assert EXTRACTOR.analyze("class A", "A.KT").language == "kotlin"
    assert EXTRACTOR.analyze("val x = 1", "build.gradle.kts").language == "kotlin"
    # Pas d'extension : reniflage du contenu
    assert EXTRACTOR.analyze("fun main() {\n    val x = 1\n}\n", "snippet").language == "kotlin"
    assert EXTRACTOR.analyze("class A {\n    int x = 1;\n}\n", "snippet").language == "java"
    assert EXTRACTOR.analyze("", "").language == "java"
    fa = EXTRACTOR.analyze("data class P(val x: Int)", None)  # chemin absent
    assert fa.language == "kotlin" and fa.models[0].kind == "dataclass"


def test_fixture_php_file_yields_no_false_endpoints(sample_project: Path):
    """Un fichier PHP analysé comme Java ne produit ni endpoint ni modèle (pas de faux positifs grossiers)."""
    text = (sample_project / "backend/app/Http/Controllers/OrderController.php").read_text(encoding="utf-8")
    fa = EXTRACTOR.analyze(text, "OrderController.java")
    assert fa.endpoints == []
    assert fa.models == []


# --------------------------------------------------------------------------- Spring MVC

CONTROLLER = '''\
package com.acme.orders.web;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import java.util.List;
import static java.util.Objects.requireNonNull;

/** Contrôleur des commandes. */
@RestController
@RequestMapping("/api/orders")
public class OrderController {

    private final OrderService service; // injecté

    public OrderController(OrderService service) {
        this.service = service;
    }

    @GetMapping
    public List<Order> list(@RequestParam(required = false) String status) {
        return service.findAll(status);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Order> show(@PathVariable Long id) {
        return ResponseEntity.ok(service.find(id)); // "{" dans une chaîne
    }

    @RequestMapping(value = "/{id}/confirm", method = RequestMethod.POST)
    public Order confirm(@PathVariable("id") Long id,
                         @RequestBody ConfirmRequest body) {
        return service.confirm(id, body);
    }

    @DeleteMapping(path = {"/{id}", "/legacy/{id}"})
    public void remove(@PathVariable Long id) {
        service.delete(id);
    }
}
'''


def test_spring_controller_symbols_and_lines():
    fa = EXTRACTOR.analyze(CONTROLLER, "src/main/java/com/acme/orders/web/OrderController.java")
    assert fa.warnings == []
    assert fa.language == "java"
    pkg = _sym(fa, "com.acme.orders.web", "module")
    assert (pkg.line_start, pkg.line_end, pkg.meta["package"]) == (1, 1, "com.acme.orders.web")
    cls = _sym(fa, "OrderController", "class")
    assert (cls.line_start, cls.line_end, cls.parent) == (11, 39, None)
    assert cls.meta["annotations"] == ["RestController", "RequestMapping"]
    assert cls.meta["modifiers"] == ["public"]
    assert cls.meta["package"] == "com.acme.orders.web"
    field = _sym(fa, "service", "property", "OrderController")
    assert (field.line_start, field.meta["type"], field.meta["modifiers"]) == (13, "OrderService", ["private", "final"])
    ctor = _sym(fa, "OrderController", "method", "OrderController")
    assert (ctor.line_start, ctor.line_end, ctor.meta.get("constructor")) == (15, 17, True)
    assert ctor.signature == "public OrderController(OrderService service)"
    lst = _sym(fa, "list", "method", "OrderController")
    assert (lst.line_start, lst.line_end) == (20, 22)
    assert lst.meta["returns"] == "List<Order>"
    assert lst.meta["annotations"] == ["GetMapping"]  # les annotations de paramètres ne sont pas attachées
    assert lst.signature == "public List<Order> list(@RequestParam(required = false) String status)"
    assert lst.qualified_name == "OrderController.list"
    show = _sym(fa, "show", "method")
    assert (show.line_start, show.line_end) == (25, 27)  # l'accolade dans la chaîne n'a pas perturbé le comptage
    confirm = _sym(fa, "confirm", "method")
    assert (confirm.line_start, confirm.line_end) == (30, 33)  # signature multi-lignes
    remove = _sym(fa, "remove", "method")
    assert (remove.line_start, remove.line_end, remove.meta["returns"]) == (36, 38, "void")
    assert [s.name for s in fa.symbols if s.kind == "method"] == ["OrderController", "list", "show", "confirm", "remove"]


def test_spring_controller_endpoints():
    fa = EXTRACTOR.analyze(CONTROLLER, "OrderController.java")
    eps = [(e.method, e.path, e.handler, e.line) for e in fa.endpoints]
    assert eps == [
        ("GET", "/api/orders", "OrderController@list", 19),
        ("GET", "/api/orders/{id}", "OrderController@show", 24),
        ("POST", "/api/orders/{id}/confirm", "OrderController@confirm", 29),
        ("DELETE", "/api/orders/{id}", "OrderController@remove", 35),
        ("DELETE", "/api/orders/legacy/{id}", "OrderController@remove", 35),
    ]
    for e in fa.endpoints:
        assert e.side == "server" and e.framework == "spring"
        assert e.meta["prefix"] == "/api/orders"
    assert fa.endpoints[1].meta["route"] == "/{id}"
    assert fa.endpoints[1].normalized_path == "/api/orders/:param"
    assert normalize_route_path(fa.endpoints[2].path) == "/api/orders/:param/confirm"


def test_java_imports():
    fa = EXTRACTOR.analyze(CONTROLLER, "OrderController.java")
    imps = [(i.module, i.names, i.line, i.is_relative) for i in fa.imports]
    assert imps == [
        ("org.springframework.web.bind.annotation.*", [], 3, False),
        ("org.springframework.http.ResponseEntity", ["ResponseEntity"], 4, False),
        ("java.util.List", ["List"], 5, False),
        ("java.util.Objects.requireNonNull", ["requireNonNull"], 6, False),
    ]


def test_spring_request_mapping_with_several_methods_and_class_prefixes():
    src = '''\
@Controller
@RequestMapping({"/admin", "/backoffice"})
public class AdminController {
    @RequestMapping(path = "/users", method = {RequestMethod.GET, RequestMethod.POST})
    public String users() { return "users"; }

    @RequestMapping("/ping")
    public String ping() { return "pong"; }
}
'''
    fa = EXTRACTOR.analyze(src, "AdminController.java")
    eps = sorted((e.method, e.path) for e in fa.endpoints)
    assert eps == [
        ("ANY", "/admin/ping"), ("ANY", "/backoffice/ping"),
        ("GET", "/admin/users"), ("GET", "/backoffice/users"),
        ("POST", "/admin/users"), ("POST", "/backoffice/users"),
    ]
    assert {e.framework for e in fa.endpoints} == {"spring"}
    assert {e.handler for e in fa.endpoints} == {"AdminController@users", "AdminController@ping"}


def test_spring_put_patch_and_multiline_annotation():
    src = '''\
@RestController
public class ItemController {
    @PutMapping("/items/{id}")
    public Item replace(@PathVariable Long id, @RequestBody Item i) { return i; }
    @PatchMapping(value = "/items/{id}", consumes = "application/json")
    public Item patch(@PathVariable Long id) { return null; }
    @RequestMapping(
        value = "/items/{id}/archive",
        method = RequestMethod.POST,
        produces = {"application/json"})
    public void archive(@PathVariable Long id) {}
}
'''
    fa = EXTRACTOR.analyze(src, "ItemController.java")
    assert fa.warnings == []
    assert [(e.method, e.path, e.handler, e.line) for e in fa.endpoints] == [
        ("PUT", "/items/{id}", "ItemController@replace", 3),
        ("PATCH", "/items/{id}", "ItemController@patch", 5),
        ("POST", "/items/{id}/archive", "ItemController@archive", 7),
    ]
    assert all(e.meta.get("prefix") is None for e in fa.endpoints)  # pas de @RequestMapping de classe : meta nettoyée
    assert fa.endpoints[1].meta["consumes"] == ["application/json"]
    assert fa.endpoints[2].meta["produces"] == ["application/json"]
    archive = _sym(fa, "archive", "method")
    assert (archive.line_start, archive.line_end, archive.meta["annotations"]) == (11, 11, ["RequestMapping"])


def test_several_declarations_on_one_line():
    """Corps de classe écrits sur une seule ligne : chaque instruction est un segment."""
    src = '''\
package a; import b.C; import static d.E.f;
public class One { private int x = 1; String s = "a;b"; public int get() { return x; } void m() {} }
@Entity class Two { @Id Long id; @Column(name = "n", nullable = false) String name; @ManyToOne Three t; }
interface Three { void a(); int b(); }
enum Four { A, B; Four() {} }
'''
    fa = EXTRACTOR.analyze(src, "One.java")
    assert fa.warnings == []
    assert _sym(fa, "a", "module").line_start == 1
    assert [(i.module, i.names) for i in fa.imports] == [("b.C", ["C"]), ("d.E.f", ["f"])]
    assert [(s.name, s.kind, s.parent, s.line_end) for s in fa.symbols if s.parent == "One"] == [
        ("x", "property", "One", 2), ("s", "property", "One", 2), ("get", "method", "One", 2), ("m", "method", "One", 2)]
    assert _sym(fa, "x", "property").signature == "private int x" and _sym(fa, "s", "property").meta["type"] == "String"
    assert _sym(fa, "get", "method").signature == "public int get()"
    assert (_sym(fa, "One").line_start, _sym(fa, "One").line_end) == (2, 2)
    m = fa.models[0]
    assert (m.name, m.table_name, m.line) == ("Two", "two", 3)
    assert [(f.name, f.type, f.nullable, f.meta) for f in m.fields] == [
        ("id", "Long", False, {"primary": True}), ("name", "String", False, {"column": "n"}),
        ("t", "Three", None, {"relation": "belongs_to"})]
    assert [(r.kind, r.target, r.field) for r in m.relations] == [("belongs_to", "Three", "t")]
    assert [(s.name, s.line_end) for s in fa.symbols if s.parent == "Three"] == [("a", 4), ("b", 4)]
    assert [s.name for s in fa.symbols if s.parent == "Four"] == ["A", "B", "Four"]
    assert [(s.name, s.kind, s.line_start) for s in fa.symbols if s.parent is None] == [
        ("a", "module", 1), ("One", "class", 2), ("Two", "class", 3), ("Three", "interface", 4), ("Four", "enum", 5)]
    assert len(fa.symbols) == 17


def test_spring_functional_routes():
    src = '''\
public class Routes {
    public RouterFunction<ServerResponse> routes(OrderHandler h) {
        return route(GET("/fn/orders"), h::list)
            .andRoute(POST("/fn/orders"), h::create);
    }
}
'''
    fa = EXTRACTOR.analyze(src, "Routes.java")
    assert [(e.method, e.path, e.handler, e.line) for e in fa.endpoints] == [
        ("GET", "/fn/orders", "h@list", 3), ("POST", "/fn/orders", "h@create", 4)]
    assert fa.endpoints[0].meta == {"functional": True}


# --------------------------------------------------------------------------- JPA

ENTITY = '''\
package com.acme.orders.domain;

import javax.persistence.*;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "orders")
public class Order {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "status", nullable = false, length = 20)
    private String status = "draft";

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "customer_id")
    private Customer customer;

    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL)
    private List<OrderLine> lines = new ArrayList<>();

    @Transient
    private int cachedTotal;

    public static final String DEFAULT_STATUS = "draft";

    public Long getId() { return id; }

    public enum Status {
        DRAFT, CONFIRMED("c"), SHIPPED;
        private final String code;
        Status() { this.code = ""; }
        Status(String code) { this.code = code; }
    }
}
'''


def test_jpa_entity_model():
    fa = EXTRACTOR.analyze(ENTITY, "Order.java")
    assert fa.warnings == []
    assert len(fa.models) == 1
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("Order", "jpa", "orders", 9)
    assert [f.name for f in m.fields] == ["id", "status", "customer", "lines"]  # @Transient et static exclus
    fid, fstatus, fcustomer, flines = m.fields
    assert (fid.type, fid.nullable, fid.meta) == ("Long", False, {"primary": True, "generated": True, "strategy": "IDENTITY"})
    assert (fstatus.type, fstatus.nullable, fstatus.default, fstatus.meta) == ("String", False, '"draft"', {"length": 20})
    assert fcustomer.meta == {"relation": "belongs_to", "join_column": "customer_id"}
    assert flines.type == "List<OrderLine>" and flines.default == "new ArrayList<>()"
    assert [(r.kind, r.target, r.field) for r in m.relations] == [
        ("belongs_to", "Customer", "customer"), ("has_many", "OrderLine", "lines")]


def test_jpa_entity_symbols_enum_and_constants():
    fa = EXTRACTOR.analyze(ENTITY, "Order.java")
    cls = _sym(fa, "Order", "class")
    assert (cls.line_start, cls.line_end) == (9, 38)
    assert _sym(fa, "id", "property", "Order").meta["annotations"] == ["Id", "GeneratedValue"]
    assert _sym(fa, "status", "property", "Order").line_start == 16
    assert _sym(fa, "customer", "property", "Order").line_start == 20
    assert _sym(fa, "lines", "property", "Order").line_start == 23
    assert _sym(fa, "cachedTotal", "property", "Order").line_start == 26
    const = _sym(fa, "DEFAULT_STATUS", "constant", "Order")
    assert (const.line_start, const.meta["modifiers"]) == (28, ["public", "static", "final"])
    getter = _sym(fa, "getId", "method", "Order")
    assert (getter.line_start, getter.line_end, getter.meta["returns"]) == (30, 30, "Long")
    enum = _sym(fa, "Status", "enum", "Order")
    assert (enum.line_start, enum.line_end) == (32, 37)
    assert [(s.name, s.line_start) for s in fa.symbols if s.kind == "constant" and s.parent == "Status"] == [
        ("DRAFT", 33), ("CONFIRMED", 33), ("SHIPPED", 33)]
    assert _sym(fa, "code", "property", "Status").line_start == 34
    ctors = [s for s in fa.symbols if s.name == "Status" and s.kind == "method"]
    assert [(c.line_start, c.signature) for c in ctors] == [(35, "Status()"), (36, "Status(String code)")]


def test_jpa_default_table_name_is_snake_case_and_mapped_superclass():
    src = '''\
@Entity
class OrderLineItem {
    @Id private Long id;
    @Column(nullable = true, unique = true) private String sku;
    @ManyToMany(targetEntity = Tag.class) private Set<Object> tags;
    @OneToOne private Address address;
}
@MappedSuperclass
public abstract class Auditable { @Column(name = "created_at") private Instant createdAt; }
'''
    fa = EXTRACTOR.analyze(src, "OrderLineItem.java")
    assert [(m.name, m.table_name) for m in fa.models] == [("OrderLineItem", "order_line_item"), ("Auditable", None)]
    m = fa.models[0]
    assert [(f.name, f.nullable) for f in m.fields] == [("id", False), ("sku", True), ("tags", None), ("address", None)]
    assert m.fields[1].meta == {"unique": True}
    assert [(r.kind, r.target) for r in m.relations] == [("many_to_many", "Tag"), ("has_one", "Address")]
    assert fa.models[1].meta.get("mapped_superclass") is True
    assert fa.models[1].fields[0].meta == {"column": "created_at"}


def test_java_record_is_a_dataclass_model():
    src = "public record Point(int x, int y, String label) {}\n"
    fa = EXTRACTOR.analyze(src, "Point.java")
    assert len(fa.models) == 1 and fa.models[0].kind == "dataclass" and fa.models[0].meta["record"] is True
    assert [(f.name, f.type) for f in fa.models[0].fields] == [("x", "int"), ("y", "int"), ("label", "String")]
    assert [(s.name, s.kind, s.parent) for s in fa.symbols] == [
        ("Point", "class", None), ("x", "property", "Point"), ("y", "property", "Point"), ("label", "property", "Point")]
    assert _sym(fa, "Point").meta["record"] is True


# --------------------------------------------------------------------------- RestTemplate

SERVICE = '''\
package com.acme.orders.service;

import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpMethod;

public class ShippingClient {
    private final RestTemplate restTemplate = new RestTemplate();
    private final String baseUrl = "http://shipping:8080";

    public Quote quote(Long orderId) {
        return restTemplate.getForObject(baseUrl + "/quotes/" + orderId, Quote.class);
    }

    public void ship(Long orderId, ShipRequest req) {
        restTemplate.postForObject("http://shipping:8080/shipments", req, Void.class);
        ResponseEntity<Void> r = restTemplate.exchange("/shipments/{id}/cancel", HttpMethod.DELETE,
                null, Void.class, orderId);
        restTemplate.put(baseUrl + "/shipments/" + orderId, req);
    }
}
'''


def test_resttemplate_client_calls():
    fa = EXTRACTOR.analyze(SERVICE, "ShippingClient.java")
    assert fa.warnings == []
    eps = [(e.method, e.path, e.handler, e.line, e.meta.get("base")) for e in fa.endpoints]
    assert eps == [
        ("GET", "${baseUrl}/quotes/${orderId}", "ShippingClient@quote", 11, "baseUrl"),
        ("POST", "http://shipping:8080/shipments", "ShippingClient@ship", 15, None),
        ("DELETE", "/shipments/{id}/cancel", "ShippingClient@ship", 16, None),
        ("PUT", "${baseUrl}/shipments/${orderId}", "ShippingClient@ship", 18, "baseUrl"),
    ]
    assert {e.side for e in fa.endpoints} == {"client"}
    assert {e.framework for e in fa.endpoints} == {"resttemplate"}
    assert fa.endpoints[0].normalized_path == "/quotes/:param"
    assert fa.endpoints[1].normalized_path == "/shipments"
    assert fa.endpoints[2].meta["call"] == "exchange"
    assert _sym(fa, "ship", "method").line_end == 19


# --------------------------------------------------------------------------- WebClient / Feign

WEB = '''\
package a;

import org.springframework.web.reactive.function.client.WebClient;

@Service
public class CatalogClient {
    private final WebClient client = WebClient.create("http://catalog:8080");

    public Mono<Product> product(String sku) {
        return client.get()
                .uri("/products/{sku}", sku)
                .retrieve()
                .bodyToMono(Product.class);
    }

    public Mono<Void> reserve(String sku) {
        return client.post().uri("/products/" + sku + "/reserve").retrieve().bodyToMono(Void.class);
    }
}

@FeignClient(name = "billing", url = "${billing.url}")
interface BillingApi {
    @GetMapping("/invoices/{id}")
    Invoice get(@PathVariable("id") Long id);

    @PostMapping(value = "/invoices")
    Invoice create(@RequestBody Invoice invoice);
}
'''


def test_webclient_and_feign_clients():
    fa = EXTRACTOR.analyze(WEB, "CatalogClient.java")
    assert fa.warnings == []
    eps = [(e.framework, e.method, e.path, e.handler, e.line) for e in fa.endpoints]
    assert eps == [
        ("webclient", "GET", "/products/{sku}", "CatalogClient@product", 10),
        ("webclient", "POST", "/products/${sku}/reserve", "CatalogClient@reserve", 17),
        ("feign", "GET", "/invoices/{id}", "BillingApi@get", 23),
        ("feign", "POST", "/invoices", "BillingApi@create", 26),
    ]
    assert {e.side for e in fa.endpoints} == {"client"}
    assert fa.endpoints[0].meta["base"] == "http://catalog:8080"  # WebClient.create(...) repéré dans le fichier
    assert fa.endpoints[2].meta["base"] == "${billing.url}" and fa.endpoints[2].meta["client"] == "billing"
    api = _sym(fa, "BillingApi", "interface")
    assert (api.line_start, api.line_end, api.meta["annotations"]) == (22, 28, ["FeignClient"])
    get = _sym(fa, "get", "method", "BillingApi")
    assert (get.line_start, get.line_end, get.signature) == (24, 24, 'Invoice get(@PathVariable("id") Long id)')
    assert _sym(fa, "product", "method").line_end == 14


# --------------------------------------------------------------------------- JAX-RS / Micronaut / interfaces

JAXRS = '''\
package a;

import javax.ws.rs.GET;
import javax.ws.rs.Path;

@Path("/customers")
@Produces(MediaType.APPLICATION_JSON)
public class CustomerResource {
    @GET
    public List<Customer> all() { return List.of(); }

    @GET
    @Path("{id}")
    public Customer one(@PathParam("id") long id) { return null; }

    @POST
    @Consumes("application/json")
    public Response create(Customer c)
    {
        return Response.ok().build();
    }
}

@Controller("/api/v2/items")
class ItemController {
    @Get("/{id}")
    HttpResponse<Item> item(Long id) { return null; }
    @Post(uri = "/")
    HttpResponse<Item> add(@Body Item i) { return null; }
}

public interface Repo<T> extends Base<T> {
    List<T> findAll();
    Optional<T> find(long id);
    default void refresh() {}
}
'''


def test_jaxrs_and_micronaut_endpoints():
    fa = EXTRACTOR.analyze(JAXRS, "CustomerResource.java")
    assert fa.warnings == []
    eps = [(e.framework, e.method, e.path, e.handler, e.line) for e in fa.endpoints]
    assert eps == [
        ("jaxrs", "GET", "/customers", "CustomerResource@all", 9),
        ("jaxrs", "GET", "/customers/{id}", "CustomerResource@one", 12),
        ("jaxrs", "POST", "/customers", "CustomerResource@create", 16),
        ("micronaut", "GET", "/api/v2/items/{id}", "ItemController@item", 26),
        ("micronaut", "POST", "/api/v2/items/", "ItemController@add", 28),
    ]
    assert {e.side for e in fa.endpoints} == {"server"}
    assert fa.endpoints[1].meta["route"] == "{id}" and fa.endpoints[1].meta["prefix"] == "/customers"
    assert fa.endpoints[2].meta["consumes"] == ["application/json"]
    assert fa.endpoints[4].normalized_path == "/api/v2/items"


def test_allman_braces_generic_interface_and_abstract_methods():
    fa = EXTRACTOR.analyze(JAXRS, "CustomerResource.java")
    create = _sym(fa, "create", "method", "CustomerResource")
    assert (create.line_start, create.line_end) == (18, 21)  # accolade ouvrante sur la ligne suivante
    repo = _sym(fa, "Repo", "interface")
    assert (repo.line_start, repo.line_end, repo.signature) == (32, 36, "public interface Repo<T> extends Base<T>")
    assert [(s.name, s.line_start, s.line_end, s.meta["returns"]) for s in fa.symbols if s.parent == "Repo"] == [
        ("findAll", 33, 33, "List<T>"), ("find", 34, 34, "Optional<T>"), ("refresh", 35, 35, "void")]
    assert _sym(fa, "findAll").signature == "List<T> findAll()"
    assert _sym(fa, "CustomerResource").line_end == 22
    assert _sym(fa, "ItemController").line_end == 30


# --------------------------------------------------------------------------- OkHttp / HttpClient

HTTP = '''\
package a;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;

public class Gateway {
    public String ping() throws Exception {
        Request request = new Request.Builder()
                .url("https://api.example.com/ping")
                .build();
        Request del = new Request.Builder().url(base + "/items/" + id).delete().build();
        HttpRequest req = HttpRequest.newBuilder(URI.create("https://api.example.com/v1/users"))
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{}"))
                .build();
        HttpRequest req2 = HttpRequest.newBuilder()
                .uri(URI.create(String.format("https://api.example.com/v1/users/%s", id)))
                .GET()
                .build();
        return "";
    }
}
'''


def test_okhttp_and_java11_httpclient():
    fa = EXTRACTOR.analyze(HTTP, "Gateway.java")
    assert fa.warnings == []
    eps = [(e.framework, e.method, e.path, e.line, e.meta.get("base")) for e in fa.endpoints]
    assert eps == [
        ("okhttp", "GET", "https://api.example.com/ping", 10, None),
        ("okhttp", "DELETE", "${base}/items/${id}", 13, "base"),
        ("httpclient", "POST", "https://api.example.com/v1/users", 14, None),
        ("httpclient", "GET", "https://api.example.com/v1/users/{param}", 18, None),
    ]
    assert {e.handler for e in fa.endpoints} == {"Gateway@ping"}
    assert fa.endpoints[3].normalized_path == "/v1/users/:param"
    ping = _sym(fa, "ping", "method")
    assert (ping.line_start, ping.line_end, ping.signature) == (9, 23, "public String ping() throws Exception")


# --------------------------------------------------------------------------- Kotlin

KT = '''\
package com.acme.orders

import kotlinx.serialization.Serializable
import com.acme.orders.domain.Customer as Cust

@Serializable
data class OrderDto(
    val id: Long,
    val status: String = "draft",
    @SerialName("customer_id") val customerId: Long?,
    var lines: List<LineDto> = emptyList(),
) {
    fun total(): Long = lines.sumOf { it.price }
}

class OrderRepository(private val db: Database) {
    val count: Int
        get() = 0

    fun findAll(): List<OrderDto> {
        return emptyList()
    }

    suspend fun load(id: Long): OrderDto? =
        db.query(id)

    companion object {
        const val TABLE = "orders"
    }
}

fun String.slug(): String = lowercase()
'''


def test_kotlin_data_class_model():
    fa = EXTRACTOR.analyze(KT, "src/main/kotlin/com/acme/orders/OrderDto.kt")
    assert fa.language == "kotlin" and fa.warnings == []
    assert len(fa.models) == 1
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("OrderDto", "dataclass", None, 7)
    assert [(f.name, f.type, f.nullable, f.default) for f in m.fields] == [
        ("id", "Long", False, None),
        ("status", "String", False, '"draft"'),
        ("customerId", "Long?", True, None),
        ("lines", "List<LineDto>", False, "emptyList()"),
    ]
    assert m.fields[3].meta["mutable"] is True and m.fields[0].meta["mutable"] is False


def test_kotlin_symbols_and_lines():
    fa = EXTRACTOR.analyze(KT, "OrderDto.kt")
    cls = _sym(fa, "OrderDto", "class")
    assert (cls.line_start, cls.line_end, cls.meta["data"], cls.meta["annotations"]) == (7, 14, True, ["Serializable"])
    assert [(s.name, s.line_start) for s in fa.symbols if s.kind == "property" and s.parent == "OrderDto"] == [
        ("id", 8), ("status", 9), ("customerId", 10), ("lines", 11)]
    assert _sym(fa, "customerId").meta["annotations"] == ["SerialName"]
    total = _sym(fa, "total", "method", "OrderDto")
    assert (total.line_start, total.line_end, total.meta["expression_body"], total.signature) == (13, 13, True, "fun total(): Long")
    repo = _sym(fa, "OrderRepository", "class")
    assert (repo.line_start, repo.line_end) == (16, 30)
    assert _sym(fa, "db", "property", "OrderRepository").meta["modifiers"] == ["private"]
    assert _sym(fa, "count", "property", "OrderRepository").line_start == 17
    find_all = _sym(fa, "findAll", "method", "OrderRepository")
    assert (find_all.line_start, find_all.line_end) == (20, 22)
    load = _sym(fa, "load", "method", "OrderRepository")
    assert (load.line_start, load.meta["async"], load.meta["expression_body"]) == (24, True, True)
    comp = _sym(fa, "Companion", "class", "OrderRepository")
    assert (comp.line_start, comp.line_end, comp.meta["companion"]) == (27, 29, True)
    assert _sym(fa, "TABLE", "constant", "Companion").line_start == 28
    slug = _sym(fa, "slug", "function")
    assert (slug.line_start, slug.parent, slug.meta["receiver"]) == (32, None, "String")
    assert [(i.module, i.names, i.line) for i in fa.imports] == [
        ("kotlinx.serialization.Serializable", ["Serializable"], 3),
        ("com.acme.orders.domain.Customer", ["Customer"], 4)]


KT_ENTITY = '''\
package com.acme.orders.domain

import jakarta.persistence.*

@Entity
@Table(name = "order_lines")
class OrderLine(
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    val id: Long = 0,
    @Column(name = "qty", nullable = false)
    var quantity: Int,
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "order_id")
    val order: Order,
    var note: String?
) {
    @OneToMany(mappedBy = "line")
    var discounts: MutableList<Discount> = mutableListOf()

    @Transient
    var cache: Int = 0
}

interface Api {
    @GET("users/{id}")
    suspend fun user(@Path("id") id: Long): User

    @POST("users")
    fun create(@Body u: User): Call<User>
}

fun Application.configureRouting() {
    routing {
        route("/api") {
            get("/health") { call.respondText("ok") }
            route("/orders") {
                get { call.respond(repo.all()) }
                post("/{id}/confirm") {
                    call.respond(HttpStatusCode.OK)
                }
            }
        }
    }
}

class Fetcher(private val client: HttpClient) {
    suspend fun load(id: Long): Order = client.get("$baseUrl/orders/$id").body()
    suspend fun push(o: Order) {
        client.post("/orders") { setBody(o) }
        val m = mapOf("a" to 1); m.get("a")
    }
}
'''


def test_kotlin_jpa_entity_with_constructor_properties():
    fa = EXTRACTOR.analyze(KT_ENTITY, "OrderLine.kt")
    assert fa.warnings == []
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("OrderLine", "jpa", "order_lines", 7)
    assert [(f.name, f.type, f.nullable) for f in m.fields] == [
        ("id", "Long", False), ("quantity", "Int", False), ("order", "Order", False),
        ("note", "String?", True), ("discounts", "MutableList<Discount>", False)]
    assert m.fields[0].meta["primary"] is True and m.fields[0].meta["strategy"] == "IDENTITY"
    assert m.fields[1].meta["column"] == "qty"
    assert m.fields[2].meta["join_column"] == "order_id"
    assert [(r.kind, r.target, r.field) for r in m.relations] == [
        ("belongs_to", "Order", "order"), ("has_many", "Discount", "discounts")]
    assert [(s.name, s.line_start) for s in fa.symbols if s.kind == "property" and s.parent == "OrderLine"] == [
        ("id", 9), ("quantity", 11), ("order", 14), ("note", 15), ("discounts", 18), ("cache", 21)]
    assert _sym(fa, "OrderLine", "class").line_end == 22


def test_retrofit_ktor_server_and_ktor_client():
    fa = EXTRACTOR.analyze(KT_ENTITY, "OrderLine.kt")
    eps = [(e.side, e.framework, e.method, e.path, e.handler, e.line) for e in fa.endpoints]
    assert eps == [
        ("client", "retrofit", "GET", "users/{id}", "Api@user", 25),
        ("client", "retrofit", "POST", "users", "Api@create", 28),
        ("server", "ktor", "GET", "/api/health", "configureRouting", 35),
        ("server", "ktor", "GET", "/api/orders", "configureRouting", 37),
        ("server", "ktor", "POST", "/api/orders/{id}/confirm", "configureRouting", 38),
        ("client", "ktor-client", "GET", "$baseUrl/orders/$id", "Fetcher@load", 47),
        ("client", "ktor-client", "POST", "/orders", "Fetcher@push", 49),
    ]
    assert fa.endpoints[5].meta["base"] == "baseUrl"
    assert fa.endpoints[5].normalized_path == "/orders/:param"
    assert fa.endpoints[4].meta == {"prefix": "/api/orders", "route": "/{id}/confirm"}
    routing = _sym(fa, "configureRouting", "function")
    assert (routing.line_start, routing.line_end, routing.meta["receiver"]) == (32, 44, "Application")


def test_kotlin_misc_declarations():
    src = '''\
import a.b.*
object Config : Base() {
    val url = "x"
}
sealed interface Shape
enum class Color(val hex: String) { RED("#f00"), GREEN("#0f0"); fun x() = 1 }
typealias Handler = (Int) -> Unit
class Outer {
    inner class Inner
    constructor(x: Int) : this()
    private fun <T> List<T>.second(): T? = getOrNull(1)
}
const val MAX = 10
'''
    fa = EXTRACTOR.analyze(src, "Misc.kt")
    assert fa.warnings == []
    assert fa.imports[0].module == "a.b.*" and fa.imports[0].names == []
    cfg = _sym(fa, "Config", "class")
    assert (cfg.line_start, cfg.line_end, cfg.meta["object"]) == (2, 4, True)
    assert _sym(fa, "url", "property", "Config").line_start == 3
    shape = _sym(fa, "Shape", "interface")
    assert (shape.line_start, shape.line_end, shape.meta["sealed"]) == (5, 5, True)
    color = _sym(fa, "Color", "enum")
    assert (color.line_start, color.line_end) == (6, 6)
    assert [s.name for s in fa.symbols if s.kind == "constant" and s.parent == "Color"] == ["RED", "GREEN"]
    assert _sym(fa, "hex", "property", "Color").line_start == 6
    assert _sym(fa, "Handler", "type").meta["typealias"] is True
    assert _sym(fa, "Inner", "class", "Outer").line_start == 9
    ctor = _sym(fa, "Outer", "method", "Outer")
    assert (ctor.line_start, ctor.meta["constructor"]) == (10, True)
    second = _sym(fa, "second", "method", "Outer")
    assert (second.meta["receiver"], second.meta["expression_body"]) == ("List<T>", True)
    mx = _sym(fa, "MAX", "constant")
    assert (mx.line_start, mx.parent) == (13, None)


# --------------------------------------------------------------------------- cas limites

def test_invalid_truncated_file_gives_warnings_not_exceptions():
    src = '''\
public class Broken {
    public void x( {
        if (true) {
    String s = "unterminated;
    @GetMapping("/x"
'''
    fa = EXTRACTOR.analyze(src, "Broken.java")
    assert fa.warnings == ["déclaration incomplète à la ligne 2 (fichier tronqué ?)", "3 accolade(s) non fermée(s) en fin de fichier"]
    cls = _sym(fa, "Broken", "class")
    assert (cls.line_start, cls.line_end) == (1, 6)
    assert fa.endpoints == []
    # Kotlin tronqué en plein mot-clé : ``class`` sans nom n'est pas une déclaration, et aucune erreur interne
    for tail in ("class", "data class", "enum class ", "object", "fun", "val", "@Column("):
        fa = EXTRACTOR.analyze("@Entity\n" + tail, "T.kt")
        assert not any("erreur interne" in w or "échec" in w for w in fa.warnings), (tail, fa.warnings)
        assert fa.symbols == [], (tail, fa.symbols)
    fa = EXTRACTOR.analyze("class OrderLine(\n    val id: Long,\n    @Column(name = \"q\"\n", "T2.kt")
    assert [s.name for s in fa.symbols] == ["OrderLine"]  # constructeur primaire jamais fermé : ignoré
    assert any("incomplète" in w for w in fa.warnings)
    assert not any("erreur interne" in w for w in fa.warnings)


def test_empty_binary_and_brace_only_inputs():
    assert EXTRACTOR.analyze("", "Empty.java").warnings == ["fichier vide"]
    assert EXTRACTOR.analyze("   \n\n", "Blank.kt").warnings == ["fichier vide"]
    assert EXTRACTOR.analyze("\x00\x01\x02", "Bin.java").warnings == ["contenu binaire ou encodage inattendu : fichier ignoré"]
    fa = EXTRACTOR.analyze("}}}}{{{{", "B.java")
    assert fa.warnings == ["4 accolade(s) non fermée(s) en fin de fichier", "4 accolade(s) fermante(s) en trop"]
    assert fa.symbols == []
    weird = "public class Ünïcode { String s = \"héllo \\\" {\"; char c = '{'; /* { */ void m() { } }\n"
    fa = EXTRACTOR.analyze(weird, "U.java")
    assert fa.warnings == [] and _sym(fa, "m", "method").line_end == 1
    assert EXTRACTOR.analyze(b"class B {}".decode(), "B.java").symbols[0].name == "B"


def test_huge_minified_line_and_unbalanced_quotes_are_fast():
    one = "public class A { " + "int x = 1; " * 20000 + "}"
    t = time.time()
    fa = EXTRACTOR.analyze(one, "A.java")
    assert time.time() - t < 2.5
    assert fa.warnings == ["ligne de plus de 5000 caractères (fichier minifié ?) : déclarations analysées partiellement"]
    assert _sym(fa, "A", "class").line_end == 1
    t = time.time()
    fa = EXTRACTOR.analyze('"' * 500000 + "/*" + "a" * 400000, "Odd.java")
    assert time.time() - t < 2.5 and isinstance(fa, FileAnalysis)
    t = time.time()
    fa = EXTRACTOR.analyze("(" * 300000 + "\n" + "@A(" * 100000 + "\n" + "<" * 100000, "Par.java")
    assert time.time() - t < 2.5 and isinstance(fa, FileAnalysis)


def test_one_megabyte_file_under_one_second():
    big = (CONTROLLER + ENTITY + SERVICE + WEB + JAXRS + HTTP) * 200
    assert len(big) > 1_000_000
    t = time.time()
    fa = EXTRACTOR.analyze(big, "Big.java")
    assert time.time() - t < 2.5
    assert fa.warnings == []
    assert len(fa.endpoints) == 200 * 22
    assert len(fa.models) == 200
    bigk = (KT + KT_ENTITY) * 520
    assert len(bigk) > 1_000_000
    t = time.time()
    fa = EXTRACTOR.analyze(bigk, "Big.kt")
    assert time.time() - t < 2.5
    assert fa.warnings == [] and len(fa.models) == 2 * 520


def test_registry_integration_and_to_dict():
    fa = analyze_file(CONTROLLER, "java", "OrderController.java")
    d = fa.to_dict()
    assert d["language"] == "java"
    assert d["endpoints"][1]["normalized_path"] == "/api/orders/:param"
    assert d["symbols"][1]["name"] == "OrderController" and "qualified_name" not in d["symbols"][1]  # propriété, non sérialisée
    kt = analyze_file(KT, "kotlin", "OrderDto.kt")
    assert kt.language == "kotlin" and kt.to_dict()["models"][0]["name"] == "OrderDto"
    assert get_extractor("java") is get_extractor("kotlin") is EXTRACTOR
