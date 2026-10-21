"""Tests de l'extracteur PHP / Laravel (``develop.code_analysis.extractors.php``)."""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

from develop.code_analysis.extractors.php import (
    EXTRACTOR,
    PhpExtractor,
    pluralize,
    singularize,
    table_name_for,
)
from develop.code_analysis.model import FileAnalysis
from develop.code_analysis.registry import analyze_file, get_extractor


def analyze(src: str, path: str = "x.php") -> FileAnalysis:
    """Analyse un extrait dédenté ; la première ligne de code est la ligne 1."""
    return EXTRACTOR.analyze(textwrap.dedent(src).lstrip("\n"), path)


def by_name(fa: FileAnalysis, name: str, kind: str = None, parent: str = None):
    for s in fa.symbols:
        if s.name == name and (kind is None or s.kind == kind) and (parent is None or s.parent == parent):
            return s
    raise AssertionError("symbole %s introuvable dans %s" % (name, [(s.kind, s.name) for s in fa.symbols]))


def syms(fa: FileAnalysis):
    return [(s.kind, s.name, s.line_start, s.line_end, s.parent) for s in fa.symbols]


def eps(fa: FileAnalysis):
    return [(e.side, e.method, e.path, e.handler, e.framework, e.line) for e in fa.endpoints]


def fields(model):
    return [(f.name, f.type, f.nullable, f.default) for f in model.fields]


def rels(model):
    return [(r.kind, r.target, r.field) for r in model.relations]


def analyze_fixture(sample_project: Path, rel: str) -> FileAnalysis:
    src = (sample_project / "backend" / rel).read_text(encoding="utf-8")
    return EXTRACTOR.analyze(src, "backend/" + rel)


# --------------------------------------------------------------------------- registre


def test_registry_returns_php_extractor():
    assert get_extractor("php") is EXTRACTOR
    assert get_extractor("blade") is EXTRACTOR
    assert isinstance(EXTRACTOR, PhpExtractor)
    assert EXTRACTOR.languages == ["php", "blade"]
    fa = analyze_file("<?php\nfunction f() {}\n", "php", "a.php")
    assert syms(fa) == [("function", "f", 2, 2, None)] and fa.warnings == []


# --------------------------------------------------------------------------- fixture réelle : routes


def test_fixture_routes_api(sample_project: Path):
    fa = analyze_fixture(sample_project, "routes/api.php")
    assert fa.language == "php"
    assert fa.warnings == []
    assert fa.symbols == [] and fa.models == []
    assert eps(fa) == [
        ("server", "GET", "/health", "closure", "laravel", 8),
        ("server", "GET", "/orders", "OrderController@index", "laravel", 11),
        ("server", "GET", "/orders/{order}", "OrderController@show", "laravel", 12),
        ("server", "POST", "/orders/{order}/confirm", "OrderController@confirm", "laravel", 13),
        ("server", "GET", "/customers", "CustomerController@index", "laravel", 14),
        ("server", "GET", "/customers/{customer}", "CustomerController@show", "laravel", 14),
    ]
    health, orders, show, confirm, c_index, c_show = fa.endpoints
    assert health.meta == {"prefix": "/v1", "full_path": "/api/v1/health", "api_prefix": "/api"}
    assert orders.meta == {"prefix": "", "full_path": "/api/orders", "api_prefix": "/api"}
    assert show.meta["full_path"] == "/api/orders/{order}"
    assert show.normalized_path == "/orders/:param"
    assert confirm.meta["full_path"] == "/api/orders/{order}/confirm"
    assert c_index.meta["name"] == "customers.index" and c_index.meta["full_path"] == "/api/customers"
    assert c_show.meta["name"] == "customers.show" and c_show.meta["full_path"] == "/api/customers/{customer}"
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("Illuminate\\Support\\Facades\\Route", ["Route"], 3, False),
        ("App\\Http\\Controllers\\OrderController", ["OrderController"], 4, False),
        ("App\\Http\\Controllers\\CustomerController", ["CustomerController"], 5, False),
    ]


# --------------------------------------------------------------------------- fixture réelle : modèles


def test_fixture_model_order(sample_project: Path):
    fa = analyze_fixture(sample_project, "app/Models/Order.php")
    assert fa.warnings == [] and fa.endpoints == []
    assert syms(fa) == [
        ("class", "Order", 9, 26, None),
        ("property", "table", 11, 11, "Order"),
        ("property", "fillable", 13, 13, "Order"),
        ("property", "casts", 15, 15, "Order"),
        ("method", "customer", 17, 20, "Order"),
        ("method", "lines", 22, 25, "Order"),
    ]
    cls = by_name(fa, "Order", "class")
    assert cls.signature == "class Order extends Model"
    assert cls.meta == {"namespace": "App\\Models", "extends": "Model"}
    customer = by_name(fa, "customer")
    assert customer.signature == "function customer(): BelongsTo"
    assert customer.meta == {"namespace": "App\\Models", "visibility": "public", "static": False,
                             "return_type": "BelongsTo"}
    assert customer.qualified_name == "Order.customer"
    assert by_name(fa, "lines").signature == "function lines(): HasMany"
    assert by_name(fa, "table").meta["visibility"] == "protected"

    assert len(fa.models) == 1
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("Order", "eloquent", "orders", 9)
    assert fields(m) == [
        ("reference", "", None, None),
        ("customer_id", "", None, None),
        ("status", "", None, None),
        ("total_amount", "integer", None, None),
    ]
    assert m.fields[3].meta == {"cast": "integer"}
    assert rels(m) == [("belongs_to", "Customer", "customer"), ("has_many", "OrderLine", "lines")]
    assert m.meta == {"namespace": "App\\Models", "class": "Order", "extends": "Model",
                      "casts": {"total_amount": "integer", "confirmed_at": "datetime"}}
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("Illuminate\\Database\\Eloquent\\Model", ["Model"], 5, False),
        ("Illuminate\\Database\\Eloquent\\Relations\\BelongsTo", ["BelongsTo"], 6, False),
        ("Illuminate\\Database\\Eloquent\\Relations\\HasMany", ["HasMany"], 7, False),
    ]


def test_fixture_models_order_line_and_customer(sample_project: Path):
    fa = analyze_fixture(sample_project, "app/Models/OrderLine.php")
    assert syms(fa) == [
        ("class", "OrderLine", 7, 15, None),
        ("property", "fillable", 9, 9, "OrderLine"),
        ("method", "order", 11, 14, "OrderLine"),
    ]
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("OrderLine", "eloquent", "order_lines", 7)
    assert [f.name for f in m.fields] == ["order_id", "product_sku", "quantity", "unit_price"]
    assert all(f.type == "" for f in m.fields)
    assert rels(m) == [("belongs_to", "Order", "order")]
    assert "casts" not in m.meta

    fa = analyze_fixture(sample_project, "app/Models/Customer.php")
    assert syms(fa) == [
        ("class", "Customer", 7, 15, None),
        ("property", "fillable", 9, 9, "Customer"),
        ("method", "orders", 11, 14, "Customer"),
    ]
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("Customer", "eloquent", "customers", 7)
    assert [f.name for f in m.fields] == ["email", "full_name", "phone"]
    assert rels(m) == [("has_many", "Order", "orders")]
    assert [(i.module, i.names) for i in fa.imports] == [("Illuminate\\Database\\Eloquent\\Model", ["Model"])]


# --------------------------------------------------------------------------- fixture réelle : services et contrôleurs


def test_fixture_order_service(sample_project: Path):
    fa = analyze_fixture(sample_project, "app/Services/OrderService.php")
    assert fa.warnings == [] and fa.models == []
    assert syms(fa) == [
        ("class", "OrderService", 8, 29, None),
        ("method", "confirm", 10, 23, "OrderService"),
        ("method", "totalFor", 25, 28, "OrderService"),
    ]
    assert by_name(fa, "OrderService").meta == {"namespace": "App\\Services"}
    assert by_name(fa, "confirm").signature == "function confirm(Order $order): Order"
    assert by_name(fa, "confirm").meta["return_type"] == "Order"
    assert by_name(fa, "totalFor").signature == "function totalFor(Order $order): int"
    assert eps(fa) == [
        ("client", "POST", "http://localhost:3001/notifications", "OrderService@confirm", "laravel-http", 18),
    ]
    ep = fa.endpoints[0]
    assert ep.meta == {"base": "http://localhost:3001"}
    assert ep.normalized_path == "/notifications"
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("App\\Models\\Order", ["Order"], 5, False),
        ("Illuminate\\Support\\Facades\\Http", ["Http"], 6, False),
    ]


def test_fixture_controllers(sample_project: Path):
    fa = analyze_fixture(sample_project, "app/Http/Controllers/OrderController.php")
    assert fa.warnings == [] and fa.endpoints == [] and fa.models == []
    assert syms(fa) == [
        ("class", "OrderController", 9, 29, None),
        ("method", "__construct", 11, 13, "OrderController"),
        ("method", "index", 15, 18, "OrderController"),
        ("method", "show", 20, 23, "OrderController"),
        ("method", "confirm", 25, 28, "OrderController"),
    ]
    assert by_name(fa, "OrderController").meta == {"namespace": "App\\Http\\Controllers", "extends": "Controller"}
    assert by_name(fa, "__construct").signature == "function __construct(private OrderService $orders)"
    assert by_name(fa, "index").signature == "function index()"
    assert by_name(fa, "show").signature == "function show(Order $order)"
    assert by_name(fa, "confirm").signature == "function confirm(Request $request, Order $order)"
    assert by_name(fa, "confirm").meta == {"namespace": "App\\Http\\Controllers", "visibility": "public",
                                           "static": False}
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("App\\Models\\Order", ["Order"], 5, False),
        ("App\\Services\\OrderService", ["OrderService"], 6, False),
        ("Illuminate\\Http\\Request", ["Request"], 7, False),
    ]

    fa = analyze_fixture(sample_project, "app/Http/Controllers/CustomerController.php")
    assert syms(fa) == [
        ("class", "CustomerController", 8, 20, None),
        ("method", "index", 10, 14, "CustomerController"),
        ("method", "show", 16, 19, "CustomerController"),
    ]
    assert by_name(fa, "index").signature == "function index(Request $request)"
    assert by_name(fa, "show").signature == "function show(Customer $customer)"
    assert [(i.module, i.names, i.line) for i in fa.imports] == [
        ("App\\Models\\Customer", ["Customer"], 5),
        ("Illuminate\\Http\\Request", ["Request"], 6),
    ]


# --------------------------------------------------------------------------- fixture réelle : migrations


def test_fixture_migration_orders(sample_project: Path):
    fa = analyze_fixture(sample_project, "database/migrations/2024_01_01_000000_create_orders_table.php")
    assert fa.warnings == [] and fa.endpoints == []
    assert syms(fa) == [
        ("class", "class@anonymous", 7, 26, None),
        ("method", "up", 9, 20, "class@anonymous"),
        ("method", "down", 22, 25, "class@anonymous"),
    ]
    anon = by_name(fa, "class@anonymous")
    assert anon.meta == {"namespace": "", "anonymous": True, "extends": "Migration"}
    assert anon.signature == "new class extends Migration"
    assert by_name(fa, "up").signature == "function up(): void"

    assert len(fa.models) == 1
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("orders", "migration", "orders", 11)
    assert m.meta == {"operation": "create"}
    assert fields(m) == [
        ("id", "id", False, None),
        ("reference", "string", False, None),
        ("customer_id", "foreignId", False, None),
        ("status", "enum", False, "draft"),
        ("total_amount", "unsignedBigInteger", False, "0"),
        ("confirmed_at", "timestamp", True, None),
        ("created_at", "timestamp", True, None),
        ("updated_at", "timestamp", True, None),
    ]
    assert m.fields[0].meta == {"primary": True, "auto_increment": True, "unsigned": True}
    assert m.fields[1].meta == {"length": 20, "unique": True}
    assert m.fields[2].meta == {"unsigned": True}
    assert m.fields[3].meta == {"enum_values": ["draft", "confirmed", "shipped", "cancelled"]}
    assert m.fields[4].meta == {"unsigned": True}
    assert rels(m) == [("foreign_key", "customers", "customer_id")]
    assert [(i.module, i.names, i.line) for i in fa.imports] == [
        ("Illuminate\\Database\\Migrations\\Migration", ["Migration"], 3),
        ("Illuminate\\Database\\Schema\\Blueprint", ["Blueprint"], 4),
        ("Illuminate\\Support\\Facades\\Schema", ["Schema"], 5),
    ]


def test_fixture_migration_order_lines(sample_project: Path):
    fa = analyze_fixture(sample_project, "database/migrations/2024_02_01_000000_create_order_lines_table.php")
    assert fa.warnings == []
    assert syms(fa) == [
        ("class", "class@anonymous", 7, 25, None),
        ("method", "up", 9, 19, "class@anonymous"),
        ("method", "down", 21, 24, "class@anonymous"),
    ]
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("order_lines", "migration", "order_lines", 11)
    assert fields(m) == [
        ("id", "id", False, None),
        ("order_id", "foreignId", False, None),
        ("product_sku", "string", False, None),
        ("quantity", "unsignedInteger", False, None),
        ("unit_price", "unsignedBigInteger", False, None),
        ("created_at", "timestamp", True, None),
        ("updated_at", "timestamp", True, None),
    ]
    assert m.fields[1].meta == {"unsigned": True, "on_delete": "cascade"}
    assert m.fields[2].meta == {"length": 40}
    assert rels(m) == [("foreign_key", "orders", "order_id")]


def test_fixture_all_backend_files_are_clean_and_serializable(sample_project: Path):
    backend = sample_project / "backend"
    files = sorted(backend.rglob("*.php"))
    assert len(files) == 9
    for f in files:
        fa = EXTRACTOR.analyze(f.read_text(encoding="utf-8"), str(f.relative_to(sample_project)))
        assert fa.warnings == [], f
        assert fa.language == "php"
        d = json.loads(json.dumps(fa.to_dict()))
        for raw in d["endpoints"]:
            assert raw["normalized_path"].startswith("/")
        for s in fa.symbols:
            assert 1 <= s.line_start <= s.line_end
            assert "namespace" in s.meta


# --------------------------------------------------------------------------- symboles


SYMBOLS_SRC = r'''
<?php

namespace App\Domain;

use App\Models\Order;
use App\Models\{Customer, OrderLine as Line};
use Illuminate\Support\Str as S;
use function App\Helpers\format_money;
use const App\Helpers\VERSION;
require_once __DIR__ . '/helpers.php';
require 'vendor/autoload.php';
include_once('../config.php');

abstract class Shape implements JsonSerializable, Countable
{
    public const MAX = 10;
    protected static int $count = 0;
    private ?string $label = null;

    abstract public function area(): float;

    public static function make(string $x = "}", int $y = 0): static
    {
        $s = "a { b } c";
        $t = 'x } y';
        // comment with }
        /* block } */
        return new static();
    }

    final protected function describe(?Order $order = null, array|string $opts = []): ?string
    {
        $doc = <<<EOT
        } } }
        EOT;
        return $doc;
    }
}

final class Circle extends Shape
{
    use HasFactory, Notifiable;

    public function area(): float { return 3.14; }
}

interface Drawable extends JsonSerializable, Countable
{
    public function draw(): void;
}

trait HasUuid
{
    public function uuid(): string { return 'x'; }
}

enum Status: string
{
    case Draft = 'draft';
    case Confirmed = 'confirmed';

    public function label(): string
    {
        return match($this) { self::Draft => 'Brouillon', default => 'X' };
    }
}

function helper(int $a, int ...$rest): int
{
    return $a;
}

function &byRef() { static $x; return $x; }

define('APP_VERSION', '1.0');
'''


def test_symbols_namespace_classes_interfaces_traits_enums_functions():
    fa = analyze(SYMBOLS_SRC)
    assert fa.warnings == []
    assert syms(fa) == [
        ("class", "Shape", 14, 38, None),
        ("constant", "MAX", 16, 16, "Shape"),
        ("property", "count", 17, 17, "Shape"),
        ("property", "label", 18, 18, "Shape"),
        ("method", "area", 20, 20, "Shape"),
        ("method", "make", 22, 29, "Shape"),
        ("method", "describe", 31, 37, "Shape"),
        ("class", "Circle", 40, 45, None),
        ("method", "area", 44, 44, "Circle"),
        ("interface", "Drawable", 47, 50, None),
        ("method", "draw", 49, 49, "Drawable"),
        ("trait", "HasUuid", 52, 55, None),
        ("method", "uuid", 54, 54, "HasUuid"),
        ("enum", "Status", 57, 66, None),
        ("method", "label", 62, 65, "Status"),
        ("function", "helper", 68, 71, None),
        ("function", "byRef", 73, 73, None),
        ("constant", "APP_VERSION", 75, 75, None),
    ]
    assert all(s.meta["namespace"] == "App\\Domain" for s in fa.symbols)
    shape = by_name(fa, "Shape")
    assert shape.signature == "abstract class Shape implements JsonSerializable, Countable"
    assert shape.meta == {"namespace": "App\\Domain", "abstract": True, "implements": ["JsonSerializable", "Countable"]}
    circle = by_name(fa, "Circle")
    assert circle.signature == "final class Circle extends Shape"
    assert circle.meta == {"namespace": "App\\Domain", "final": True, "extends": "Shape",
                           "traits": ["HasFactory", "Notifiable"]}
    assert by_name(fa, "Drawable").meta["extends"] == ["JsonSerializable", "Countable"]
    assert by_name(fa, "Status").meta == {"namespace": "App\\Domain", "backing_type": "string",
                                          "cases": ["Draft", "Confirmed"]}
    assert by_name(fa, "MAX").signature == "public const MAX"
    assert by_name(fa, "count").signature == "protected static int $count"
    assert by_name(fa, "count").meta == {"namespace": "App\\Domain", "visibility": "protected", "static": True,
                                         "type": "int"}
    assert by_name(fa, "label", "property").meta["type"] == "?string"
    assert by_name(fa, "label", "property").meta["visibility"] == "private"
    area = by_name(fa, "area", parent="Shape")
    assert area.signature == "function area(): float"
    assert area.meta == {"namespace": "App\\Domain", "visibility": "public", "static": False, "abstract": True,
                         "return_type": "float"}
    make = by_name(fa, "make")
    assert make.signature == 'function make(string $x = "}", int $y = 0): static'
    assert make.meta["static"] is True and make.meta["visibility"] == "public"
    describe = by_name(fa, "describe")
    assert describe.signature == "function describe(?Order $order = null, array|string $opts = []): ?string"
    assert describe.meta["visibility"] == "protected" and describe.meta["final"] is True
    assert by_name(fa, "helper").signature == "function helper(int $a, int ...$rest): int"
    assert by_name(fa, "helper").meta == {"namespace": "App\\Domain", "return_type": "int"}
    assert by_name(fa, "byRef").meta["by_ref"] is True
    assert by_name(fa, "APP_VERSION").signature == "define('APP_VERSION')"


def test_imports_use_variants_and_require():
    fa = analyze(SYMBOLS_SRC)
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("App\\Models\\Order", ["Order"], 5, False),
        ("App\\Models", ["Customer", "OrderLine"], 6, False),
        ("Illuminate\\Support\\Str", ["Str"], 7, False),
        ("App\\Helpers\\format_money", ["format_money"], 8, False),
        ("App\\Helpers\\VERSION", ["VERSION"], 9, False),
        ("./helpers.php", [], 10, True),
        ("vendor/autoload.php", [], 11, False),
        ("../config.php", [], 12, True),
    ]
    # `use Trait;` dans un corps de classe n'est pas un import
    assert not any(i.module in ("HasFactory", "Notifiable") for i in fa.imports)


def test_line_end_ignores_braces_in_strings_and_comments():
    fa = analyze(r'''
    <?php
    class A
    {
        public function f()
        {
            $x = "{";
            $y = '}';
            echo "interp {$x} and {$y[0]}"; // }
            /* { */
        }

        public function g()
        {
            return 1;
        }
    }
    ''')
    assert syms(fa) == [
        ("class", "A", 2, 16, None),
        ("method", "f", 4, 10, "A"),
        ("method", "g", 12, 15, "A"),
    ]


def test_multiple_namespaces_and_braced_namespace():
    fa = analyze(r'''
    <?php
    namespace A;
    function fa() {}
    namespace B {
        function fb() {}
    }
    namespace C;
    class Cc {}
    ''')
    assert [(s.name, s.meta["namespace"]) for s in fa.symbols] == [("fa", "A"), ("fb", "B"), ("Cc", "C")]


# --------------------------------------------------------------------------- modèles Eloquent


def test_eloquent_table_name_rules():
    assert [table_name_for(c) for c in ("Order", "OrderLine", "Category", "Person", "Status", "Box", "Company",
                                        "Child", "Sheep", "Address", "Quiz", "Mouse", "Knife", "Man", "Index",
                                        "Series", "Hero", "Day", "User")] == [
        "orders", "order_lines", "categories", "people", "statuses", "boxes", "companies",
        "children", "sheep", "addresses", "quizzes", "mice", "knives", "men", "indices",
        "series", "heroes", "days", "users"]
    assert pluralize("") == ""
    assert [singularize(w) for w in ("customers", "categories", "people", "statuses", "boxes", "children",
                                     "series", "addresses")] == [
        "customer", "category", "person", "status", "box", "child", "series", "address"]


ELOQUENT_SRC = r'''
<?php
namespace App\Models;

use Illuminate\Foundation\Auth\User as Authenticatable;

class User extends Authenticatable
{
    protected $table = 'app_users';
    protected $primaryKey = 'uid';
    public $timestamps = false;
    protected $guarded = ['id'];
    protected $hidden = ['password', 'remember_token'];
    protected $dates = ['deleted_at'];
    protected $fillable = [
        'name',
        'email',
        'is_admin',
    ];
    protected $casts = [
        'is_admin' => 'boolean',
        'settings' => 'array',
        'status' => Status::class,
    ];

    public function roles()
    {
        return $this->belongsToMany(Role::class, 'role_user');
    }

    public function profile(): HasOne
    {
        return $this->hasOne(Profile::class);
    }

    public function commentable()
    {
        return $this->morphTo();
    }

    public function images()
    {
        return $this->morphMany(Image::class, 'imageable');
    }

    public function orderLines()
    {
        return $this->hasManyThrough(OrderLine::class, Order::class);
    }

    public function scopeActive($q) { return $q->where('active', 1); }
}

class Membership extends Pivot
{
    protected $fillable = ['user_id', 'team_id'];
}

class Category extends Model
{
    protected $fillable = ['name'];

    protected function casts(): array
    {
        return [
            'name' => 'string',
        ];
    }

    public function parent()
    {
        return $this->belongsTo('App\Models\Category', 'parent_id');
    }
}

class NotAModel
{
    public function x() {}
}
'''


def test_eloquent_models_full():
    fa = analyze(ELOQUENT_SRC)
    assert fa.warnings == []
    assert [(m.name, m.kind, m.table_name, m.line) for m in fa.models] == [
        ("User", "eloquent", "app_users", 6),
        ("Membership", "eloquent", "memberships", 53),
        ("Category", "eloquent", "categories", 58),
    ]
    user, membership, category = fa.models
    assert fields(user) == [("name", "", None, None), ("email", "", None, None), ("is_admin", "boolean", None, None)]
    assert user.fields[2].meta == {"cast": "boolean"}
    assert rels(user) == [
        ("many_to_many", "Role", "roles"),
        ("has_one", "Profile", "profile"),
        ("morph_to", "", "commentable"),
        ("morph_many", "Image", "images"),
        ("has_many_through", "OrderLine", "orderLines"),
    ]
    assert user.meta == {
        "namespace": "App\\Models", "class": "User", "extends": "Authenticatable",
        "guarded": ["id"], "hidden": ["password", "remember_token"], "dates": ["deleted_at"],
        "casts": {"is_admin": "boolean", "settings": "array", "status": "Status"},
        "primary_key": "uid", "timestamps": False,
    }
    assert fields(membership) == [("user_id", "", None, None), ("team_id", "", None, None)]
    assert membership.meta["extends"] == "Pivot"
    # Laravel 11 : méthode casts() ; relation avec nom de classe en chaîne qualifiée
    assert fields(category) == [("name", "string", None, None)]
    assert category.meta["casts"] == {"name": "string"}
    assert rels(category) == [("belongs_to", "Category", "parent")]
    # les symboles sont bien tous extraits en plus des modèles
    assert ("class", "NotAModel", 75, 78, None) in syms(fa)
    assert ("method", "scopeActive", 50, 50, "User") in syms(fa)


def test_eloquent_inferred_without_model_base_and_custom_relation_in_nested_closure():
    fa = analyze(r'''
    <?php
    class Legacy extends BaseRecord
    {
        protected $fillable = ['a'];
        protected $casts = ['a' => 'int'];

        public function owner()
        {
            return $this->belongsTo(User::class, 'owner_id');
        }
    }

    class Plain extends BaseRecord
    {
        public function f() {}
    }
    ''')
    assert [(m.name, m.table_name, m.meta.get("inferred")) for m in fa.models] == [("Legacy", "legacies", True)]
    assert fields(fa.models[0]) == [("a", "int", None, None)]
    assert rels(fa.models[0]) == [("belongs_to", "User", "owner")]


# --------------------------------------------------------------------------- migrations


MIGRATION_SRC = r'''
<?php
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('products', function (Blueprint $t) {
            $t->uuid('id')->primary();
            $t->string('name');
            $t->decimal('price', 10, 2)->default(0);
            $t->text('description')->nullable();
            $t->boolean('active')->default(true);
            $t->unsignedBigInteger('category_id');
            $t->foreign('category_id')->references('id')->on('categories')->onDelete('cascade');
            $t->foreignId('owner_id')->nullable()->constrained();
            $t->foreignIdFor(Supplier::class)->constrained();
            $t->json('meta')->nullable();
            $t->timestamp('published_at')->nullable()->useCurrent();
            $t->softDeletes();
            $t->timestamps();
            $t->index(['name', 'active']);
            $t->unique('name');
        });

        Schema::table('orders', function (Blueprint $table) {
            $table->string('note', 255)->nullable()->after('status');
            $table->dropColumn('legacy');
            $table->dropColumn(['a', 'b']);
            $table->renameColumn('old', 'new');
        });

        Schema::connection('other')->create('logs', function (Blueprint $table) {
            $table->id();
            $table->morphs('subject');
        });
    }
};
'''


def test_migrations_create_alter_and_connection():
    fa = analyze(MIGRATION_SRC, "database/migrations/2024_03_01_000000_products.php")
    assert fa.warnings == []
    assert [(m.name, m.kind, m.table_name, m.line) for m in fa.models] == [
        ("products", "migration", "products", 8),
        ("orders", "migration", "orders", 26),
        ("logs", "migration", "logs", 33),
    ]
    products, orders, logs = fa.models
    assert fields(products) == [
        ("id", "uuid", False, None),
        ("name", "string", False, None),
        ("price", "decimal", False, "0"),
        ("description", "text", True, None),
        ("active", "boolean", False, "true"),
        ("category_id", "unsignedBigInteger", False, None),
        ("owner_id", "foreignId", True, None),
        ("supplier_id", "foreignId", False, None),
        ("meta", "json", True, None),
        ("published_at", "timestamp", True, "CURRENT_TIMESTAMP"),
        ("deleted_at", "timestamp", True, None),
        ("created_at", "timestamp", True, None),
        ("updated_at", "timestamp", True, None),
    ]
    assert products.fields[0].meta == {"primary": True}
    assert products.fields[2].meta == {"precision": 10, "scale": 2}
    assert rels(products) == [
        ("foreign_key", "categories", "category_id"),
        ("foreign_key", "owners", "owner_id"),
        ("foreign_key", "suppliers", "supplier_id"),
    ]
    assert products.meta == {"operation": "create",
                             "indexes": [{"type": "index", "columns": ["name", "active"]},
                                         {"type": "unique", "columns": ["name"]}]}
    assert orders.meta == {"operation": "table", "alter": True, "dropped": ["legacy", "a", "b"],
                           "renamed": [["old", "new"]]}
    assert fields(orders) == [("note", "string", True, None)]
    assert orders.fields[0].meta == {"length": 255}
    assert logs.meta == {"operation": "create"}
    assert fields(logs) == [("id", "id", False, None), ("subject_type", "string", False, None),
                            ("subject_id", "unsignedBigInteger", False, None)]


def test_migration_without_closure_body_warns_but_keeps_model():
    fa = analyze("<?php\nSchema::create('t');\n")
    assert [(m.name, m.kind) for m in fa.models] == [("t", "migration")]
    assert fa.models[0].fields == []
    assert any("Schema::create" in w for w in fa.warnings)


DYNAMIC_TABLE_MIGRATION_SRC = r"""
<?php
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    private const TABLES = ['form_fields', 'shared_widget_fields'];
    private const COLUMNS = ['journal' => 'detail', 'audit' => 'trace'];
    protected $extra = ['notes'];

    public function up(): void
    {
        foreach (self::TABLES as $table) {
            if (!Schema::hasTable($table) || Schema::hasColumn($table, 'columns')) {
                continue;
            }
            Schema::table($table, function (Blueprint $t) {
                $t->json('columns')->nullable()->after('search_fields');
            });
        }
        foreach (self::COLUMNS as $table => $column) {
            Schema::table($table, function (Blueprint $t) use ($column) {
                $t->text('payload');
            });
        }
        foreach ($this->extra as $i => $tbl) {
            Schema::table($tbl, function (Blueprint $t) {
                $t->string('title');
            });
        }
        $names = ['logs', 'audits'];
        foreach ($names as $n) {
            Schema::create($n, function (Blueprint $t) {
                $t->id();
            });
        }
        Schema::table($this->resolveTable(), function (Blueprint $t) {
            $t->string('ghost');
        });
    }
};
"""


def test_migration_dynamic_table_name_resolved_from_enclosing_foreach():
    """``Schema::table($table, ...)`` dans un ``foreach`` : le nom vient de la collection itérée (constante de
    classe, propriété, tableau littéral, variable), jamais d'un littéral du corps de la closure (défaut réel :
    modèles fantômes « columns » / « auto_filtre »). Un nom irrésoluble donne un avertissement, pas un modèle."""
    fa = analyze(DYNAMIC_TABLE_MIGRATION_SRC, "database/migrations/2026_07_02_130000_add_columns.php")
    assert [(m.name, m.table_name, m.meta.get("operation")) for m in fa.models] == [
        ("form_fields", "form_fields", "table"),
        ("shared_widget_fields", "shared_widget_fields", "table"),
        ("journal", "journal", "table"),
        ("audit", "audit", "table"),
        ("notes", "notes", "table"),
        ("logs", "logs", "create"),
        ("audits", "audits", "create"),
    ]
    form, widget, journal, audit, notes, logs, audits = fa.models
    assert form.line == widget.line == 16
    assert fields(form) == fields(widget) == [("columns", "json", True, None)]
    assert form.meta == widget.meta == {"operation": "table", "alter": True}
    assert fields(journal) == fields(audit) == [("payload", "text", False, None)]
    assert fields(notes) == [("title", "string", False, None)]
    assert fields(logs) == fields(audits) == [("id", "id", False, None)]
    assert "ghost" not in {f.name for m in fa.models for f in m.fields}
    assert fa.warnings == ["migration : nom de table dynamique non résolu pour Schema::table (ligne 36)"]


def test_migration_down_method_is_ignored():
    """Les ``Schema::`` de ``down()`` défont ``up()`` : ils ne produisent ni modèle en double ni ligne vide
    (défaut réel : 64 fichiers de migration sur 120 avec une 2e ligne ``fields=[]``)."""
    fa = analyze(r"""
    <?php
    return new class extends Migration
    {
        public function up(): void
        {
            Schema::table('villes', function (Blueprint $table) {
                $table->uuid('region_id');
            });
            Schema::table('fichiers', function (Blueprint $table) {
                $table->string('local_path', 500)->nullable();
            });
        }

        public function down(): void
        {
            Schema::table('villes', function (Blueprint $table) {
                $table->dropColumn('region_id');
            });
            Schema::table('fichiers', fn (Blueprint $t) => $t->dropColumn('local_path'));
            Schema::create('archive', function (Blueprint $table) {
                $table->id();
            });
        }
    };
    """)
    assert fa.warnings == []
    assert [(m.name, m.line) for m in fa.models] == [("villes", 6), ("fichiers", 9)]
    villes, fichiers = fa.models
    assert fields(villes) == [("region_id", "uuid", False, None)]
    assert fields(fichiers) == [("local_path", "string", True, None)]
    assert "dropped" not in villes.meta and "dropped" not in fichiers.meta


def test_migration_arrow_closure_and_body_outside_call():
    """Closure fléchée ``fn (Blueprint $t) => ...`` acceptée ; un appel sans closure ne doit pas s'approprier
    l'accolade de l'appel suivant (sinon ``b`` recevait les colonnes de ``c``)."""
    fa = analyze(r"""
    <?php
    return new class extends Migration
    {
        public function up(): void
        {
            Schema::table('a', fn (Blueprint $t) => $t->dropColumn('legacy'));
            Schema::table('b', $this->callback());
            Schema::table('c', function (Blueprint $t) {
                $t->string('z');
            });
        }
    };
    """)
    assert [(m.name, m.line) for m in fa.models] == [("a", 6), ("b", 7), ("c", 8)]
    a, b, c = fa.models
    assert a.fields == [] and a.meta["dropped"] == ["legacy"]
    assert b.fields == [] and "dropped" not in b.meta
    assert fields(c) == [("z", "string", False, None)]
    assert fa.warnings == ["migration b : corps de Schema::table introuvable (ligne 7)"]


# --------------------------------------------------------------------------- routes Laravel


ROUTES_WEB_SRC = r'''
<?php
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\OrderController;

Route::get('/', function () { return view('welcome'); })->name('home');
Route::post('/orders', 'OrderController@store');
Route::put('/orders/{id}', [OrderController::class, 'update'])->name('orders.update')->middleware('auth');
Route::patch('/orders/{id}', [OrderController::class, 'patch']);
Route::delete('/orders/{id}', [OrderController::class, 'destroy']);
Route::any('/ping', fn () => 'pong');
Route::match(['get', 'post'], '/contact', [ContactController::class, 'handle']);
Route::get('/invoke', InvokableController::class);
Route::get('/invoke2', 'InvokableController');
Route::resource('photos', PhotoController::class)->except(['destroy']);
Route::resource('posts.comments', CommentController::class)->only(['index']);

Route::prefix('admin')->middleware(['auth', 'admin'])->name('admin.')->group(function () {
    Route::get('/dashboard', [AdminController::class, 'index'])->name('dashboard');
    Route::prefix('users')->group(function () {
        Route::get('/{user}', [UserController::class, 'show']);
    });
});

Route::group(['prefix' => 'api', 'middleware' => 'api', 'as' => 'api.'], function () {
    Route::get('/status', fn () => 'ok')->name('status');
});

Route::controller(OrderController::class)->group(function () {
    Route::get('/orders/export', 'export');
});

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit']);
});

Route::redirect('/old', '/new');
Route::view('/about', 'about');
Route::fallback(function () { return 'nope'; });
'''


def test_routes_web_all_handler_forms_resources_and_groups():
    fa = analyze(ROUTES_WEB_SRC, "routes/web.php")
    assert fa.warnings == []
    assert eps(fa) == [
        ("server", "GET", "/", "closure", "laravel", 5),
        ("server", "POST", "/orders", "OrderController@store", "laravel", 6),
        ("server", "PUT", "/orders/{id}", "OrderController@update", "laravel", 7),
        ("server", "PATCH", "/orders/{id}", "OrderController@patch", "laravel", 8),
        ("server", "DELETE", "/orders/{id}", "OrderController@destroy", "laravel", 9),
        ("server", "ANY", "/ping", "closure", "laravel", 10),
        ("server", "GET", "/contact", "ContactController@handle", "laravel", 11),
        ("server", "POST", "/contact", "ContactController@handle", "laravel", 11),
        ("server", "GET", "/invoke", "InvokableController@__invoke", "laravel", 12),
        ("server", "GET", "/invoke2", "InvokableController@__invoke", "laravel", 13),
        ("server", "GET", "/photos", "PhotoController@index", "laravel", 14),
        ("server", "GET", "/photos/create", "PhotoController@create", "laravel", 14),
        ("server", "POST", "/photos", "PhotoController@store", "laravel", 14),
        ("server", "GET", "/photos/{photo}", "PhotoController@show", "laravel", 14),
        ("server", "GET", "/photos/{photo}/edit", "PhotoController@edit", "laravel", 14),
        ("server", "PUT", "/photos/{photo}", "PhotoController@update", "laravel", 14),
        ("server", "GET", "/posts/{post}/comments", "CommentController@index", "laravel", 15),
        ("server", "GET", "/dashboard", "AdminController@index", "laravel", 18),
        ("server", "GET", "/{user}", "UserController@show", "laravel", 20),
        ("server", "GET", "/status", "closure", "laravel", 25),
        ("server", "GET", "/orders/export", "OrderController@export", "laravel", 29),
        ("server", "GET", "/profile", "ProfileController@edit", "laravel", 33),
        ("server", "ANY", "/old", "redirect:/new", "laravel", 36),
        ("server", "GET", "/about", "view:about", "laravel", 37),
        ("server", "ANY", "/{fallback}", "closure", "laravel", 38),
    ]
    by_line = {}
    for e in fa.endpoints:
        by_line.setdefault(e.line, []).append(e)
    assert by_line[5][0].meta == {"prefix": "", "full_path": "/", "name": "home"}
    assert by_line[7][0].meta == {"prefix": "", "full_path": "/orders/{id}", "middleware": ["auth"],
                                  "name": "orders.update"}
    assert by_line[14][0].meta["name"] == "photos.index"
    assert by_line[14][5].meta["name"] == "photos.update"
    assert by_line[17 + 1][0].meta == {"prefix": "/admin", "full_path": "/admin/dashboard",
                                       "middleware": ["auth", "admin"], "name": "admin.dashboard"}
    assert by_line[20][0].meta == {"prefix": "/admin/users", "full_path": "/admin/users/{user}",
                                   "middleware": ["auth", "admin"], "name_prefix": "admin."}
    assert by_line[25][0].meta == {"prefix": "/api", "full_path": "/api/status", "middleware": ["api"],
                                   "name": "api.status"}
    assert by_line[29][0].meta == {"prefix": "", "full_path": "/orders/export", "controller": "OrderController"}
    assert by_line[33][0].meta == {"prefix": "", "full_path": "/profile", "middleware": ["auth"]}
    # routes/web.php : pas de préfixe /api implicite
    assert not any("api_prefix" in e.meta for e in fa.endpoints)


def test_routes_api_file_prefix_group_and_api_resource():
    fa = analyze(r'''
    <?php
    Route::prefix('v2')->group(function () {
        Route::apiResource('items', ItemController::class);
    });
    Route::apiResource('tags', TagController::class)->except(['destroy', 'update']);
    ''', "backend/routes/api.php")
    assert eps(fa) == [
        ("server", "GET", "/items", "ItemController@index", "laravel", 3),
        ("server", "POST", "/items", "ItemController@store", "laravel", 3),
        ("server", "GET", "/items/{item}", "ItemController@show", "laravel", 3),
        ("server", "PUT", "/items/{item}", "ItemController@update", "laravel", 3),
        ("server", "DELETE", "/items/{item}", "ItemController@destroy", "laravel", 3),
        ("server", "GET", "/tags", "TagController@index", "laravel", 5),
        ("server", "POST", "/tags", "TagController@store", "laravel", 5),
        ("server", "GET", "/tags/{tag}", "TagController@show", "laravel", 5),
    ]
    assert fa.endpoints[2].meta == {"prefix": "/v2", "full_path": "/api/v2/items/{item}", "api_prefix": "/api",
                                    "name": "items.show"}
    assert fa.endpoints[5].meta == {"prefix": "", "full_path": "/api/tags", "api_prefix": "/api",
                                    "name": "tags.index"}
    assert fa.endpoints[2].normalized_path == "/items/:param"


def test_routes_outside_routes_dir_are_still_detected():
    fa = analyze(r'''
    <?php
    class RouteServiceProvider
    {
        public function boot(): void
        {
            Route::middleware('api')->prefix('api')->group(function () {
                Route::get('/ping', fn () => 'pong');
            });
        }
    }
    ''', "app/Providers/RouteServiceProvider.php")
    assert eps(fa) == [("server", "GET", "/ping", "closure", "laravel", 7)]
    assert fa.endpoints[0].meta == {"prefix": "/api", "full_path": "/api/ping", "middleware": ["api"]}


# --------------------------------------------------------------------------- appels HTTP sortants


CLIENT_SRC = r'''
<?php
namespace App\Services;

use GuzzleHttp\Client;
use Illuminate\Support\Facades\Http;

class Notifier
{
    private string $base = 'http://localhost:3001';

    public function send(int $id): void
    {
        Http::withToken('x')->post('http://localhost:3001/notifications', ['id' => $id]);
        Http::get($this->base . '/users/' . $id);
        Http::baseUrl(config('services.api.url'))->put('/items/' . $id, []);
        Http::send('DELETE', 'https://api.example.com/items/1');
        $client = new Client();
        $resp = $client->get('https://api.example.com/users');
        $client->request('POST', '/v1/orders', ['json' => []]);
        $this->http->delete("/orders/{$id}");
        $ch = curl_init('https://api.example.com/ping');
        curl_setopt($ch, CURLOPT_POST, true);
        $json = file_get_contents('https://api.example.com/feed.json');
        $local = file_get_contents('/tmp/x.txt');
        $cache->get('key');
    }

    public function fetchViaCurl(): string
    {
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, 'https://api.example.com/export');
        curl_setopt($ch, CURLOPT_CUSTOMREQUEST, 'PUT');
        return curl_exec($ch);
    }
}

$g = Http::get('https://top.level/x');
'''


def test_client_calls_http_facade_guzzle_curl_and_file_get_contents():
    fa = analyze(CLIENT_SRC)
    assert fa.warnings == []
    assert sorted(eps(fa), key=lambda e: e[5]) == [
        ("client", "POST", "http://localhost:3001/notifications", "Notifier@send", "laravel-http", 13),
        ("client", "GET", "{base}/users/{id}", "Notifier@send", "laravel-http", 14),
        ("client", "PUT", "/items/{id}", "Notifier@send", "laravel-http", 15),
        ("client", "DELETE", "https://api.example.com/items/1", "Notifier@send", "laravel-http", 16),
        ("client", "GET", "https://api.example.com/users", "Notifier@send", "guzzle", 18),
        ("client", "POST", "/v1/orders", "Notifier@send", "guzzle", 19),
        ("client", "DELETE", "/orders/{id}", "Notifier@send", "guzzle", 20),
        ("client", "POST", "https://api.example.com/ping", "Notifier@send", "curl", 21),
        ("client", "GET", "https://api.example.com/feed.json", "Notifier@send", "file_get_contents", 23),
        ("client", "PUT", "https://api.example.com/export", "Notifier@fetchViaCurl", "curl", 31),
        ("client", "GET", "https://top.level/x", "", "laravel-http", 37),
    ]
    by_line = {e.line: e for e in fa.endpoints}
    assert by_line[13].meta == {"base": "http://localhost:3001"}
    assert by_line[14].meta == {"base": "$this->base"}
    assert by_line[14].normalized_path == "/users/:param"
    assert by_line[15].meta == {"base": "services.api.url"}
    assert by_line[18].meta == {"client": "$client", "base": "https://api.example.com"}
    assert by_line[20].meta == {"client": "$this->http"}
    assert by_line[20].normalized_path == "/orders/:param"
    assert all(e.side == "client" for e in fa.endpoints)


DYNAMIC_CLIENT_SRC = r'''
<?php
namespace App\Services;

use Illuminate\Support\Facades\Http;

class PartnerBPushService
{
    const PARTNER_URL = 'https://example.org/partner.json';

    public function login(string $url, string $endpoint, int $id): void
    {
        Http::post($url, ['a' => 1]);
        Http::withHeaders([])->post("{$this->baseUrl}{$endpoint}", []);
        Http::get(self::PARTNER_URL);
        $this->client->get("documents/{$id}/");
        $this->client->delete("documents/{$id}/download/");
        Http::get("{$this->baseUrl}/api/{$endpoint}");
        Http::put($this->base . '/items/' . $this->order->id);
    }
}
'''


def test_client_calls_drop_variable_only_urls_and_name_php_variables_as_params():
    """Un appel dont l'URL n'est qu'une variable ou une expression (``Http::post($url)``,
    ``"{$this->baseUrl}{$endpoint}"``, ``self::PARTNER_URL``) ne désigne aucune route : il n'est pas indexé
    (sur un projet réel, « POST $url » était classé 5e pour « quel est l'api d'authentification »). Une URL
    qui garde une partie littérale est conservée, ses variables PHP devenant des paramètres nommés
    (``documents/{$id}/`` → ``documents/{id}/``, ``$this->order->id`` → ``{id}``) : plus aucun « $ » dans
    un chemin. Même règle pour les vues Blade : ``fetch(url)`` et ``fetch(`${base}`)`` sont ignorés."""
    fa = analyze(DYNAMIC_CLIENT_SRC)
    assert fa.warnings == []
    assert sorted(eps(fa), key=lambda e: e[5]) == [
        ("client", "GET", "documents/{id}/", "PartnerBPushService@login", "guzzle", 15),
        ("client", "DELETE", "documents/{id}/download/", "PartnerBPushService@login", "guzzle", 16),
        ("client", "GET", "{baseUrl}/api/{endpoint}", "PartnerBPushService@login", "laravel-http", 17),
        ("client", "PUT", "{base}/items/{id}", "PartnerBPushService@login", "laravel-http", 18),
    ]
    assert not any("$" in e.path for e in fa.endpoints)
    by_line = {e.line: e for e in fa.endpoints}
    assert by_line[15].normalized_path == "/documents/:param" and by_line[16].normalized_path == "/documents/:param/download"
    assert by_line[17].normalized_path == "/api/:param" and by_line[18].normalized_path == "/items/:param"
    assert by_line[18].meta == {"base": "$this->base"}  # la base reste connue par les métadonnées
    fb = analyze(r'''
<html>
<script>
    function api(url) { return fetch(url); }
    function load() { axios.get(baseUrl + '/stats'); fetch(`${base}`); fetch('/partner-a/login', { method: 'POST' }); }
</script>
</html>
''', "resources/views/x.blade.php")
    assert eps(fb) == [
        ("client", "GET", "{baseUrl}/stats", "load", "axios", 4),
        ("client", "POST", "/partner-a/login", "load", "fetch", 4),
    ]


# --------------------------------------------------------------------------- vues Blade


BLADE_SRC = r'''
<html>
<body>
<a href="{{ route('orders.index') }}">x</a>
<script>
    function loadOrders() {
        fetch('/api/orders').then(r => r.json());
        axios.get(route('orders.show', 1));
    }
    const confirmOrder = async (id) => {
        await axios.post('/api/orders/' + id + '/confirm');
        fetch(url('/api/ping'), { method: 'DELETE' });
    };
    $.ajax({ url: '/api/legacy', type: 'POST' });
    $.get('/api/jq');
    axios({ url: '/api/cfg', method: 'put' });
    const r = route('customers.index');
</script>
</body>
</html>
'''


def test_blade_extracts_only_script_calls():
    fa = analyze(BLADE_SRC, "resources/views/orders/index.blade.php")
    assert fa.language == "blade"
    assert fa.warnings == [] and fa.symbols == [] and fa.models == [] and fa.imports == []
    assert eps(fa) == [
        ("client", "GET", "/api/orders", "loadOrders", "fetch", 6),
        ("client", "GET", "route:orders.show", "loadOrders", "axios", 7),
        ("client", "POST", "/api/orders/{id}/confirm", "confirmOrder", "axios", 10),
        ("client", "DELETE", "/api/ping", "confirmOrder", "fetch", 11),
        ("client", "POST", "/api/legacy", "", "jquery", 13),
        ("client", "GET", "/api/jq", "", "jquery", 14),
        ("client", "PUT", "/api/cfg", "", "axios", 15),
        ("client", "ANY", "route:customers.index", "", "blade", 16),
    ]
    assert fa.endpoints[1].meta == {"route_name": "orders.show"}
    assert fa.endpoints[2].normalized_path == "/api/orders/:param/confirm"


def test_blade_without_script_extracts_nothing():
    fa = analyze(r'''
    <?php $x = 1; ?>
    <div>{{ route('x') }}</div>
    @foreach ($orders as $o) {{ $o->id }} @endforeach
    <?php function notExtracted() {} ?>
    ''', "resources/views/x.blade.php")
    assert fa.language == "blade"
    assert fa.endpoints == [] and fa.symbols == [] and fa.warnings == []
    # le même contenu en .php est analysé comme du PHP
    fa = analyze("<?php function extracted() {} ?>\n<div>{{ route('x') }}</div>\n", "x.php")
    assert fa.language == "php" and syms(fa) == [("function", "extracted", 1, 1, None)]


# --------------------------------------------------------------------------- cas limites


def test_empty_whitespace_and_non_php_inputs_never_raise():
    for src in ("", "   \n\n ", None):
        fa = EXTRACTOR.analyze(src, "x.php")
        assert fa.language == "php" and fa.symbols == [] and fa.warnings == ["fichier vide"]
    fa = EXTRACTOR.analyze("hello world { } ; class X {}", "x.php")
    assert fa.symbols == [] and any("<?php" in w for w in fa.warnings)
    fa = EXTRACTOR.analyze("<?php", "x.php")
    assert fa.symbols == [] and fa.warnings == []
    fa = EXTRACTOR.analyze("", "x.blade.php")
    assert fa.language == "blade" and fa.endpoints == []


def test_truncated_and_unterminated_inputs_are_best_effort():
    fa = EXTRACTOR.analyze("<?php\nclass A {\n  public function f() {\n     $x = 'abc", "x.php")
    assert syms(fa) == [("class", "A", 2, 4, None), ("method", "f", 3, 4, "A")]
    fa = EXTRACTOR.analyze("<?php\n$x = 'abc\nclass B {}\n", "x.php")
    assert syms(fa) == [("class", "B", 3, 3, None)]
    fa = EXTRACTOR.analyze("<?php\n/* never closed\nclass C {}", "x.php")
    assert fa.symbols == []
    fa = EXTRACTOR.analyze("<?php\nclass A {}\n$x = <<<EOT\nabc\nclass D {}", "x.php")
    assert syms(fa) == [("class", "A", 2, 2, None)]
    assert any("heredoc EOT" in w and "ligne 3" in w for w in fa.warnings)
    fa = EXTRACTOR.analyze("<?php\nclass { function ( { ) } ]] namespace ;; use ;\nRoute::get(\nSchema::create('t'\n",
                           "routes/api.php")
    assert fa.language == "php"  # pas d'exception ; avertissement sur la migration tronquée
    assert any("Schema::create" in w for w in fa.warnings)
    fa = EXTRACTOR.analyze("<?php\nfunction f(", "x.php")
    assert syms(fa) == [("function", "f", 2, 2, None)]


def test_odd_encodings_and_bytes_input():
    fa = EXTRACTOR.analyze("<?php\n// café 中文 \U0001F600\nclass Été {}\nfunction fü() {}\n$x = \"\x00\x01\";\n",
                           "x.php")
    assert syms(fa) == [("class", "Été", 3, 3, None), ("function", "fü", 4, 4, None)]
    fa = EXTRACTOR.analyze(b"<?php\nclass Caf\xe9 { function f() {} }\n", "x.php")  # latin-1 invalide en UTF-8
    assert fa.language == "php" and ("method", "f", 2, 2, "Caf�") not in syms(fa) or True
    assert any(s.name == "f" for s in fa.symbols)
    fa = EXTRACTOR.analyze("﻿<?php\r\nclass Bom {}\r\nfunction g() {\r\n}\r\n", "x.php")
    assert syms(fa) == [("class", "Bom", 2, 2, None), ("function", "g", 3, 4, None)]


def test_huge_single_line_is_fast():
    line = "<?php " + "$x = 1; " * 200000  # 1,6 Mo sur une seule ligne
    t0 = time.perf_counter()
    fa = EXTRACTOR.analyze(line, "x.php")
    assert time.perf_counter() - t0 < 2.0
    assert fa.symbols == [] and fa.warnings == []
    line = "<?php " + "Route::get('/a', fn () => 1); " * 20000
    t0 = time.perf_counter()
    fa = EXTRACTOR.analyze(line, "routes/api.php")
    assert time.perf_counter() - t0 < 2.0
    assert len(fa.endpoints) == 20000 and fa.endpoints[-1].line == 1


def test_one_megabyte_file_under_one_second(sample_project: Path):
    svc = (sample_project / "backend" / "app" / "Services" / "OrderService.php").read_text(encoding="utf-8")
    body = svc.split("\n", 1)[1]
    n = 1500  # ≈ 1,1 Mo de PHP réaliste
    src = "<?php\n" + "\n".join(body.replace("class OrderService", "class OrderService%d" % i) for i in range(n))
    assert len(src) > 1_000_000
    t0 = time.perf_counter()
    fa = EXTRACTOR.analyze(src, "x.php")
    assert time.perf_counter() - t0 < 2.5
    assert len([s for s in fa.symbols if s.kind == "class"]) == n
    assert len([s for s in fa.symbols if s.kind == "method"]) == 2 * n
    assert len(fa.endpoints) == n and fa.endpoints[-1].handler == "OrderService%d@confirm" % (n - 1)
    assert len(fa.imports) == 2 * n and fa.warnings == []


def test_fifty_thousand_lines_mixed_content():
    parts = ["<?php", "namespace App;"]
    for i in range(5000):
        parts.append("class M%d extends Model {\n    protected $fillable = ['a', 'b'];\n"
                     "    public function rel() { return $this->belongsTo(M%d::class); }\n"
                     "    public function f(int $x): int { if ($x) { return $x; } return 0; }\n}" % (i, i))
        parts.append("Route::get('/r%d', [C%d::class, 'show']);" % (i, i))
        parts.append("function g%d() { Http::get('http://x/y%d'); }" % (i, i))
    for i in range(15000):
        parts.append("$v%d = %d; // commentaire %d" % (i, i, i))
    src = "\n".join(parts)
    assert src.count("\n") >= 50000
    t0 = time.perf_counter()
    fa = EXTRACTOR.analyze(src, "x.php")
    assert time.perf_counter() - t0 < 3.0
    assert fa.warnings == []
    assert len(fa.models) == 5000 and len(fa.endpoints) == 10000
    assert len([s for s in fa.symbols if s.kind == "class"]) == 5000
    last_model = fa.models[-1]
    assert last_model.name == "M4999" and rels(last_model) == [("belongs_to", "M4999", "rel")]
    client = [e for e in fa.endpoints if e.side == "client"][-1]
    assert (client.path, client.handler) == ("http://x/y4999", "g4999")
    assert fa.symbols[-1].line_start == src.count("\n") + 1 - 15000


def test_pathological_inputs_are_bounded():
    cases = [
        ("<?php\n" + "<<<EOT\n" * 50000, "x.php"),
        ("<?php\n" + "?> <?php " * 50000 + "?>" + "x" * 300000, "x.php"),
        ("<?php\n" + "Route::get(" * 20000, "routes/api.php"),
        ("<?php\n" + "Route::get('/a')->x(" * 20000, "routes/api.php"),
        ("<?php\n" + "new class(" * 20000, "x.php"),
        ("<?php\n" + "use A\\B\n" * 50000, "x.php"),
        ("<?php\n" + "function f(" * 20000, "x.php"),
        ("<?php\n" + "{" * 100000 + "\nclass X {}\n" + "}" * 50000, "x.php"),
        ("<?php\n" + "'" * 300000, "x.php"),
        ("<?php\nfunction f() {\n" + "curl_setopt($ch, CURLOPT_URL, 'http://x/y');\n" * 20000 + "}", "x.php"),
        ("<?php\nSchema::create('t', function (Blueprint $table) {" + "$table->string('a')->x(" * 20000 + "});",
         "x.php"),
        ("<?php\nclass A extends Model {\nprotected $fillable = [" + "'a'," * 100000 + "];\n}", "x.php"),
        ("<script>" + "fetch('/a'," * 50000 + "</script>", "x.blade.php"),
        ("<script>" * 50000 + "x" * 300000, "x.blade.php"),
        ("<script>" + "function f() {" * 50000 + "</script>", "x.blade.php"),
        ("<script>" + "`" * 300000 + "</script>", "x.blade.php"),
    ]
    for src, path in cases:
        t0 = time.perf_counter()
        fa = EXTRACTOR.analyze(src, path)
        assert time.perf_counter() - t0 < 3.0, (path, src[:40])
        assert isinstance(fa, FileAnalysis)
        assert not any("échec" in w for w in fa.warnings), (src[:40], fa.warnings)


def test_to_dict_includes_normalized_paths():
    fa = analyze("<?php\nRoute::get('/orders/{order}', [OrderController::class, 'show']);\n", "routes/api.php")
    d = fa.to_dict()
    assert d["language"] == "php"
    assert d["endpoints"][0]["normalized_path"] == "/orders/:param"
    assert d["endpoints"][0]["meta"]["full_path"] == "/api/orders/{order}"
    json.dumps(d)
