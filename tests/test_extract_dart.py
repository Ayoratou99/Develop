"""Tests de l'extracteur Dart / Flutter (http, dio, retrofit/chopper, shelf, dart_frog, modèles, widgets, imports)."""
from __future__ import annotations

import time
from pathlib import Path

from develop.code_analysis import analyze_file
from develop.code_analysis.extractors.dart import EXTRACTOR
from develop.code_analysis.model import FileAnalysis, normalize_route_path, route_paths_match
from develop.code_analysis.registry import get_extractor


# --------------------------------------------------------------------------- aides

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


def _ep(fa: FileAnalysis, **filters):
    found = _eps(fa, **filters)
    assert len(found) == 1, "attendu 1 endpoint %s, trouvé %s" % (
        filters, [(e.side, e.method, e.path, e.handler, e.line) for e in fa.endpoints])
    return found[0]


def _imp(fa: FileAnalysis, module: str):
    for i in fa.imports:
        if i.module == module:
            return i
    raise AssertionError("import %s introuvable dans %s" % (module, [i.module for i in fa.imports]))


def _model(fa: FileAnalysis, name: str):
    for m in fa.models:
        if m.name == name:
            return m
    raise AssertionError("modèle %s introuvable dans %s" % (name, [m.name for m in fa.models]))


def _field(model, name: str):
    for f in model.fields:
        if f.name == name:
            return f
    raise AssertionError("champ %s introuvable dans %s" % (name, [f.name for f in model.fields]))


# --------------------------------------------------------------------------- extraits synthétiques

# Service API : package http + dio, interpolations, Uri.parse / Uri.https, variable résolue.
API_SERVICE = '''import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:dio/dio.dart';
import '../models/order.dart';
import './env.dart' show Env, apiBase;
export 'order_repository.dart';
part 'api_service.g.dart';

const String kApiBase = 'https://api.example.com';
final _dio = Dio(BaseOptions(baseUrl: 'https://api.example.com'));

class ApiService {
  final http.Client client;
  final String base;
  ApiService(this.client, {required this.base});

  Future<List<Order>> fetchOrders() async {
    final res = await http.get(Uri.parse('$base/orders'));
    final data = jsonDecode(res.body) as List;
    return data.map((e) => Order.fromJson(e)).toList();
  }

  Future<Order> createOrder(Map<String, dynamic> body) async {
    final res = await http.post(Uri.parse('${Env.api}/orders'), body: jsonEncode(body));
    return Order.fromJson(jsonDecode(res.body));
  }

  Future<void> confirm(int id) async {
    await client.get(Uri.https('api.example.com', '/orders/$id'));
    await _dio.post('/orders/$id/confirm');
    await Dio().get('/orders/${id}/lines');
    await _dio.delete("/orders/${order.id}");
  }

  Future<void> ping() async {
    final uri = Uri.parse("http://localhost:8000/api/health");
    await client.get(uri);
  }
}

Future<void> main() async {
  final s = ApiService(http.Client(), base: kApiBase);
  await s.fetchOrders();
}
'''

# Modèle Order : fromJson/toJson, champs nullable, @JsonKey, enum, extension, mixin, getter/setter.
ORDER_MODEL = '''import 'package:json_annotation/json_annotation.dart';

part 'order.g.dart';

enum OrderStatus { draft, confirmed, cancelled }

@JsonSerializable()
class Order {
  final int id;
  final String reference;
  final double? total;
  @JsonKey(name: 'customer_id')
  final int customerId;
  late DateTime createdAt;
  OrderStatus status;

  Order({required this.id, required this.reference, this.total, required this.customerId, this.status = OrderStatus.draft});

  factory Order.fromJson(Map<String, dynamic> json) => Order(
        id: json['id'] as int,
        reference: json['reference'] as String,
        total: (json['total'] as num?)?.toDouble(),
        customerId: json['customer_id'] as int,
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'reference': reference,
        'total': total,
        'customer_id': customerId,
      };

  bool get isConfirmed => status == OrderStatus.confirmed;
  set label(String v) { reference = v; }
}

extension OrderX on Order {
  String get display => '#$id $reference';
}

mixin Loggable {
  void log(String m) => print(m);
}
'''

# Widgets Flutter : StatelessWidget, StatefulWidget, State<…> avec mixin.
WIDGET = '''import 'package:flutter/material.dart';

class OrderCard extends StatelessWidget {
  final Order order;
  const OrderCard({super.key, required this.order});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Text('Order #${order.id}'),
    );
  }
}

class OrderPage extends StatefulWidget {
  const OrderPage({Key? key}) : super(key: key);
  @override
  State<OrderPage> createState() => _OrderPageState();
}

class _OrderPageState extends State<OrderPage> with TickerProviderStateMixin {
  int _count = 0;

  void _increment() {
    setState(() {
      _count++;
    });
  }

  @override
  Widget build(BuildContext context) => Scaffold(body: Text('$_count'));
}
'''

# Client retrofit : @RestApi(baseUrl) + @GET/@POST/@DELETE sur méthodes abstraites.
RETROFIT = '''import 'package:retrofit/retrofit.dart';
import 'package:dio/dio.dart';

part 'client.g.dart';

@RestApi(baseUrl: 'https://api.example.com/v1')
abstract class OrderClient {
  factory OrderClient(Dio dio, {String baseUrl}) = _OrderClient;

  @GET('/orders')
  Future<List<Order>> list();

  @POST('/orders/{id}/confirm')
  Future<Order> confirm(@Path('id') int id);

  @DELETE("/orders/{id}")
  Future<void> remove(@Path() int id);
}
'''

# Service chopper : @ChopperApi(baseUrl) + @Get(path: …) / @Post(...).
CHOPPER = '''import 'package:chopper/chopper.dart';

@ChopperApi(baseUrl: '/customers')
abstract class CustomerService extends ChopperService {
  @Get(path: '/{id}')
  Future<Response> getCustomer(@Path() int id);

  @Post()
  Future<Response> create(@Body() Map<String, dynamic> body);
}
'''

# Serveur shelf_router : router.get / router.post (closure) / router.mount, cascade et annotation Route.
SHELF = '''import 'package:shelf/shelf.dart';
import 'package:shelf_router/shelf_router.dart';

Router buildRouter() {
  final router = Router();
  router.get('/orders', listOrders);
  router.post('/orders/<id>/confirm', (Request req, String id) async {
    return Response.ok('ok');
  });
  router.mount('/api/', apiRouter);
  return router;
}

Response listOrders(Request request) => Response.ok('[]');

final cascade = Router()
  ..get('/health', (Request r) => Response.ok('up'))
  ..delete('/orders/<id>', removeOrder);

class Api {
  @Route.get('/customers/<id>')
  Future<Response> customer(Request request, String id) async => Response.ok(id);
}
'''

# Handler dart_frog (le chemin du fichier donne la route).
DART_FROG = '''import 'package:dart_frog/dart_frog.dart';

Future<Response> onRequest(RequestContext context, String id) async {
  if (context.request.method == HttpMethod.get) {
    return Response.json(body: {'id': id});
  }
  if (context.request.method == HttpMethod.delete) {
    return Response(statusCode: 204);
  }
  return Response(statusCode: 405);
}
'''

# Chaînes multi-lignes contenant des accolades + interpolation avec littéral de map.
MULTILINE = """class Foo {
  static const String template = '''
    { "a": { "b": 1 } }
    } } } {
  ''';
  final String other = \"\"\"
  {{{{ not code }}}}
  \"\"\";
  void bar() {
    print('${template.length} ${ {"k": 1}['k'] }');
  }
}
int after() => 1;
"""

# Divers : freezed, annotation multi-ligne, type fonction, closures, http.Request, Uri.parse(base + …), enum enrichi.
EXTRA = '''import 'dart:io' if (dart.library.html) 'dart:html' as io;
import 'package:freezed_annotation/freezed_annotation.dart';

@freezed
class Customer with _$Customer {
  const factory Customer({required int id, String? email, @Default(true) bool active}) = _Customer;
  factory Customer.fromJson(Map<String, dynamic> json) => _$CustomerFromJson(json);
}

@JsonSerializable(
  explicitToJson: true,
)
class Line {
  final int qty;
  final void Function(int) onChange;
  Line(this.qty, this.onChange);
  factory Line.fromMap(Map<String, dynamic> map) => Line(map['qty'] as int, (_) {});
  double get total => qty * 2.0;
}

final Dio api = Dio(BaseOptions(baseUrl: 'https://shop.test/api'));
final client = http.Client();
const base = 'https://shop.test';

final fetchAll = () async {
  final r = await api.get<Map<String, dynamic>>('/customers');
  return r.data;
};

Future<void> upload(String id) async {
  final req = http.MultipartRequest('POST', Uri.parse('$base/customers/$id/avatar'));
  await client.get(Uri.parse(base + '/customers'));
  await client.get(Uri.parse(kBase + "/ping"));
  final u = Uri.https('shop.test', '/health');
  print(req);
  final url = Uri.parse('https://shop.test/standalone/${id}');
  final launch = Uri.parse(nothing);
}

void Function(int) makeCb(int a) => (x) => print(x);

typedef Json = Map<String, dynamic>;

enum Status {
  draft('d'),
  done('x');

  final String code;
  const Status(this.code);
  static const int kMax = 2;
  String get label => code.toUpperCase();
}
'''


# --------------------------------------------------------------------------- registre et fixture réelle

def test_extractor_is_registered_for_dart():
    assert EXTRACTOR.languages == ["dart"]
    assert get_extractor("dart") is EXTRACTOR
    fa = analyze_file("class A {}\n", "dart", "lib/a.dart")
    assert fa.language == "dart"
    assert _sym(fa, "A", "class").line_start == 1


def test_fixture_has_no_dart_but_analysis_never_raises(sample_project: Path):
    """Le mini-projet ne contient pas de Dart : on vérifie la robustesse sur tous ses fichiers texte."""
    files = [p for p in sample_project.rglob("*") if p.is_file() and p.suffix != ".png"]
    assert files, "fixture vide ?"
    assert not [p for p in files if p.suffix == ".dart"]
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        fa = EXTRACTOR.analyze(text, str(p))
        assert isinstance(fa, FileAnalysis)
        assert fa.language == "dart"
        for s in fa.symbols:
            assert 1 <= s.line_start <= s.line_end


def test_fixture_php_controller_analyzed_as_dart(sample_project: Path):
    """Un contrôleur Laravel forcé en Dart : la classe est repérée (accolades équilibrées), sans endpoint ni modèle."""
    text = (sample_project / "backend/app/Http/Controllers/OrderController.php").read_text(encoding="utf-8")
    fa = EXTRACTOR.analyze(text, "backend/app/Http/Controllers/OrderController.php")
    cls = _sym(fa, "OrderController", "class")
    assert (cls.line_start, cls.line_end) == (9, 29)
    assert cls.meta["extends"] == "Controller"
    assert fa.endpoints == []
    assert fa.models == []
    assert fa.imports == []


def test_fixture_typescript_client_analyzed_as_dart(sample_project: Path):
    """Un client TypeScript : ni symbole, ni modèle, ni import Dart ; seul l'appel ``http.get<Order[]>("/orders")``
    (ligne 23) ressemble à un appel client Dart et est relevé au mieux."""
    text = (sample_project / "frontend/src/api/client.ts").read_text(encoding="utf-8")
    fa = EXTRACTOR.analyze(text, "frontend/src/api/client.ts")
    assert fa.symbols == []
    assert fa.models == []
    assert fa.imports == []
    assert [(e.side, e.method, e.path, e.handler, e.framework, e.line) for e in fa.endpoints] == [
        ("client", "GET", "/orders", "", "http", 23)]


def test_fixture_express_server_routes_seen_as_shelf_style(sample_project: Path):
    """``app.get('/x', (req, res) => …)`` ressemble à shelf_router : les routes sont relevées au mieux."""
    text = (sample_project / "services/notifier/src/server.js").read_text(encoding="utf-8")
    fa = EXTRACTOR.analyze(text, "services/notifier/src/server.js")
    assert [(e.side, e.method, e.path, e.line, e.framework) for e in fa.endpoints] == [
        ("server", "GET", "/notifications/:userId", 7, "shelf"),
        ("server", "POST", "/notifications", 11, "shelf"),
        ("server", "DELETE", "/notifications/:id", 16, "shelf"),
    ]
    assert fa.endpoints[0].normalized_path == "/notifications/:param"
    assert _sym(fa, "app", "constant").line_start == 4
    assert fa.models == []


# --------------------------------------------------------------------------- service API (http + dio)

def test_api_service_symbols():
    fa = EXTRACTOR.analyze(API_SERVICE, "lib/services/api_service.dart")
    assert fa.warnings == []
    k = _sym(fa, "kApiBase", "constant")
    assert (k.line_start, k.line_end, k.parent) == (9, 9, None)
    assert k.meta["type"] == "String"
    assert k.signature == "const String kApiBase"
    d = _sym(fa, "_dio", "variable")
    assert d.line_start == 10 and d.meta["exported"] is False
    cls = _sym(fa, "ApiService", "class")
    assert (cls.line_start, cls.line_end) == (12, 39)
    assert cls.meta.get("widget") is None
    ctor = _sym(fa, "ApiService", "method", parent="ApiService")
    assert ctor.meta["constructor"] is True and ctor.line_start == 15
    assert ctor.meta["params"] == ["client", "base"]
    fetch = _sym(fa, "fetchOrders", "method", parent="ApiService")
    assert (fetch.line_start, fetch.line_end) == (17, 21)
    assert fetch.meta["async"] is True
    assert fetch.meta["return_type"] == "Future<List<Order>>"
    assert fetch.signature == "Future<List<Order>> fetchOrders() async"
    assert fetch.qualified_name == "ApiService.fetchOrders"
    create = _sym(fa, "createOrder", "method", parent="ApiService")
    assert (create.line_start, create.line_end) == (23, 26)
    assert create.meta["params"] == ["body"]
    confirm = _sym(fa, "confirm", "method", parent="ApiService")
    assert (confirm.line_start, confirm.line_end) == (28, 33)
    ping = _sym(fa, "ping", "method", parent="ApiService")
    assert (ping.line_start, ping.line_end) == (35, 38)
    main = _sym(fa, "main", "function")
    assert (main.line_start, main.line_end, main.parent) == (41, 44, None)
    assert main.meta["async"] is True
    # pas de symbole parasite pour les variables locales ni les appels
    assert [s.name for s in fa.symbols] == [
        "kApiBase", "_dio", "ApiService", "ApiService", "fetchOrders", "createOrder", "confirm", "ping", "main"]


def test_api_service_http_endpoints():
    fa = EXTRACTOR.analyze(API_SERVICE, "lib/services/api_service.dart")
    assert all(e.side == "client" for e in fa.endpoints)
    e = _ep(fa, line=18)
    assert (e.method, e.path, e.handler, e.framework) == ("GET", "$base/orders", "ApiService@fetchOrders", "http")
    assert e.meta["base"] == "base"
    assert e.meta["receiver"] == "http" and e.meta["call"] == "get"
    assert e.normalized_path == "/orders"
    e = _ep(fa, line=24)
    assert (e.method, e.path, e.handler, e.framework) == ("POST", "${Env.api}/orders", "ApiService@createOrder", "http")
    assert e.meta["base"] == "Env.api"
    assert e.normalized_path == "/orders"
    e = _ep(fa, line=29)
    assert (e.method, e.path, e.handler, e.framework) == ("GET", "/orders/$id", "ApiService@confirm", "http")
    assert e.meta["host"] == "api.example.com"
    assert e.meta["base"] == "https://api.example.com"
    assert e.meta["source"] == "Uri.https"
    assert e.normalized_path == "/orders/:param"
    # variable ``uri`` résolue à sa définition (ligne 36), endpoint cité à la ligne de l'appel
    e = _ep(fa, line=37)
    assert (e.method, e.path, e.handler) == ("GET", "http://localhost:8000/api/health", "ApiService@ping")
    assert e.meta["base"] == "http://localhost:8000"
    assert e.meta["source"] == "variable:uri"
    assert e.normalized_path == "/api/health"


def test_api_service_dio_endpoints():
    fa = EXTRACTOR.analyze(API_SERVICE, "lib/services/api_service.dart")
    e = _ep(fa, line=30)
    assert (e.method, e.path, e.handler, e.framework) == ("POST", "/orders/$id/confirm", "ApiService@confirm", "dio")
    assert e.meta["receiver"] == "_dio"
    assert e.meta["base"] == "https://api.example.com"  # BaseOptions(baseUrl: …) du fichier
    e = _ep(fa, line=31)
    assert (e.method, e.path, e.framework) == ("GET", "/orders/${id}/lines", "dio")
    assert e.meta["receiver"] == "Dio"
    e = _ep(fa, line=32)
    assert (e.method, e.path, e.framework) == ("DELETE", "/orders/${order.id}", "dio")
    assert e.normalized_path == "/orders/:param"
    assert len(fa.endpoints) == 7
    assert sorted(e.line for e in fa.endpoints) == [18, 24, 29, 30, 31, 32, 37]


def test_api_service_imports():
    fa = EXTRACTOR.analyze(API_SERVICE, "lib/services/api_service.dart")
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("dart:convert", [], 1, False),
        ("package:http/http.dart", ["http"], 2, False),
        ("package:dio/dio.dart", [], 3, False),
        ("../models/order.dart", [], 4, True),
        ("./env.dart", ["Env", "apiBase"], 5, True),
        ("order_repository.dart", [], 6, True),
        ("api_service.g.dart", [], 7, True),
    ]
    assert all(i.resolved_path is None for i in fa.imports)


# --------------------------------------------------------------------------- modèle Order, enum, extension, mixin

def test_order_model_symbols():
    fa = EXTRACTOR.analyze(ORDER_MODEL, "lib/models/order.dart")
    assert fa.warnings == []
    en = _sym(fa, "OrderStatus", "enum")
    assert (en.line_start, en.line_end) == (5, 5)
    assert en.meta["values"] == ["draft", "confirmed", "cancelled"]
    cls = _sym(fa, "Order", "class")
    assert (cls.line_start, cls.line_end) == (8, 35)
    assert cls.meta["annotations"] == ["JsonSerializable"]
    assert cls.signature == "class Order"
    ctor = _sym(fa, "Order", "method", parent="Order")
    assert ctor.line_start == 17 and ctor.meta["constructor"] is True
    assert ctor.meta["params"] == ["id", "reference", "total", "customerId", "status"]
    fj = _sym(fa, "fromJson", "method", parent="Order")
    assert (fj.line_start, fj.line_end) == (19, 24)
    assert fj.meta["factory"] is True and fj.meta["constructor"] is True
    tj = _sym(fa, "toJson", "method", parent="Order")
    assert (tj.line_start, tj.line_end) == (26, 31)
    assert tj.meta["return_type"] == "Map<String, dynamic>"
    g = _sym(fa, "isConfirmed", "property", parent="Order")
    assert g.line_start == 33 and g.meta["accessor"] == "get" and g.meta["return_type"] == "bool"
    s = _sym(fa, "label", "property", parent="Order")
    assert s.line_start == 34 and s.meta["accessor"] == "set" and s.meta["params"] == ["v"]
    assert s.signature == "set label(String v)"
    ext = _sym(fa, "OrderX", "type")
    assert (ext.line_start, ext.line_end) == (37, 39)
    assert ext.meta["declaration"] == "extension" and ext.meta["on"] == ["Order"]
    disp = _sym(fa, "display", "property", parent="OrderX")
    assert disp.line_start == 38
    mx = _sym(fa, "Loggable", "trait")
    assert (mx.line_start, mx.line_end) == (41, 43) and mx.meta["declaration"] == "mixin"
    log = _sym(fa, "log", "method", parent="Loggable")
    assert log.line_start == 42 and log.meta["params"] == ["m"]
    # les champs d'instance ne sont pas des symboles de haut niveau
    assert "id" not in [s.name for s in fa.symbols]


def test_order_model_fields_and_json_keys():
    fa = EXTRACTOR.analyze(ORDER_MODEL, "lib/models/order.dart")
    assert [m.name for m in fa.models] == ["Order"]
    m = _model(fa, "Order")
    assert m.kind == "dart_class" and m.line == 8 and m.table_name is None
    assert m.relations == []
    assert m.meta["reason"] == "from_json"
    assert m.meta["has_from_json"] is True and m.meta["has_to_json"] is True
    assert m.meta["json_serializable"] is True
    assert m.meta["json_keys"] == ["id", "reference", "total", "customer_id"]
    assert m.meta["constructors"] == ["Order", "fromJson"]
    assert [f.name for f in m.fields] == ["id", "reference", "total", "customerId", "createdAt", "status"]
    assert _field(m, "id").type == "int" and _field(m, "id").nullable is False
    assert _field(m, "id").meta["final"] is True
    total = _field(m, "total")
    assert total.type == "double" and total.nullable is True
    cid = _field(m, "customerId")
    assert cid.meta["json_key"] == "customer_id"
    created = _field(m, "createdAt")
    assert created.type == "DateTime" and created.meta["late"] is True and created.meta["final"] is False
    assert _field(m, "status").type == "OrderStatus"
    assert fa.imports[0].module == "package:json_annotation/json_annotation.dart"
    part = _imp(fa, "order.g.dart")
    assert part.line == 3 and part.is_relative is True
    assert fa.endpoints == []


def test_model_detection_rules():
    code = '''class Point {
  final int x;
  final int y;
  const Point(this.x, this.y);
  Point copyWith({int? x, int? y}) => Point(x ?? this.x, y ?? this.y);
  @override
  bool operator ==(Object o) => o is Point && o.x == x && o.y == y;
  @override
  int get hashCode => x ^ y;
}

class Counter {
  final int start;
  Counter(this.start);
  int next() => start + 1;
}

class Mutable {
  int a = 0;
  Mutable();
}

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
}

class Money {
  final int cents;
  Money(this.cents);
  static Money fromJson(Map<String, dynamic> j) => Money(j['cents'] as int);
  Map<String, dynamic> toMap() => {'cents': cents};
}

class Pair<A, B> {
  final A first;
  final B? second;
  Pair({required this.first, this.second});
}
'''
    fa = EXTRACTOR.analyze(code, "lib/models/misc.dart")
    assert [m.name for m in fa.models] == ["Point", "Money", "Pair"]
    pt = _model(fa, "Point")
    assert pt.meta["reason"] == "final_fields" and pt.meta["has_from_json"] is False
    assert [(f.name, f.type, f.nullable) for f in pt.fields] == [("x", "int", False), ("y", "int", False)]
    money = _model(fa, "Money")
    assert money.meta["reason"] == "from_json" and money.meta["json_keys"] == ["cents"]
    pair = _model(fa, "Pair")
    assert pair.line == 35
    assert [(f.name, f.type, f.nullable) for f in pair.fields] == [("first", "A", False), ("second", "B", True)]
    assert _sym(fa, "Pair", "class").meta["generics"] == "<A, B>"
    op = _sym(fa, "operator ==", "method", parent="Point")
    assert op.line_start == 7 and op.meta["operator"] is True
    nxt = _sym(fa, "next", "method", parent="Counter")
    assert nxt.line_start == 15


def test_freezed_model_fields_come_from_constructor():
    fa = EXTRACTOR.analyze(EXTRA, "lib/extra.dart")
    cust = _model(fa, "Customer")
    assert cust.line == 5 and cust.meta["freezed"] is True and cust.meta["reason"] == "from_json"
    assert cust.meta["annotations"] == ["freezed"]
    assert cust.meta["with"] == ["_$Customer"]
    assert [(f.name, f.type, f.nullable, f.default) for f in cust.fields] == [
        ("id", "int", False, None), ("email", "String", True, None), ("active", "bool", False, "true")]
    assert _field(cust, "id").meta == {"final": True, "required": True, "named": True}
    line = _model(fa, "Line")
    assert line.meta["annotations"] == ["JsonSerializable"]  # annotation sur plusieurs lignes
    assert line.meta["json_keys"] == ["qty"]  # fromMap(map['qty'])
    assert _field(line, "onChange").type == "void Function(int)"


# --------------------------------------------------------------------------- widgets Flutter

def test_flutter_widgets_are_components():
    fa = EXTRACTOR.analyze(WIDGET, "lib/widgets/order_card.dart")
    assert fa.warnings == []
    card = _sym(fa, "OrderCard", "component")
    assert (card.line_start, card.line_end) == (3, 13)
    assert card.meta["widget"] is True and card.meta["extends"] == "StatelessWidget"
    assert card.signature == "class OrderCard extends StatelessWidget"
    ctor = _sym(fa, "OrderCard", "method", parent="OrderCard")
    assert ctor.line_start == 5 and ctor.meta["const"] is True and ctor.meta["params"] == ["key", "order"]
    build = _sym(fa, "build", "method", parent="OrderCard")
    assert (build.line_start, build.line_end) == (8, 12)
    assert build.meta["annotations"] == ["override"] and build.meta["params"] == ["context"]
    page = _sym(fa, "OrderPage", "component")
    assert (page.line_start, page.line_end) == (15, 19) and page.meta["extends"] == "StatefulWidget"
    cs = _sym(fa, "createState", "method", parent="OrderPage")
    assert cs.line_start == 18 and cs.meta["return_type"] == "State<OrderPage>"
    state = _sym(fa, "_OrderPageState", "component")
    assert (state.line_start, state.line_end) == (21, 32)
    assert state.meta["extends"] == "State<OrderPage>" and state.meta["with"] == ["TickerProviderStateMixin"]
    assert state.meta["exported"] is False
    inc = _sym(fa, "_increment", "method", parent="_OrderPageState")
    assert (inc.line_start, inc.line_end) == (24, 28)
    build2 = _sym(fa, "build", "method", parent="_OrderPageState")
    assert (build2.line_start, build2.line_end) == (31, 31)
    # un widget n'est jamais un modèle de données
    assert fa.models == []
    assert fa.endpoints == []
    assert [(i.module, i.is_relative) for i in fa.imports] == [("package:flutter/material.dart", False)]


def test_other_widget_bases_and_non_widgets():
    code = '''class A extends ConsumerWidget {}
class B extends HookWidget {}
class C extends MyFancyWidget {}
class D extends CustomPainter {}
abstract class E {}
'''
    fa = EXTRACTOR.analyze(code, "w.dart")
    assert [(s.name, s.kind) for s in fa.symbols] == [
        ("A", "component"), ("B", "component"), ("C", "component"), ("D", "class"), ("E", "class")]
    assert _sym(fa, "E").meta["abstract"] is True and _sym(fa, "E").meta["modifiers"] == ["abstract"]


# --------------------------------------------------------------------------- retrofit / chopper

def test_retrofit_annotations():
    fa = EXTRACTOR.analyze(RETROFIT, "lib/api/client.dart")
    cls = _sym(fa, "OrderClient", "class")
    assert (cls.line_start, cls.line_end) == (7, 18)
    assert cls.meta["annotations"] == ["RestApi"] and cls.meta["abstract"] is True
    lst = _sym(fa, "list", "method", parent="OrderClient")
    assert lst.meta["abstract"] is True and lst.meta["annotations"] == ["GET"]
    assert [(e.side, e.method, e.path, e.handler, e.framework, e.line) for e in fa.endpoints] == [
        ("client", "GET", "/orders", "OrderClient@list", "retrofit", 11),
        ("client", "POST", "/orders/{id}/confirm", "OrderClient@confirm", "retrofit", 14),
        ("client", "DELETE", "/orders/{id}", "OrderClient@remove", "retrofit", 17),
    ]
    for e in fa.endpoints:
        assert e.meta["base"] == "https://api.example.com/v1"
        assert e.meta["annotation"] == e.method
    assert fa.endpoints[1].normalized_path == "/orders/:param/confirm"
    assert fa.models == []  # classe abstraite : pas un modèle
    assert _imp(fa, "client.g.dart").is_relative is True


def test_chopper_annotations():
    fa = EXTRACTOR.analyze(CHOPPER, "lib/api/customer_service.dart")
    assert [(e.method, e.path, e.handler, e.framework, e.line, e.meta.get("base")) for e in fa.endpoints] == [
        ("GET", "/{id}", "CustomerService@getCustomer", "retrofit", 6, "/customers"),
        ("POST", "", "CustomerService@create", "retrofit", 9, "/customers"),
    ]
    assert fa.endpoints[0].meta["annotation"] == "Get"


def test_retrofit_and_dio_paths_normalize_to_the_same_route():
    assert normalize_route_path("/orders/{id}/confirm") == normalize_route_path("/orders/$id/confirm")
    assert normalize_route_path("$base/orders/${id}/lines") == "/orders/:param/lines"
    assert route_paths_match("/orders/<id>/confirm", "${Env.api}/orders/$id/confirm")


# --------------------------------------------------------------------------- shelf / dart_frog

def test_shelf_router_endpoints():
    fa = EXTRACTOR.analyze(SHELF, "bin/server.dart")
    assert _sym(fa, "buildRouter", "function").line_end == 12
    assert _sym(fa, "listOrders", "function").line_start == 14
    e = _ep(fa, line=6)
    assert (e.side, e.method, e.path, e.handler, e.framework) == ("server", "GET", "/orders", "listOrders", "shelf")
    assert e.meta["receiver"] == "router"
    e = _ep(fa, line=7)
    assert (e.method, e.path, e.handler) == ("POST", "/orders/<id>/confirm", "buildRouter")  # closure : fonction englobante
    assert e.normalized_path == "/orders/:param/confirm"
    e = _ep(fa, line=10)
    assert (e.method, e.path, e.handler) == ("ANY", "/api/", "apiRouter")
    # cascade ``Router()..get(...)``
    e = _ep(fa, line=17)
    assert (e.side, e.method, e.path, e.framework, e.handler) == ("server", "GET", "/health", "shelf", "")
    assert e.meta["receiver"] == ".."
    e = _ep(fa, line=18)
    assert (e.method, e.path, e.handler) == ("DELETE", "/orders/<id>", "removeOrder")
    # annotation @Route.get du générateur shelf_router
    e = _ep(fa, path="/customers/<id>")
    assert (e.side, e.method, e.handler, e.framework, e.line) == ("server", "GET", "Api@customer", "shelf", 22)
    assert e.meta["annotation"] == "Route.get"
    assert len(fa.endpoints) == 6


def test_dart_frog_route_derived_from_path():
    fa = EXTRACTOR.analyze(DART_FROG, "routes/orders/[id].dart")
    assert _sym(fa, "onRequest", "function").line_end == 11
    assert [(e.side, e.method, e.path, e.handler, e.framework, e.line) for e in fa.endpoints] == [
        ("server", "DELETE", "/orders/[id]", "onRequest", "dart_frog", 3),
        ("server", "GET", "/orders/[id]", "onRequest", "dart_frog", 3),
    ]
    assert fa.endpoints[0].meta["derived_from_path"] is True
    assert fa.endpoints[0].normalized_path == "/orders/:param"
    # index.dart → racine du dossier ; sans HttpMethod → ANY ; chemin Windows accepté
    fa = EXTRACTOR.analyze("Response onRequest(RequestContext c) => Response(body: 'hi');\n", "my_app\\routes\\index.dart")
    assert [(e.method, e.path) for e in fa.endpoints] == [("ANY", "/")]
    fa = EXTRACTOR.analyze("Response onRequest(RequestContext c) => Response();\n", "routes/api/v1/index.dart")
    assert [e.path for e in fa.endpoints] == ["/api/v1"]
    # middleware et fichiers hors routes/ : rien
    assert EXTRACTOR.analyze("Handler middleware(Handler h) => h;\n", "routes/_middleware.dart").endpoints == []
    assert EXTRACTOR.analyze(DART_FROG, "lib/handlers/orders.dart").endpoints == []


# --------------------------------------------------------------------------- autres formes d'appels client

def test_misc_client_calls_and_closures():
    fa = EXTRACTOR.analyze(EXTRA, "lib/extra.dart")
    fetch_all = _sym(fa, "fetchAll", "function")
    assert (fetch_all.line_start, fetch_all.line_end) == (25, 28)
    assert fetch_all.meta["closure"] is True and fetch_all.meta["async"] is True
    assert fetch_all.signature == "final fetchAll"
    # dio.get<T>(…) sur une instance déclarée ``final Dio api`` dans une closure nommée
    e = _ep(fa, line=26)
    assert (e.method, e.path, e.handler, e.framework) == ("GET", "/customers", "fetchAll", "dio")
    assert e.meta["base"] == "https://shop.test/api"
    # http.MultipartRequest('POST', Uri.parse(…))
    e = _ep(fa, line=31)
    assert (e.method, e.path, e.handler, e.framework) == ("POST", "$base/customers/$id/avatar", "upload", "http")
    assert e.meta["base"] == "base" and e.meta["call"] == "Request"
    # Uri.parse(base + '/customers') : chemin brut + base
    e = _ep(fa, line=32)
    assert (e.method, e.path, e.meta["base"], e.framework) == ("GET", "/customers", "base", "http")
    e = _ep(fa, line=33)
    assert (e.path, e.meta["base"]) == ("/ping", "kBase")
    # Uri.https / Uri.parse autonomes → endpoint ANY ; Uri.parse(variable inconnue) ignoré
    e = _ep(fa, line=34)
    assert (e.method, e.path, e.meta["host"], e.meta["standalone"]) == ("ANY", "/health", "shop.test", True)
    e = _ep(fa, line=36)
    assert (e.method, e.path, e.meta["base"]) == ("ANY", "https://shop.test/standalone/${id}", "https://shop.test")
    assert e.normalized_path == "/standalone/:param"
    assert sorted(e.line for e in fa.endpoints) == [26, 31, 32, 33, 34, 36]
    assert all(e.side == "client" for e in fa.endpoints)


def test_dio_request_with_method_option_and_named_instances():
    code = '''class Repo {
  final Dio _client = Dio();
  Dio get dio => _client;
  Future<void> run() async {
    await _client.request('/jobs/$id', options: Options(method: 'PUT'));
    await _client.put('/jobs/${job.id}/done', data: {});
    final r = await http.Client().post(Uri.parse('https://api.test/v1/jobs'));
    await http.delete(Uri.parse("$kBase/jobs/$id"), headers: {'a': 'b'});
    await http.read(Uri.parse('$kBase/jobs.txt'));
  }
}
'''
    fa = EXTRACTOR.analyze(code, "lib/repo.dart")
    assert [(e.method, e.path, e.framework, e.handler, e.line) for e in fa.endpoints] == [
        ("PUT", "/jobs/$id", "dio", "Repo@run", 5),
        ("PUT", "/jobs/${job.id}/done", "dio", "Repo@run", 6),
        ("POST", "https://api.test/v1/jobs", "http", "Repo@run", 7),
        ("DELETE", "$kBase/jobs/$id", "http", "Repo@run", 8),
        ("GET", "$kBase/jobs.txt", "http", "Repo@run", 9),
    ]
    assert fa.endpoints[2].meta["base"] == "https://api.test"
    assert fa.endpoints[3].meta["base"] == "kBase"
    assert fa.endpoints[2].normalized_path == "/v1/jobs"


def test_calls_in_strings_and_comments_are_ignored():
    code = '''void f() {
  // http.get(Uri.parse('/commented'));
  /* dio.post('/block'); */
  print("http.get(Uri.parse('/in-string'))");
  final list = [1, 2].map((e) => e).toList();
  map.get('key');
  prefs.get('token');
}
'''
    fa = EXTRACTOR.analyze(code, "lib/f.dart")
    assert fa.endpoints == []


# --------------------------------------------------------------------------- imports

def test_import_forms():
    code = '''library my_app.services;

import 'dart:async';
import 'package:flutter/widgets.dart' show Widget, BuildContext hide Text;
import 'package:collection/collection.dart' deferred as coll;
import 'utils/money.dart';
import '../../core/env.dart' as env;
import 'dart:io' if (dart.library.html) 'dart:html' as io;
export 'src/models.dart' show Order;
export 'package:meta/meta.dart';
part 'services.g.dart';
part of 'app.dart';
part of other.library;
import 'multi/line.dart'
    as ml;
'''
    fa = EXTRACTOR.analyze(code, "lib/services.dart")
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("dart:async", [], 3, False),
        ("package:flutter/widgets.dart", ["Widget", "BuildContext"], 4, False),
        ("package:collection/collection.dart", ["coll"], 5, False),
        ("utils/money.dart", [], 6, True),
        ("../../core/env.dart", ["env"], 7, True),
        ("dart:io", ["io"], 8, False),
        ("src/models.dart", ["Order"], 9, True),
        ("package:meta/meta.dart", [], 10, False),
        ("services.g.dart", [], 11, True),
        ("app.dart", [], 12, True),
        ("other.library", [], 13, False),
        ("multi/line.dart", ["ml"], 14, True),
    ]
    lib = _sym(fa, "my_app.services", "module")
    assert lib.line_start == 1 and lib.meta["declaration"] == "library"


# --------------------------------------------------------------------------- symboles divers

def test_top_level_functions_typedef_constants_and_enhanced_enum():
    fa = EXTRACTOR.analyze(EXTRA, "lib/extra.dart")
    mk = _sym(fa, "makeCb", "function")
    assert (mk.line_start, mk.line_end) == (40, 40)
    assert mk.meta["return_type"] == "void Function(int)" and mk.meta["params"] == ["a"]
    td = _sym(fa, "Json", "type")
    assert td.line_start == 42 and td.meta["declaration"] == "typedef"
    assert td.signature == "typedef Json = Map<String, dynamic>"
    api = _sym(fa, "api", "variable")
    assert api.line_start == 21 and api.meta["type"] == "Dio"
    base = _sym(fa, "base", "constant")
    assert base.line_start == 23
    st = _sym(fa, "Status", "enum")
    assert (st.line_start, st.line_end) == (44, 52)
    assert st.meta["values"] == ["draft", "done"]
    ctor = _sym(fa, "Status", "method", parent="Status")
    assert ctor.line_start == 49 and ctor.meta["const"] is True and ctor.meta["constructor"] is True
    kmax = _sym(fa, "kMax", "constant", parent="Status")
    assert kmax.line_start == 50 and kmax.meta["modifiers"] == ["static", "const"]
    label = _sym(fa, "label", "property", parent="Status")
    assert label.line_start == 51
    up = _sym(fa, "upload", "function")
    assert (up.line_start, up.line_end) == (30, 38)
    assert fa.models and [m.name for m in fa.models] == ["Customer", "Line"]
    assert _imp(fa, "dart:io").names == ["io"]


def test_multiline_signatures_and_initializer_lists():
    code = '''class Repo extends Base
    with Logging
    implements Store {
  final String name;
  Repo(this.name) : assert(name.isNotEmpty), super(name) {
    init();
  }
  Future<void> save(
    Order order, {
    bool notify = false,
  }) async {
    await http.post(Uri.parse('$base/orders/${order.id}'), body: order.toJson());
  }
  static Repo of(String n) => Repo(n);
  external int nativeCall(int x);
  Stream<int> ticks() async* {
    yield 1;
  }
}
'''
    fa = EXTRACTOR.analyze(code, "lib/repo.dart")
    cls = _sym(fa, "Repo", "class")
    assert (cls.line_start, cls.line_end) == (1, 19)
    assert cls.meta["extends"] == "Base" and cls.meta["with"] == ["Logging"] and cls.meta["implements"] == ["Store"]
    ctor = _sym(fa, "Repo", "method", parent="Repo")
    assert (ctor.line_start, ctor.line_end) == (5, 7)
    save = _sym(fa, "save", "method", parent="Repo")
    assert (save.line_start, save.line_end) == (8, 13)
    assert save.meta["params"] == ["order", "notify"] and save.meta["async"] is True
    e = _ep(fa, line=12)
    assert (e.method, e.path, e.handler) == ("POST", "$base/orders/${order.id}", "Repo@save")
    of = _sym(fa, "of", "method", parent="Repo")
    assert of.meta["static"] is True and of.line_start == 14
    nat = _sym(fa, "nativeCall", "method", parent="Repo")
    assert nat.line_start == 15 and nat.meta.get("abstract") is None
    ticks = _sym(fa, "ticks", "method", parent="Repo")
    assert (ticks.line_start, ticks.line_end) == (16, 18) and ticks.meta["generator"] is True
    assert [s.name for s in fa.symbols if s.kind == "method"] == ["Repo", "save", "of", "nativeCall", "ticks"]


def test_abstract_class_and_getters_setters_at_top_level():
    code = '''abstract class Repository<T> {
  Future<T?> find(int id);
  Future<void> save(T item);
  int get count;
  set limit(int v);
}

int _counter = 0;
int get counter => _counter;
set counter(int v) => _counter = v;
'''
    fa = EXTRACTOR.analyze(code, "lib/repository.dart")
    cls = _sym(fa, "Repository", "class")
    assert cls.meta["abstract"] is True and cls.meta["generics"] == "<T>" and cls.line_end == 6
    find = _sym(fa, "find", "method", parent="Repository")
    assert find.meta["abstract"] is True and find.meta["return_type"] == "Future<T?>"
    cnt = _sym(fa, "count", "property", parent="Repository")
    assert cnt.meta["abstract"] is True
    lim = _sym(fa, "limit", "property", parent="Repository")
    assert lim.meta["accessor"] == "set"
    g = _sym(fa, "counter", "property", parent=None)
    assert g.line_start == 9 and g.meta["accessor"] == "get"
    assert [s.line_start for s in fa.symbols if s.name == "counter"] == [9, 10]
    assert _sym(fa, "_counter", "variable").line_start == 8
    assert fa.models == []


# --------------------------------------------------------------------------- chaînes multi-lignes et accolades

def test_multiline_strings_do_not_break_brace_balancing():
    fa = EXTRACTOR.analyze(MULTILINE, "lib/foo.dart")
    assert fa.warnings == []
    foo = _sym(fa, "Foo", "class")
    assert (foo.line_start, foo.line_end) == (1, 12)
    tpl = _sym(fa, "template", "constant", parent="Foo")
    assert (tpl.line_start, tpl.line_end) == (2, 5)
    assert tpl.signature == "static const String template"
    bar = _sym(fa, "bar", "method", parent="Foo")
    assert (bar.line_start, bar.line_end) == (9, 11)
    after = _sym(fa, "after", "function")
    assert (after.line_start, after.line_end) == (13, 13)
    assert fa.endpoints == []


def test_interpolations_with_nested_strings_and_braces():
    code = '''void f() {
  print('${'${'${'${1}'}'}'}');
  print("a ${map['k']} b ${ {'x': 1}['x'] } c");
  print(r'raw $notInterpolated {');
  final s = 'esc \\' quote { ${"inner \\" }"}';
}
int g() => 1;
'''
    fa = EXTRACTOR.analyze(code, "lib/i.dart")
    assert fa.warnings == []
    assert (_sym(fa, "f").line_start, _sym(fa, "f").line_end) == (1, 6)
    assert _sym(fa, "g").line_start == 7


# --------------------------------------------------------------------------- cas limites

def test_empty_and_whitespace_only():
    for text in ("", "\n\n", "   \t\n", "// seulement un commentaire\n", "/* bloc */"):
        fa = EXTRACTOR.analyze(text, "lib/empty.dart")
        assert fa.language == "dart"
        assert fa.symbols == [] and fa.endpoints == [] and fa.models == [] and fa.imports == []
        assert fa.warnings == []


def test_invalid_syntax_yields_warning_not_exception():
    fa = EXTRACTOR.analyze("class {{{{ ((( \n  void (((\n", "lib/bad.dart")
    assert fa.symbols == []
    assert any("déséquilibr" in w for w in fa.warnings)
    fa = EXTRACTOR.analyze("class A {\n  void f() {\n", "lib/trunc.dart")
    a = _sym(fa, "A", "class")
    assert a.line_start == 1 and a.line_end >= 1  # fin approximative
    assert fa.warnings


def test_unterminated_strings():
    # chaîne simple non terminée : limitée à sa ligne, le reste du fichier est encore analysé
    fa = EXTRACTOR.analyze("class A {\n  final s = 'oops;\n  int b = 1;\n}\nint c() => 2;\n", "u.dart")
    assert any("non terminée" in w and "ligne 2" in w for w in fa.warnings)
    assert (_sym(fa, "A").line_start, _sym(fa, "A").line_end) == (1, 4)
    assert _sym(fa, "c", "function").line_start == 5
    # chaîne multi-ligne non terminée : tout le reste est une chaîne, sans exception
    fa = EXTRACTOR.analyze("class A {\n  final s = '''oops;\n  int b = 1;\n}\n", "u.dart")
    assert any("non terminée" in w for w in fa.warnings)
    assert _sym(fa, "A").line_start == 1


def test_odd_inputs_never_raise():
    assert _sym(EXTRACTOR.analyze(b"class A {}\n", "b.dart"), "A").line_start == 1  # bytes
    fa = EXTRACTOR.analyze("class A {\r\n  void f() {}\r\n}\r\n", "crlf.dart")
    assert (_sym(fa, "A").line_end, _sym(fa, "f").line_start) == (3, 2)
    assert _sym(EXTRACTOR.analyze("class A {\x00 void f() {} }\n", "nul.dart"), "A").line_start == 1
    # Unicode dans les chaînes/commentaires (les identifiants Dart restent ASCII)
    fa = EXTRACTOR.analyze("class A { // Ünïcødé\n  String f() => 'été ${x} 日本';\n}\n", "u.dart")
    assert (_sym(fa, "A").line_end, _sym(fa, "f").line_start) == (3, 2)
    assert EXTRACTOR.analyze("class Ünïcødé {}\n", "u.dart").warnings == []
    assert EXTRACTOR.analyze(None, "none.dart").symbols == []  # type: ignore[arg-type]
    minified = "class A{final int a;A(this.a);}class B extends StatelessWidget{Widget build(BuildContext c)=>Text('x');}"
    fa = EXTRACTOR.analyze(minified, "min.dart")
    assert _sym(fa, "A", "class").line_start == 1  # au mieux : la première déclaration de la ligne
    # pas d'endpoint fantôme sur du JSON ou du Markdown
    assert EXTRACTOR.analyze('{"get": "/x", "routes": ["a", "b"]}', "x.json").endpoints == []
    assert EXTRACTOR.analyze("# Titre\n\nTexte avec 'apostrophe et `code`.\n", "x.md").symbols == []


def test_huge_line_and_many_lines_are_fast():
    t = time.time()
    fa = EXTRACTOR.analyze("var x = '" + "a" * 2_000_000 + "';\n", "huge.dart")
    assert time.time() - t < 2.0
    assert _sym(fa, "x", "variable").line_start == 1
    many = "int v%d = %d;\n" % (0, 0) + "".join("final v%d = f(\n  %d,\n);\n" % (i, i) for i in range(1, 16668))
    t = time.time()
    fa = EXTRACTOR.analyze(many, "many.dart")
    assert time.time() - t < 2.0
    assert many.count("\n") == 50002
    assert len(fa.symbols) == 16668
    last = fa.symbols[-1]
    assert (last.line_start, last.line_end) == (50000, 50002)  # initialiseur sur trois lignes


def test_one_megabyte_file_under_one_second():
    unit = '''import 'package:http/http.dart' as http;

/// Service %(i)d
class Service%(i)d extends BaseService with Loggable {
  final String base;
  final int? retries;
  static const int kMax%(i)d = 3;
  Service%(i)d({required this.base, this.retries});

  Future<List<Order>> fetch(int id) async {
    final url = '$base/orders/$id';
    final res = await http.get(Uri.parse(url), headers: {'a': 'b'});
    if (res.statusCode != 200) {
      throw Exception("bad ${res.statusCode}");
    }
    return (jsonDecode(res.body) as List).map((e) => Order.fromJson(e)).toList();
  }

  bool get ok => retries != null; // commentaire
  /* bloc
     commentaire */
}
'''
    text = "".join(unit % {"i": i} for i in range(1600))
    assert len(text) > 1_000_000
    t = time.time()
    fa = EXTRACTOR.analyze(text, "lib/big.dart")
    elapsed = time.time() - t
    assert elapsed < 1.0, "1 Mo analysé en %.2fs" % elapsed
    assert len([s for s in fa.symbols if s.kind == "class"]) == 1600
    assert len(fa.endpoints) == 1600
    assert fa.endpoints[-1].path == "$base/orders/$id"
    assert fa.endpoints[-1].handler == "Service1599@fetch"
    assert fa.warnings == []


def test_pathological_inputs_are_bounded():
    t = time.time()
    fa = EXTRACTOR.analyze("var x = '''\n" + "{ } ( ) ${a} ' \" \n" * 50000, "p.dart")
    assert time.time() - t < 2.0 and fa.warnings
    t = time.time()
    fa = EXTRACTOR.analyze("void f() {\n" + "  print('${a} ${b['k']} ${ {\"x\": 1} }');\n" * 30000 + "}\n", "p.dart")
    assert time.time() - t < 2.0 and _sym(fa, "f").line_end == 30002
    t = time.time()
    fa = EXTRACTOR.analyze("x" * 100000 + " " + "y" * 100000 + ".get('/a');\n", "p.dart")
    assert time.time() - t < 2.0
    t = time.time()
    fa = EXTRACTOR.analyze("(" * 50000 + "{" * 50000 + "<" * 50000, "p.dart")
    assert time.time() - t < 2.0 and fa.warnings


def test_to_dict_round_trip():
    fa = EXTRACTOR.analyze(API_SERVICE, "lib/services/api_service.dart")
    d = fa.to_dict()
    assert d["language"] == "dart"
    assert d["endpoints"][0]["normalized_path"] == "/orders"
    assert d["imports"][1]["names"] == ["http"]
