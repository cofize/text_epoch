"""HTTP-слой на чистом WSGI: страница, статика и API. Зависимостей нет.

Локально:   python -m service --port 8000
Продакшен:  gunicorn service.wsgi:application   (или waitress, uwsgi)

Маршруты: GET /, GET /static/<файл>, GET /api/health, POST /api/analyze.
Те же API-пути работают и без префикса /api (/analyze, /health).
"""
import json
import mimetypes
import os

from epoch.analyzer import Analyzer
from epoch.config import CONTRACT_VERSION, DICT_DIR, GENRES
from epoch.api import handle_request
from epoch.dictionary import load_dictionaries
from epoch.models import load_models

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY = 1_000_000
CORS = [("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type")]


def build_analyzer(dict_dir=DICT_DIR):
    dicts = load_dictionaries(dict_dir)
    genre_model, epoch_model, version, stub = load_models(dicts)
    return Analyzer(dicts, genre_model, epoch_model, version, stub)


def make_app(analyzer):
    def respond(start_response, status, body, ctype, extra=()):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        phrase = {200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found",
                  405: "Method Not Allowed", 413: "Payload Too Large"}.get(status, "Error")
        start_response(f"{status} {phrase}", [("Content-Type", ctype),
                       ("Content-Length", str(len(data)))] + CORS + list(extra))
        return [data]

    def send_json(start_response, status, obj):
        return respond(start_response, status, json.dumps(obj, ensure_ascii=False),
                       "application/json; charset=utf-8", [("Cache-Control", "no-store")])

    def health():
        d = analyzer.dicts
        ok = bool(d.by_genre)
        return (200 if ok else 503), {
            "status": "ok" if ok else "no_dictionaries",
            "contract": CONTRACT_VERSION, "model_version": analyzer.model_version,
            "stub": analyzer.stub, "genres": GENRES,
            "genre_method": getattr(analyzer.genre_model, "method", "model"), "dictionaries": d.info(),
            "dictionaries_demo": d.demo, "warnings": d.warnings}

    def static(start_response, name):
        path = os.path.normpath(os.path.join(STATIC, name))
        if not path.startswith(STATIC + os.sep) or not os.path.isfile(path):
            return send_json(start_response, 404, {"error": "not_found", "message": "Не найдено."})
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            return respond(start_response, 200, f.read(),
                           ctype + ("; charset=utf-8" if ctype.startswith("text") or "javascript" in ctype else ""),
                           [("Cache-Control", "no-cache")])

    def application(environ, start_response):
        method, path = environ["REQUEST_METHOD"], environ.get("PATH_INFO", "/")
        if method == "OPTIONS":
            return respond(start_response, 204, b"", "text/plain")
        route = path[4:] if path.startswith("/api/") else path
        if route in ("/analyze", "/health"):
            if route == "/health" and method == "GET":
                return send_json(start_response, *health())
            if route == "/analyze" and method == "POST":
                try:
                    n = int(environ.get("CONTENT_LENGTH") or 0)
                except ValueError:
                    n = 0
                if n > MAX_BODY:
                    return send_json(start_response, 413, {"error": "text_too_long",
                                     "message": "Слишком большой запрос."})
                return send_json(start_response, *handle_request(environ["wsgi.input"].read(n), analyzer))
            return send_json(start_response, 405, {"error": "invalid_request",
                             "message": "Метод не поддерживается."})
        if method != "GET":
            return send_json(start_response, 405, {"error": "invalid_request",
                             "message": "Метод не поддерживается."})
        if path == "/":
            return static(start_response, "index.html")
        if path.startswith("/static/"):
            return static(start_response, path[len("/static/"):])
        return send_json(start_response, 404, {"error": "not_found", "message": "Не найдено."})

    return application


analyzer = build_analyzer()
application = make_app(analyzer)
