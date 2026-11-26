"""Запуск: python -m unittest discover tests -v"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from epoch.analyzer import Analyzer
from epoch.api import handle_request
from epoch.config import GENRES
from epoch.dictionary import load_dictionaries
from epoch.normalize import normalize_orth, token_key, tokens
from epoch.stubs import StubEpochModel, StubGenreModel
from service.wsgi import make_app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICTS = load_dictionaries(os.path.join(ROOT, "dictionaries"))
PROSE = ("Помещик Ладыгин приехал в уездный город. В трактире говорили, что становой и чиновник "
         "повздорили, а барыня писала из деревни, что приказчик винит погоду.")


def analyzer(g=None, e=None):
    return Analyzer(DICTS, g or StubGenreModel(), e or StubEpochModel(), "test", True)


class Normalize(unittest.TestCase):
    def test_positions_survive_old_orthography(self):
        text = "Онъ вернулся домой поздно вечеромъ, и въ комнатѣ было тихо."
        for s, e, raw, key in tokens(text):
            self.assertEqual(text[s:e], raw)
        keys = [k for *_, k in tokens(text)]
        self.assertIn("он", keys)
        self.assertIn("комнате", keys)
        self.assertIn("вечером", keys)

    def test_key(self):
        self.assertEqual(token_key("Ещё"), "еще")
        self.assertEqual(token_key("Онъ"), "он")
        self.assertEqual(token_key("объявление"), "объявление")  # ъ внутри слова остаётся

    def test_same_as_step2(self):
        path = os.environ.get("STEP2_PATH", "/mnt/user-data/uploads/02_extract.py")
        if not os.path.exists(path):
            self.skipTest("02_extract.py не найден (задайте STEP2_PATH)")
        sys.modules.setdefault("lxml", mock.MagicMock())
        sys.modules.setdefault("lxml.etree", mock.MagicMock())
        spec = importlib.util.spec_from_file_location("step2", path)
        step2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step2)
        for t in ["Онъ вернулся въ комнатѣ", "Ѳедоръ и Ѵижица, міръ, объявленіе", "ёлка, съѣздъ, подъѣздъ-то"]:
            self.assertEqual(normalize_orth(t), step2.normalize_orth(t))


class Dictionaries(unittest.TestCase):
    def test_demo_loaded_for_all_genres(self):
        self.assertEqual(set(DICTS.by_genre), set(GENRES))
        for gd in DICTS.by_genre.values():
            self.assertTrue(gd.demo)
            for e in gd.entries.values():
                self.assertTrue(0 <= gd.weight(e) <= 1)

    def test_bad_files_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            files = {"dict_проза_form.json": {"genre": "проза", "unit": "lemma", "eras": ["1800-1849"],
                                              "markers": {"сей": {"era": "1800-1849", "log_ratio": 2}}},
                     "dict_поэзия_form.json": {"genre": "поэзия", "unit": "form", "eras": ["1800-1849"],
                                               "markers": {"сей": {"era": "1800-1849", "log_ratio": 2},
                                                           "ныне": {"era": "1999-2000", "log_ratio": 2}}}}
            for name, obj in files.items():
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False)
            r = load_dictionaries(d)
        self.assertEqual(list(r.by_genre), ["поэзия"])
        self.assertEqual(list(r.by_genre["поэзия"].entries), ["сей"])
        self.assertTrue(any("lemma" in w or "unit" in w for w in r.warnings))


class Pipeline(unittest.TestCase):
    def test_genre_first_then_era_with_that_genres_dictionary(self):
        seen = {}

        class G:
            def predict(self, text):
                return {"поэзия": 0.1, "проза": 0.8, "публицистика": 0.1}

        class E:
            def predict(self, ctx):
                seen["genre"], seen["hits"], seen["eras"] = ctx.genre, ctx.hits, ctx.eras
                return {e: 0.1 for e in ctx.eras} | {"1850-1899": 0.7}

        r = analyzer(G(), E()).analyze(PROSE)
        self.assertEqual((r["genre"], seen["genre"]), ("проза", "проза"))
        self.assertEqual(r["period"], "1850-1899")
        self.assertTrue(seen["hits"])
        self.assertEqual(r["dictionary"]["genre"], "проза")

    def test_forced_genre_switches_dictionary(self):
        text = "Сей тихий дол, где дремлет лира, где юный певец зрел утренний пламень и глас."
        poetry = analyzer().analyze(text, "поэзия")
        prose = analyzer().analyze(text, "проза")
        self.assertEqual(poetry["genre_source"], "user")
        self.assertTrue(poetry["markers_found"] > prose["markers_found"])

    def test_marker_offsets_and_order(self):
        r = analyzer().analyze(PROSE)
        starts = [m["start"] for m in r["dated_markers"]]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(r["dated_markers"])
        for m in r["dated_markers"]:
            self.assertEqual(PROSE[m["start"]:m["end"]], m["text"])
            self.assertEqual(m["points_to"], r["period"])
            self.assertTrue(0 <= m["weight"] <= 1)

    def test_scores_are_distributions(self):
        r = analyzer().analyze(PROSE)
        self.assertAlmostEqual(sum(x["score"] for x in r["period_scores"]), 1, places=2)
        self.assertAlmostEqual(sum(x["score"] for x in r["genre_scores"]), 1, places=2)
        self.assertEqual(r["granularity"], r["period_range"]["to"] - r["period_range"]["from"] + 1)

    def test_old_orthography_text_finds_markers(self):
        r = analyzer().analyze("Помѣщикъ Ладыгинъ пріѣхалъ въ городъ; становой и чиновникъ поссорились.", "проза")
        self.assertGreaterEqual(r["markers_found"], 3)


class Api(unittest.TestCase):
    def call(self, body, a="default"):
        raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        return handle_request(raw, analyzer() if a == "default" else a)

    def test_ok_and_contract_fields(self):
        st, b = self.call({"text": PROSE})
        self.assertEqual(st, 200)
        for k in ("genre", "genre_score", "period", "period_range", "granularity", "score",
                  "alternatives", "dated_markers", "model_version"):
            self.assertIn(k, b)

    def test_errors(self):
        cases = [({"text": ""}, 400, "empty_text"), ({"text": "коротко"}, 400, "text_too_short"),
                 ({"text": "а" * 10001}, 400, "text_too_long"), ({"x": 1}, 400, "invalid_request"),
                 ({"text": 5}, 400, "invalid_request"), (b"{bad", 400, "invalid_request"),
                 ({"text": PROSE, "genre": "драма"}, 400, "unknown_genre"),
                 ({"text": "This sentence is definitely written in English and is long enough."},
                  422, "unsupported_language")]
        for body, status, code in cases:
            st, b = self.call(body)
            self.assertEqual((st, b["error"]), (status, code), body)
        st, b = self.call({"text": PROSE}, a=None)
        self.assertEqual((st, b["error"]), (503, "model_unavailable"))

    def test_internal_error_hides_details(self):
        class Boom:
            def analyze(self, *a):
                raise RuntimeError("secret")
        with self.assertLogs("epoch", "ERROR"):
            st, b = handle_request(json.dumps({"text": PROSE}).encode(), Boom())
        self.assertEqual((st, b["error"]), (500, "internal_error"))
        self.assertNotIn("secret", json.dumps(b))


class Http(unittest.TestCase):
    app = staticmethod(make_app(analyzer()))

    def req(self, method, path, body=b""):
        out = {}
        env = {"REQUEST_METHOD": method, "PATH_INFO": path, "CONTENT_LENGTH": str(len(body)),
               "wsgi.input": io.BytesIO(body)}
        data = b"".join(self.app(env, lambda s, h: out.update(status=int(s.split()[0]), headers=dict(h))))
        return out["status"], out["headers"], data

    def test_routes(self):
        st, h, data = self.req("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn("text/html", h["Content-Type"])
        st, _, data = self.req("GET", "/api/health")
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(data)["stub"])
        body = json.dumps({"text": PROSE}, ensure_ascii=False).encode()
        for path in ("/api/analyze", "/analyze"):
            st, h, data = self.req("POST", path, body)
            self.assertEqual(st, 200)
            self.assertIn("utf-8", h["Content-Type"])
            self.assertEqual(json.loads(data)["genre"], "проза")
        self.assertEqual(self.req("GET", "/api/analyze")[0], 405)
        self.assertEqual(self.req("GET", "/nope")[0], 404)
        self.assertEqual(self.req("OPTIONS", "/api/analyze")[0], 204)

    def test_static_traversal_blocked(self):
        for p in ("/static/../epoch/config.py", "/static/%2e%2e/epoch/config.py", "/static//etc/passwd"):
            self.assertEqual(self.req("GET", p)[0], 404, p)

    def test_health_without_dictionaries_is_503(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Analyzer(load_dictionaries(d), StubGenreModel(), StubEpochModel(), "t", True)
        app = make_app(empty)
        out = {}
        b"".join(app({"REQUEST_METHOD": "GET", "PATH_INFO": "/api/health", "wsgi.input": io.BytesIO()},
                     lambda s, h: out.update(s=s)))
        self.assertTrue(out["s"].startswith("503"))


if __name__ == "__main__":
    unittest.main()
