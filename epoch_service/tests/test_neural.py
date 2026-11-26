"""Всё, что не требует torch: нарезка, агрегирование, чтение папки модели, совместимость эпох,
автоподхват модели и работа конвейера с моделью на 6 периодов и словарём на 4 эпохи.
Сам torch-код (RunModel) здесь не запускается: нужны веса."""
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from epoch import config
from epoch.analyzer import Analyzer
from epoch.config import GENRES
from epoch.dictionary import Dictionaries, Entry, GenreDictionary, eras_overlap, load_dictionaries
from epoch.models import load_models
from epoch.stubs import StubGenreModel
from epoch import neural
from epoch.neural import (NeuralEpochModel, NeuralGenreModel, check_compatibility, log_mean, read_run,
                          word_chunks)

ERAS6 = ["1800-1824", "1825-1849", "1850-1874", "1875-1899", "1900-1924", "1925-1949"]
ERAS4 = ["1800-1849", "1850-1899", "1900-1924", "1925-1944"]


class FakeRun:
    """Подставная модель с тем же интерфейсом, что RunModel; запоминает, что ей передали."""
    genres, eras, frag_words = GENRES, ERAS6, 40

    def __init__(self, era_pick=2):
        self.era_pick, self.calls = era_pick, []

    def genre_proba(self, texts):
        self.calls.append(("genre", list(texts), None))
        return np.tile([0.1, 0.8, 0.1], (len(texts), 1))

    def era_proba(self, texts, genres):
        self.calls.append(("era", list(texts), list(genres)))
        p = np.full(6, 0.04)
        p[self.era_pick] = 0.8
        return np.tile(p, (len(texts), 1))


def dicts4():
    gd = GenreDictionary("проза", ERAS4, {"помещик": Entry("1850-1899", 3.0), "сей": Entry("1800-1849", 4.0),
                                           "колхоз": Entry("1925-1944", 5.0)})
    return Dictionaries({"проза": gd})


class Chunks(unittest.TestCase):
    TEXT = " ".join(f"слово{i}" for i in range(100))

    def test_split_by_words_and_no_loss(self):
        ch = word_chunks(self.TEXT, 40)
        self.assertEqual([len(c.split()) for c in ch], [40, 40, 20])   # хвост 20 слов >= 40//4: остаётся отдельным
        self.assertEqual(" ".join(ch).split(), self.TEXT.split())

    def test_short_tail_is_merged(self):
        text = " ".join(f"w{i}" for i in range(85))
        ch = word_chunks(text, 40)                                     # хвост 5 слов < 10: присоединяется
        self.assertEqual([len(c.split()) for c in ch], [40, 45])
        self.assertEqual(" ".join(ch).split(), text.split())           # ничего не потеряно и не продублировано

    def test_short_text_and_cap(self):
        self.assertEqual(word_chunks("мало слов", 40), ["мало слов"])
        long = " ".join(f"w{i}" for i in range(40 * 50))
        self.assertEqual(len(word_chunks(long, 40)), 16)

    def test_log_mean_is_distribution_and_geometric(self):
        p = log_mean(np.array([[0.9, 0.1], [0.5, 0.5]]))
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertAlmostEqual(p[0] / p[1], (0.9 * 0.5) ** 0.5 / (0.1 * 0.5) ** 0.5, places=5)


class Models(unittest.TestCase):
    def test_era_model_gets_genre_with_every_chunk_and_normalized_text(self):
        run = FakeRun()
        text = "Онъ вернулся въ городъ " * 30                          # 120 слов -> 3 фрагмента по 40
        probs = NeuralEpochModel(run).predict(mock.Mock(text=text, genre="проза"))
        kind, texts, genres = run.calls[-1]
        self.assertEqual((kind, len(texts), genres), ("era", 3, ["проза"] * 3))
        self.assertNotIn("ъ", " ".join(texts))                          # ъ снят до модели, как в обучении
        self.assertEqual(max(probs, key=probs.get), "1850-1874")
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_genre_model(self):
        g = NeuralGenreModel(FakeRun()).predict("Сей тихий дол " * 30)
        self.assertEqual(max(g, key=g.get), "проза")


class Folder(unittest.TestCase):
    def make(self, d, **kw):
        m = os.path.join(d, "model")
        for n in ("genre", "era", "tok"):
            os.makedirs(os.path.join(m, n), exist_ok=True)
        if kw.get("info", True):
            with open(os.path.join(m, "model_info.json"), "w", encoding="utf-8") as f:
                json.dump({"name": "rubert", "genres": GENRES, "eras": ERAS6, "max_len": 512}, f)
        if "words" in kw:
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"frag_words_median": kw["words"]}, f)

    def test_ok_and_meta(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, words=187)
            info, words = read_run(d)
        self.assertEqual((info["eras"], words), (ERAS6, 187))

    def test_default_words_without_meta(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d)
            self.assertEqual(read_run(d)[1], 220)

    def test_friendly_errors(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "model_run"):
                read_run(d)
            self.make(d, info=False)
            with self.assertRaisesRegex(RuntimeError, "model_info.json"):
                read_run(d)
            self.make(d)
            open(os.path.join(d, "model", "booster.joblib"), "w").close()
            with self.assertRaisesRegex(RuntimeError, "без словаря"):
                read_run(d)


class Compatibility(unittest.TestCase):
    def test_overlap(self):
        self.assertTrue(eras_overlap("1850-1899", "1875-1899"))
        self.assertTrue(eras_overlap("1925-1944", "1925-1949"))
        self.assertFalse(eras_overlap("1800-1849", "1850-1874"))

    def test_mismatch_is_warning_not_error(self):
        d = dicts4()
        check_compatibility(GENRES, ERAS6, d)
        self.assertEqual(len(d.warnings), 1)
        self.assertIn("пересекается", d.warnings[0])

    def test_same_eras_no_warning(self):
        gd = GenreDictionary("проза", ERAS6, {"сей": Entry("1800-1824", 4.0)})
        d = Dictionaries({"проза": gd})
        check_compatibility(GENRES, ERAS6, d)
        self.assertEqual(d.warnings, [])

    def test_unknown_genre_fails_at_startup(self):
        with self.assertRaisesRegex(RuntimeError, "GENRES"):
            check_compatibility(["драма"], ERAS6, dicts4())

    def test_non_overlapping_dictionary_era_is_flagged(self):
        gd = GenreDictionary("проза", ["1950-1999"], {"x": Entry("1950-1999", 2.0)})
        d = Dictionaries({"проза": gd})
        check_compatibility(GENRES, ERAS6, d)
        self.assertIn("не покажутся", d.warnings[0])


class Pipeline(unittest.TestCase):
    TEXT = "Помещик Ладыгин приехал в город, и сей помещик долго молчал, а колхоз был ещё далеко. " * 3

    def analyzer(self, pick):
        run, d = FakeRun(pick), dicts4()
        check_compatibility(GENRES, ERAS6, d)
        return Analyzer(d, NeuralGenreModel(run), NeuralEpochModel(run), "test", stub=False)

    def test_periods_come_from_model_and_markers_by_overlap(self):
        r = self.analyzer(2).analyze(self.TEXT)                          # модель: 1850-1874
        self.assertEqual((r["genre"], r["period"], r["stub"]), ("проза", "1850-1874", False))
        self.assertEqual([x["period"] for x in r["period_scores"]], ERAS6)
        self.assertEqual({m["text"].lower() for m in r["dated_markers"]}, {"помещик"})
        self.assertEqual({m["points_to"] for m in r["dated_markers"]}, {"1850-1899"})   # эпоха слова из словаря
        self.assertEqual((r["markers_found"], len(r["dated_markers"])), (12, 6))

    def test_last_period_matches_shorter_dictionary_era(self):
        r = self.analyzer(5).analyze(self.TEXT)                          # модель: 1925-1949, словарь: 1925-1944
        self.assertEqual({m["text"].lower() for m in r["dated_markers"]}, {"колхоз"})

    def test_genre_without_dictionary_uses_model_periods(self):
        r = self.analyzer(2).analyze("Сей тихий дол, где дремлет лира, и юный певец зрел пламень.\n" * 4, "поэзия")
        self.assertEqual([x["period"] for x in r["period_scores"]], ERAS6)
        self.assertIn("no_dictionary", r["warnings"])


class GenreChoice(unittest.TestCase):
    """Жанр: по умолчанию эвристика (жанровая сеть в проверках ошибалась), период: модель."""

    def build(self, env):
        made = {}

        class Run(FakeRun):
            info = {"name": "rubert"}

            def __init__(self, path, load_genre=True):
                super().__init__()
                made["load_genre"] = load_genre

        clean = {k: v for k, v in os.environ.items() if k != "EPOCH_GENRE"}
        with mock.patch.object(neural, "RunModel", Run), mock.patch.dict(os.environ, {**clean, **env}, clear=True):
            g, e, version = neural.build(Dictionaries())
        return g, e, made

    def test_default_is_heuristic_and_genre_net_not_loaded(self):
        g, e, made = self.build({})
        self.assertIsInstance(g, StubGenreModel)
        self.assertIsInstance(e, NeuralEpochModel)
        self.assertFalse(made["load_genre"])

    def test_model_genre_can_be_enabled(self):
        g, e, made = self.build({"EPOCH_GENRE": "model"})
        self.assertIsInstance(g, NeuralGenreModel)
        self.assertTrue(made["load_genre"])

    def test_genre_method_in_response(self):
        d, run = dicts4(), FakeRun(2)
        prose = "Помещик Ладыгин приехал в город, и сей помещик долго молчал у окна. " * 3
        heur = Analyzer(d, StubGenreModel(), NeuralEpochModel(run), "t", stub=False)
        net = Analyzer(d, NeuralGenreModel(run), NeuralEpochModel(run), "t", stub=False)
        self.assertEqual(heur.analyze(prose)["genre_method"], "heuristic")
        self.assertEqual(net.analyze(prose)["genre_method"], "model")
        r = heur.analyze(prose, "поэзия")
        self.assertEqual((r["genre_method"], r["genre_source"]), ("user", "user"))
        self.assertEqual(run.calls[-1][2][0], "поэзия")           # период получает жанр, выбранный пользователем

    def test_genre_net_refuses_when_not_loaded(self):
        run = FakeRun()
        run.genre_proba = lambda texts: (_ for _ in ()).throw(RuntimeError("не загружена"))
        with self.assertRaises(RuntimeError):
            NeuralGenreModel(run).predict("текст " * 50)


class Autodetect(unittest.TestCase):
    def test_model_run_folder_is_picked_up(self):
        d = Dictionaries()
        sentinel = (FakeRun(), FakeRun(), "v")
        with tempfile.TemporaryDirectory() as run:
            os.makedirs(os.path.join(run, "model"))
            with open(os.path.join(run, "model", "model_info.json"), "w") as f:
                f.write("{}")
            env = {k: v for k, v in os.environ.items() if k != "EPOCH_MODELS_FACTORY"}
            with mock.patch.object(config, "RUN_DIR", run), mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch("epoch.neural.build", return_value=sentinel):
                g, e, version, stub = load_models(d)
                self.assertEqual((version, stub), ("v", False))
                with mock.patch.dict(os.environ, {"EPOCH_MODELS_FACTORY": "stub"}):
                    self.assertTrue(load_models(d)[3])                    # stub принудительно
        with tempfile.TemporaryDirectory() as empty:
            with mock.patch.object(config, "RUN_DIR", empty):
                env = {k: v for k, v in os.environ.items() if k != "EPOCH_MODELS_FACTORY"}
                with mock.patch.dict(os.environ, env, clear=True):
                    self.assertTrue(load_models(d)[3])                    # папки нет: заглушки


if __name__ == "__main__":
    unittest.main()
