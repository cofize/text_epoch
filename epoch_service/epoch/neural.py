"""Подключение обученной модели напарницы (папка запуска model_run из Colab).

Ожидаемая структура (preds_*.jsonl не нужны):
    model_run/meta.json                      необязательно: frag_words_median (длина фрагмента в словах)
    model_run/model/model_info.json          genres, eras, max_len
    model_run/model/genre/  era/  tok/       веса (safetensors) и токенизатор

Как устроена её модель (из era_pipeline.py): две сети. Жанр: по тексту. Период: по ПАРЕ «жанр, текст»,
то есть период зависит от жанра, и жанр надо передавать. Текст режется на фрагменты по словам,
вероятности фрагментов усредняются по логарифмам.

Отличие от её сервиса: мы приводим текст к тому же виду, на котором модель училась (шаг 2:
ѣ, і, ѳ, ѵ и конечный ъ), её сервис кормит модель сырым текстом.
Нужны torch и transformers (requirements-nn.txt). Словарь маркеров на оценку не влияет, только на подсветку.
"""
import json
import os

import numpy as np

from . import config
from .config import GENRES
from .dictionary import eras_overlap
from .normalize import normalize_orth

DEFAULT_FRAG_WORDS = 220    # если нет meta.json; в обучающих данных медиана около 220 слов
MAX_CHUNKS = 16             # длиннее: берём 16 фрагментов равномерно (скорость)


def word_chunks(text, n):
    """Фрагменты по n слов; короткий хвост (меньше n/4 слов) присоединяется к предыдущему."""
    w = text.split()
    ch = [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [text.strip()]
    if len(ch) > 1 and len(ch[-1].split()) < n // 4:
        tail = ch.pop()                 # именно так: `ch[-2] += ch.pop()` сдвигает индекс и теряет фрагмент
        ch[-1] += " " + tail
    if len(ch) > MAX_CHUNKS:
        ch = [ch[i] for i in sorted({int(round(i)) for i in np.linspace(0, len(ch) - 1, MAX_CHUNKS)})]
    return ch


def log_mean(P):
    """Усреднение вероятностей фрагментов по логарифмам (как в её сервисе)."""
    l = np.log(np.asarray(P) + 1e-9).mean(axis=0)
    p = np.exp(l - l.max())
    return p / p.sum()


def read_run(run_dir):
    """Проверяет папку запуска -> (model_info, длина фрагмента в словах). Понятные ошибки вместо трассировок."""
    mdir = os.path.join(run_dir, "model")
    if not os.path.isdir(mdir):
        raise RuntimeError(f"Не найдена папка модели {mdir}. Положите папку model_run рядом с проектом "
                           f"или задайте путь в EPOCH_NN_DIR.")
    for f in ("booster.joblib", "model.joblib"):
        if os.path.exists(os.path.join(mdir, f)):
            raise RuntimeError(f"В {mdir} лежит {f}: это запуск tfidf или модель со словарём. Поддерживаются запуски "
                               f"rubert/xlmr без словаря (папки genre, era, tok и model_info.json).")
    missing = [n for n in ("model_info.json", "genre", "era", "tok") if not os.path.exists(os.path.join(mdir, n))]
    if missing:
        raise RuntimeError(f"В {mdir} не хватает: {', '.join(missing)}. Скачайте папку model целиком.")
    with open(os.path.join(mdir, "model_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    for k in ("genres", "eras"):
        if not info.get(k):
            raise RuntimeError(f"В model_info.json нет поля {k}.")
    words = DEFAULT_FRAG_WORDS
    meta = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            words = int(json.load(f).get("frag_words_median") or words)
    return info, words


class RunModel:
    def __init__(self, run_dir, batch_size=8, device=None, load_genre=True):
        info, self.frag_words = read_run(run_dir)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("Для работы модели нужны torch и transformers: pip install torch transformers") from e
        self.info, self.genres, self.eras = info, list(info["genres"]), list(info["eras"])
        self.max_len, self.batch_size, self._torch = int(info.get("max_len", 512)), batch_size, torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        mdir = os.path.join(run_dir, "model")
        self.tok = AutoTokenizer.from_pretrained(os.path.join(mdir, "tok"))
        load = lambda name: AutoModelForSequenceClassification.from_pretrained(
            os.path.join(mdir, name)).to(self.device).eval()
        self.genre_model = load("genre") if load_genre else None   # сеть жанра можно не грузить: экономия памяти
        self.era_model = load("era")

    def _proba(self, model, texts, genres=None):
        torch, out = self._torch, []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                t = list(texts[i:i + self.batch_size])
                if genres is None:
                    enc = self.tok(t, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
                else:   # период: пара «жанр, текст», как при обучении
                    enc = self.tok(list(genres[i:i + self.batch_size]), t, truncation="only_second",
                                   max_length=self.max_len, padding=True, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    def genre_proba(self, texts):
        if self.genre_model is None:
            raise RuntimeError("Сеть жанра не загружена (EPOCH_GENRE=model включает её).")
        return self._proba(self.genre_model, texts)

    def era_proba(self, texts, genres):
        return self._proba(self.era_model, texts, genres)


class NeuralGenreModel:
    method = "model"

    def __init__(self, run):
        self.run = run

    def predict(self, text):
        chunks = word_chunks(normalize_orth(text), self.run.frag_words)
        return dict(zip(self.run.genres, (float(v) for v in log_mean(self.run.genre_proba(chunks)))))


class NeuralEpochModel:
    def __init__(self, run):
        self.run = run
        self.eras = list(run.eras)

    def predict(self, ctx):
        chunks = word_chunks(normalize_orth(ctx.text), self.run.frag_words)
        p = log_mean(self.run.era_proba(chunks, [ctx.genre] * len(chunks)))
        return dict(zip(self.eras, (float(v) for v in p)))


def check_compatibility(genre_labels, era_labels, dictionaries):
    """Неизвестные жанры: ошибка при запуске. Разные эпохи словаря и модели: допустимо (слова подсвечиваются
    по пересечению периодов), но об этом предупреждаем в /api/health и на странице."""
    unknown = set(genre_labels) - set(GENRES)
    if unknown:
        raise RuntimeError(f"Модель знает жанры {sorted(unknown)}, которых нет в epoch/config.py GENRES.")
    groups = {}
    for genre, gd in dictionaries.by_genre.items():
        if set(gd.eras) != set(era_labels):
            groups.setdefault(tuple(gd.eras), []).append(genre)
    for eras, genres in groups.items():
        lost = [e for e in eras if not any(eras_overlap(e, m) for m in era_labels)]
        text = (f"Периоды словарей ({', '.join(genres)}): {', '.join(eras)}; периоды модели: {', '.join(era_labels)}. "
                f"Слова подсвечиваются, если их период пересекается с найденным.")
        if lost:
            text += f" Периоды {', '.join(lost)} не пересекаются ни с одним периодом модели: их слова не покажутся."
        dictionaries.warnings.append(text)


def build(dictionaries):
    """Период определяет модель. Жанр: по умолчанию эвристика по признакам текста (epoch/stubs.py),
    потому что жанровая сеть в проверках ошибалась; EPOCH_GENRE=model включает жанровую сеть модели."""
    use_model_genre = os.environ.get("EPOCH_GENRE", "heuristic").strip().lower() == "model"
    run = RunModel(config.RUN_DIR, load_genre=use_model_genre)
    check_compatibility(run.genres, run.eras, dictionaries)
    if use_model_genre:
        genre = NeuralGenreModel(run)
    else:
        from .stubs import StubGenreModel
        genre = StubGenreModel()
    folder = os.path.basename(os.path.normpath(config.RUN_DIR))
    return genre, NeuralEpochModel(run), f"{run.info.get('name', 'model')} ({folder})"
