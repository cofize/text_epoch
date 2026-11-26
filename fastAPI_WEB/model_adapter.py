"""Адаптер: подключает обученную модель к веб-сервису. Интерфейс тот же: analyze(text, genre).

Что нужно рядом (в папке fastAPI_WEB):
  era_pipeline.py        - файл из обучения
  model_run/             - ОДНА папка запуска из Colab (например runs_new/xlmr целиком: meta.json + model/)
                          для *_dict-запусков положите рядом и базовую папку (см. base_run в booster.joblib)
Путь можно задать переменной окружения MODEL_RUN.
Если модели нет или она не загрузилась - сервис продолжает работать на словарях (заглушка).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import nlp  # словарная заглушка: берём из неё подсветку маркеров и запасной режим

RUN_DIR = Path(os.environ.get("MODEL_RUN", Path(__file__).resolve().parent / "model_run"))
_MODEL = None
_META = {}
_ERROR = None


def _load():
    global _MODEL, _META, _ERROR
    try:
        import era_pipeline as ep
        _MODEL = ep.load_model(str(RUN_DIR))          # грузится один раз при старте сервера
        mp = RUN_DIR / "meta.json"
        _META = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    except Exception as e:                            # нет папки, нет torch и т.д.
        _MODEL, _ERROR = None, f"{type(e).__name__}: {e}"


_load()


def is_loaded() -> bool:
    return _MODEL is not None


def status() -> dict:
    return {"model_loaded": is_loaded(), "run": RUN_DIR.name, "error": _ERROR}


def _chunks(text: str, n: int):
    w = text.split()
    ch = [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [text]
    if len(ch) > 1 and len(ch[-1].split()) < n // 4:
        ch[-2] += " " + ch.pop()
    return ch


def _agg(P):                                           # усреднение лог-вероятностей по фрагментам
    l = np.log(P + 1e-9).mean(0)
    p = np.exp(l - l.max())
    return p / p.sum()


def analyze(text: str, genre: str | None = None) -> dict:
    """genre: 'поэзия' | 'проза' | 'публицистика' - жанр задан пользователем;
    None или 'auto' - жанр определяет модель."""
    if not is_loaded():
        return nlp.analyze(text, genre if genre in nlp.DICTIONARIES else "проза")
    if not text or not text.strip():
        raise ValueError("Введите текст для анализа")
    m = _MODEL
    chunks = _chunks(text, int(_META.get("frag_words_median", 80)) or 80)

    if genre in (None, "", "auto"):
        pg = _agg(m.genre_proba(chunks))
        genre, gscore, gsource = m.genres[int(pg.argmax())], float(pg.max()), "model"
    else:
        if genre not in m.genres:
            raise ValueError(f"Неизвестный жанр: {genre}")
        gscore, gsource = 1.0, "user"

    pe = _agg(m.era_proba(chunks, [genre] * len(chunks)))
    order = np.argsort(-pe)
    best = m.eras[int(order[0])]
    start = int(best.split("-")[0])
    end = int(best.split("-")[1])

    marks = nlp.analyze(text, genre) if genre in nlp.DICTIONARIES else {}
    return {
        "genre": genre,
        "genre_score": round(gscore, 3),
        "genre_source": gsource,
        "period": best,
        "period_range": {"from": start, "to": end},
        "granularity": end - start + 1,
        "score": round(float(pe[order[0]]), 3),
        "alternatives": [{"period": m.eras[int(i)], "score": round(float(pe[i]), 3)} for i in order[1:3]],
        "era_probs": {m.eras[i]: round(float(pe[i]), 3) for i in range(len(m.eras))},
        "dated_markers": marks.get("dated_markers", []),
        "markers_found": marks.get("markers_found", 0),
        "marker_counts_by_period": marks.get("marker_counts_by_period", {}),
        "fragments": len(chunks),
        "model_version": f"{RUN_DIR.name}",
        "stub": False,
        "message": None,
    }
