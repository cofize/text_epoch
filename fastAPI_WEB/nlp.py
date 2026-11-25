"""Анализатор-заглушка на основе жанровых словарей markers/dict_*_form.json.

Сейчас жанр задаёт пользователь. После его выбора анализируются только маркеры
соответствующего жанра. Модуль не зависит от FastAPI: позже сюда можно подключить
обученную модель, сохранив интерфейс analyze(text, genre).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MARKERS_DIR = BASE_DIR / "markers"
GENRE_FILES = {
    "поэзия": "dict_поэзия_form.json",
    "проза": "dict_проза_form.json",
    "публицистика": "dict_публицистика_form.json",
}
PERIODS = ["1800-1849", "1850-1899", "1900-1924", "1925-1944"]
TOKEN_RE = re.compile(r"[а-яё]+", re.IGNORECASE)


def _normalize(value: str) -> str:
    return value.lower().replace("ё", "е")


def _load_dictionaries() -> dict:
    dictionaries = {}
    for genre, filename in GENRE_FILES.items():
        path = MARKERS_DIR / filename
        with path.open(encoding="utf-8-sig") as file:
            payload = json.load(file)
        dictionaries[genre] = {
            _normalize(token): info for token, info in payload["markers"].items()
        }
    return dictionaries


DICTIONARIES = _load_dictionaries()


def analyze(text: str, genre: str) -> dict:
    """Вернуть период и найденные маркеры для заранее выбранного жанра."""
    if genre not in DICTIONARIES:
        raise ValueError(f"Неизвестный жанр: {genre}")
    if not text or not text.strip():
        raise ValueError("Введите текст для анализа")

    dictionary = DICTIONARIES[genre]
    scores = defaultdict(float)
    occurrences = defaultdict(int)
    found = []

    for match in TOKEN_RE.finditer(text):
        token = _normalize(match.group(0))
        info = dictionary.get(token)
        if not info:
            continue
        era = info.get("era")
        if era not in PERIODS:
            continue
        # log_ratio отражает выраженность маркера в соответствующем периоде.
        # Сглаживание не даёт единичному маркеру полностью определить результат.
        strength = max(float(info.get("log_ratio", 0.0)), 0.0)
        scores[era] += min(strength, 6.0)
        occurrences[era] += 1
        found.append({
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "category": "лексика",
            "weight": round(min(strength / 6.0, 1.0), 3),
            "points_to": era,
            "source": "rule",
            "count": 1,
        })

    # Суммируем свидетельства, но нормируем на число найденных признаков,
    # чтобы длинный текст не получал искусственно более высокую уверенность.
    total = sum(scores.values())
    if total:
        probabilities = {p: scores[p] / total for p in PERIODS}
        best = max(PERIODS, key=lambda p: probabilities[p])
        confidence = probabilities[best]
    else:
        probabilities = {p: 0.0 for p in PERIODS}
        best = None
        confidence = 0.0

    alternatives = []
    if best:
        alternatives = [
            {"period": p, "score": round(probabilities[p], 3)}
            for p in sorted(PERIODS, key=lambda p: probabilities[p], reverse=True)
            if p != best
        ][:2]

    found = found[:30]
    return {
        "genre": genre,
        "genre_score": 1.0,  # жанр выбран пользователем, а не предсказан заглушкой
        "genre_source": "user",
        "period": best,
        "period_range": (
            {"from": int(best[:4]), "to": int(best[5:])} if best else None
        ),
        "granularity": (50 if best in {"1800-1849", "1850-1899"} else 25) if best else None,
        "score": round(confidence, 3),
        "alternatives": alternatives,
        "dated_markers": found,
        "markers_found": len(found),
        "marker_counts_by_period": dict(occurrences),
        "model_version": "markers-stub-0.1",
        "stub": True,
        "message": None if best else "В тексте не найдены маркеры из выбранного жанрового словаря.",
    }


if __name__ == "__main__":
    sample = "Сей человек говорил о своей судьбе и о том, что видел вокруг."
    for genre in GENRE_FILES:
        print(genre, analyze(sample, genre))
