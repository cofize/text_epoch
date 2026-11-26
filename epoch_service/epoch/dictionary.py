"""Загрузка словарей маркеров: файлы dict_<жанр>_form.json из шага 4.

Формат: {"genre", "unit": "form", "eras": [...], "markers": {слово: {"era", "log_ratio", ...}}}.
Жанр берётся из поля genre, а не из имени файла. Некорректные записи пропускаются
и попадают в warnings (видны в /api/health).
"""
import glob
import json
import os
from dataclasses import dataclass, field

from .config import GENRES


def era_range(label):
    a, b = label.split("-")
    return int(a), int(b)


def eras_overlap(a, b):
    """Пересекаются ли периоды, например 1850-1899 и 1875-1899 (словарь по 50 лет, модель по 25)."""
    (a1, a2), (b1, b2) = era_range(a), era_range(b)
    return a1 <= b2 and b1 <= a2


@dataclass
class Entry:
    era: str
    log_ratio: float


@dataclass
class GenreDictionary:
    genre: str
    eras: list
    entries: dict
    demo: bool = False
    scale: float = 1.0

    def weight(self, entry):
        """0..1: log_ratio относительно 95-го процентиля словаря этого жанра."""
        return float(min(1.0, max(0.0, entry.log_ratio / self.scale)))


@dataclass
class Dictionaries:
    by_genre: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def info(self):
        return {g: {"markers": len(d.entries), "eras": d.eras, "demo": d.demo}
                for g, d in self.by_genre.items()}

    @property
    def demo(self):
        return any(d.demo for d in self.by_genre.values())


def _load_one(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    genre = raw.get("genre")
    if genre not in GENRES:
        raise ValueError(f"неизвестный жанр {genre!r}")
    if raw.get("unit", "form") != "form":
        raise ValueError("поддерживается только unit=form (лемматизация на сайте не подключена)")
    eras = raw.get("eras") or []
    for e in eras:
        era_range(e)
    entries, bad = {}, 0
    for tok, v in (raw.get("markers") or {}).items():
        try:
            if v["era"] not in eras:
                raise KeyError(v["era"])
            entries[tok] = Entry(v["era"], float(v["log_ratio"]))
        except (KeyError, TypeError, ValueError):
            bad += 1
    if not entries:
        raise ValueError("в словаре нет ни одного корректного маркера")
    lr = sorted(e.log_ratio for e in entries.values())
    scale = lr[int(0.95 * (len(lr) - 1) + 0.5)] or 1.0       # 95-й процентиль
    return GenreDictionary(genre, eras, entries, bool(raw.get("demo")), scale), bad


def load_dictionaries(directory):
    out = Dictionaries()
    for path in sorted(glob.glob(os.path.join(directory, "dict_*_form.json"))):
        name = os.path.basename(path)
        try:
            gd, bad = _load_one(path)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            out.warnings.append(f"{name}: пропущен ({e})")
            continue
        if gd.genre in out.by_genre:
            out.warnings.append(f"{name}: второй словарь жанра «{gd.genre}», пропущен")
            continue
        if bad:
            out.warnings.append(f"{name}: пропущено некорректных записей: {bad}")
        out.by_genre[gd.genre] = gd
    for g in GENRES:
        if g not in out.by_genre:
            out.warnings.append(f"нет словаря для жанра «{g}»: подсветки не будет")
    return out
