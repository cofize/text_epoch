"""Заглушки. Жанр: простые признаки текста. Эпоха: голосование по словарю маркеров.
Это НЕ модель: вероятности не откалиброваны, нужны только чтобы интерфейс жил."""
import math
import re

from .config import GENRES

TEMP = 40.0


def _softmax(scores):
    m = max(scores.values())
    ex = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


class StubGenreModel:
    method = "heuristic"            # жанр по признакам текста, не нейросеть
    PAST = re.compile(r"\w+(?:л|ла|ло|ли)$")
    ABSTRACT = re.compile(r"\w+(?:ость|ости|ение|ения|ением|ство|ства|изм|ция|ции)$")

    def predict(self, text):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        words = re.findall(r"[а-яёА-ЯЁ]+", text.lower()) or ["_"]
        short = sum(len(l) <= 60 for l in lines) / max(len(lines), 1)
        dialog = len(re.findall(r"(?:^|\n)\s*[—–-]\s", text)) + text.count("— ")
        past = sum(bool(self.PAST.match(w)) for w in words) / len(words)
        abstract = sum(bool(self.ABSTRACT.match(w)) for w in words) / len(words)
        poem = 4.0 * short if len(lines) >= 4 else 0.0
        scores = {
            "поэзия": poem,
            "проза": 1.0 + min(dialog, 6) * 0.3 + past * 12 - poem * 0.3,
            "публицистика": 1.0 + abstract * 25 - poem * 0.3,
        }
        return _softmax({g: scores[g] for g in GENRES})


class StubEpochModel:
    def predict(self, ctx):
        dens = {e: 0.0 for e in ctx.eras}
        for h in ctx.hits:
            dens[h.era] += h.log_ratio
        n = max(ctx.n_tokens, 1)
        return _softmax({e: v / n * 100 / TEMP for e, v in dens.items()})
