"""Конвейер: жанр -> словарь этого жанра -> эпоха -> маркеры."""
from .config import DEFAULT_ERAS, GENRES, MAX_MARKERS, SHORT_TEXT_CHARS
from .dictionary import eras_overlap, era_range
from .models import EpochContext, Hit
from .normalize import tokens


def _normalized(scores, keys):
    clean = {k: max(0.0, float(scores.get(k, 0.0))) for k in keys}
    z = sum(clean.values())
    if z <= 0:
        raise ValueError("модель вернула нулевые вероятности")
    return {k: v / z for k, v in clean.items()}


class Analyzer:
    def __init__(self, dictionaries, genre_model, epoch_model, model_version, stub):
        self.dicts, self.genre_model, self.epoch_model = dictionaries, genre_model, epoch_model
        self.model_version, self.stub = model_version, stub

    def find_hits(self, toks, gd):
        hits = []
        for start, end, raw, key in toks:
            e = gd.entries.get(key) if gd else None
            if e:
                hits.append(Hit(start, end, raw, key, e.era, e.log_ratio, gd.weight(e)))
        return hits

    def analyze(self, text, genre=None):
        # 1. жанр
        if genre:
            genre_scores, source = {g: float(g == genre) for g in GENRES}, "user"
        else:
            genre_scores, source = _normalized(self.genre_model.predict(text), GENRES), "model"
        genre = max(genre_scores, key=genre_scores.get)

        # 2. словарь этого жанра
        gd = self.dicts.by_genre.get(genre)
        # периоды ответа задаёт модель; у заглушки их нет, тогда берутся эпохи словаря
        eras = (getattr(self.epoch_model, "eras", None) or (gd.eras if gd else None)
                or (next(iter(self.dicts.by_genre.values())).eras if self.dicts.by_genre else DEFAULT_ERAS))
        toks = tokens(text)
        hits = self.find_hits(toks, gd)

        # 3. эпоха внутри жанра
        ctx = EpochContext(text, genre, eras, hits, len(toks))
        era_scores = _normalized(self.epoch_model.predict(ctx), eras)
        best = max(era_scores, key=era_scores.get)
        y1, y2 = era_range(best)
        ranked = sorted(era_scores, key=era_scores.get, reverse=True)

        # 4. маркеры: слова словаря жанра, указывающие на найденную эпоху
        # (эпоха слова может быть крупнее периода модели: 1850-1899 подходит к 1850-1874)
        pointing = sorted((h for h in hits if eras_overlap(h.era, best)), key=lambda h: -h.weight)
        pointing = sorted(pointing[:MAX_MARKERS], key=lambda h: h.start)
        warnings = []
        if len(text.strip()) < SHORT_TEXT_CHARS:
            warnings.append("short_text")
        if gd is None:
            warnings.append("no_dictionary")
        return {
            "genre": genre, "genre_score": round(genre_scores[genre], 3), "genre_source": source,
            "genre_method": "user" if source == "user" else getattr(self.genre_model, "method", "model"),
            "genre_scores": [{"genre": g, "score": round(genre_scores[g], 3)} for g in GENRES],
            "period": best, "period_range": {"from": y1, "to": y2}, "granularity": y2 - y1 + 1,
            "score": round(era_scores[best], 3),
            "alternatives": [{"period": e, "score": round(era_scores[e], 3)}
                             for e in ranked[1:3] if era_scores[e] >= 0.02],
            "period_scores": [{"period": e, "score": round(era_scores[e], 3)} for e in eras],
            "dated_markers": [{
                "text": h.text, "start": h.start, "end": h.end, "category": "лексика",
                "weight": round(h.weight, 3), "log_ratio": round(h.log_ratio, 2),
                "points_to": h.era, "source": "dictionary"} for h in pointing],
            "markers_found": len(hits),
            "text_length": len(text),
            "warnings": warnings,
            "dictionary": {"genre": genre, "markers": len(gd.entries) if gd else 0,
                           "demo": bool(gd and gd.demo)},
            "stub": self.stub,
            "model_version": self.model_version,
        }
