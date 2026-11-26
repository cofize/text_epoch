"""Интерфейсы моделей. Сюда подключается нейросеть.

Порядок работы сервиса: 1) GenreModel определяет жанр; 2) по жанру берётся его
словарь, ищутся совпадения (hits); 3) EpochModel получает текст, жанр и совпадения
и возвращает вероятности эпох ЭТОГО жанра.

Обученная модель из папки model_run подхватывается сама (epoch/neural.py). Любые другие модели:
    def build(dictionaries) -> (genre_model, epoch_model, model_version)
и запуск с EPOCH_MODELS_FACTORY=ваш_модуль:build (см. README).
"""
import importlib
import os
from dataclasses import dataclass


@dataclass
class Hit:
    start: int
    end: int
    text: str
    key: str
    era: str
    log_ratio: float
    weight: float


@dataclass
class EpochContext:
    text: str            # исходный текст пользователя
    genre: str
    eras: list           # метки эпох жанра по порядку, например "1850-1899"
    hits: list           # список Hit: совпадения со словарём этого жанра
    n_tokens: int


class GenreModel:
    def predict(self, text):
        """-> {жанр: вероятность}, сумма 1."""
        raise NotImplementedError


class EpochModel:
    def predict(self, ctx):
        """-> {метка эпохи из ctx.eras: вероятность}, сумма 1."""
        raise NotImplementedError


def load_models(dictionaries):
    """-> (genre_model, epoch_model, model_version, is_stub).

    Порядок выбора: EPOCH_MODELS_FACTORY=модуль:функция; иначе, если рядом лежит папка model_run,
    берётся она (epoch.neural); иначе заглушки. EPOCH_MODELS_FACTORY=stub принудительно включает заглушки.
    """
    from . import config
    spec = os.environ.get("EPOCH_MODELS_FACTORY")
    if not spec and os.path.exists(os.path.join(config.RUN_DIR, "model", "model_info.json")):
        spec = "epoch.neural:build"
    if spec and spec != "stub":
        mod, func = spec.split(":")
        g, e, version = getattr(importlib.import_module(mod), func)(dictionaries)
        return g, e, version, False
    from .stubs import StubEpochModel, StubGenreModel
    return StubGenreModel(), StubEpochModel(), "stub-0.1", True
