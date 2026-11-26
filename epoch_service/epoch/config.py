"""Константы сервиса. Жанры должны совпадать с жанрами из шага 1 пайплайна."""
import os

GENRES = ["поэзия", "проза", "публицистика"]
MIN_CHARS, MAX_CHARS = 50, 10_000
SHORT_TEXT_CHARS = 400          # короче обучающих фрагментов (1000-3000 знаков): предупреждаем
MAX_MARKERS = 300
CONTRACT_VERSION = "1.2"
DEFAULT_ERAS = ["1800-1849", "1850-1899", "1900-1924", "1925-1944"]
DICT_DIR = os.environ.get(
    "EPOCH_DICT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dictionaries"))

# Папка запуска модели из Colab (внутри: model/{genre,era,tok,model_info.json}, рядом meta.json).
# Если она лежит рядом с проектом, сервис подхватывает её сам.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.environ.get("EPOCH_NN_DIR", os.path.join(PROJECT_DIR, "model_run"))
