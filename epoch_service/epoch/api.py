"""Валидация запроса и ответы по контракту (см. README): handle_request(тело, анализатор)."""
import json
import logging

from .config import GENRES, MAX_CHARS, MIN_CHARS

ERRORS = {
    "empty_text": (400, "Введите текст."),
    "text_too_long": (400, f"Текст длиннее {MAX_CHARS} символов. Вставьте отрывок покороче."),
    "text_too_short": (400, f"Текст короче {MIN_CHARS} символов: для определения этого мало."),
    "invalid_request": (400, 'Запрос должен быть JSON вида {"text": "..."}.'),
    "unknown_genre": (400, "Неизвестный жанр. Допустимы: " + ", ".join(GENRES) + "."),
    "unsupported_language": (422, "Сервис работает только с текстами на русском языке."),
    "model_unavailable": (503, "Модель временно недоступна. Попробуйте позже."),
    "internal_error": (500, "Внутренняя ошибка сервера."),
}


class ApiError(Exception):
    def __init__(self, code):
        self.code = code
        self.http, self.message = ERRORS[code]
        super().__init__(self.message)


def russian_share(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("\u0400" <= c <= "\u052f" for c in letters) / len(letters)


def validate(body):
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise ApiError("invalid_request")
    text = body["text"]
    if not text.strip():
        raise ApiError("empty_text")
    if len(text) > MAX_CHARS:
        raise ApiError("text_too_long")
    if len(text.strip()) < MIN_CHARS:
        raise ApiError("text_too_short")
    if russian_share(text) < 0.6:
        raise ApiError("unsupported_language")
    genre = body.get("genre")
    if genre in (None, "", "auto"):
        genre = None
    elif genre not in GENRES:
        raise ApiError("unknown_genre")
    return text, genre


def handle_request(raw, analyzer):
    """-> (HTTP-статус, словарь ответа)."""
    try:
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError("invalid_request")
        text, genre = validate(body)
        if analyzer is None:
            raise ApiError("model_unavailable")
        return 200, analyzer.analyze(text, genre)
    except ApiError as e:
        return e.http, {"error": e.code, "message": e.message}
    except Exception:
        logging.getLogger("epoch").exception("internal error")
        return 500, {"error": "internal_error", "message": ERRORS["internal_error"][1]}
