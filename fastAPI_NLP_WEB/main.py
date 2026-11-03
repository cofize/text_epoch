"""Сервис NLP+WEB. Пары 5–6.

Запуск:  uvicorn main:app --reload
Проверка: http://127.0.0.1:8000/docs

Шаг 1 уже написан — остальные делаете вы. После каждого шага сохраняйте файл
(uvicorn перезапустится сам) и запускайте проверку: python test_api.py
"""

import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field

from nlp import MODES, analyze

app = FastAPI(
    title="Определитель эпохи текста",
    description="Учебный проект курса «Основы web-программирования»: текст → век, оценка и датирующие признаки",
    version="0.1.0",
)

# Это понадобится на паре 13, когда страница начнёт обращаться к серверу.
# Без этих строк браузер не пустит её запросы: страница и сервер будут
# по разным адресам, а это для браузера «чужой сайт».
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY = []        # история запросов живёт в памяти процесса: перезапуск её обнуляет
REQUESTS = 0        # сколько всего запросов к /api/analyze было
INDEX = Path(__file__).parent / "static" / "index.html"


# ——————————————————————————————————————————————————————————————
# Шаг 1. Готово — смотрите, как устроен эндпоинт
# ——————————————————————————————————————————————————————————————
# @app.get(...) — декоратор: он говорит FastAPI «по этому адресу вызывай
# функцию ниже». Что функция вернёт, то FastAPI превратит в JSON.

@app.get("/hello")
def hello():
    return {"message": "Hello, FastAPI!"}


# ——————————————————————————————————————————————————————————————
# Шаг 2. Эндпоинт /greet с параметром запроса
# ——————————————————————————————————————————————————————————————
# Нужно: GET /greet?name=Alex  →  {"message": "Hello, Alex!"}
#        GET /greet            →  {"message": "Hello, stranger!"}
#
# Query("stranger", min_length=1, max_length=50) задаёт значение по умолчанию
# и ограничения: FastAPI проверит их сам и вернёт 422, если они нарушены.
#
# (сделано)

@app.get("/greet")
def greet(name: str = Query("stranger", min_length=1, max_length=50)):
    return {"message": f"Hello, {name}!"}


# ——————————————————————————————————————————————————————————————
# Шаг 3. POST /feedback с Pydantic-моделью
# ——————————————————————————————————————————————————————————————
# Нужно: POST /feedback с телом {"username": "Alex", "comment": "норм"}
#        →  {"message": "Thanks for your feedback, Alex!"}
#
# Класс-наследник BaseModel описывает, что должно прийти в теле запроса.
# Проверку типов и ограничений FastAPI делает сам.
#
# (сделано)

class Feedback(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    comment: str = Field(min_length=1, max_length=1000)


@app.post("/feedback")
def feedback(item: Feedback):
    return {"message": f"Thanks for your feedback, {item.username}!"}


# ——————————————————————————————————————————————————————————————
# Шаг 4. Главный эндпоинт проекта: POST /api/analyze
# ——————————————————————————————————————————————————————————————
# Это шаг 3, переименованный под ваш проект. Ответ должен совпадать
# с тем, что записано в вашем файле API.md:
#
#   {"ok": true,
#    "result": {...},                                  ← то, что вернула analyze()
#    "meta": {"chars": 0, "mode": "stats", "took_ms": 0}}
#
# Если analyze() подняла ValueError (пустой текст, неизвестный режим) —
# отвечаем кодом 400 и текстом ошибки.
#
# (сделано)

class AnalyzeIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    mode: str = "epoch"


@app.post("/api/analyze")
def analyze_text(item: AnalyzeIn):
    global REQUESTS
    started = time.perf_counter()
    try:
        result = analyze(item.text, item.mode)
    except ValueError as error:
        raise HTTPException(400, str(error))

    REQUESTS += 1
    response = {
        "ok": True,
        "result": result,
        "meta": {
            "chars": len(item.text),
            "mode": item.mode,
            "took_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    }
    HISTORY.append({"text": item.text[:80], "mode": item.mode, "result": result})
    return response


# ——————————————————————————————————————————————————————————————
# Шаг 5. Режимы анализа — в файле nlp.py
# ——————————————————————————————————————————————————————————————
# Откройте nlp.py: режим "stats" написан, "freq" и "longest" ваши.
# Здесь ничего менять не нужно, кроме эндпоинта ниже.
#
# (сделано) Он отдаёт список доступных режимов,
# чтобы страница на паре 13 могла построить выпадающий список сама

@app.get("/api/modes")
def modes():
    return {"modes": MODES}


# ——————————————————————————————————————————————————————————————
# Со звёздочкой
# ——————————————————————————————————————————————————————————————
# 1. (сделано) GET /api/health → {"status": "ok", "requests": <сколько запросов было>}
# 2. (сделано) GET /api/history?limit=10 → последние запросы из HISTORY, новые сверху
# 3. (сделано: режим epoch) Четвёртый режим анализа на вашем материале — в nlp.py
# 4. Сохранять историю не в список, а в MongoDB из занятия 3


@app.get("/api/health")
def health():
    return {"status": "ok", "requests": REQUESTS}


@app.get("/api/history")
def history(limit: int = Query(10, ge=1, le=100)):
    return {"items": HISTORY[-limit:][::-1]}      # новые сверху


# ——————————————————————————————————————————————————————————————
# Страница сайта. Отдаётся тем же сервером, поэтому открывается на
# http://127.0.0.1:8000/ и обращается к /api/analyze без настройки адресов.
# ——————————————————————————————————————————————————————————————

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX)
