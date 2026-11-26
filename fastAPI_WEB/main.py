"""FastAPI-сервис text_epoch: сначала выбор жанра, затем датировка текста."""
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from nlp import GENRE_FILES
from model_adapter import analyze, status

app = FastAPI(
    title="Text Epoch — определитель периода текста",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY = []
REQUESTS = 0
INDEX = Path(__file__).parent / "static" / "index.html"


class AnalyzeIn(BaseModel):
    genre: str = "auto"          # "auto" - жанр определит модель
    text: str = Field(min_length=50, max_length=10000)


@app.get("/api/genres")
def genres():
    return {"genres": list(GENRE_FILES)}


@app.post("/api/analyze")
def analyze_text(item: AnalyzeIn):
    global REQUESTS
    started = time.perf_counter()
    try:
        result = analyze(item.text, item.genre)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    REQUESTS += 1
    response = {
        "ok": True,
        "result": result,
        "meta": {
            "chars": len(item.text),
            "genre": item.genre,
            "took_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    }
    HISTORY.append({"genre": item.genre, "text": item.text[:80], "result": result})
    return response


@app.get("/api/health")
def health():
    return {"status": "ok", "requests": REQUESTS, "analyzer": status()}


@app.get("/api/history")
def history(limit: int = Query(10, ge=1, le=100)):
    return {"items": HISTORY[-limit:][::-1]}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX)
