"""Интеграционная проверка сервиса. Сначала запустить uvicorn main:app --reload."""
import requests

BASE = "http://127.0.0.1:8000"
TEXT = ("Сей человек долго смотрел на дорогу и думал о своей судьбе. "
        "Вокруг стояла тишина, а вечерний свет медленно угасал за деревьями.")


def check(name, condition, details=""):
    mark = "OK" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f" — {details}" if details else ""))
    return bool(condition)


def main():
    passed = 0
    total = 0
    def run(name, condition, details=""):
        nonlocal passed, total
        total += 1
        passed += check(name, condition, details)

    health = requests.get(f"{BASE}/api/health", timeout=10)
    run("health endpoint", health.status_code == 200 and health.json().get("status") == "ok")

    genres = requests.get(f"{BASE}/api/genres", timeout=10)
    run("список жанров", genres.status_code == 200 and set(genres.json().get("genres", [])) == {"поэзия", "проза", "публицистика"})

    response = requests.post(f"{BASE}/api/analyze", json={"genre": "проза", "text": TEXT}, timeout=10)
    ok = response.status_code == 200
    run("анализ текста", ok, response.text[:160] if not ok else "")
    if ok:
        result = response.json()["result"]
        run("жанр отражён в результате", result.get("genre") == "проза")
        run("заглушка явно помечена", result.get("stub") is False)
        run("найденные маркеры имеют позиции", all(0 <= m["start"] < m["end"] <= len(TEXT) for m in result.get("dated_markers", [])))
        run("маркеры отсортированы", [m["start"] for m in result.get("dated_markers", [])] == sorted(m["start"] for m in result.get("dated_markers", [])))

    short = requests.post(f"{BASE}/api/analyze", json={"genre": "проза", "text": "слишком коротко"}, timeout=10)
    run("короткий текст отклоняется", short.status_code == 422)

    bad_genre = requests.post(f"{BASE}/api/analyze", json={"genre": "драма", "text": TEXT}, timeout=10)
    run("неизвестный жанр отклоняется", bad_genre.status_code == 400)

    page = requests.get(BASE, timeout=10)
    run("главная страница доступна", page.status_code == 200 and "TEXT EPOCH" in page.text)
    docs = requests.get(f"{BASE}/docs", timeout=10)
    run("Swagger доступен", docs.status_code == 200)
    print(f"\nИтого: {passed}/{total} проверок пройдено")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
