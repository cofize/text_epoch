"""Анализ текста. Единственная точка входа — функция analyze(text, mode).

Этот файл — шов между курсами. Сейчас здесь простая обработка на стандартной
библиотеке; на паре 13 сюда встанет ваша настоящая функция анализа из АОЕЯ.
Остальной проект об этом не узнает, если сохранится сигнатура:

    analyze(text, mode) -> dict

Правило: ошибки в данных (пустой текст, неизвестный режим) здесь поднимаются
как ValueError. Превращать их в HTTP-ответ — дело main.py. Модуль анализа
ничего не знает про веб, и это нарочно: его можно вызвать из ноутбука,
из скрипта, из другого сервиса.
"""

import re
from collections import Counter

# Определитель эпохи. Когда будет готова настоящая модель, меняется только эта строка:
#   from model import predict
from epoch_stub import predict

STOPWORDS = {
    "и", "в", "во", "не", "на", "что", "с", "со", "как", "а", "то", "все", "она",
    "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "ее",
    "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "когда",
    "даже", "ну", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
    "вам", "там", "потом", "себя", "ничего", "ей", "они", "тут", "где", "есть",
    "надо", "для", "мы", "их", "чем", "была", "без", "чего", "раз", "себе", "под",
    "будет", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "это",
}

MODES = ["epoch", "stats", "freq", "longest"]


def tokenize(text):
    """Слова текста: только буквы, всё в нижнем регистре.

    Регулярное выражение [а-яёa-z]+ оставляет кириллицу и латиницу,
    а цифры, знаки препинания и дефисы отбрасывает. Способ грубый —
    «кто-то» распадётся на два слова, — но для начала этого хватает.
    """
    return re.findall(r"[а-яёa-z]+", text.lower())


def analyze(text, mode="epoch"):
    # ——— Режим epoch: определение эпохи текста (проект) ———
    # Токенизация здесь не нужна: признаки эпохи включают буквы вроде «ѣ» и «і»,
    # которые tokenize() отбросил бы. Результат: {period, score, dated_markers, ...}
    if mode == "epoch":
        if not text.strip():
            raise ValueError("в тексте нет слов")
        return predict(text)

    words = tokenize(text)
    if not words:
        raise ValueError("в тексте нет слов")

    # ——— Режим 1: готов, разберитесь, как он устроен ———
    if mode == "stats":
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        return {
            "words": len(words),                                  # всего слов
            "unique": len(set(words)),                            # уникальных
            "ttr": round(len(set(words)) / len(words), 3),        # лексическое разнообразие
            "sentences": len(sentences),
            "avg_sentence": round(len(words) / max(len(sentences), 1), 1),
        }

    # ——— Режим 2: частотный список. TODO ———
    # Верните десять самых частых знаменательных слов:
    #   {"top": [{"word": "язык", "count": 7}, ...], "total_counted": 123}
    # Знаменательные — те, которых нет в STOPWORDS и длина которых больше двух.
    # Пригодится: Counter(список).most_common(10)
    if mode == "freq":
        counted = [w for w in words if w not in STOPWORDS and len(w) > 2]
        return {
            "top": [{"word": w, "count": c} for w, c in Counter(counted).most_common(10)],
            "total_counted": len(counted),
        }

    # ——— Режим 3: самые длинные слова. TODO ———
    # Верните десять самых длинных разных слов и длину самого длинного:
    #   {"words": ["словообразование", ...], "max_length": 15}
    # Пригодится: sorted(set(words), key=len, reverse=True)
    if mode == "longest":
        longest = sorted(set(words), key=len, reverse=True)[:10]
        return {"words": longest, "max_length": len(longest[0])}

    raise ValueError(f"неизвестный режим: {mode}. Доступны: {', '.join(MODES)}")


# Проверить модуль можно и без сервера:  python nlp.py
if __name__ == "__main__":
    sample = ("Язык устроен так, что небольшое число элементов порождает "
              "бесконечное число сообщений. Эта способность называется продуктивностью.")
    for mode in MODES:
        try:
            print(mode, "→", analyze(sample, mode))
        except ValueError as error:
            print(mode, "→ пока не работает:", error)
