"""Нормализация и токенизация. ДОЛЖНА совпадать с 02_extract.py (normalize_orth)
и 04_marker_candidates.py (WORD, нижний регистр, ё -> е), иначе слова словаря
не найдутся в тексте, а нейросеть увидит не то, что видела при обучении.

Замены букв (ѣ -> е и т. д.) идут один к одному по длине, поэтому позиции слов
в нормализованном тексте равны позициям в исходном: подсветка попадает точно.
Конечный ъ снимается уже у отдельного слова (при поиске в словаре), а не в тексте.
"""
import re

CONS = "бвгджзйклмнпрстфхцчшщБВГДЖЗЙКЛМНПРСТФХЦЧШЩ"
ORTH_MAP = str.maketrans({
    "\u0463": "е", "\u0462": "Е", "\u0473": "ф", "\u0472": "Ф",
    "\u0456": "и", "\u0406": "И", "\u0475": "и", "\u0474": "И",
})
FINAL_HARD_SIGN = re.compile("(?<=[" + CONS + "])[ъЪ](?![а-яёА-ЯЁ])")
WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")


def normalize_orth(text):
    """Как в шаге 2: тот текст, на котором учится модель."""
    return FINAL_HARD_SIGN.sub("", text.translate(ORTH_MAP))


def token_key(raw):
    return FINAL_HARD_SIGN.sub("", raw).lower().replace("ё", "е")


def tokens(text):
    """[(start, end, слово_как_в_тексте, ключ_словаря)] с позициями в ИСХОДНОМ тексте."""
    mapped = text.translate(ORTH_MAP)
    out = []
    for m in WORD.finditer(mapped):
        key = token_key(m.group(0))
        if len(key) >= 2:                      # как в шаге 4: однобуквенные слова не считаются
            out.append((m.start(), m.end(), text[m.start():m.end()], key))
    return out
