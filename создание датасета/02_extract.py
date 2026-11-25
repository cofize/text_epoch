#!/usr/bin/env python3
"""
Шаг 2. Достаём тексты из XML-дампа Викитеки и режем на фрагменты.

Вход:  дамп (.xml или .xml.bz2) и selected.csv из шага 1
Выход: fragments.jsonl (page_id, frag_id, text, n_words, old_orth)

Что делает с каждой выбранной страницей (только пространство имён 0):
  1. убирает вики-разметку: шаблоны (в том числе шапку {{Отексте}} с автором
     и названием, чтобы не было утечки), категории, ссылки, таблицы, теги,
     заголовки разделов, строки с источниками ("Печатается по...");
  2. приводит дореформенную орфографию к современной (ѣ, і, ѳ, ѵ, конечный ъ),
     иначе модель научится различать "до 1918" по буквам, а не по языку;
  3. режет текст на фрагменты примерно по 70 слов для прозы и публицистики
     и по 40 слов для поэзии (стихи режутся по строкам, строки сохраняются);
  4. оставляет не больше --per-page случайных фрагментов с одной страницы.

Зависит только от pandas и lxml. Дамп читается потоком, память почти не растёт.

Запуск:
python 02_extract.py --dump ruwikisource.xml.bz2 --selected selected.csv --out fragments.jsonl

"""
import argparse
import bz2
import html
import json
import random
import re

import pandas as pd
from lxml import etree

# --- орфография ------------------------------------------------------------
# \u0463 ѣ  \u0462 Ѣ  \u0473 ѳ  \u0472 Ѳ  \u0456 і  \u0406 І  \u0475 ѵ  \u0474 Ѵ
CONS = "бвгджзйклмнпрстфхцчшщБВГДЖЗЙКЛМНПРСТФХЦЧШЩ"
OLD_ORTH = re.compile("[\u0463\u0462\u0473\u0472\u0456\u0406\u0475\u0474]|"
                      "[" + CONS + "][ъЪ](?![а-яёА-ЯЁ])")
FINAL_HARD_SIGN = re.compile("(?<=[" + CONS + "])[ъЪ](?![а-яёА-ЯЁ])")
ORTH_MAP = str.maketrans({
    "\u0463": "е", "\u0462": "Е",   # ять
    "\u0473": "ф", "\u0472": "Ф",   # фита
    "\u0456": "и", "\u0406": "И",   # десятеричное и
    "\u0475": "и", "\u0474": "И",   # ижица
})


def normalize_orth(text):
    return FINAL_HARD_SIGN.sub("", text.translate(ORTH_MAP))


# --- чистка вики-разметки --------------------------------------------------
RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_REF_EMPTY = re.compile(r"<ref[^>]*/>", re.I)
RE_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I)
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_CATEGORY = re.compile(
    r"\[\[\s*(?:Категория|Category|Файл|File|Image|Изображение)\s*:[^\]]*\]\]",
    re.I)
RE_WIKILINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
RE_EXT_LINK_TEXT = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
RE_EXT_LINK = re.compile(r"\[https?://[^\]\s]*\]")
RE_BR = re.compile(r"<br\s*/?>", re.I)
RE_TAG = re.compile(r"<[^>\n]{1,200}>")
RE_QUOTES = re.compile(r"'{2,}")
RE_HEADING = re.compile(r"^=+.*?=+\s*$")
RE_LIST = re.compile(r"^[*#:;]+\s*")
RE_MAGIC = re.compile(r"__[A-Z]+__")
RE_SOURCE_LINE = re.compile(
    r"^(Источники?\s*:|Печатается по|Впервые\s*(опубликовано|напечатано|"
    r"[—–:-])|Опубликовано|Текст по|ЦАМО|АП РФ|ГАРФ|РГАСПИ|"
    r"Примечани[ея]|См\. |Прим\. )")
RE_CYR = re.compile(r"[А-Яа-яЁё]")


def clean_wikitext(raw):
    t = RE_COMMENT.sub("", raw)
    t = RE_REF.sub("", RE_REF_EMPTY.sub("", t))
    t = RE_TABLE.sub("", t)
    prev = None
    while prev != t:                       # вложенные шаблоны, изнутри наружу
        prev = t
        t = RE_TEMPLATE.sub("", t)
    t = RE_CATEGORY.sub("", t)
    t = RE_WIKILINK.sub(r"\1", t)
    t = RE_EXT_LINK_TEXT.sub(r"\1", t)
    t = RE_EXT_LINK.sub("", t)
    t = RE_BR.sub("\n", t)
    t = RE_TAG.sub("", t)
    t = html.unescape(t).replace("\xa0", " ")
    t = RE_MAGIC.sub("", RE_QUOTES.sub("", t))
    lines = []
    for line in t.split("\n"):
        line = RE_LIST.sub("", line).strip()
        line = re.sub(r"[ \t]+", " ", line)
        if not line or RE_HEADING.match(line) or RE_SOURCE_LINE.match(line):
            continue
        if not RE_CYR.search(line):         # "* * *", номера, латиница
            continue
        lines.append(line)
    return "\n".join(lines)


# --- нарезка на фрагменты --------------------------------------------------
SENT_SPLIT = re.compile(
    r"(?:(?<=[.!?…])|(?<=[.!?…][\"»”)\]]))\s+"
    r"(?=[\"«“(\[—–-]*[А-ЯЁA-Z0-9])")
# длина фрагмента в СИМВОЛАХ
TARGET_CHARS = {"поэзия": 800}
DEFAULT_TARGET_CHARS = 1800
MIN_CHARS = {"поэзия": 300}
DEFAULT_MIN_CHARS = 1000
MAX_CHARS = {"поэзия": 1500}
DEFAULT_MAX_CHARS = 3000


def n_words(s):
    return len(s.split())


def split_units(text, genre):
    if genre == "поэзия":
        return text.split("\n"), "\n"
    units = []
    for para in text.split("\n"):
        units.extend(s.strip() for s in SENT_SPLIT.split(para) if s.strip())
    return units, " "


def pack(units, joiner, target, min_chars):
    out, cur, size = [], [], 0
    for u in units:
        cur.append(u)
        size += len(u) + 1
        if size >= target:
            out.append(joiner.join(cur))
            cur, size = [], 0
    if cur and size >= min_chars:
        out.append(joiner.join(cur))
    return out


def good(fragment, genre):
    n = len(fragment)
    if not (MIN_CHARS.get(genre, DEFAULT_MIN_CHARS) <= n
            <= MAX_CHARS.get(genre, DEFAULT_MAX_CHARS)):
        return False
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", fragment)
    if not letters:
        return False
    cyr_share = sum(1 for c in letters if RE_CYR.match(c)) / len(letters)
    digit_share = sum(c.isdigit() for c in fragment) / n
    return cyr_share >= 0.9 and digit_share < 0.05


def make_fragments(raw, genre, rnd, per_page):
    text = clean_wikitext(raw)
    old = bool(OLD_ORTH.search(text))
    text = normalize_orth(text)
    units, joiner = split_units(text, genre)
    frags = [f for f in pack(units, joiner,
                             TARGET_CHARS.get(genre, DEFAULT_TARGET_CHARS),
                             MIN_CHARS.get(genre, DEFAULT_MIN_CHARS))
             if good(f, genre)]
    if len(frags) > per_page:
        frags = rnd.sample(frags, per_page)
    return frags, old


def open_dump(path):
    return bz2.open(path, "rb") if path.endswith(".bz2") else open(path, "rb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--selected", default="selected.csv")
    ap.add_argument("--out", default="fragments.jsonl")
    ap.add_argument("--per-page", type=int, default=12,
                    help="максимум фрагментов с одной страницы")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sel = pd.read_csv(args.selected)
    genre_of = dict(zip(sel.page_id.astype(int), sel.genre))
    rnd = random.Random(args.seed)
    scanned = used = n_frag = 0

    with open_dump(args.dump) as fh, \
            open(args.out, "w", encoding="utf-8") as out:
        for _, page in etree.iterparse(fh, events=("end",),
                                       tag="{*}page", huge_tree=True):
            scanned += 1
            if page.findtext("{*}ns") == "0":
                pid = int(page.findtext("{*}id"))
                if pid in genre_of:
                    raw = page.findtext("{*}revision/{*}text") or ""
                    frags, old = make_fragments(raw, genre_of[pid], rnd,
                                                args.per_page)
                    if frags:
                        used += 1
                    for i, f in enumerate(frags):
                        out.write(json.dumps({
                            "page_id": pid, "frag_id": f"{pid}_{i}",
                            "text": f, "n_words": n_words(f),
                            "old_orth": old}, ensure_ascii=False) + "\n")
                        n_frag += 1
            page.clear()
            while page.getprevious() is not None:
                del page.getparent()[0]
            if scanned % 50000 == 0:
                print(f"просмотрено страниц: {scanned}, "
                      f"с фрагментами: {used}, фрагментов: {n_frag}",
                      flush=True)

    print(f"Готово. Просмотрено: {scanned}, использовано страниц: {used}, "
          f"фрагментов: {n_frag}. Файл: {args.out}")
    missing = len(genre_of) - used
    if missing > 0:
        print(f"Из {len(genre_of)} выбранных страниц фрагментов не дали "
              f"{missing} (нет в дампе, слишком короткие или не кириллица).")


if __name__ == "__main__":
    main()
