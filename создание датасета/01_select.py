#!/usr/bin/env python3
"""
Шаг 1. Выбираем страницы Викитеки для эксперимента.

Вход:  meta.jsonl (по строке на страницу: title, page_id, length, categories,
       year, year_source, author, translator, ...)
Выход: selected.csv

Что делает:
  - берёт только страницы с годом в нужном диапазоне, без переводов,
    без конфликтных годов и достаточной длины;
  - определяет жанр по категориям Викитеки (если в записи уже есть поле
    "genre" из вашего пайплайна, берёт его, а свою функцию игнорирует);
  - страницы с несколькими жанрами или без жанра отбрасывает;
  - считает корзину по годам (по умолчанию 25 лет);
  - задаёт "произведение" (work): часть названия до первого "/", то есть
    все главы одной книги и все редакции одного закона получают один work;
  - задаёт "группу" для разбиения на train/val/test: автор, а если автора нет,
    то произведение. Так один автор не окажется сразу в train и в test.
    (Если у автора в разметке несколько написаний имени, они считаются
    разными авторами, это можно поправить позже словарём замен.)

Запуск:
  python 01_select.py --meta meta.jsonl --out selected.csv
"""
import argparse
import json

import pandas as pd

# Жанры по точным названиям категорий Викитеки. Список можно расширять.
GENRES = {
    "поэзия": ["Стихотворения", "Русская поэзия", "Поэмы",
               "Сборники стихов", "Восьмистишия"],
    "проза": ["Рассказы", "Повести", "Романы", "Русская проза"],
    "публицистика": ["Статьи", "Критика", "Очерки", "Публицистика",
                     "Фельетоны"],
}


def genre_from_categories(cats):
    found = [g for g, names in GENRES.items() if any(n in cats for n in names)]
    return found[0] if len(found) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta.jsonl")
    ap.add_argument("--out", default="selected.csv")
    ap.add_argument("--year-min", type=int, default=1800)
    ap.add_argument("--year-max", type=int, default=1944)
    ap.add_argument("--bin-years", type=int, default=25)
    ap.add_argument("--min-length", type=int, default=500)
    ap.add_argument("--genres", default="проза,публицистика,поэзия")
    args = ap.parse_args()

    keep = set(args.genres.split(","))
    rows = []
    with open(args.meta, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            year = r.get("year")
            if year is None:
                continue
            if r.get("translator") or r.get("year_source") == "conflict":
                continue
            year = int(year)
            if not (args.year_min <= year <= args.year_max):
                continue
            if r.get("length", 0) < args.min_length:
                continue
            genre = r.get("genre") or genre_from_categories(
                r.get("categories", []))
            if genre not in keep:
                continue
            title = r["title"]
            rows.append({
                "page_id": r["page_id"],
                "title": title,
                "work": title.split("/")[0].strip(),
                "author": (r.get("author") or "").strip(),
                "year": year,
                "year_source": r.get("year_source", ""),
                "genre": genre,
                "length": r["length"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Ничего не выбралось: проверьте фильтры и поля meta.")
    df["bin"] = args.year_min + ((df.year - args.year_min)
                                 // args.bin_years) * args.bin_years
    df["group"] = df.author.where(df.author != "", df.work)
    df.to_csv(args.out, index=False)

    print(f"Выбрано страниц: {len(df)}, произведений: {df.work.nunique()}, "
          f"групп: {df.group.nunique()}")
    pd.set_option("display.width", 200)
    print("\nСтраниц по жанру и корзине:")
    print(df.pivot_table(index="bin", columns="genre", values="page_id",
                         aggfunc="count", fill_value=0))
    print("\nПроизведений по жанру и корзине:")
    print(df.pivot_table(index="bin", columns="genre", values="work",
                         aggfunc="nunique", fill_value=0))
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
