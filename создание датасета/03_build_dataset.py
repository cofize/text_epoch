#!/usr/bin/env python3
"""
Шаг 3. Собираем датасет и делим его на три файла: train.jsonl, val.jsonl,
test.jsonl.

Главное правило: автор целиком, со всеми своими произведениями, попадает
ровно в одну выборку. Для этого строятся группы (связные компоненты):
  - автор нормализуется (без [[ ]], регистр, ё/е, пунктуация);
  - автор и все его произведения объединяются в одну группу;
  - если одно произведение встретилось под разными написаниями автора
    или как с автором, так и без, эти варианты тоже склеиваются.
В конце проверяется, что ни группа, ни автор, ни произведение не встречаются
в двух выборках (иначе скрипт падает).

Порядок:
  1. группы -> 2. лимит на произведение и на группу -> 3. отбрасывание
  мелких ячеек "жанр x корзина" и выравнивание -> 4. жадное распределение
  групп по выборкам с учётом размера, внутри каждой ячейки -> 5. запись.

Запуск:
  python 03_build_dataset.py --fragments fragments.jsonl \
      --selected selected.csv --out-dir data
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]


# --- нормализация и группы -------------------------------------------------
def clean_author(a):
    a = str(a or "")
    a = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", a)
    return a.replace("[", "").replace("]", "").strip()


def norm(s):
    s = clean_author(s).lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_groups(df, aliases):
    df["author_norm"] = df["author"].map(
        lambda a: aliases.get(norm(a), norm(a)))
    df["work_norm"] = df["work"].map(norm)
    dsu = DSU()
    keys = []
    for a, w in zip(df.author_norm, df.work_norm):
        wn = "w:" + w
        if a:
            an = "a:" + a
            dsu.union(an, wn)
            keys.append(an)
        else:
            keys.append(wn)
    df["group"] = [dsu.find(k) for k in keys]
    df["author"] = df["author"].map(clean_author)
    return df


# --- распределение групп по выборкам --------------------------------------
def assign_splits(df, fracs, seed):
    """Жадно: большие группы первыми, каждую кладём в ту выборку, где
    после добавления доля заполнения целевого размера (по ячейкам
    "жанр x корзина") получится наименьшей."""
    cell_tot = df.groupby(["genre", "bin"]).size().to_dict()
    gcell = df.groupby(["group", "genre", "bin"]).size()
    per_group = defaultdict(dict)
    for (g, ge, b), n in gcell.items():
        per_group[g][(ge, b)] = n

    rng = np.random.RandomState(seed)
    groups = list(per_group)
    rng.shuffle(groups)                                   # случайные ничьи
    groups.sort(key=lambda g: -sum(per_group[g].values()))  # стабильная

    cur = {s: defaultdict(int) for s in SPLITS}
    where = {}
    for g in groups:
        best, best_cost = None, None
        for s in SPLITS:
            cost = sum(((cur[s][c] + n) / (fracs[s] * cell_tot[c])) ** 2
                       for c, n in per_group[g].items())
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        where[g] = best
        for c, n in per_group[g].items():
            cur[best][c] += n
    return df["group"].map(where)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", default="fragments.jsonl")
    ap.add_argument("--selected", default="selected.csv")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--cap-per-work", type=int, default=40)
    ap.add_argument("--cap-per-group", type=int, default=300,
                    help="максимум фрагментов от одного автора (группы), "
                         "чтобы один автор не доминировал в ячейке")
    ap.add_argument("--min-cell", type=int, default=100)
    ap.add_argument("--per-cell", type=int, default=0,
                    help="0 = размер самой маленькой оставшейся ячейки")
    ap.add_argument("--fracs", default="0.8,0.1,0.1",
                    help="доли train,val,test")
    ap.add_argument("--aliases", default="",
                    help="json {вариант_имени: каноничное_имя} (уже в "
                         "нормализованном виде: нижний регистр, без точек)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fracs = dict(zip(SPLITS, map(float, args.fracs.split(","))))
    aliases = {}
    if args.aliases:
        with open(args.aliases, encoding="utf-8") as f:
            aliases = {norm(k): norm(v) for k, v in json.load(f).items()}

    frags = pd.read_json(args.fragments, lines=True)
    sel = pd.read_csv(args.selected, keep_default_na=False)
    sel["author"] = sel["author"].astype(str)
    df = frags.merge(
        sel[["page_id", "work", "author", "year", "bin", "genre"]],
        on="page_id", how="inner")
    print(f"Фрагментов после объединения: {len(df)}")

    df = build_groups(df, aliases)
    gs = df.groupby("group").size().sort_values(ascending=False)
    print(f"Групп (авторов/самостоятельных произведений): {len(gs)}")
    print("Крупнейшие группы (доля фрагментов):")
    print((gs.head(5) / len(df)).round(3).to_string())
    if gs.iloc[0] / len(df) > 0.1:
        print("ВНИМАНИЕ: одна группа больше 10% данных. Возможно, "
              "склеились разные авторы через одинаковое название "
              "произведения. Проверьте.")

    df = df.sample(frac=1, random_state=args.seed)
    df = df.groupby("work_norm", sort=False).head(args.cap_per_work)
    df = df.groupby("group", sort=False).head(args.cap_per_group)
    print(f"После лимитов ({args.cap_per_work} на произведение, "
          f"{args.cap_per_group} на группу): {len(df)}")

    sizes = df.groupby(["genre", "bin"]).size()
    print("\nРазмеры ячеек (жанр x корзина):")
    print(sizes.unstack(fill_value=0))
    small = sizes[sizes < args.min_cell]
    if len(small):
        print(f"\nОтброшены ячейки меньше {args.min_cell}: "
              f"{[tuple(i) for i in small.index]}")
    kept = sizes[sizes >= args.min_cell]
    if kept.empty:
        raise SystemExit("Все ячейки меньше --min-cell. Уменьшите порог.")
    per_cell = args.per_cell or int(kept.min())
    print(f"\nФрагментов в каждой ячейке (не больше): {per_cell}")

    df = df[pd.MultiIndex.from_frame(df[["genre", "bin"]]).isin(kept.index)]
    df = df.groupby(["genre", "bin"], sort=False).head(per_cell).copy()

    df["split"] = assign_splits(df, fracs, args.seed)

    # --- проверка: никаких пересечений -------------------------------------
    for col in ["group", "work_norm"]:
        bad = df.groupby(col).split.nunique()
        assert (bad == 1).all(), f"{col} встречается в нескольких выборках!"
    named = df[df.author_norm != ""]
    assert (named.groupby("author_norm").split.nunique() == 1).all(), \
        "автор встречается в нескольких выборках!"
    print("\nПроверка пройдена: автор, произведение и группа целиком "
          "лежат в одной выборке.")

    cols = ["frag_id", "text", "year", "bin", "genre", "work", "author",
            "group", "old_orth"]
    os.makedirs(args.out_dir, exist_ok=True)
    for s in SPLITS:
        part = df[df.split == s]
        path = os.path.join(args.out_dir, f"{s}.jsonl")
        part[cols].to_json(path, orient="records", lines=True,
                           force_ascii=False)
        print(f"{s}: {len(part)} фрагментов, {part.group.nunique()} групп "
              f"-> {path}")

    pd.set_option("display.width", 200)
    print("\nФрагментов по ячейкам и выборкам:")
    print(df.pivot_table(index=["genre", "bin"], columns="split",
                         values="frag_id", aggfunc="count",
                         fill_value=0)[SPLITS])
    print("\nДоля старой орфографии по корзинам и выборкам "
          "(до нормализации):")
    print(df.pivot_table(index="bin", columns="split", values="old_orth",
                         aggfunc="mean").round(3)[SPLITS])


if __name__ == "__main__":
    main()
