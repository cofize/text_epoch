#!/usr/bin/env python3
"""
Шаг 3. Собираем итоговый датасет: ограничиваем вклад одного произведения,
выравниваем ячейки "жанр x период" и делим на train / val / test.

Вход:  fragments.jsonl (шаг 2) и selected.csv (шаг 1)
Выход: dataset.jsonl с полями
       frag_id, text, year, bin, genre, work, author, group, split, old_orth

Логика:
  1. не больше --cap-per-work фрагментов от одного произведения
     (иначе книга из 300 глав перетянет на себя весь период);
  2. ячейки "жанр x корзина" меньше --min-cell отбрасываются;
  3. каждая оставшаяся ячейка урезается до одного размера
     (--per-cell или, по умолчанию, размер самой маленькой оставшейся);
  4. split считается по хэшу ГРУППЫ (автор, а если его нет, то произведение),
     поэтому один автор целиком попадает в одну часть. Это строгий тест:
     модель не видела ни одной книги этого автора. Если нужен более мягкий
     вариант (по произведениям), запустите с --split-by work.

Запуск:
  python 03_build_dataset.py --fragments fragments.jsonl \
      --selected selected.csv --out dataset.jsonl
"""
import argparse
import hashlib

import pandas as pd


def bucket(key, seed):
    h = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h, 16) % 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", default="fragments.jsonl")
    ap.add_argument("--selected", default="selected.csv")
    ap.add_argument("--out", default="dataset.jsonl")
    ap.add_argument("--cap-per-work", type=int, default=40)
    ap.add_argument("--min-cell", type=int, default=1000,
                    help="ячейки меньше этого размера отбрасываются")
    ap.add_argument("--per-cell", type=int, default=0,
                    help="сколько фрагментов оставить в каждой ячейке "
                         "(0 = размер самой маленькой оставшейся)")
    ap.add_argument("--split-by", choices=["group", "work"], default="group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frags = pd.read_json(args.fragments, lines=True)
    sel = pd.read_csv(args.selected, keep_default_na=False)
    sel["author"] = sel["author"].astype(str)
    df = frags.merge(
        sel[["page_id", "work", "author", "year", "bin", "genre", "group"]],
        on="page_id", how="inner")
    print(f"Фрагментов после объединения: {len(df)}")

    df = df.sample(frac=1, random_state=args.seed)       # перемешали
    df = df.groupby("work", sort=False).head(args.cap_per_work)
    print(f"После ограничения {args.cap_per_work} на произведение: {len(df)}")

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
    if per_cell > int(kept.min()):
        print(f"\nВнимание: --per-cell={per_cell} больше самой маленькой "
              f"ячейки ({int(kept.min())}); ячейки будут разного размера.")
    print(f"\nФрагментов в каждой ячейке (не больше): {per_cell}")

    keep_idx = kept.index
    df = df[pd.MultiIndex.from_frame(df[["genre", "bin"]]).isin(keep_idx)]
    df = df.groupby(["genre", "bin"], sort=False).head(per_cell)

    key = "group" if args.split_by == "group" else "work"
    b = df[key].astype(str).map(lambda k: bucket(k, args.seed))
    df["split"] = pd.cut(b, [-1, 79, 89, 99],
                         labels=["train", "val", "test"]).astype(str)

    cols = ["frag_id", "text", "year", "bin", "genre", "work", "author",
            "group", "split", "old_orth"]
    df[cols].to_json(args.out, orient="records", lines=True,
                     force_ascii=False)

    print(f"\nГотово: {len(df)} фрагментов -> {args.out}")
    print("\nCколько фрагментов в train / val / test по ячейкам:")
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["genre", "bin"], columns="split",
                         values="frag_id", aggfunc="count", fill_value=0))
    print("\nДоля фрагментов со старой орфографией по корзинам "
          "(до нормализации):")
    print(df.groupby("bin").old_orth.mean().round(3))


if __name__ == "__main__":
    main()
