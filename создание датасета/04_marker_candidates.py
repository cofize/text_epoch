#!/usr/bin/env python3
"""
Шаг 4. Словари маркеров эпохи: ОТДЕЛЬНЫЙ СЛОВАРЬ ДЛЯ КАЖДОГО ЖАНРА.

Вход:  data/train.jsonl из шага 3 (поля text, year, genre, work, group)
       data/val.jsonl   (необязательно, нужен только для проверки словаря)
Выход: в папку --out-dir на каждый жанр три файла:
         markers_<жанр>_<unit>.csv  все кандидаты со статистикой (для разбора руками)
         dict_<жанр>_<unit>.json    чистый словарь: слово -> эпоха (для сайта и нейросети)
         dropped_<жанр>_<unit>.csv  что отсеяно как "мета" и почему (для проверки)

Все расчёты идут строго внутри одного жанра: слово сравнивается с другими
эпохами ТОГО ЖЕ жанра. Словарь строится только по train. Val используется
как "второе мнение": встречается ли слово в нужной эпохе у авторов, которых
скрипт не видел (см. колонки val_*). Test этот скрипт не трогает вообще.

Что отсеивается как "не язык эпохи":
  - meta       редакторский аппарат: стр, рукописи, автограф, варианты, перевод...
  - meta_soft  разговор о литературе: стихи, поэт, рассказ, послание, басни...
               (выключается флагом --keep-soft-meta)
  - orth       остатки дореформенной орфографии: котораго, ея, разсказ, ужь
               (выключается флагом --orth keep; об этом см. пояснение в чате)
  - abbr       сокращения с точкой перед цифрой или строчной: "стр. 5", "ст. 12"
  - name       слова, которые почти всегда пишутся с заглавной в середине
               предложения (имена, топонимы)
Дополнительно: числа с суффиксом ("1845-го"), слова ЦЕЛИКОМ ЗАГЛАВНЫМИ
(заголовки) и вики-заголовки "=== ... ===" в счёт не идут.

Запуск:
  python 04_marker_candidates.py --train data/train.jsonl --val data/val.jsonl
  python 04_marker_candidates.py --train data/train.jsonl --unit lemma
  python 04_marker_candidates.py --train data/train.jsonl --genres поэзия

Свои стоп-слова:  --stoplist my_stop.txt  (одно слово в строке, "рукопис*" = по префиксу)
Вернуть слово:    --allow my_allow.txt     (то же самое, перекрывает все фильтры)

Зависимости: pandas, numpy (для --unit lemma ещё pymorphy3).
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
RE_WIKI_HEAD = re.compile(r"={2,}[^=\n]{0,200}={2,}")
DEFAULT_ERAS = "1800-1849,1850-1899,1900-1924,1925-1944"
BOUNDARY = set(".!?…:;—–-\"«“„([\n=*")


# --- словари "не языка эпохи" ---------------------------------------------
def _rx(patterns):
    return re.compile(r"^(?:" + "|".join(patterns) + r")$")


# Редакторский и библиографический аппарат: в тексте самого произведения
# таких слов нет, они попали из примечаний и шапок страниц Викитеки.
HARD_PATTERNS = [
    r"стр|ст|кн|ред|др|пр|гр|го|проф|коп|руб|изд|ср|см|прим|пер|тт|гл|сб|вып"
    r"|рис|табл|акад|ок",
    r"г-(?:н|ж)\w*",
    r"рукопис\w*", r"автограф\w*", r"вариант\w*", r"тетрад\w*",
    r"библиотек\w*", r"архив\w*", r"примечан\w*", r"комментар\w*",
    r"издан\w*", r"издател\w*", r"издал|издала", r"опубликов\w*",
    r"напечатан\w*", r"печатает\w*", r"типограф\w*", r"редакц\w*",
    r"редактор\w*", r"страниц\w*", r"перевод\w*", r"перевел\w*",
    r"переведен\w*", r"переводч\w*", r"оригинал\w*", r"подлинник\w*",
    r"заглави\w*", r"отдел(?:е|ом|а|ы|ов|ах)?", r"раздел(?:а|е|ом|ы|ов)?",
    r"дат[аы]|датир\w*", r"текст\w*", r"томах|томов", r"черновик\w*",
    r"списк\w*|список",
]
# Разговор о литературе и подзаголовки: это слова языка, но они описывают
# сам текст, а не эпоху. Выключаются флагом --keep-soft-meta.
SOFT_PATTERNS = [
    r"стих(?:и|ов|ам|ами|ах|а|у|ом|е)?", r"стихотвор\w*", r"поэт\w*",
    r"поэм\w*", r"поэз\w*", r"роман\w*", r"рассказ\w*", r"повест\w*",
    r"басн\w*|басня", r"эпиграмм\w*", r"послани\w*",
    r"од|ода|оды|оде", r"сонет\w*", r"цикл\w*", r"пародия\w*|пародии",
    r"подражани\w*", r"писател\w*", r"автор\w*", r"критик\w*",
    r"литератур\w*", r"журнал\w*", r"газет\w*", r"вестник\w*",
    r"ведомост\w*", r"заметк\w*|заметки", r"очерк\w*", r"сочинени\w*",
    r"печать", r"имеется|имеются", r"относится|относятся", r"следующ\w*",
    r"написан[аоы]?", r"впервые",
]
# Остатки дореформенной орфографии, которые шаг 2 не нормализует.
ORTH_PATTERNS = [
    r"ея|нея|оне|ето|ужь|жь|естьли|еслиб[ы]?|однакож[ь]?|\w+-жь|тожь|чтожь",
    r"\w+(?:аго|яго)", r"\w*ыя", r"\w+кия",
    r"(?:без|раз|роз)[кпстфхцчшщ]\w*",
]


def load_words(path):
    pats = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            w = line.strip().lower().replace("ё", "е")
            if not w or w.startswith("#"):
                continue
            pats.append(re.escape(w[:-1]) + r"\w*" if w.endswith("*")
                        else re.escape(w))
    return pats


class Lexicon:
    def __init__(self, extra, allow, keep_soft, orth):
        self.hard = _rx(HARD_PATTERNS + extra)
        self.soft = None if keep_soft else _rx(SOFT_PATTERNS)
        self.orth = _rx(ORTH_PATTERNS) if orth == "drop" else None
        self.allow = _rx(allow) if allow else None

    def reason(self, tok):
        if self.allow and self.allow.match(tok):
            return ""
        if self.hard.match(tok):
            return "meta"
        if self.soft and self.soft.match(tok):
            return "meta_soft"
        if self.orth and self.orth.match(tok):
            return "orth"
        return ""


# --- токенизация -----------------------------------------------------------
def parse_eras(spec):
    return [tuple(int(x) for x in p.strip().split("-")) for p in spec.split(",")]


def era_index(year, eras):
    for i, (a, b) in enumerate(eras):
        if a <= year <= b:
            return i
    return -1


def make_lemmatizer():
    try:
        import pymorphy3 as pm
    except ImportError:
        try:
            import pymorphy2 as pm
        except ImportError:
            raise SystemExit("Для --unit lemma нужен pymorphy3: pip install pymorphy3")
    morph = pm.MorphAnalyzer()

    @lru_cache(maxsize=None)
    def lemma(w):
        return morph.parse(w)[0].normal_form
    return lemma


def is_sentence_initial(text, s):
    j = s - 1
    while j >= 0 and text[j] in " \t\xa0":
        j -= 1
    return j < 0 or text[j] in BOUNDARY


def is_abbreviation(text, e):
    """За словом точка, а дальше цифра или строчная буква: 'стр. 5', 'ст. 12'."""
    if text[e:e + 1] != ".":
        return False
    k = e + 1
    while k < len(text) and text[k] in " \t\xa0":
        k += 1
    return k < len(text) and (text[k].isdigit() or text[k].islower())


def tokens_of(text, unit, lemma):
    """Список (токен, с_заглавной_внутри_предложения, внутри_предложения, сокращение)."""
    text = RE_WIKI_HEAD.sub(" ", text)
    out = []
    for m in WORD.finditer(text):
        raw = m.group(0)
        s, e = m.span()
        if len(raw) < 2 or raw.isupper():            # заголовки КАПСОМ
            continue
        if s >= 2 and text[s - 1] in "-–" and text[s - 2].isdigit():
            continue                                  # "1845-го", "3-му"
        low = raw.lower().replace("ё", "е")
        init = is_sentence_initial(text, s)
        out.append((lemma(low) if unit == "lemma" else low,
                    raw[0].isupper() and not init, not init,
                    is_abbreviation(text, e)))
    return out


# --- статистика -------------------------------------------------------------
def g2(a, b, n1, n2):
    """Log-likelihood G2: слово в эпохе (a из n1) против остальных эпох (b из n2)."""
    a = a.astype(float)
    b = b.astype(float)
    tot = a + b
    e1 = n1 * tot / (n1 + n2)
    e2 = n2 * tot / (n1 + n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(a > 0, a * np.log(a / e1), 0.0)
        t2 = np.where(b > 0, b * np.log(b / e2), 0.0)
    return 2 * (t1 + t2)


def spearman_trend(row):
    r = np.argsort(np.argsort(row)).astype(float)
    if r.std() == 0:
        return 0.0
    return float(np.corrcoef(np.arange(len(row)), r)[0, 1])


def count_by_era(toks_list, eras_of, idx, E):
    cnt = np.zeros((len(idx), E), dtype=np.int64)
    for toks, e in zip(toks_list, eras_of):
        c = Counter(idx[t[0]] for t in toks if t[0] in idx)
        if c:
            cnt[np.fromiter(c.keys(), np.int64, len(c)), e] += \
                np.fromiter(c.values(), np.int64, len(c))
    return cnt


def analyse_genre(gdf, vdf, genre, eras, args, lex, lemma):
    E = len(eras)
    toks_cache = [tokens_of(t, args.unit, lemma) for t in gdf["text"]]

    # проход 1: частоты, заглавные, сокращения, размеры эпох
    total, noninit, caps, abbr = Counter(), Counter(), Counter(), Counter()
    N_era = np.zeros(E, dtype=np.int64)
    for toks, e in zip(toks_cache, gdf["era"]):
        N_era[e] += len(toks)
        for tok, cap, ni, ab in toks:
            total[tok] += 1
            if ni:
                noninit[tok] += 1
                if cap:
                    caps[tok] += 1
            if ab:
                abbr[tok] += 1
    vocab = [t for t, c in total.items() if c >= args.min_total]
    idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    print(f"  [{genre}] фрагментов: {len(gdf)}, токенов: {int(N_era.sum())}, "
          f"слов (>= {args.min_total}): {V}")
    if V == 0:
        return [], []

    name_ratio = np.array([caps[t] / noninit[t] if noninit[t] else 0.0
                           for t in vocab])
    abbr_ratio = np.array([abbr[t] / total[t] for t in vocab])
    reason = []
    for i, t in enumerate(vocab):
        r = lex.reason(t)
        if not r and abbr_ratio[i] >= args.max_abbr_ratio:
            r = "abbr"
        if not r and name_ratio[i] >= args.max_name_ratio:
            r = "name"
        reason.append(r)

    # проход 2: счётчики по (слово, эпоха), разброс по произведениям и авторам
    cnt_era = np.zeros((V, E), dtype=np.int64)
    work_cnt, grp_cnt = defaultdict(Counter), defaultdict(Counter)
    for toks, e, work, group in zip(toks_cache, gdf["era"], gdf["work"],
                                    gdf["group"]):
        c = Counter(idx[t[0]] for t in toks if t[0] in idx)
        if not c:
            continue
        ids = np.fromiter(c.keys(), np.int64, len(c))
        cnt_era[ids, e] += np.fromiter(c.values(), np.int64, len(c))
        work_cnt[(e, work)].update(c)
        grp_cnt[(e, str(group))].update(c)
    df_works = np.zeros((V, E), dtype=np.int32)
    df_groups = np.zeros((V, E), dtype=np.int32)
    max_grp = np.zeros((V, E), dtype=np.int64)
    for (e, _), c in work_cnt.items():
        df_works[np.fromiter(c.keys(), np.int64, len(c)), e] += 1
    for (e, _), c in grp_cnt.items():
        ids = np.fromiter(c.keys(), np.int64, len(c))
        vals = np.fromiter(c.values(), np.int64, len(c))
        df_groups[ids, e] += 1
        max_grp[ids, e] = np.maximum(max_grp[ids, e], vals)

    # val: те же слова, но у авторов, которых словарь не видел
    cnt_val, N_val = None, None
    if vdf is not None and len(vdf):
        vt = [tokens_of(t, args.unit, lemma) for t in vdf["text"]]
        N_val = np.zeros(E, dtype=np.int64)
        for toks, e in zip(vt, vdf["era"]):
            N_val[e] += len(toks)
        cnt_val = count_by_era(vt, vdf["era"], idx, E)

    N_all = int(N_era.sum())
    fpm = cnt_era / np.maximum(N_era, 1) * 1e6
    rows, dropped = [], []
    for e, (y0, y1) in enumerate(eras):
        a = cnt_era[:, e]
        b = cnt_era.sum(axis=1) - a
        n1, n2 = int(N_era[e]), N_all - int(N_era[e])
        if n1 == 0 or n2 == 0:
            print(f"  [{genre}] эпоха {y0}-{y1}: нет данных, пропуск")
            continue
        g2v = g2(a, b, n1, n2)
        logratio = np.log2(((a + 0.5) / n1) / ((b + 0.5) / n2))
        top_share = max_grp[:, e] / np.maximum(a, 1)
        stat_ok = ((a >= args.min_count) & (df_works[:, e] >= args.min_works)
                   & (df_groups[:, e] >= args.min_groups)
                   & (logratio >= args.min_logratio)
                   & (top_share <= args.max_top_group_share))

        val_lr = val_cnt = None
        if cnt_val is not None and N_val[e] > 0 and N_val.sum() - N_val[e] > 0:
            va = cnt_val[:, e]
            vb = cnt_val.sum(axis=1) - va
            val_lr = np.log2(((va + 0.5) / N_val[e])
                             / ((vb + 0.5) / (N_val.sum() - N_val[e])))
            val_cnt = va
            if args.require_replication:
                stat_ok &= (va >= 1) & (val_lr > 0)

        lex_ok = np.array([r == "" for r in reason])
        order = np.argsort(-g2v)
        keep = [i for i in order if stat_ok[i] and lex_ok[i]][:args.top]
        drop = [i for i in order if stat_ok[i] and not lex_ok[i]][:args.dropped_top]

        for rank, i in enumerate(keep, 1):
            peak = int(np.argmax(fpm[i]))
            active = [f"{eras[k][0]}-{eras[k][1]}" for k in range(E)
                      if fpm[i, k] >= 0.5 * fpm[i, peak]]
            rows.append({
                "genre": genre, "unit": args.unit, "era": f"{y0}-{y1}",
                "rank": rank, "token": vocab[i],
                "count_era": int(a[i]),
                "fpm_era": round(float(fpm[i, e]), 1),
                "fpm_rest": round(float(b[i] / n2 * 1e6), 1),
                "log_ratio": round(float(logratio[i]), 2),
                "G2": round(float(g2v[i]), 1),
                "era_share": round(float(fpm[i, e] / fpm[i].sum()), 2),
                "works_era": int(df_works[i, e]),
                "groups_era": int(df_groups[i, e]),
                "top_group_share": round(float(top_share[i]), 2),
                "peak_era": f"{eras[peak][0]}-{eras[peak][1]}",
                "active_eras": "|".join(active),
                "trend": round(spearman_trend(fpm[i]), 2) if E > 2 else 0.0,
                "fpm_by_era": json.dumps([round(float(x), 1) for x in fpm[i]]),
                "name_ratio": round(float(name_ratio[i]), 2),
                "abbr_ratio": round(float(abbr_ratio[i]), 2),
                "val_count": int(val_cnt[i]) if val_cnt is not None else "",
                "val_log_ratio": round(float(val_lr[i]), 2)
                if val_lr is not None else "",
                "replicates": (bool(val_cnt[i] >= 1 and val_lr[i] > 0)
                               if val_lr is not None else ""),
            })
        for i in drop:
            dropped.append({
                "genre": genre, "era": f"{y0}-{y1}", "token": vocab[i],
                "reason": reason[i], "count_era": int(a[i]),
                "log_ratio": round(float(logratio[i]), 2),
                "G2": round(float(g2v[i]), 1),
                "works_era": int(df_works[i, e]),
                "groups_era": int(df_groups[i, e]),
                "name_ratio": round(float(name_ratio[i]), 2),
                "abbr_ratio": round(float(abbr_ratio[i]), 2)})
    return rows, dropped


def build_final_dict(res, genre, eras, args, norm_note):
    """Одно слово -> одна эпоха (где оно сильнее всего по G2), топ-N на эпоху."""
    best = res.sort_values("G2", ascending=False).drop_duplicates("token")
    parts = [p.head(args.final_top) for _, p in best.groupby("era")]
    final = pd.concat(parts).sort_values(["era", "G2"], ascending=[True, False])
    markers = {r.token: {"era": r.era, "log_ratio": r.log_ratio, "G2": r.G2,
                         "era_share": r.era_share}
               for r in final.itertuples()}
    return {"genre": genre, "unit": args.unit, "normalization": norm_note,
            "eras": [f"{a}-{b}" for a, b in eras], "markers": markers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="",
                    help="data/val.jsonl: только для колонок val_* (необязательно)")
    ap.add_argument("--out-dir", default="markers")
    ap.add_argument("--unit", choices=["form", "lemma"], default="form")
    ap.add_argument("--eras", default=DEFAULT_ERAS)
    ap.add_argument("--genres", default="",
                    help="жанры через запятую; по умолчанию все из train")
    ap.add_argument("--min-total", type=int, default=20,
                    help="слова реже этого числа в жанре не рассматриваем")
    ap.add_argument("--min-count", type=int, default=20,
                    help="минимум употреблений в эпохе")
    ap.add_argument("--min-works", type=int, default=15,
                    help="минимум разных произведений в эпохе")
    ap.add_argument("--min-groups", type=int, default=8,
                    help="минимум разных авторов (групп) в эпохе")
    ap.add_argument("--min-logratio", type=float, default=1.0,
                    help="log2 отношения частот; 1 = в 2 раза чаще, чем в остальных эпохах")
    ap.add_argument("--max-top-group-share", type=float, default=0.4,
                    help="доля употреблений в эпохе от ОДНОГО автора не больше этого")
    ap.add_argument("--max-name-ratio", type=float, default=0.6,
                    help="доля написаний с заглавной внутри предложения; выше = имя")
    ap.add_argument("--max-abbr-ratio", type=float, default=0.3,
                    help="доля употреблений вида 'слово. 5' / 'слово. и'; выше = сокращение")
    ap.add_argument("--keep-soft-meta", action="store_true",
                    help="не отсеивать слова про литературу (стихи, поэт, рассказ...)")
    ap.add_argument("--orth", choices=["drop", "keep"], default="drop",
                    help="drop: убрать остатки дореформенной орфографии (котораго, ея)")
    ap.add_argument("--stoplist", default="", help="файл с доп. стоп-словами")
    ap.add_argument("--allow", default="", help="файл со словами, которые нельзя отсеивать")
    ap.add_argument("--require-replication", action="store_true",
                    help="оставлять только слова, которые подтверждаются на val")
    ap.add_argument("--top", type=int, default=2000,
                    help="сколько кандидатов оставить на (жанр, эпоху)")
    ap.add_argument("--final-top", type=int, default=300,
                    help="сколько слов на эпоху попадёт в итоговый dict_*.json")
    ap.add_argument("--dropped-top", type=int, default=200)
    args = ap.parse_args()

    eras = parse_eras(args.eras)
    lemma = make_lemmatizer() if args.unit == "lemma" else None
    lex = Lexicon(load_words(args.stoplist) if args.stoplist else [],
                  load_words(args.allow) if args.allow else [],
                  args.keep_soft_meta, args.orth)

    def load(path):
        d = pd.read_json(path, lines=True)
        d = d.assign(era=d["year"].map(lambda y: era_index(int(y), eras)))
        return d[d["era"] >= 0].reset_index(drop=True)

    train = load(args.train)
    val = load(args.val) if args.val else None
    if train.empty:
        raise SystemExit("В train нет фрагментов в заданных эпохах.")
    genres = ([g.strip() for g in args.genres.split(",") if g.strip()]
              or sorted(train["genre"].unique()))
    print(f"train: {len(train)} фрагментов; val: "
          f"{len(val) if val is not None else 'не задан'}; "
          f"жанры: {genres}; эпохи: {eras}")
    os.makedirs(args.out_dir, exist_ok=True)
    norm_note = ("нижний регистр, ё->е" + ("; лемма pymorphy" if args.unit == "lemma" else ""))

    for genre in genres:
        gdf = train[train["genre"] == genre].reset_index(drop=True)
        if gdf.empty:
            print(f"  [{genre}] данных нет, пропуск")
            continue
        vdf = (val[val["genre"] == genre].reset_index(drop=True)
               if val is not None else None)
        rows, dropped = analyse_genre(gdf, vdf, genre, eras, args, lex, lemma)
        if not rows:
            print(f"  [{genre}] кандидатов нет: ослабьте --min-count/--min-works/--min-groups")
            continue
        res = pd.DataFrame(rows)
        tag = f"{genre}_{args.unit}"
        res.to_csv(os.path.join(args.out_dir, f"markers_{tag}.csv"),
                   index=False, encoding="utf-8-sig")
        pd.DataFrame(dropped).to_csv(
            os.path.join(args.out_dir, f"dropped_{tag}.csv"),
            index=False, encoding="utf-8-sig")
        final = build_final_dict(res, genre, eras, args, norm_note)
        with open(os.path.join(args.out_dir, f"dict_{tag}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=1)

        print(f"\n=== {genre}: кандидатов {len(res)}, в словаре {len(final['markers'])} ===")
        if dropped:
            print("  отсеяно как не-язык эпохи (среди сильных):",
                  {k: int(v) for k, v in
                   pd.DataFrame(dropped).reason.value_counts().items()})
        for era, part in res.groupby("era"):
            print(f"  {era}: " + ", ".join(part.head(12)["token"]))
        if "replicates" in res and res["replicates"].astype(str).isin(["True", "False"]).any():
            rep = res[res["replicates"].astype(str).isin(["True", "False"])]
            print(f"  подтверждаются на val: {(rep.replicates == True).mean():.0%} кандидатов")  # noqa: E712
    print(f"\nГотово. Файлы в папке: {args.out_dir}")


if __name__ == "__main__":
    main()
