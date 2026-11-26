"""
Библиотека: данные, три модели (tfidf / rubert / xlmr) и двухступенчатое предсказание
«жанр -> период». Напрямую не запускается: используйте train.py, make_report.py, predict.py.

Формат записи dataset.json (список объектов или JSONL):
  {"frag_id", "text", "year", "genre", "work", "author", "group", "split", "old_orth", ("bin")}
"""
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

SPLITS = ("train", "val", "test")
_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
            "validation": "val", "dev": "val", "test": "test"}


# =====================================================================
# Данные
# =====================================================================
def load_records(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


_FILE_STEMS = {"train": ("train",), "val": ("val", "valid", "validation", "dev"), "test": ("test",)}


def load_dataset(path):
    """
    Один файл (поле split внутри записей) ИЛИ папка с train.jsonl / val.jsonl / test.jsonl
    (тогда split берётся из имени файла; папка может называться как угодно, хоть «dataset.jsonl»).
    """
    if not os.path.isdir(path):
        return load_records(path)
    names = {f.lower(): f for f in os.listdir(path)}
    out = []
    for split, stems in _FILE_STEMS.items():
        fname = next((names[s + ext] for s in stems for ext in (".jsonl", ".json") if s + ext in names), None)
        if not fname:
            continue
        for r in load_records(os.path.join(path, fname)):
            r["split"] = split
            out.append(r)
    if not out:
        raise FileNotFoundError(f"в папке {path} нет train.jsonl / val.jsonl / test.jsonl")
    return out


def era_label(year, bin_years):
    start = (int(year) // bin_years) * bin_years
    return f"{start}-{start + bin_years - 1}"


def era_start(label):
    return int(str(label).split("-")[0])


def _unit(key, seed):
    return int(hashlib.md5(f"{seed}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def prepare(records, bin_years=25, seed=42, log=print):
    """Нормализует записи и возвращает {'train': [...], 'val': [...], 'test': [...]}."""
    recs, skipped = [], 0
    for i, r in enumerate(records):
        text, year, genre = r.get("text"), r.get("year"), r.get("genre")
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None
        if not text or year is None or not genre:
            skipped += 1
            continue
        fid = r.get("frag_id", i)
        work = str(r.get("work") or f"frag{fid}")
        group = str(r.get("group") or r.get("author") or work)
        recs.append({"id": fid, "text": text, "genre": str(genre), "year": year,
                     "era": era_label(year, bin_years), "bin": r.get("bin"), "work": work, "group": group,
                     "split": _ALIASES.get(str(r.get("split", "")).strip().lower()),
                     "old_orth": bool(r.get("old_orth", False)), "n_words": len(text.split())})
    if skipped:
        log(f"пропущено записей без text/year/genre: {skipped}")

    if not any(r["split"] for r in recs):
        log("поле split не найдено: делю по группам 80/10/10")
        for r in recs:
            u = _unit(r["group"], seed)
            r["split"] = "test" if u < 0.1 else "val" if u < 0.2 else "train"
    else:
        dropped = [r for r in recs if not r["split"]]
        if dropped:
            log(f"записей с неизвестным split отброшено: {len(dropped)}")
        recs = [r for r in recs if r["split"]]
        if not any(r["split"] == "val" for r in recs):
            log("в данных нет val: беру 10% групп из train")
            for r in recs:
                if r["split"] == "train" and _unit(r["group"], seed + 1) < 0.1:
                    r["split"] = "val"
    by = {s: [] for s in SPLITS}
    for r in recs:
        by[r["split"]].append(r)
    return by


def _norm_hash(text):
    return hashlib.md5(" ".join(text.lower().split()).encode()).hexdigest()


def data_summary(by, bin_years):
    """Статистика и предупреждения о проблемах в данных (утечки, редкие классы, орфография)."""
    tr = by["train"]
    warns = []
    s = {"bin_years": bin_years, "counts": {k: len(v) for k, v in by.items()},
         "works": {k: len({r["work"] for r in v}) for k, v in by.items()},
         "groups": {k: len({r["group"] for r in v}) for k, v in by.items()}}
    s["genres"] = {k: dict(Counter(r["genre"] for r in v)) for k, v in by.items()}
    s["eras"] = {k: dict(sorted(Counter(r["era"] for r in v).items())) for k, v in by.items()}
    if tr:
        s["majority"] = {"genre": Counter(r["genre"] for r in tr).most_common(1)[0][0],
                         "era": Counter(r["era"] for r in tr).most_common(1)[0][0],
                         "pair": list(Counter((r["genre"], r["era"]) for r in tr).most_common(1)[0][0])}
        s["frag_words_median"] = int(np.median([r["n_words"] for r in tr]))

    for other in ("val", "test"):
        if not by[other]:
            continue
        g_tr, g_ot = {r["group"] for r in tr}, {r["group"] for r in by[other]}
        w_tr, w_ot = {r["work"] for r in tr}, {r["work"] for r in by[other]}
        h_tr, h_ot = {_norm_hash(r["text"]) for r in tr}, {_norm_hash(r["text"]) for r in by[other]}
        if g_tr & g_ot:
            warns.append(f"{len(g_tr & g_ot)} групп (авторов) есть и в train, и в {other}: метрики могут быть завышены.")
        if w_tr & w_ot:
            warns.append(f"{len(w_tr & w_ot)} произведений есть и в train, и в {other}: утечка.")
        if h_tr & h_ot:
            warns.append(f"{len(h_tr & h_ot)} фрагментов {other} дословно совпадают с train: дубликаты.")
        miss_e = {r["era"] for r in by[other]} - {r["era"] for r in tr}
        miss_g = {r["genre"] for r in by[other]} - {r["genre"] for r in tr}
        if miss_e or miss_g:
            warns.append(f"В {other} есть классы без примеров в train: {sorted(miss_e | miss_g)}.")

    rare = [k for k, v in Counter(r["era"] for r in tr).items() if v < 50]
    rare += [k for k, v in Counter(r["genre"] for r in tr).items() if v < 50]
    if rare:
        warns.append(f"Меньше 50 фрагментов в train у классов: {sorted(set(rare))}.")

    allr = [r for v in by.values() for r in v]
    bad_bin, n_bin = 0, 0
    for r in allr:
        try:
            n_bin += 1
            bad_bin += int(r["bin"]) != era_start(r["era"])
        except (TypeError, ValueError):
            n_bin -= 1
    if n_bin and bad_bin:
        warns.append(f"У {bad_bin} из {n_bin} записей поле bin не совпадает с периодом, посчитанным по year "
                     f"(ширина периода {bin_years} лет): проверьте --bin-years и поле bin.")
    by_era = defaultdict(list)
    for r in allr:
        by_era[r["era"]].append(r["old_orth"])
    shares = {e: float(np.mean(v)) for e, v in sorted(by_era.items())}
    s["old_orth_share_by_era"] = shares
    big = [v for e, v in shares.items() if len(by_era[e]) >= 30]
    if len(big) >= 2 and max(big) - min(big) > 0.25:
        warns.append("Доля фрагментов со старой орфографией (old_orth) сильно различается по периодам: "
                     "модель может определять период по орфографии, а не по стилю.")
    s["warnings"] = warns
    return s


def cap_train(train, max_per_cell=0, max_total=0, seed=42):
    """Ограничивает обучающую выборку (для очень больших корпусов)."""
    rnd = random.Random(seed)
    out = train
    if max_per_cell:
        cells = defaultdict(list)
        for r in out:
            cells[(r["genre"], r["era"])].append(r)
        out = []
        for v in cells.values():
            out += rnd.sample(v, max_per_cell) if len(v) > max_per_cell else v
    if max_total and len(out) > max_total:
        out = rnd.sample(out, max_total)
    return out


# =====================================================================
# Модели
# =====================================================================
class BaseModel:
    kind = "base"

    def genre_proba(self, texts):          # -> (n, число жанров)
        raise NotImplementedError

    def era_proba(self, texts, genres):    # -> (n, число периодов), период зависит от жанра
        raise NotImplementedError


class TfidfModel(BaseModel):
    """TF-IDF (слова + символьные n-граммы, хэшированные) + SGD (логистическая функция потерь)."""
    kind = "tfidf"

    def __init__(self, name="tfidf"):
        self.name = name

    def _new_clf(self):
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="log_loss", alpha=1e-6, class_weight="balanced",
                             max_iter=20, tol=1e-3, n_jobs=-1, random_state=0)

    def _raw(self, texts):
        from scipy.sparse import hstack
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, train, val=None, log=print, ckpt_dir=None):
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        self.word = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 21, alternate_sign=False,
                                      norm=None, lowercase=True)
        self.char = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 20,
                                      alternate_sign=False, norm=None, lowercase=True)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.genres = sorted({r["genre"] for r in train})
        self.eras = sorted({r["era"] for r in train})
        self.e2i = {e: i for i, e in enumerate(self.eras)}
        texts = [r["text"] for r in train]
        G = np.array([r["genre"] for r in train])
        E = np.array([r["era"] for r in train])
        log("tfidf: считаю признаки...")
        X = self.tfidf.fit_transform(self._raw(texts))
        log("tfidf: обучаю классификатор жанра...")
        self.genre_clf = self._new_clf().fit(X, G)
        self.global_era = self._new_clf().fit(X, E) if len(set(E)) > 1 else E[0]
        self.era_clf = {}
        for g in self.genres:
            m = G == g
            eras_g = set(E[m])
            if len(eras_g) == 1:
                self.era_clf[g] = next(iter(eras_g))          # единственный период
            elif m.sum() >= 20:
                log(f"tfidf: классификатор периода для жанра «{g}» ({int(m.sum())} фрагм.)")
                self.era_clf[g] = self._new_clf().fit(X[m], E[m])
        return self

    def genre_proba(self, texts):
        return self.genre_clf.predict_proba(self.tfidf.transform(self._raw(texts)))

    def era_proba(self, texts, genres):
        X = self.tfidf.transform(self._raw(texts))
        P = np.zeros((len(texts), len(self.eras)))
        genres = np.array(genres)
        for g in set(genres):
            idx = np.where(genres == g)[0]
            clf = self.era_clf.get(g, self.global_era)
            if isinstance(clf, str):
                P[idx, self.e2i[clf]] = 1.0
                continue
            pr = clf.predict_proba(X[idx])
            for j, c in enumerate(clf.classes_):
                P[np.ix_(idx, [self.e2i[c]])] = pr[:, [j]]
        return P

    def save(self, d):
        import joblib
        os.makedirs(d, exist_ok=True)
        joblib.dump(self, os.path.join(d, "model.joblib"))


class TransformerModel(BaseModel):
    """Дообучение трансформера. Две сети: жанр по тексту; период по паре «жанр [SEP] текст»."""
    kind = "transformer"

    def __init__(self, name, hf_name, epochs=3, lr=2e-5, batch=8, accum=2, max_len=512, seed=42):
        self.name, self.hf_name = name, hf_name
        self.epochs, self.lr, self.batch, self.accum, self.max_len, self.seed = epochs, lr, batch, accum, max_len, seed

    # ---- служебное ----
    @staticmethod
    def _device():
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _encode(self, texts, genres=None):
        if genres is None:
            return self.tok(list(texts), truncation=True, max_length=self.max_len)
        return self.tok(list(genres), list(texts), truncation="only_second", max_length=self.max_len)

    def _batches(self, enc, order, bs):
        keys = list(enc.keys())
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            feats = [{k: enc[k][j] for k in keys} for j in b]
            yield b, self.tok.pad(feats, return_tensors="pt")

    def _logits(self, model, enc):
        import torch
        n = len(enc["input_ids"])
        out = np.zeros((n, model.config.num_labels), dtype=np.float32)
        order = sorted(range(n), key=lambda j: len(enc["input_ids"][j]))   # близкие длины -> меньше паддинга
        dev = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for b, batch in self._batches(enc, order, 32):
                batch = {k: v.to(dev) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
                    lg = model(**batch).logits
                out[b] = lg.float().cpu().numpy()
        return out

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    # ---- обучение одной ступени ----
    def _fit_stage(self, enc_tr, y_tr, enc_va, y_va, n_labels, tag, log, metric="f1"):
        import gc
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import f1_score
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

        dev = self._device()
        torch.manual_seed(self.seed)
        model = AutoModelForSequenceClassification.from_pretrained(self.hf_name, num_labels=n_labels).to(dev)
        y_tr = np.asarray(y_tr)
        counts = np.bincount(y_tr, minlength=n_labels)
        w = (len(y_tr) / (n_labels * np.maximum(counts, 1))) ** 0.5      # смягчённые веса классов
        weights = torch.tensor(w, dtype=torch.float, device=dev)

        n = len(y_tr)
        accum = max(1, self.accum)
        n_batches = (n + self.batch - 1) // self.batch
        steps = self.epochs * ((n_batches + accum - 1) // accum)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

        amp_dtype, scaler = None, None
        if dev == "cuda":
            # bf16 только на видеокартах Ampere и новее (T4 его не поддерживает аппаратно), иначе fp16
            if torch.cuda.get_device_capability(0)[0] >= 8:
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                try:
                    scaler = torch.amp.GradScaler("cuda")
                except (AttributeError, TypeError):
                    scaler = torch.cuda.amp.GradScaler()
        else:
            log("ВНИМАНИЕ: GPU не найден, обучение трансформера на CPU будет очень медленным.")
        log(f"{self.name}/{tag}: батч {self.batch} x накопление {accum}, точность {amp_dtype}, max_len {self.max_len}")

        rnd = random.Random(self.seed)
        best_f1, best_state = -1.0, None
        for ep in range(1, self.epochs + 1):
            model.train()
            order = list(range(n))
            rnd.shuffle(order)
            total, nb = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for bi, (b, batch) in enumerate(self._batches(enc_tr, order, self.batch)):
                batch = {k: v.to(dev) for k, v in batch.items()}
                y = torch.tensor(y_tr[b], dtype=torch.long, device=dev)
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(**batch).logits
                else:
                    logits = model(**batch).logits
                loss = F.cross_entropy(logits.float(), y, weight=weights)
                scaled = loss / accum
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if (bi + 1) % accum == 0 or (bi + 1) == n_batches:
                    if scaler is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                total += float(loss.detach())
                nb += 1
            msg = f"{self.name}/{tag}: эпоха {ep}/{self.epochs}, loss {total / max(nb, 1):.4f}"
            if enc_va is not None and len(y_va):
                pred = self._logits(model, enc_va).argmax(1)
                if metric == "qwk":
                    score, mname = qwk_idx(np.asarray(y_va), pred, n_labels), "QWK"
                else:
                    score, mname = f1_score(y_va, pred, average="macro"), "macro-F1"
                if score != score:                      # nan (например, в val один класс)
                    score = -0.99
                msg += f", val {mname} {score:.4f}"
                if score > best_f1:
                    best_f1 = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(msg)
        if best_state is not None:
            model.load_state_dict(best_state)
        # освобождаем память: состояние оптимизатора и градиенты больше не нужны
        del opt, sched, scaler, best_state
        model.zero_grad(set_to_none=True)
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
        return model

    def fit(self, train, val=None, log=print, ckpt_dir=None):
        """ckpt_dir: сюда сохраняется готовая genre-сеть; при повторном запуске с теми же настройками она не переобучается."""
        import gc
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.hf_name)
        self.genres = sorted({r["genre"] for r in train})
        self.eras = sorted({r["era"] for r in train})
        g2i = {g: i for i, g in enumerate(self.genres)}
        e2i = {e: i for i, e in enumerate(self.eras)}
        val = [r for r in (val or []) if r["genre"] in g2i and r["era"] in e2i]
        T = [r["text"] for r in train]
        Tv = [r["text"] for r in val]
        dev = self._device()

        sig = {"hf_name": self.hf_name, "epochs": self.epochs, "lr": self.lr, "max_len": self.max_len,
               "n_train": len(train), "genres": self.genres, "eras": self.eras}
        gdir = os.path.join(ckpt_dir, "genre") if ckpt_dir else None
        sigp = os.path.join(ckpt_dir, "genre_sig.json") if ckpt_dir else None
        loaded = False
        if ckpt_dir and os.path.exists(sigp) and os.path.isdir(gdir):
            try:
                with open(sigp, encoding="utf-8") as f:
                    loaded = json.load(f) == sig
            except (OSError, ValueError):
                loaded = False
            if loaded:
                log(f"{self.name}/genre: найдена готовая сеть жанра в {ckpt_dir}, обучение этой стадии пропущено")
                self.genre_model = AutoModelForSequenceClassification.from_pretrained(gdir)
        if not loaded:
            enc_v = self._encode(Tv) if val else None
            self.genre_model = self._fit_stage(self._encode(T), [g2i[r["genre"]] for r in train],
                                               enc_v, [g2i[r["genre"]] for r in val],
                                               len(self.genres), "genre", log)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
                self.genre_model.save_pretrained(gdir)
                with open(sigp, "w", encoding="utf-8") as f:
                    json.dump(sig, f, ensure_ascii=False)

        # на время обучения сети периода убираем сеть жанра с видеокарты
        self.genre_model.to("cpu")
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

        enc_v = self._encode(Tv, [r["genre"] for r in val]) if val else None
        self.era_model = self._fit_stage(self._encode(T, [r["genre"] for r in train]),
                                         [e2i[r["era"]] for r in train],
                                         enc_v, [e2i[r["era"]] for r in val],
                                         len(self.eras), "era", log, metric="qwk")
        self.genre_model.to(dev)
        return self

    def genre_proba(self, texts):
        return self._softmax(self._logits(self.genre_model, self._encode(texts)))

    def era_proba(self, texts, genres):
        return self._softmax(self._logits(self.era_model, self._encode(texts, genres)))

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        self.genre_model.save_pretrained(os.path.join(d, "genre"))
        self.era_model.save_pretrained(os.path.join(d, "era"))
        self.tok.save_pretrained(os.path.join(d, "tok"))
        with open(os.path.join(d, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "hf_name": self.hf_name, "max_len": self.max_len,
                       "genres": self.genres, "eras": self.eras}, f, ensure_ascii=False)


# Какие модели есть. Названия HF можно переопределить в train.py (--hf-name).
MODEL_ZOO = {
    "tfidf": {"kind": "tfidf"},
    "rubert": {"kind": "transformer", "hf_name": "DeepPavlov/rubert-base-cased"},
    "xlmr": {"kind": "transformer", "hf_name": "xlm-roberta-base"},
}


def build_model(name, hf_name=None, **kw):
    spec = MODEL_ZOO[name]
    if spec["kind"] == "tfidf":
        return TfidfModel(name)
    return TransformerModel(name, hf_name or spec["hf_name"], **kw)


def load_model(d):
    """Загружает сохранённую модель из папки запуска (runs/<имя>)."""
    mdir = os.path.join(d, "model")
    if os.path.exists(os.path.join(mdir, "booster.joblib")):
        import joblib
        info = joblib.load(os.path.join(mdir, "booster.joblib"))
        base = load_model(os.path.join(os.path.dirname(os.path.abspath(d)), info["base_run"]))
        return BoostedModel(base, info["feat"], info["clf"], info["name"], info["base_run"])
    if os.path.exists(os.path.join(mdir, "model.joblib")):
        import joblib
        return joblib.load(os.path.join(mdir, "model.joblib"))
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    with open(os.path.join(mdir, "model_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    m = TransformerModel(info["name"], info["hf_name"], max_len=info["max_len"])
    m.tok = AutoTokenizer.from_pretrained(os.path.join(mdir, "tok"))
    m.genres, m.eras = info["genres"], info["eras"]
    dev = m._device()
    m.genre_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(mdir, "genre")).to(dev)
    m.era_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(mdir, "era")).to(dev)
    return m


# =====================================================================
# Двухступенчатое предсказание
# =====================================================================
def _mode(xs):
    return Counter(xs).most_common(1)[0][0]


def predict_records(model, recs):
    """
    Для каждого фрагмента:
      ступень 1: жанр; ступень 2: период при предсказанном жанре (уровень фрагмента);
      а также то же по произведению (work): вероятности фрагментов усредняются в лог-пространстве,
      сначала жанр произведения, затем период при этом жанре.
    """
    texts = [r["text"] for r in recs]
    Pg = model.genre_proba(texts)
    g_frag = [model.genres[i] for i in Pg.argmax(1)]
    Pe = model.era_proba(texts, g_frag)

    works = defaultdict(list)
    for i, r in enumerate(recs):
        works[r["work"]].append(i)
    w_genre = {w: model.genres[int(np.log(Pg[ix] + 1e-9).mean(0).argmax())] for w, ix in works.items()}
    g_work = [w_genre[r["work"]] for r in recs]

    Pe_w = Pe.copy()
    diff = [i for i in range(len(recs)) if g_work[i] != g_frag[i]]
    if diff:                                    # пересчитываем период только там, где жанр произведения иной
        Pe_w[diff] = model.era_proba([texts[i] for i in diff], [g_work[i] for i in diff])
    w_era = {w: model.eras[int(np.log(Pe_w[ix] + 1e-9).mean(0).argmax())] for w, ix in works.items()}
    w_true_g = {w: _mode([recs[i]["genre"] for i in ix]) for w, ix in works.items()}
    w_true_e = {w: _mode([recs[i]["era"] for i in ix]) for w, ix in works.items()}

    rows = []
    for i, r in enumerate(recs):
        w = r["work"]
        rows.append({"frag_id": r["id"], "work": w, "group": r["group"], "year": r["year"],
                     "old_orth": r["old_orth"], "n_words": r["n_words"],
                     "true_genre": r["genre"], "true_era": r["era"],
                     "pred_genre": g_frag[i], "pred_era": model.eras[int(Pe[i].argmax())],
                     "genre_conf": round(float(Pg[i].max()), 4), "era_conf": round(float(Pe[i].max()), 4),
                     "work_true_genre": w_true_g[w], "work_true_era": w_true_e[w],
                     "work_pred_genre": w_genre[w], "work_pred_era": w_era[w],
                     "text_start": r["text"][:200].replace("\n", " ")})
    return rows


def predict_document(model, text, frag_words=80, topk=3):
    """Предсказание для произвольного текста: режем на фрагменты, считаем как одно произведение."""
    words = text.split()
    chunks = [" ".join(words[i:i + frag_words]) for i in range(0, len(words), frag_words)]
    if len(chunks) > 1 and len(chunks[-1].split()) < frag_words // 4:
        chunks[-2] += " " + chunks.pop()
    chunks = chunks or [text]
    Pg = model.genre_proba(chunks)
    lg = np.log(Pg + 1e-9).mean(0)
    pg = np.exp(lg - lg.max())
    pg /= pg.sum()
    genre = model.genres[int(pg.argmax())]
    Pe = model.era_proba(chunks, [genre] * len(chunks))
    le = np.log(Pe + 1e-9).mean(0)
    pe = np.exp(le - le.max())
    pe /= pe.sum()
    top = lambda p, names: [(names[i], float(p[i])) for i in np.argsort(-p)[:topk]]
    return {"genre": genre, "era": model.eras[int(pe.argmax())], "fragments": len(chunks),
            "genre_top": top(pg, model.genres), "era_top": top(pe, model.eras)}


# =====================================================================
# QWK - главная метрика качества периода
# =====================================================================
def qwk_idx(a, b, n):
    """Квадратично взвешенная каппа Коэна для порядковых меток 0..n-1 (периоды упорядочены по времени)."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if len(a) == 0 or n < 2:
        return float("nan")
    O = np.zeros((n, n))
    np.add.at(O, (a, b), 1)
    W = (np.arange(n)[:, None] - np.arange(n)[None, :]) ** 2 / float((n - 1) ** 2)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 0 else float("nan")


def qwk(y_true, y_pred, order):
    """То же по строковым меткам; order - список периодов в хронологическом порядке."""
    ix = {e: i for i, e in enumerate(order)}
    return qwk_idx([ix[e] for e in y_true], [ix[e] for e in y_pred], len(order))


# =====================================================================
# Словарь маркеров: признаки и «добавка» к любой модели
# =====================================================================
import re as _re
_TOKEN = _re.compile(r"[а-яa-z]+")


def tokenize_forms(text):
    """Нормализация как в словарях: нижний регистр, ё -> е; единица - словоформа."""
    return _TOKEN.findall(text.lower().replace("ё", "е"))


def load_dictionaries(path, only_replicated=True, log=print):
    """
    path: папка со словарями dict_<жанр>_form.json (рядом могут лежать markers_<жанр>_form.csv) или один json.
    only_replicated: оставить только маркеры, подтверждённые на val (колонка replicates в csv).
    """
    import csv
    import glob
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "dict_*.json")))
    if not files:
        raise FileNotFoundError(f"в {path} не найдено файлов dict_*.json")
    dicts = {}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        markers = {t: (m["era"], float(m["log_ratio"])) for t, m in d["markers"].items()}
        note = ""
        if only_replicated:
            csvp = os.path.join(os.path.dirname(fp), os.path.basename(fp).replace("dict_", "markers_", 1)
                                .rsplit(".", 1)[0] + ".csv")
            if os.path.exists(csvp):
                with open(csvp, encoding="utf-8-sig") as f:
                    ok = {r["token"] for r in csv.DictReader(f) if r.get("replicates", "True") == "True"}
                before = len(markers)
                markers = {t: v for t, v in markers.items() if t in ok}
                note = f" (оставлено воспроизводимых {len(markers)} из {before})"
            else:
                log(f"ВНИМАНИЕ: нет {os.path.basename(csvp)}, фильтр по воспроизводимости не применён")
        dicts[d["genre"]] = {"eras": d["eras"], "markers": markers}
        log(f"словарь «{d['genre']}»: {len(markers)} маркеров{note}")
    return dicts


class DictFeaturizer:
    """
    Для каждого фрагмента по словарю ЕГО жанра: для каждой эпохи словаря
      (1) взвешенная частота маркеров (сумма log_ratio на 100 слов),
      (2) частота маркеров (на 100 слов).
    Признаки стандартизуются отдельно по каждому жанру (по train).
    """
    def __init__(self, dicts):
        self.dicts = dicts
        self.slots = sorted({e for d in dicts.values() for e in d["eras"]}, key=era_start)
        self.slot_ix = {e: i for i, e in enumerate(self.slots)}
        self.dim = 2 * len(self.slots)
        self.stats = {}

    def raw(self, texts, genres):
        J = len(self.slots)
        X = np.zeros((len(texts), self.dim))
        for i, (t, g) in enumerate(zip(texts, genres)):
            d = self.dicts.get(g)
            if not d:
                continue
            toks = tokenize_forms(t)
            for tok in toks:
                m = d["markers"].get(tok)
                if m:
                    j = self.slot_ix[m[0]]
                    X[i, j] += m[1]
                    X[i, J + j] += 1.0
            X[i] *= 100.0 / max(len(toks), 1)
        return X

    def fit(self, texts, genres):
        X = self.raw(texts, genres)
        genres = np.array(genres)
        for g in set(genres):
            m = genres == g
            mu, sd = X[m].mean(0), X[m].std(0)
            self.stats[g] = (mu, np.where(sd < 1e-9, 1.0, sd))
        return self

    def transform(self, texts, genres):
        X = self.raw(texts, genres)
        for i, g in enumerate(genres):
            if g in self.dicts and g in self.stats:
                mu, sd = self.stats[g]
                X[i] = np.clip((X[i] - mu) / sd, -5, 5)
            else:
                X[i] = 0.0
        return X


class BoostedModel(BaseModel):
    """
    Базовая модель + словарь. Жанр - как у базовой модели. Период: логистическая регрессия над
    [log-вероятности периода базовой модели, признаки словаря]. Работает с любой из трёх моделей.
    """
    kind = "boosted"

    def __init__(self, base, feat, clf, name, base_run):
        self.base, self.feat, self.clf, self.name, self.base_run = base, feat, clf, name, base_run
        self.genres, self.eras = base.genres, base.eras

    def genre_proba(self, texts):
        return self.base.genre_proba(texts)

    def design(self, texts, genres, P):
        return np.hstack([np.log(np.clip(P, 1e-4, 1.0)), self.feat.transform(texts, genres)])

    def era_proba(self, texts, genres):
        P = self.base.era_proba(texts, genres)
        pr = self.clf.predict_proba(self.design(texts, genres, P))
        out = np.zeros((len(texts), len(self.eras)))
        out[:, [self.eras.index(c) for c in self.clf.classes_]] = pr
        return out

    def save(self, d):
        import joblib
        os.makedirs(d, exist_ok=True)
        joblib.dump({"feat": self.feat, "clf": self.clf, "name": self.name, "base_run": self.base_run},
                    os.path.join(d, "booster.joblib"))


def fit_booster(base, dicts, train, val, name, base_run, log=print):
    """
    Обучает «добавку» со словарём на val (честные, не виденные базовой моделью вероятности).
    Силу регуляризации C выбирает по QWK на кросс-валидации по группам (авторам) внутри val.
    Возвращает (BoostedModel, статистика).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    feat = DictFeaturizer(dicts).fit([r["text"] for r in train], [r["genre"] for r in train])
    val = [r for r in val if r["era"] in base.eras and r["genre"] in base.genres]
    if len(val) < 30 or len({r["era"] for r in val}) < 2:
        raise SystemExit("для обучения добавки со словарём в val слишком мало данных")
    texts, gen = [r["text"] for r in val], [r["genre"] for r in val]
    P = base.era_proba(texts, gen)                     # истинный жанр - как при обучении сети периода
    X = np.hstack([np.log(np.clip(P, 1e-4, 1.0)), feat.transform(texts, gen)])
    y = np.array([r["era"] for r in val])
    groups = [r["group"] for r in val]
    order = base.eras
    stats = {"val_n": len(val), "val_qwk_base": qwk(y, np.array(order)[P.argmax(1)], order)}

    k = min(5, len(set(groups)))
    best = (float("-inf"), 1.0)
    if k >= 2:
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            oof = np.empty(len(y), dtype=object)
            try:
                for tr, te in GroupKFold(k).split(X, y, groups):
                    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
                    oof[te] = clf.predict(X[te])
            except ValueError:
                continue
            sc = qwk(y, oof.astype(str), order)
            if sc == sc and sc > best[0]:
                best = (sc, C)
    clf = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(X, y)
    stats.update({"best_C": best[1], "val_qwk_boost_cv": best[0] if best[0] > float("-inf") else None})
    log(f"{name}: на val QWK периода: базовая {stats['val_qwk_base']:.3f} -> со словарём (кросс-вал., C={best[1]}) "
        f"{stats['val_qwk_boost_cv']:.3f}" if stats["val_qwk_boost_cv"] is not None else f"{name}: добавка обучена")
    return BoostedModel(base, feat, clf, name, base_run), stats
