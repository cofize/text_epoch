"use strict";
const $ = (id) => document.getElementById(id);
const MAX = 10000;

const EXAMPLES = {
  prose: "Помещик Ладыгин приехал в уездный город на третий день после ярмарки. В трактире только и разговоров было о том, что земство отложило дело, что становой и чиновник повздорили и что купец уехал, не заплатив. Барыня писала из деревни, что мужик опять не вышел на работу, а приказчик по обыкновению винит погоду. Гимназист, племянник Ладыгина, слушал эти разговоры с усмешкой и уверял, что он нигилист, чем немало смущал старого барина.",
  poetry: "Сей тихий дол, где дремлет лира,\nГде юный певец зрел утренний пламень,\nНыне пуст, и в песнях мира\nНе слышен глас, как прежде, и камень\nХранит одну лишь память о златом\nБылом, о деве, что у брега ждала,\nИ снова ночь над тёмным и пустым\nПростором, где река тиха и мала.",
  publicism: "Пятилетка поставила перед страной задачи, которых не знала ни одна эпоха. Колхоз стал основной формой деревенской жизни, а партия требует от каждого ударник труда и в городе, и в деревне. Оборона страны неотделима от работы в тылу, и фашизм, готовящий новую войну, встретит единый советский фронт. Индустриализация и коллективизация изменили лицо страны за считанные годы, а вредитель и кулак, пытающиеся помешать общему делу, неизбежно оказываются в изоляции.",
};
const WARNINGS = {
  short_text: "Текст короче 400 символов: оценка менее надёжна. Модель обучена на отрывках от 1000 символов.",
  no_dictionary: "Для этого жанра нет словаря маркеров, подсветки не будет.",
};

const GENRE_INFO = {
  "поэзия": "Стихотворения и поэтические тексты",
  "проза": "Художественная проза и повествование",
  "публицистика": "Статьи, очерки и общественные тексты",
};
let eraOrder = [];          // метки эпох по порядку: от них зависит цвет
let busy = false;
let selectedGenre = "";     // "" = определить автоматически

const fmtEra = (s) => s.replace("-", "–");
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const pct = (p) => Math.round(p * 100) + "%";
const eraClass = (era) => "era-" + (Math.max(0, eraOrder.indexOf(era)) % 6);
function plural(n, one, few, many) {
  const a = n % 100, b = n % 10;
  if (a >= 11 && a <= 14) return many;
  return b === 1 ? one : (b >= 2 && b <= 4 ? few : many);
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/* ---------- состояние окна: правка или просмотр ---------- */
function setMode(view) {
  $("text").hidden = view;
  $("viewer").hidden = !view;
  $("inspector").hidden = !view;
  $("go").hidden = view;
  $("edit").hidden = !view;
  $("clear").hidden = view;
}
function showError(msg) { $("error").textContent = msg || ""; $("error").hidden = !msg; }
function updateCount() {
  const n = $("text").value.length;
  $("count").textContent = n.toLocaleString("ru-RU") + " / " + MAX.toLocaleString("ru-RU");
}

/* ---------- старт: жанры и пометки о заглушке/демо-словаре ---------- */
async function init() {
  const pick = $("genre-pick");
  let genres = ["поэзия", "проза", "публицистика"];
  try {
    const h = await (await fetch("/api/health")).json();
    genres = h.genres || genres;
    (h.dictionaries && Object.values(h.dictionaries)[0] || { eras: [] }).eras.forEach((e) => eraOrder.push(e));
    const box = $("notices");
    if (!h.stub && h.genre_method === "heuristic") box.append(el("p", "notice", "Жанр определяется по простым признакам текста (длина строк, диалоги, отвлечённая лексика), период оценивает обученная модель. Если жанр определился неверно, выберите его справа."));
    if (h.stub) box.append(el("p", "notice", "Сейчас работает заглушка: жанр определяется по простым признакам текста, эпоха — по словарю маркеров. Нейросеть ещё не подключена."));
    if (h.dictionaries_demo) box.append(el("p", "notice", "Словарь маркеров демонстрационный: слова подобраны вручную, а не посчитаны по корпусу."));
    (h.warnings || []).forEach((w) => box.append(el("p", "notice", w)));
  } catch (e) {
    showError("Сервис не отвечает. Проверьте, что сервер запущен.");
  }
  genres.forEach((g) => {
    const b = el("button", "genre");
    b.type = "button"; b.dataset.genre = g; b.setAttribute("aria-pressed", "false");
    b.append(el("strong", "", cap(g)));
    if (GENRE_INFO[g]) b.append(el("small", "", GENRE_INFO[g]));
    b.addEventListener("click", () => {
      selectedGenre = selectedGenre === g ? "" : g;          // повторный клик снимает выбор
      pick.querySelectorAll(".genre").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.genre === selectedGenre)));
      if (!$("viewer").hidden) analyze();                    // результат уже на экране: пересчитать с новым жанром
    });
    pick.append(b);
  });
}

/* ---------- запрос ---------- */
async function analyze() {
  if (busy) return;
  showError("");
  const text = $("text").value;
  const genre = selectedGenre;
  busy = true; $("go").disabled = true; $("go").textContent = "Определяю…";
  try {
    const r = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(genre ? { text, genre } : { text }),
    });
    const data = await r.json();
    if (!r.ok) { showError(data.message || "Не удалось обработать запрос."); return; }
    render(text, data);
  } catch (e) {
    showError("Сервис не отвечает. Проверьте, что сервер запущен, и повторите.");
  } finally {
    busy = false; $("go").disabled = false; $("go").textContent = "Определить";
  }
}

/* ---------- вывод ---------- */
function bars(ul, rows, winKey, keyOf, labelOf, withEra) {
  ul.replaceChildren();
  rows.forEach((row) => {
    const li = el("li", (withEra ? eraClass(keyOf(row)) + " " : "") + (keyOf(row) === winKey ? "win" : ""));
    const fill = el("span", "fill"), track = el("span", "track");
    track.append(fill);
    li.append(el("span", "lab", labelOf(row)), track, el("span", "pct", pct(row.score)));
    ul.append(li);
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = Math.max(row.score * 100, 1) + "%"; }));
  });
}

function render(text, d) {
  const eras = d.period_scores.map((p) => p.period);
  if (eras.join() !== eraOrder.join()) eraOrder = eras;

  $("empty").hidden = true; $("result").hidden = false;
  const wl = $("warnings"); wl.replaceChildren();
  d.warnings.forEach((w) => WARNINGS[w] && wl.append(el("li", "", WARNINGS[w])));

  $("genre-word").textContent = cap(d.genre);
  $("genre-conf").textContent = d.genre_source === "user" ? "задан вручную"
    : d.genre_method === "heuristic" ? "по признакам текста" : "уверенность " + pct(d.genre_score);
  bars($("genre-bars"), d.genre_scores, d.genre, (r) => r.genre, (r) => cap(r.genre), false);

  $("era-title").textContent = "Эпоха среди текстов жанра «" + d.genre + "»";
  $("era-word").textContent = fmtEra(d.period);
  $("era-conf").textContent = "уверенность " + pct(d.score);
  bars($("era-bars"), d.period_scores, d.period, (r) => r.period, (r) => fmtEra(r.period), true);

  renderMarkers(d);
  renderViewer(text, d);
  setMode(true);
  $("inspector").className = "inspector";
  $("inspector").textContent = d.dated_markers.length
    ? "Наведите на выделенное слово: здесь появится объяснение."
    : "В тексте нет слов словаря, указывающих на эту эпоху.";
}

function renderMarkers(d) {
  const groups = new Map();
  d.dated_markers.forEach((m) => {
    const k = m.text.toLowerCase();
    const g = groups.get(k) || { word: k, n: 0, w: 0 };
    g.n += 1; g.w = Math.max(g.w, m.weight);
    groups.set(k, g);
  });
  const chips = $("chips"); chips.replaceChildren();
  [...groups.values()].sort((a, b) => b.w - a.w || b.n - a.n).forEach((g) => {
    const li = el("li", eraClass(d.period), g.word);
    if (g.n > 1) li.append(el("span", "n", "×" + g.n));
    chips.append(li);
  });
  const era = fmtEra(d.period);
  $("marker-note").textContent = groups.size
    ? "Слова из словаря жанра «" + d.genre + "», которые указывают на " + era + ". Всего в тексте найдено слов словаря: " + d.markers_found + "."
    : d.markers_found
      ? "Слова словаря жанра «" + d.genre + "» в тексте есть (" + d.markers_found + "), но они указывают на другие эпохи, не на " + era + "."
      : "Слов из словаря жанра «" + d.genre + "» в тексте не найдено.";
}

function renderViewer(text, d) {
  const v = $("viewer"); v.replaceChildren();
  const insp = $("inspector");
  let pos = 0;
  d.dated_markers.forEach((m) => {
    if (m.start > pos) v.append(document.createTextNode(text.slice(pos, m.start)));
    const mk = el("mark", "m " + eraClass(d.period), text.slice(m.start, m.end));
    mk.dataset.w = m.weight >= 0.7 ? "3" : m.weight >= 0.4 ? "2" : "1";
    mk.tabIndex = 0;
    const show = () => {
      v.querySelectorAll("mark.on").forEach((x) => x.classList.remove("on"));
      mk.classList.add("on");
      insp.className = "inspector on " + eraClass(d.period);
      const times = Math.max(1, Math.round(Math.pow(2, m.log_ratio)));
      insp.textContent = "«" + m.text.toLowerCase() + "» указывает на " + fmtEra(m.points_to) + ": в жанре «" + d.genre +
        "» это слово встречается примерно в " + times + " " + plural(times, "раз", "раза", "раз") + " чаще, чем в других эпохах.";
    };
    mk.addEventListener("mouseenter", show);
    mk.addEventListener("focus", show);
    v.append(mk);
    pos = m.end;
  });
  if (pos < text.length) v.append(document.createTextNode(text.slice(pos)));
}

/* ---------- события ---------- */
$("text").addEventListener("input", updateCount);
$("go").addEventListener("click", analyze);
$("edit").addEventListener("click", () => { setMode(false); $("text").focus(); });
$("clear").addEventListener("click", () => { $("text").value = ""; updateCount(); showError(""); $("text").focus(); });
document.querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", () => {
  setMode(false); $("text").value = EXAMPLES[b.dataset.ex]; updateCount(); showError("");
}));
$("text").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze(); });

setMode(false);
updateCount();
init();
