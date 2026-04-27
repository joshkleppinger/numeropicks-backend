"""
NumeroPicks — Prediction Engine
Extracted from numero.py, adapted for use as a FastAPI backend.
All tkinter / GUI code removed. Pure Python logic only.
"""

import csv, json, math, os, random, time
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import numpy as np;    HAS_NP    = True
except ImportError:        HAS_NP    = False
try:
    import torch;          HAS_TORCH = True
except ImportError:        HAS_TORCH = False
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing  import StandardScaler
    HAS_SK = True
except ImportError:        HAS_SK    = False
try:
    from scipy import stats as scipy_stats; HAS_SCIPY = True
except Exception:                           HAS_SCIPY = False
try:
    import requests
    from bs4 import BeautifulSoup;  HAS_REQUESTS = True
except ImportError:                 HAS_REQUESTS = False

# ── Data directory ─────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("NUMERO_DATA_DIR", "/opt/render/project/src/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCRAPE_STATE = DATA_DIR / "scrape_state.json"

# ══════════════════════════════════════════════════════════════════════════════
#  GAME DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

GAMES = {
    "powerball": {
        "key":           "powerball",
        "display_name":  "Powerball",
        "white_max":     69,
        "special_max":   26,
        "white_count":   5,
        "special_name":  "PB",
        "draw_days":     {0, 2, 5},
        "url_base":      "https://www.lottery.net/powerball/numbers",
        "csv":           DATA_DIR / "Powerball_draws.csv",
        "pred_csv":      DATA_DIR / "Powerball_predictions.csv",
        "acc_csv":       DATA_DIR / "Powerball_accuracy.csv",
        "history_start": 1992,
        "era_changes":   [49, 59, 69],
    },
    "megamillions": {
        "key":           "megamillions",
        "display_name":  "Mega Millions",
        "white_max":     70,
        "special_max":   25,
        "white_count":   5,
        "special_name":  "MB",
        "draw_days":     {1, 4},
        "url_base":      "https://www.lottery.net/mega-millions/numbers",
        "csv":           DATA_DIR / "MegaMillions_draws.csv",
        "pred_csv":      DATA_DIR / "MegaMillions_predictions.csv",
        "acc_csv":       DATA_DIR / "MegaMillions_accuracy.csv",
        "history_start": 1996,
        "era_changes":   [56, 75, 70],
    },
    "superlotto": {
        "key":           "superlotto",
        "display_name":  "SuperLotto Plus",
        "white_max":     47,
        "special_max":   27,
        "white_count":   5,
        "special_name":  "MN",
        "draw_days":     {2, 5},
        "url_base":      "https://www.lottery.net/california/superlotto-plus/numbers",
        "csv":           DATA_DIR / "SuperLotto_draws.csv",
        "pred_csv":      DATA_DIR / "SuperLotto_predictions.csv",
        "acc_csv":       DATA_DIR / "SuperLotto_accuracy.csv",
        "history_start": 1986,
        "era_changes":   [47],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATE PARSING
# ══════════════════════════════════════════════════════════════════════════════

_DATE_FMTS = [
    "%a, %b %d, %Y", "%a, %b, %d, %Y", "%a, %b %d %Y",
    "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d",
]

def parse_date(s: str):
    s = s.strip().replace("  ", " ")
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

# ══════════════════════════════════════════════════════════════════════════════
#  CSV I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_draws(game: dict) -> list:
    path = game["csv"]
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                n     = game["white_count"]
                balls   = [int(r[f"ball_{i}"]) for i in range(1, n + 1)]
                special = int(r["special"])
                rows.append({"date": r["date"].strip(), "balls": balls, "special": special})
            except Exception:
                pass
    return rows


def save_draws(game: dict, rows: list):
    path = game["csv"]
    if path.exists():
        path.with_suffix(".bak.csv").write_bytes(path.read_bytes())
    n          = game["white_count"]
    fieldnames = ["date"] + [f"ball_{i}" for i in range(1, n + 1)] + ["special"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {"date": r["date"], "special": r["special"]}
            for i, b in enumerate(r["balls"], 1):
                row[f"ball_{i}"] = b
            writer.writerow(row)

# ══════════════════════════════════════════════════════════════════════════════
#  SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_lottery_net_page(html: str, white_max: int, special_max: int,
                             white_count: int) -> list:
    soup    = BeautifulSoup(html, "html.parser")
    results = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        a = tds[0].find("a")
        if not a:
            continue
        dt = parse_date(a.get_text(" ", strip=True))
        if dt is None:
            href  = a.get("href", "").rstrip("/")
            slug  = href.split("/")[-1]
            parts = slug.split("-")
            if len(parts) == 3:
                try:
                    dt = datetime(int(parts[2]), int(parts[0]), int(parts[1]))
                except (ValueError, IndexError):
                    pass
        if dt is None:
            continue
        nums = []
        for li in tds[-1].find_all("li"):
            txt = li.get_text(strip=True)
            if txt.isdigit():
                nums.append(int(txt))
            if len(nums) == white_count + 1:
                break
        if len(nums) < white_count + 1:
            continue
        balls   = sorted(nums[:white_count])
        special = nums[white_count]
        if not (all(1 <= b <= white_max for b in balls)
                and 1 <= special <= special_max):
            continue
        results.append({
            "date_str": dt.strftime("%a, %b %d, %Y"),
            "dt":       dt,
            "balls":    balls,
            "special":  special,
        })
    return results


def scrape_game(game: dict, existing_rows: list, log_fn=None) -> tuple:
    if not HAS_REQUESTS:
        return 0, "requests/beautifulsoup4 not installed."

    def log(m):
        if log_fn: log_fn(m)

    existing_dates = set()
    latest_dt      = None
    earliest_dt    = None
    for r in existing_rows:
        dt = parse_date(r["date"])
        if dt:
            existing_dates.add(dt.date())
            if latest_dt is None or dt > latest_dt:   latest_dt   = dt
            if earliest_dt is None or dt < earliest_dt: earliest_dt = dt

    current_year  = datetime.now().year
    history_start = game.get("history_start", 2002)
    earliest_year = earliest_dt.year if earliest_dt else current_year

    if latest_dt is None:
        years_to_fetch = list(range(history_start, current_year + 1))
        log(f"First run — fetching full history from {history_start} …")
    elif earliest_year > history_start + 1:
        years_to_fetch = list(range(history_start, current_year + 1))
        log(f"Incomplete history (earliest={earliest_year}) — fetching all …")
    elif latest_dt.year < current_year:
        years_to_fetch = list(range(latest_dt.year, current_year + 1))
    else:
        years_to_fetch = [current_year]

    all_parsed = []
    for year in years_to_fetch:
        url = f"{game['url_base']}/{year}"
        log(f"Fetching {url} …")
        try:
            resp   = requests.get(url, timeout=25, headers=_HEADERS)
            resp.raise_for_status()
            parsed = _parse_lottery_net_page(
                resp.text, game["white_max"], game["special_max"], game["white_count"]
            )
            log(f"  {len(parsed)} draws on {year} page")
            all_parsed.extend(parsed)
        except Exception as e:
            log(f"  Error fetching {year}: {e}")

    new_rows = []
    seen     = set()
    for p in all_parsed:
        d = p["dt"].date()
        if d in existing_dates or d in seen:
            continue
        seen.add(d)
        new_rows.append({"date": p["date_str"], "balls": p["balls"], "special": p["special"]})

    if new_rows:
        new_rows.sort(key=lambda r: parse_date(r["date"]) or datetime.min)
        existing_rows.extend(new_rows)
        save_draws(game, existing_rows)
        _save_scrape_state()
        return len(new_rows), f"Added {len(new_rows)} new draw(s). Total: {len(existing_rows):,}"
    _save_scrape_state()
    return 0, f"Up to date ({len(existing_rows):,} rows)"


def _save_scrape_state():
    with open(SCRAPE_STATE, "w") as f:
        json.dump({"last_scrape": datetime.now().isoformat()}, f)


def load_scrape_state():
    if not SCRAPE_STATE.exists():
        return None
    try:
        with open(SCRAPE_STATE) as f:
            return datetime.fromisoformat(json.load(f)["last_scrape"])
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def soft_floor(d: dict, floor: float = 0.15) -> dict:
    mn = min(d.values()); mx = max(d.values()); span = mx - mn + 1e-12
    return {k: floor + (1.0 - floor) * (v - mn) / span for k, v in d.items()}


def arith_blend(score_dicts: list, weights: list, ball_range) -> dict:
    total_w = sum(weights)
    return {
        n: sum(w * d.get(n, 0.5) for d, w in zip(score_dicts, weights)) / total_w
        for n in ball_range
    }


def _halton_sequence(n: int, base: int) -> list:
    seq = []
    for i in range(n):
        f, r, idx = 1.0, 0.0, i + 1
        while idx > 0:
            f /= base; r += f * (idx % base); idx //= base
        seq.append(r)
    return seq


def _make_cdf(prob_dict: dict, keys) -> list:
    cumulative = 0.0
    cdf = []
    for k in sorted(keys):
        cumulative += prob_dict.get(k, 0)
        cdf.append((cumulative, k))
    return cdf


def _sample_cdf(cdf: list, u: float):
    for threshold, val in cdf:
        if u <= threshold:
            return val
    return cdf[-1][1]


def analyze_and_predict(rows: list, game: dict, progress_cb=None) -> list:
    """Full 7-method prediction engine. Returns list of 5 ticket dicts."""
    def prog(pct, msg=""):
        if progress_cb: progress_cb(pct, msg)

    if not rows:
        return []

    WHITE_MAX   = game["white_max"]
    SPECIAL_MAX = game["special_max"]
    WHITE_COUNT = game["white_count"]
    WHITE_RANGE = range(1, WHITE_MAX + 1)
    SPEC_RANGE  = range(1, SPECIAL_MAX + 1)
    N           = len(rows)

    prog(2, "Loading data …")

    if HAS_NP:
        all_balls   = np.array([r["balls"]   for r in rows])
        all_special = np.array([r["special"] for r in rows])
    else:
        all_balls   = [r["balls"]   for r in rows]
        all_special = [r["special"] for r in rows]

    ERA3_START = 0
    era_changes = game.get("era_changes", [WHITE_MAX])
    for i, r in enumerate(rows):
        if any(b > era_changes[0] for b in r["balls"]) and ERA3_START == 0:
            ERA3_START = i
    era3_rows = rows[ERA3_START:]

    if HAS_NP:
        era3_balls   = np.array([r["balls"]   for r in era3_rows])
        era3_special = np.array([r["special"] for r in era3_rows])
        flat_era3    = era3_balls.flatten().tolist()
    else:
        era3_balls   = [r["balls"]   for r in era3_rows]
        era3_special = [r["special"] for r in era3_rows]
        flat_era3    = [b for r in era3_balls for b in r]

    ERA3_DRAWS = len(era3_rows)

    prog(8, "Method 1 — Frequency …")
    w_count = Counter(flat_era3)
    s_count = Counter(era3_special.tolist() if HAS_NP else
                      [r["special"] for r in era3_rows])
    alpha   = 2.0
    w_freq  = {n: (w_count.get(n, 0) + alpha) / (ERA3_DRAWS * WHITE_COUNT + alpha * WHITE_MAX)
               for n in WHITE_RANGE}
    s_freq  = {n: (s_count.get(n, 0) + alpha) / (ERA3_DRAWS + alpha * SPECIAL_MAX)
               for n in SPEC_RANGE}
    w_freq  = soft_floor(w_freq, 0.30)
    s_freq  = soft_floor(s_freq, 0.50)

    prog(18, "Method 2 — Markov …")
    m2 = {n: 0.0 for n in WHITE_RANGE}
    if HAS_NP:
        for pos in range(WHITE_COUNT):
            col   = era3_balls[:, pos].tolist()
            trans = defaultdict(Counter)
            for i in range(len(col) - 1):
                trans[col[i]][col[i + 1]] += 1
            last  = col[-1]; total = sum(trans[last].values())
            if total:
                for n in WHITE_RANGE:
                    m2[n] += (trans[last].get(n, 0) + 0.5) / (total + 0.5 * WHITE_MAX)
    m2 = soft_floor(m2, 0.20)

    prog(28, "Method 3 — Spectral …")
    m3_w = {n: 1.0 for n in WHITE_RANGE}
    m3_s = {n: 1.0 for n in SPEC_RANGE}
    if HAS_NP and ERA3_DRAWS > 20:
        for pos in range(WHITE_COUNT):
            col = era3_balls[:, pos].astype(float)
            mu  = col.mean(); sig = col.std() + 1e-6
            tgt = max(1, min(WHITE_MAX, mu + 0.2 * (mu - col[-1])))
            for n in WHITE_RANGE:
                m3_w[n] *= math.exp(-0.5 * ((n - tgt) / (sig * 4.0)) ** 2)
        s_arr = era3_special.astype(float)
        s_tgt = max(1, min(SPECIAL_MAX, s_arr.mean() + 0.2 * (s_arr.mean() - s_arr[-1])))
        s_sig = s_arr.std() * 4.0 + 1e-6
        for n in SPEC_RANGE:
            m3_s[n] *= math.exp(-0.5 * ((n - s_tgt) / s_sig) ** 2)
    m3_w = soft_floor(m3_w, 0.40)
    m3_s = soft_floor(m3_s, 0.60)

    prog(38, "Method 4 — Gap …")
    last_w = {}; last_s = {}
    for i, r in enumerate(rows):
        for b in r["balls"]: last_w[b] = i
        last_s[r["special"]] = i
    eg_w = WHITE_MAX / float(WHITE_COUNT)
    eg_s = float(SPECIAL_MAX)
    m4_w = {n: (lambda g: g if g >= 1 else g ** 2)(
                (N - last_w.get(n, N - int(eg_w))) / eg_w) for n in WHITE_RANGE}
    m4_s = {n: (lambda g: g if g >= 1 else g ** 2)(
                (N - last_s.get(n, N - int(eg_s))) / eg_s) for n in SPEC_RANGE}
    m4_w = soft_floor(m4_w, 0.20)
    m4_s = soft_floor(m4_s, 0.20)

    prog(50, "Method 5 — Neural …")
    m5_w = {n: 1.0 for n in WHITE_RANGE}
    m5_s = {n: 1.0 for n in SPEC_RANGE}
    if HAS_TORCH and ERA3_DRAWS > 50 and HAS_NP:
        try:
            m5_w, m5_s = _torch_predict(era3_balls, era3_special, ERA3_DRAWS,
                                         WHITE_MAX, SPECIAL_MAX)
            m5_w = soft_floor(m5_w, 0.20)
            m5_s = soft_floor(m5_s, 0.20)
        except Exception:
            pass
    if all(v == 1.0 for v in m5_w.values()) and HAS_SK and HAS_NP and ERA3_DRAWS > 50:
        try:
            m5_w, m5_s = _sklearn_predict(era3_balls, era3_special, ERA3_DRAWS,
                                           WHITE_MAX, SPECIAL_MAX)
            m5_w = soft_floor(m5_w, 0.20)
            m5_s = soft_floor(m5_s, 0.20)
        except Exception:
            pass

    prog(62, "Method 6 — Monte Carlo …")
    combo_w = arith_blend([w_freq, m2, m3_w, m4_w, m5_w],
                          [0.20, 0.15, 0.10, 0.20, 0.35], WHITE_RANGE)
    combo_s = arith_blend([s_freq, m3_s, m4_s, m5_s],
                          [0.25, 0.15, 0.25, 0.35], SPEC_RANGE)
    tw = sum(combo_w.values()) + 1e-12
    ts = sum(combo_s.values()) + 1e-12
    w_probs = {n: combo_w[n] / tw for n in WHITE_RANGE}
    s_probs = {n: combo_s[n] / ts for n in SPEC_RANGE}
    m6_w  = {n: 1e-9 for n in WHITE_RANGE}
    m6_s  = {n: 1e-9 for n in SPEC_RANGE}
    SAMPLES = 120_000
    hal_w   = _halton_sequence(SAMPLES, 2)
    hal_s   = _halton_sequence(SAMPLES, 3)
    w_cdf   = _make_cdf(w_probs, WHITE_RANGE)
    s_cdf   = _make_cdf(s_probs, SPEC_RANGE)
    for i in range(SAMPLES): m6_w[_sample_cdf(w_cdf, hal_w[i])] += 1
    for i in range(SAMPLES): m6_s[_sample_cdf(s_cdf, hal_s[i])] += 1
    m6_w = soft_floor(m6_w, 0.15)
    m6_s = soft_floor(m6_s, 0.15)

    prog(75, "Method 7 — Signature …")
    m7_w = {n: 1.0 for n in WHITE_RANGE}
    if HAS_NP and ERA3_DRAWS > 10:
        decay = 0.9997
        for idx, r in enumerate(era3_balls.tolist() if HAS_NP else era3_balls):
            w = decay ** (ERA3_DRAWS - 1 - idx)
            for b in r: m7_w[b] = m7_w.get(b, 1.0) + w
    m7_w = soft_floor(m7_w, 0.25)

    prog(83, "Synthesising …")
    final_w = arith_blend([w_freq, m2, m3_w, m4_w, m5_w, m6_w, m7_w],
                          [0.15, 0.15, 0.08, 0.17, 0.20, 0.15, 0.10], WHITE_RANGE)
    final_s = arith_blend([s_freq, m3_s, m4_s, m5_s, m6_s],
                          [0.20, 0.10, 0.25, 0.30, 0.15], SPEC_RANGE)

    prog(90, "Selecting tickets …")
    sorted_w = sorted(WHITE_RANGE, key=lambda n: final_w[n], reverse=True)

    def get_band(n): return (n - 1) // 10

    tickets       = []
    used_sets     = []
    bands_covered = set()

    if HAS_NP:
        s_wts = np.array([final_s[n] for n in SPEC_RANGE], dtype=float)
        s_wts /= s_wts.sum()

    attempt = 0
    while len(tickets) < 5 and attempt < 200:
        attempt += 1
        pool = sorted_w[:min(35 + attempt * 3, WHITE_MAX)]
        if HAS_NP:
            wts = np.array([final_w[n] for n in pool], dtype=float)
            if len(tickets) >= 2:
                for i, n in enumerate(pool):
                    if get_band(n) not in bands_covered:
                        wts[i] *= 2.5
            wts /= wts.sum()
            chosen = list(np.random.choice(pool, size=WHITE_COUNT, replace=False, p=wts))
        else:
            chosen = pool[:WHITE_COUNT]

        cs = frozenset(chosen)
        if cs in used_sets: continue
        tb = set(get_band(n) for n in chosen)
        if len(tb) < 2: continue
        used_sets.append(cs)
        bands_covered |= tb

        if HAS_NP:
            adj = s_wts.copy()
            for prev in tickets: adj[prev["special"] - 1] *= 0.15
            adj /= adj.sum()
            special = int(np.random.choice(list(SPEC_RANGE), p=adj))
        else:
            special = random.randint(1, SPECIAL_MAX)

        tickets.append({"balls": sorted(chosen), "special": special})

    while len(tickets) < 5:
        tickets.append({
            "balls":   sorted(random.sample(list(WHITE_RANGE), WHITE_COUNT)),
            "special": random.randint(1, SPECIAL_MAX),
        })

    prog(100, "Done.")
    return tickets


def _torch_predict(era3_balls, era3_special, N, WHITE_MAX, SPECIAL_MAX):
    SEQ = 20
    if N <= SEQ + 5: raise ValueError("Not enough data")
    WHITE_COUNT = era3_balls.shape[1]
    IN_DIM = WHITE_COUNT + 1
    data = []
    for i in range(N):
        vec = [b / WHITE_MAX for b in era3_balls[i]] + [era3_special[i] / SPECIAL_MAX]
        data.append(vec)
    data = torch.tensor(data, dtype=torch.float32)
    X, Y = [], []
    for i in range(N - SEQ - 1):
        X.append(data[i:i + SEQ]); Y.append(data[i + SEQ])
    X = torch.stack(X); Y = torch.stack(Y)
    class LSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(IN_DIM, 64, 2, batch_first=True, dropout=0.2)
            self.fc   = torch.nn.Sequential(
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, IN_DIM), torch.nn.Sigmoid())
        def forward(self, x):
            out, _ = self.lstm(x); return self.fc(out[:, -1, :])
    model = LSTM()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss  = torch.nn.MSELoss()
    model.train()
    for _ in range(40):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 64):
            idx = perm[s:s+64]; p = model(X[idx])
            l   = loss(p, Y[idx]); opt.zero_grad(); l.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(data[-SEQ:].unsqueeze(0)).squeeze().numpy()
    sig = 3.0
    w_scores = {n: 1.0 for n in range(1, WHITE_MAX+1)}
    for pos in range(WHITE_COUNT):
        tgt = pred[pos] * WHITE_MAX
        for n in range(1, WHITE_MAX+1):
            w_scores[n] *= math.exp(-0.5*((n-tgt)/sig)**2) ** 0.2
    s_scores = {n: math.exp(-0.5*((n-pred[-1]*SPECIAL_MAX)/sig)**2)
                for n in range(1, SPECIAL_MAX+1)}
    return w_scores, s_scores


def _sklearn_predict(era3_balls, era3_special, N, WHITE_MAX, SPECIAL_MAX):
    SEQ = 10
    if N <= SEQ + 5: raise ValueError("Not enough data")
    WHITE_COUNT = era3_balls.shape[1]
    X, Y = [], []
    for i in range(N - SEQ - 1):
        x = []
        for j in range(SEQ):
            x.extend([b/WHITE_MAX for b in era3_balls[i+j]])
            x.append(era3_special[i+j]/SPECIAL_MAX)
        X.append(x)
        Y.append([b/WHITE_MAX for b in era3_balls[i+SEQ]] +
                 [era3_special[i+SEQ]/SPECIAL_MAX])
    X = np.array(X); Y = np.array(Y)
    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=200,
                         random_state=42, early_stopping=True)
    model.fit(X, Y)
    last = []
    for j in range(N-SEQ, N):
        last.extend([b/WHITE_MAX for b in era3_balls[j]])
        last.append(era3_special[j]/SPECIAL_MAX)
    pred = model.predict([last])[0]
    sig  = 3.0
    w_scores = {n: 1.0 for n in range(1, WHITE_MAX+1)}
    for pos in range(WHITE_COUNT):
        tgt = pred[pos]*WHITE_MAX
        for n in range(1, WHITE_MAX+1):
            w_scores[n] *= math.exp(-0.5*((n-tgt)/sig)**2)**0.2
    s_scores = {n: math.exp(-0.5*((n-pred[-1]*SPECIAL_MAX)/sig)**2)
                for n in range(1, SPECIAL_MAX+1)}
    return w_scores, s_scores

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTIONS & ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def next_draw_date(game: dict) -> str:
    today     = datetime.now().date()
    draw_days = game["draw_days"]
    for offset in range(1, 8):
        d = today + timedelta(days=offset)
        if d.weekday() in draw_days:
            return datetime(d.year, d.month, d.day).strftime("%a, %b %d, %Y")
    return (today + timedelta(days=1)).strftime("%a, %b %d, %Y")


def save_predictions(game: dict, tickets: list, target_date_str: str):
    path = game["pred_csv"]
    n    = game["white_count"]
    fns  = (["prediction_date", "target_draw_date"]
            + [f"pred_ball_{i}" for i in range(1, n+1)]
            + ["pred_special"])
    needs_header = True
    if path.exists() and path.stat().st_size > 0:
        with open(path, encoding="utf-8") as chk:
            if chk.readline().strip().startswith("prediction_date"):
                needs_header = False
    path.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%a, %b %d, %Y")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        if needs_header: w.writeheader()
        for t in tickets:
            row = {"prediction_date": today, "target_draw_date": target_date_str,
                   "pred_special": t["special"]}
            for i, b in enumerate(t["balls"], 1):
                row[f"pred_ball_{i}"] = b
            w.writerow(row)


def compute_accuracy(pred_balls, pred_special, actual_balls, actual_special) -> float:
    hits   = len(set(pred_balls) & set(actual_balls))
    sp_hit = (pred_special == actual_special)
    n      = len(pred_balls)
    if hits == n and sp_hit:   return 1.00
    if hits == n:              return 0.99
    if hits == n-1 and sp_hit: return 0.83
    if hits == n-1:            return 0.80
    if hits == n-2 and sp_hit: return 0.67
    if hits == n-2:            return 0.60
    if hits == n-3 and sp_hit: return 0.33
    if hits == n-3:            return 0.20
    if hits == 1 and sp_hit:   return 0.17
    if hits == 1:              return 0.01
    if sp_hit:                 return 0.07
    return 0.00


def compare_predictions(game: dict, draw_rows: list) -> dict:
    path = game["pred_csv"]
    if not path.exists():
        return {"evaluated": [], "pending": []}
    n = game["white_count"]
    draw_by_date = {}
    for r in draw_rows:
        dt = parse_date(r["date"])
        if dt: draw_by_date[dt.date()] = r
    today  = datetime.now().date()
    rounds = defaultdict(list)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    t    = row["target_draw_date"].strip()
                    pd_  = row["prediction_date"].strip()
                    balls = [int(row[f"pred_ball_{i}"]) for i in range(1, n+1)]
                    sp    = int(row["pred_special"])
                    rounds[(pd_, t)].append({"pred_balls": balls, "pred_special": sp})
                except Exception:
                    pass
    except Exception:
        return {"evaluated": [], "pending": []}
    evaluated, pending = [], []
    for (pred_date, target_date), tickets in rounds.items():
        tdt = parse_date(target_date)
        if not tdt: continue
        if tdt.date() > today:
            pending.append({"prediction_date": pred_date,
                            "target_draw_date": target_date, "tickets": tickets})
            continue
        actual = draw_by_date.get(tdt.date())
        if actual is None:
            pending.append({"prediction_date": pred_date,
                            "target_draw_date": target_date,
                            "tickets": tickets, "awaiting_scrape": True})
            continue
        best  = max(tickets, key=lambda t: compute_accuracy(
            t["pred_balls"], t["pred_special"], actual["balls"], actual["special"]))
        score = compute_accuracy(best["pred_balls"], best["pred_special"],
                                 actual["balls"], actual["special"])
        evaluated.append({
            "prediction_date":  pred_date,
            "target_draw_date": target_date,
            "pred_balls":       best["pred_balls"],
            "pred_special":     best["pred_special"],
            "actual_balls":     actual["balls"],
            "actual_special":   actual["special"],
            "white_matches":    len(set(best["pred_balls"]) & set(actual["balls"])),
            "sp_match":         int(best["pred_special"] == actual["special"]),
            "score":            score,
        })
    evaluated.sort(key=lambda r: parse_date(r["target_draw_date"]) or datetime.min)
    pending.sort(key=lambda r:   parse_date(r["target_draw_date"]) or datetime.min)
    return {"evaluated": evaluated, "pending": pending}

def compare_predictions_with_db(game: dict, draw_rows: list, db_preds: list) -> dict:
    """Like compare_predictions but uses DB predictions list directly."""
    from collections import defaultdict
    n = game["white_count"]
    draw_by_date = {}
    for r in draw_rows:
        dt = parse_date(r["date"])
        if dt: draw_by_date[dt.date()] = r

    from datetime import datetime as _dt
    today = _dt.now().date()
    rounds = defaultdict(list)
    for p in db_preds:
        try:
            pred_date = str(p["prediction_date"])
            tgt_date  = str(p["target_draw_date"])
            rounds[(pred_date, tgt_date)].append({
                "pred_balls":   p["pred_balls"],
                "pred_special": p["pred_special"],
            })
        except Exception:
            pass

    evaluated, pending = [], []
    for (pred_date, target_date), tickets in rounds.items():
        tdt = parse_date(target_date)
        if not tdt: continue
        if tdt.date() > today:
            pending.append({"prediction_date": pred_date,
                            "target_draw_date": target_date,
                            "tickets": tickets})
            continue
        actual = draw_by_date.get(tdt.date())
        if actual is None:
            pending.append({"prediction_date": pred_date,
                            "target_draw_date": target_date,
                            "tickets": tickets, "awaiting_scrape": True})
            continue
        best  = max(tickets, key=lambda t: compute_accuracy(
            t["pred_balls"], t["pred_special"],
            actual["balls"], actual["special"]))
        score = compute_accuracy(best["pred_balls"], best["pred_special"],
                                  actual["balls"], actual["special"])
        evaluated.append({
            "prediction_date":  pred_date,
            "target_draw_date": target_date,
            "pred_balls":       best["pred_balls"],
            "pred_special":     best["pred_special"],
            "actual_balls":     actual["balls"],
            "actual_special":   actual["special"],
            "white_matches":    len(set(best["pred_balls"]) & set(actual["balls"])),
            "sp_match":         int(best["pred_special"] == actual["special"]),
            "score":            score,
        })

    evaluated.sort(key=lambda r: parse_date(r["target_draw_date"]) or datetime.min)
    pending.sort(key=lambda r:   parse_date(r["target_draw_date"]) or datetime.min)
    return {"evaluated": evaluated, "pending": pending}
