"""
NumeroPicks — Prediction Engine
Extracted from numero.py, adapted for use as a FastAPI backend.
All tkinter / GUI code removed. Pure Python logic only.
"""

import csv, json, math, os, random, re, time
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
# Note: ca_daily_scraper is no longer needed — scrape_game() uses the
# California Lottery JSON API for all games including Daily 3 and Daily 4.

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
        "calottery_id":  12,
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
        "special_max":   24,
        "white_count":   5,
        "special_name":  "MB",
        "draw_days":     {1, 4},
        "calottery_id":  15,
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
        "calottery_id":  8,
        "csv":           DATA_DIR / "SuperLotto_draws.csv",
        "pred_csv":      DATA_DIR / "SuperLotto_predictions.csv",
        "acc_csv":       DATA_DIR / "SuperLotto_accuracy.csv",
        "history_start": 1986,
        "era_changes":   [49, 47],   # 1986-2000: 1-49 (original SuperLotto)
                                     # 2000+: 1-47 (SuperLotto Plus, current)
    },
    "daily3": {
        "key":           "daily3",
        "display_name":  "Daily 3",
        "white_max":     9,
        "special_max":   0,      # no special ball
        "white_count":   3,
        "special_name":  None,
        "draw_days":     {0,1,2,3,4,5,6},  # daily (twice daily - midday + evening)
        "calottery_id":  9,
        "has_two_draws_per_day": True,
        "csv":           DATA_DIR / "Daily3_draws.csv",
        "pred_csv":      DATA_DIR / "Daily3_predictions.csv",
        "acc_csv":       DATA_DIR / "Daily3_accuracy.csv",
        "history_start": 2004,
        "era_changes":   [9],  # always been 0-9
    },
    "daily4": {
        "key":           "daily4",
        "display_name":  "Daily 4",
        "white_max":     9,
        "special_max":   0,      # no special ball
        "white_count":   4,
        "special_name":  None,
        "draw_days":     {0,1,2,3,4,5,6},  # daily
        "calottery_id":  14,
        "csv":           DATA_DIR / "Daily4_draws.csv",
        "pred_csv":      DATA_DIR / "Daily4_predictions.csv",
        "acc_csv":       DATA_DIR / "Daily4_accuracy.csv",
        "history_start": 2010,
        "era_changes":   [9],  # always been 0-9
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
    # Remove day-of-week prefix if present (e.g., "Monday April 6, 2026" -> "April 6, 2026")
    s = re.sub(r'^\w+\s+', '', s, count=1)
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
                # Handle games with no special ball (Daily 3/4)
                if game.get("special_name") and game["special_name"] and r.get("special", ""):
                    special = int(r["special"])
                elif game.get("special_max", 0) == 0:
                    special = None
                else:
                    special = int(r["special"])
                row = {"date": r["date"].strip(), "balls": balls, "special": special}
                # Preserve draw_type if present (Daily 3 midday/evening)
                if "draw_type" in r and r["draw_type"]:
                    row["draw_type"] = r["draw_type"]
                rows.append(row)
            except Exception:
                pass
    return rows


def save_draws(game: dict, rows: list):
    path = game["csv"]
    if path.exists():
        path.with_suffix(".bak.csv").write_bytes(path.read_bytes())
    n          = game["white_count"]
    fieldnames = ["date"] + [f"ball_{i}" for i in range(1, n + 1)] + ["special"]
    # Add draw_type column for Daily 3 (twice-daily draws)
    if game.get("key") == "daily3":
        fieldnames.append("draw_type")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {"date": r["date"], "special": r.get("special", "")}
            for i, b in enumerate(r["balls"], 1):
                row[f"ball_{i}"] = b
            if game.get("key") == "daily3":
                row["draw_type"] = r.get("draw_type", "")
            writer.writerow(row)

# ══════════════════════════════════════════════════════════════════════════════
#  SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Safari/605.1.15"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_CALOTTERY_API = "https://www.calottery.com/api/DrawGameApi/DrawGamePastDrawResults"
_CALOTTERY_PAGE_SIZE = 20    # API silently caps requests > ~20 per page


def _make_calottery_session():
    """Build a session with a one-time homepage warm-up to get cookies.
    The CA Lottery API behaves better when called with cookies established."""
    if not HAS_REQUESTS:
        return None
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get("https://www.calottery.com/draw-games/powerball", timeout=15)
    except Exception:
        pass
    s.headers.update({
        "Accept": "application/json",
        "Referer": "https://www.calottery.com/draw-games/powerball",
    })
    return s


def _fetch_calottery_page(session, game_id, page, retries=5):
    """Fetch one page of past results. Retries on null/empty responses
    because the API serves null intermittently under load."""
    url = f"{_CALOTTERY_API}/{game_id}/{page}/{_CALOTTERY_PAGE_SIZE}"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            try:
                data = r.json()
            except Exception:
                data = None
            # Empty / null is treated as a soft failure (API quirk under load)
            if data is None or (isinstance(data, dict) and not data.get("PreviousDraws")):
                if attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return None
            return data
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def _parse_calottery_draws(api_draws, game):
    """Convert calottery API records into row dicts.

    For Daily 3 (two draws per date): if both Midday and Evening appear in the
    same batch, the higher DrawNumber is Evening and the lower is Midday.
    For Daily 4 and the big games: one draw per date, no draw_type needed.
    """
    n            = game["white_count"]
    special_max  = game["special_max"]
    has_two      = game.get("has_two_draws_per_day", False)

    raw = []
    for d in api_draws:
        if not d.get("DrawDate"):
            continue
        date_iso = d["DrawDate"][:10]   # "2026-06-12"
        try:
            dt = datetime.strptime(date_iso, "%Y-%m-%d")
        except ValueError:
            continue
        date_str = dt.strftime("%a, %b %d, %Y")

        wn = d.get("WinningNumbers", {}) or {}
        whites = []
        special = None
        for key in sorted(wn.keys(), key=lambda k: int(k) if k.isdigit() else 999):
            entry = wn.get(key)
            if not entry:
                continue
            num_text = entry.get("Number")
            if num_text is None or not num_text.isdigit():
                continue
            num = int(num_text)
            if entry.get("IsSpecial"):
                special = num
            else:
                whites.append(num)
        if len(whites) != n:
            continue
        if special_max > 0 and special is None:
            continue

        raw.append({
            "draw_number": d.get("DrawNumber"),
            "dt":          dt,
            "date_str":    date_str,
            "whites":      whites,
            "special":     special,
        })

    rows = []
    if has_two:
        # Group by date and assign Midday vs Evening from DrawNumber pairs
        from collections import defaultdict
        by_date = defaultdict(list)
        for r in raw:
            by_date[r["date_str"]].append(r)
        for date_str, items in by_date.items():
            items.sort(key=lambda x: x["draw_number"] or 0)
            if len(items) == 1:
                # Solo entry — default to Evening (will get cleaned up next scrape)
                rows.append({
                    "date":      date_str,
                    "dt":        items[0]["dt"],
                    "balls":     items[0]["whites"],   # ordered, NOT sorted
                    "special":   None,
                    "draw_type": "Evening",
                })
            else:
                rows.append({
                    "date":      date_str,
                    "dt":        items[0]["dt"],
                    "balls":     items[0]["whites"],
                    "special":   None,
                    "draw_type": "Midday",
                })
                rows.append({
                    "date":      date_str,
                    "dt":        items[-1]["dt"],
                    "balls":     items[-1]["whites"],
                    "special":   None,
                    "draw_type": "Evening",
                })
    else:
        for r in raw:
            # Big games sort whites ascending; Daily 4 keeps draw order
            balls = sorted(r["whites"]) if special_max > 0 else r["whites"]
            rows.append({
                "date":      r["date_str"],
                "dt":        r["dt"],
                "balls":     balls,
                "special":   r["special"],
                "draw_type": "",
            })
    return rows


def scrape_game(game: dict, existing_rows: list, log_fn=None) -> tuple:
    """Scrape recent draws from the official California Lottery JSON API.

    The API exposes the most-recent ~400 draws per game. The scheduled
    Render scrape grabs the first few pages on each run — more than enough
    to capture any new draws since the last invocation. Full history is
    populated via the standalone backfill installers (run from a laptop).
    """
    if not HAS_REQUESTS:
        return 0, "requests not installed."

    def log(m):
        if log_fn: log_fn(m)

    game_id = game.get("calottery_id")
    if not game_id:
        return 0, f"No calottery_id configured for {game.get('key')}"

    # Build set of (date, draw_type) keys we already have so we can dedup
    existing_keys = set()
    latest_dt = None
    for r in existing_rows:
        dt = parse_date(r["date"])
        if dt:
            key = (dt.date(), r.get("draw_type", "") or "")
            existing_keys.add(key)
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt

    session = _make_calottery_session()
    if session is None:
        return 0, "Could not create HTTP session."

    # How many pages to fetch?  For scheduled incremental scrapes we only
    # need the last few — but on first run (empty DB) we grab everything
    # the API will give us (~20 pages = ~400 draws).
    pages_to_fetch = 25 if latest_dt is None else 5

    log(f"Fetching {game['display_name']} via CA Lottery API (game ID {game_id}) "
        f"— up to {pages_to_fetch} pages")

    all_raw = []
    consecutive_failures = 0
    for page_num in range(1, pages_to_fetch + 1):
        data = _fetch_calottery_page(session, game_id, page_num)
        if data is None:
            consecutive_failures += 1
            log(f"  page {page_num}: failed")
            if consecutive_failures >= 3:
                log(f"  -> stopping after 3 consecutive failures")
                break
            time.sleep(5)
            continue
        consecutive_failures = 0

        draws = data.get("PreviousDraws") or []
        if not draws:
            log(f"  page {page_num}: empty, stopping")
            break
        all_raw.extend(draws)

        # On incremental runs (latest_dt is set), short-circuit once we've
        # walked past dates we already have
        if latest_dt is not None and len(all_raw) >= 10:
            oldest_in_batch = min(d.get("DrawDate", "") for d in draws if d.get("DrawDate"))
            try:
                oldest_dt = datetime.strptime(oldest_in_batch[:10], "%Y-%m-%d")
                if oldest_dt < latest_dt - timedelta(days=14):
                    # We've gone 2 weeks past the newest known row — plenty of overlap
                    break
            except Exception:
                pass

        time.sleep(0.5)   # be polite

    if not all_raw:
        return 0, "No data returned from API."

    parsed = _parse_calottery_draws(all_raw, game)
    log(f"  parsed {len(parsed)} draws from {len(all_raw)} API records")

    # Filter to new rows only
    new_rows = []
    for p in parsed:
        key = (p["dt"].date(), p["draw_type"] or "")
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_rows.append({
            "date":      p["date"],
            "balls":     p["balls"],
            "special":   p["special"],
            "draw_type": p["draw_type"],
        })

    if new_rows:
        new_rows.sort(key=lambda r: parse_date(r["date"]) or datetime.min)
        existing_rows.extend(new_rows)
        save_draws(game, existing_rows)
        _save_scrape_state()
        return (len(new_rows),
                f"Added {len(new_rows)} new draw(s). Total: {len(existing_rows):,}")
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



# ══════════════════════════════════════════════════════════════════════════════
#  BRIER SCORE + BAYESIAN WEIGHT UPDATING
# ══════════════════════════════════════════════════════════════════════════════

def brier_score(prob_dict: dict, actual_numbers: list, number_range) -> float:
    """
    Compute the Brier Score for a probabilistic forecast.

    BS = (1/N) * sum_i (p_i - o_i)^2

    where p_i = predicted probability for number i,
          o_i = 1 if number i was drawn, else 0,
          N   = total numbers in range.

    Lower is better. Random baseline for Powerball (5/69): ~0.061.
    A perfect forecast would score 0.
    """
    actual_set = set(actual_numbers)
    n          = len(list(number_range))
    total = sum(
        (prob_dict.get(i, 0.0) - (1.0 if i in actual_set else 0.0)) ** 2
        for i in number_range
    )
    return total / n if n > 0 else 1.0


def brier_score_baseline(white_count: int, white_max: int) -> float:
    """Brier Score for a uniform random forecast — the baseline to beat."""
    p = white_count / white_max
    return white_count * (1 - p) ** 2 / white_max + (white_max - white_count) * p ** 2 / white_max


def evaluate_method_brier(method_scores: dict, recent_rows: list,
                           white_range, window: int = 50) -> float:
    """
    Average Brier Score for a method over the last `window` draws.
    Lower = better predictive accuracy.
    """
    rows = recent_rows[-window:]
    if not rows:
        return 1.0
    total = 0.0
    # Normalise scores to probabilities
    s_sum = sum(method_scores.values()) + 1e-12
    probs = {k: v / s_sum for k, v in method_scores.items()}
    for r in rows:
        total += brier_score(probs, r["balls"], white_range)
    return total / len(rows)


def bayesian_weight_update(method_scores_list: list,
                           method_names: list,
                           recent_rows: list,
                           prior_weights: list,
                           white_range,
                           window: int = 60) -> list:
    """
    Update method weights using Bayesian inference based on recent Brier Scores.

    For each method compute its average Brier Score over the last `window` draws.
    Convert to a likelihood (lower BS → higher likelihood) and multiply by the
    prior weight.  Normalise to get posterior weights.

    Returns updated weight list in the same order as method_names.
    """
    if not recent_rows or not HAS_NP:
        return prior_weights

    likelihoods = []
    brier_scores_by_method = []
    for scores in method_scores_list:
        bs = evaluate_method_brier(scores, recent_rows, white_range, window)
        brier_scores_by_method.append(bs)
        # Convert Brier Score to likelihood: e^(-k*BS)
        # k=15 gives meaningful differentiation across typical BS range 0.03–0.08
        likelihoods.append(float(np.exp(-15.0 * bs)))

    # Posterior ∝ prior × likelihood
    posteriors = [p * l for p, l in zip(prior_weights, likelihoods)]
    total = sum(posteriors) + 1e-12
    updated = [p / total for p in posteriors]

    # Log the update for transparency
    print("[bayes] Method weight update:")
    for name, bs, old_w, new_w in zip(
            method_names, brier_scores_by_method, prior_weights, updated):
        print(f"  {name:12s}  BS={bs:.5f}  {old_w:.3f} → {new_w:.3f}")

    return updated


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL EVALUATION FRAMEWORK
#  Null model · Log-loss · Calibration · Backtesting · Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

import math as _math

# ── Null hypothesis (uniform) model ──────────────────────────────────────────

def null_model_probs(number_range) -> dict:
    """Uniform probability over all numbers — the baseline to beat."""
    nums = list(number_range)
    p    = 1.0 / len(nums)
    return {n: p for n in nums}


# ── Log-loss ──────────────────────────────────────────────────────────────────

def log_loss_score(prob_dict: dict, actual_numbers: list,
                   number_range, eps: float = 1e-9) -> float:
    """
    Mean log-loss over all numbers in range.
    −(1/N) Σ [ o_i·log(p_i) + (1−o_i)·log(1−p_i) ]
    Lower is better. Perfect = 0, random ≈ log(2) ≈ 0.693.
    """
    actual_set = set(actual_numbers)
    nums       = list(number_range)
    total = 0.0
    for n in nums:
        p  = max(eps, min(1 - eps, prob_dict.get(n, eps)))
        o  = 1.0 if n in actual_set else 0.0
        total += o * _math.log(p) + (1 - o) * _math.log(1 - p)
    return -total / len(nums)


def log_loss_baseline(white_count: int, white_max: int,
                      eps: float = 1e-9) -> float:
    """Log-loss for the uniform null model."""
    p      = white_count / white_max
    p      = max(eps, min(1 - eps, p))
    return -(p * _math.log(p) + (1 - p) * _math.log(1 - p))


# ── Calibration ───────────────────────────────────────────────────────────────

def calibration_curve(prob_dict: dict, draw_rows: list,
                      number_range, n_bins: int = 10) -> list:
    """
    Compute calibration: for numbers predicted at ~p%, did they actually
    appear ~p% of the time?

    Returns list of dicts: {bin_center, mean_predicted, mean_actual, count}
    """
    nums      = list(number_range)
    total_draws = len(draw_rows)
    if total_draws == 0:
        return []

    # Normalise probs
    s = sum(prob_dict.values()) + 1e-12
    probs = {n: prob_dict[n] / s for n in nums}

    # Count actual appearances per number
    appearances = {n: 0 for n in nums}
    for r in draw_rows:
        for b in r["balls"]:
            if b in appearances:
                appearances[b] += 1
    actual_freq = {n: appearances[n] / total_draws for n in nums}

    # Bin by predicted probability
    bins = [[] for _ in range(n_bins)]
    for n in nums:
        p   = probs[n]
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, actual_freq[n]))

    result = []
    for i, b in enumerate(bins):
        if not b:
            continue
        mean_pred   = sum(x[0] for x in b) / len(b)
        mean_actual = sum(x[1] for x in b) / len(b)
        result.append({
            "bin_center":    round((i + 0.5) / n_bins, 3),
            "mean_predicted": round(mean_pred,   6),
            "mean_actual":    round(mean_actual, 6),
            "count":          len(b),
        })
    return result


# ── Rolling backtest ──────────────────────────────────────────────────────────

def rolling_backtest(rows: list, game: dict,
                     train_size: int = 2000,
                     max_evals: int  = 100,
                     step: int       = 10) -> dict:
    """
    Rolling-window backtest:
      - Train on rows[0:train_size]
      - Predict row[train_size]
      - Slide forward by `step`, repeat up to `max_evals` times
      - Measure hit rate, log-loss, Brier score, calibration

    Returns summary dict.
    """
    if not HAS_NP or len(rows) < train_size + 10:
        return {"error": "Not enough data for backtesting"}

    WHITE_MAX   = game["white_max"]
    WHITE_COUNT = game["white_count"]
    WHITE_RANGE = range(1, WHITE_MAX + 1)

    null_probs  = null_model_probs(WHITE_RANGE)

    brier_scores  = []
    logloss_scores= []
    null_briers   = []
    null_loglosses= []
    hit_rates_1   = []   # at least 1 white ball match
    hit_rates_2   = []   # at least 2 white ball matches

    eval_indices = list(range(train_size,
                              min(len(rows), train_size + max_evals * step),
                              step))

    for eval_idx in eval_indices:
        train   = rows[max(0, eval_idx - train_size): eval_idx]
        actual  = rows[eval_idx]

        # Simple frequency model on training window (fast proxy for full engine)
        count   = {}
        for r in train:
            for b in r["balls"]:
                count[b] = count.get(b, 0) + 1
        total_balls = sum(count.values()) + 1e-12
        alpha = 2.0
        probs = {n: (count.get(n, 0) + alpha) /
                    (total_balls + alpha * WHITE_MAX)
                 for n in WHITE_RANGE}
        p_sum = sum(probs.values())
        probs = {n: v / p_sum for n, v in probs.items()}

        actual_balls = actual["balls"]

        # Brier
        bs = brier_score(probs, actual_balls, WHITE_RANGE)
        bs_null = brier_score(null_probs, actual_balls, WHITE_RANGE)
        brier_scores.append(bs)
        null_briers.append(bs_null)

        # Log-loss
        ll = log_loss_score(probs, actual_balls, WHITE_RANGE)
        ll_null = log_loss_score(null_probs, actual_balls, WHITE_RANGE)
        logloss_scores.append(ll)
        null_loglosses.append(ll_null)

        # Hit rate
        pred_top = sorted(WHITE_RANGE, key=lambda n: probs[n], reverse=True)
        pred_set_15 = set(pred_top[:15])   # top-15 as "predictions"
        hits = len(set(actual_balls) & pred_set_15)
        hit_rates_1.append(1 if hits >= 1 else 0)
        hit_rates_2.append(1 if hits >= 2 else 0)

    if not brier_scores:
        return {"error": "No evaluations completed"}

    n = len(brier_scores)
    return {
        "n_evaluations":        n,
        "train_window":         train_size,
        "mean_brier":           round(float(sum(brier_scores)  / n), 6),
        "mean_brier_null":      round(float(sum(null_briers)   / n), 6),
        "mean_logloss":         round(float(sum(logloss_scores)/ n), 6),
        "mean_logloss_null":    round(float(sum(null_loglosses)/ n), 6),
        "hit_rate_1plus":       round(float(sum(hit_rates_1)   / n), 4),
        "hit_rate_2plus":       round(float(sum(hit_rates_2)   / n), 4),
        "brier_improvement_pct":round(
            (sum(null_briers) - sum(brier_scores)) /
            (sum(null_briers) + 1e-12) * 100, 2),
        "logloss_improvement_pct": round(
            (sum(null_loglosses) - sum(logloss_scores)) /
            (sum(null_loglosses) + 1e-12) * 100, 2),
    }


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_confidence(rows: list, number_range,
                          n_bootstrap: int = 500,
                          ci: float = 0.95) -> dict:
    """
    Bootstrap resampling to estimate 95% CI on predicted probabilities.

    Returns {number: (lower_ci, mean, upper_ci)} for the top 20 most
    variable numbers (sorted by CI width).
    """
    if not HAS_NP or len(rows) < 50:
        return {}

    import random as _random
    nums   = list(number_range)
    n_rows = len(rows)
    alpha  = (1 - ci) / 2

    # Store bootstrap probability for each number across resamples
    boot_probs = {n: [] for n in nums}

    for _ in range(n_bootstrap):
        sample = [rows[_random.randint(0, n_rows - 1)] for _ in range(n_rows)]
        counts = {n: 0 for n in nums}
        for r in sample:
            for b in r["balls"]:
                if b in counts:
                    counts[b] += 1
        total = sum(counts.values()) + 1e-12
        for n in nums:
            boot_probs[n].append(counts[n] / total)

    result = {}
    for n in nums:
        bp = sorted(boot_probs[n])
        lo = bp[int(alpha * n_bootstrap)]
        hi = bp[int((1 - alpha) * n_bootstrap)]
        mn = sum(bp) / len(bp)
        result[n] = {
            "lower": round(lo, 6),
            "mean":  round(mn, 6),
            "upper": round(hi, 6),
            "ci_width": round(hi - lo, 6),
        }

    # Return top 20 by CI width (most uncertain numbers)
    top20 = sorted(result.items(), key=lambda x: x[1]["ci_width"], reverse=True)[:20]
    return {str(n): v for n, v in top20}

def analyze_and_predict(rows: list, game: dict, progress_cb=None) -> list:
    """Full 7-method prediction engine. Returns list of 5 ticket dicts."""
    def prog(pct, msg=""):
        if progress_cb: progress_cb(pct, msg)

    if not rows:
        return []

    # ── Ensure chronological order ────────────────────────────────────────
    # Many downstream methods (gap analysis, Markov, decay weighting, the
    # neural-network sequence model, and the era filter below) all assume
    # rows are ordered oldest → newest. The DB may return rows in insert
    # order, which is NOT chronological when backfills were applied piece-
    # meal. We sort defensively here.
    rows = sorted(rows, key=lambda r: parse_date(r.get("date", "")) or datetime.min)

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

    # ── Filter to current-era draws only ──────────────────────────────────
    # Each game went through ball-pool changes over the years. Training on
    # all eras together biases predictions:
    #   - Numbers added in recent eras look "cold" because they had fewer
    #     opportunities to be drawn (Powerball 60-69, only since 2015)
    #   - Numbers removed from the current era distort frequencies if they
    #     still appear in old training rows (old SuperLotto 48-49)
    #
    # era_changes is the list of historical white-ball caps, oldest first,
    # current era LAST. Examples:
    #   Powerball:  [49, 59, 69]   range grew over time
    #   Mega Mil.:  [56, 75, 70]   range grew then shrank (was 75, now 70)
    #   SuperLotto: [49, 47]       range shrank from 49 to 47
    #   Daily 3/4:  [9]            single era, no filter
    #
    # We combine two filters:
    #   (a) GROWING-range filter (forward): find the first draw containing
    #       a ball higher than max(previous caps). Earlier draws may have
    #       come from a smaller-range era and are excluded.
    #   (b) SHRINKING-range filter (backward): find the last draw with
    #       a ball higher than current_cap. Such a draw is from an older,
    #       larger-range era; start the current era right after it.
    # ERA3_START is the LATER (larger) of the two — making sure we satisfy
    # both conditions.
    ERA3_START = 0
    era_changes = game.get("era_changes", [WHITE_MAX])
    if len(era_changes) >= 2 and rows:
        current_cap = era_changes[-1]
        max_previous = max(era_changes[:-1])

        # (a) Growing-range filter: only applies if current cap is larger
        # than some prior cap
        growing_start = 0
        if current_cap > max_previous:
            for i, r in enumerate(rows):
                if any(b > max_previous for b in r["balls"]):
                    growing_start = i
                    break

        # (b) Shrinking-range filter: only applies if some prior cap was
        # larger than the current cap
        shrinking_start = 0
        if max_previous > current_cap:
            for i in range(len(rows) - 1, -1, -1):
                if any(b > current_cap for b in rows[i]["balls"]):
                    shrinking_start = i + 1
                    break

        # Use the later of the two starts
        ERA3_START = max(growing_start, shrinking_start)

        # Safety: if the filter eliminated nearly all data, fall back to
        # using everything (probably an era_changes config issue)
        if ERA3_START >= len(rows) - 20:
            ERA3_START = 0
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
    w_freq  = soft_floor(w_freq, 0.30)
    
    # Only analyze special ball if game has one
    if SPECIAL_MAX > 0:
        s_freq  = {n: (s_count.get(n, 0) + alpha) / (ERA3_DRAWS + alpha * SPECIAL_MAX)
                   for n in SPEC_RANGE}
        s_freq  = soft_floor(s_freq, 0.50)
    else:
        s_freq  = {}  # No special ball for Daily 3/4

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
    m3_s = {}
    if HAS_NP and ERA3_DRAWS > 20:
        for pos in range(WHITE_COUNT):
            col = era3_balls[:, pos].astype(float)
            mu  = col.mean(); sig = col.std() + 1e-6
            tgt = max(1, min(WHITE_MAX, mu + 0.2 * (mu - col[-1])))
            for n in WHITE_RANGE:
                m3_w[n] *= math.exp(-0.5 * ((n - tgt) / (sig * 4.0)) ** 2)
        if SPECIAL_MAX > 0:
            m3_s = {n: 1.0 for n in SPEC_RANGE}
            s_arr = era3_special.astype(float)
            s_tgt = max(1, min(SPECIAL_MAX, s_arr.mean() + 0.2 * (s_arr.mean() - s_arr[-1])))
            s_sig = s_arr.std() * 4.0 + 1e-6
            for n in SPEC_RANGE:
                m3_s[n] *= math.exp(-0.5 * ((n - s_tgt) / s_sig) ** 2)
    m3_w = soft_floor(m3_w, 0.40)
    if SPECIAL_MAX > 0:
        m3_s = soft_floor(m3_s, 0.60)

    prog(38, "Method 4 — Gap …")
    last_w = {}; last_s = {}
    for i, r in enumerate(rows):
        for b in r["balls"]: last_w[b] = i
        if SPECIAL_MAX > 0 and r["special"] is not None:
            last_s[r["special"]] = i
    eg_w = WHITE_MAX / float(WHITE_COUNT)
    m4_w = {n: (lambda g: g if g >= 1 else g ** 2)(
                (N - last_w.get(n, N - int(eg_w))) / eg_w) for n in WHITE_RANGE}
    m4_w = soft_floor(m4_w, 0.20)
    
    if SPECIAL_MAX > 0:
        eg_s = float(SPECIAL_MAX)
        m4_s = {n: (lambda g: g if g >= 1 else g ** 2)(
                    (N - last_s.get(n, N - int(eg_s))) / eg_s) for n in SPEC_RANGE}
        m4_s = soft_floor(m4_s, 0.20)
    else:
        m4_s = {}

    prog(50, "Method 5 — Neural …")
    m5_w = {n: 1.0 for n in WHITE_RANGE}
    m5_s = {}
    if HAS_TORCH and ERA3_DRAWS > 50 and HAS_NP:
        try:
            if SPECIAL_MAX > 0:
                m5_w, m5_s = _torch_predict(era3_balls, era3_special, ERA3_DRAWS,
                                             WHITE_MAX, SPECIAL_MAX)
                m5_w = soft_floor(m5_w, 0.20)
                m5_s = soft_floor(m5_s, 0.20)
            else:
                # Daily 3/4: predict white balls only
                m5_w, _ = _torch_predict(era3_balls, None, ERA3_DRAWS,
                                         WHITE_MAX, 0)
                m5_w = soft_floor(m5_w, 0.20)
        except Exception:
            pass
    if all(v == 1.0 for v in m5_w.values()) and HAS_SK and HAS_NP and ERA3_DRAWS > 50:
        try:
            if SPECIAL_MAX > 0:
                m5_w, m5_s = _sklearn_predict(era3_balls, era3_special, ERA3_DRAWS,
                                               WHITE_MAX, SPECIAL_MAX)
                m5_w = soft_floor(m5_w, 0.20)
                m5_s = soft_floor(m5_s, 0.20)
            else:
                # Daily 3/4: predict white balls only
                m5_w, _ = _sklearn_predict(era3_balls, None, ERA3_DRAWS,
                                           WHITE_MAX, 0)
                m5_w = soft_floor(m5_w, 0.20)
        except Exception:
            pass

    prog(62, "Method 6 — Monte Carlo …")
    combo_w = arith_blend([w_freq, m2, m3_w, m4_w, m5_w],
                          [0.20, 0.15, 0.10, 0.20, 0.35], WHITE_RANGE)
    
    if SPECIAL_MAX > 0:
        combo_s = arith_blend([s_freq, m3_s, m4_s, m5_s],
                              [0.25, 0.15, 0.25, 0.35], SPEC_RANGE)
    else:
        combo_s = {}
    
    tw = sum(combo_w.values()) + 1e-12
    w_probs = {n: combo_w[n] / tw for n in WHITE_RANGE}
    m6_w  = {n: 1e-9 for n in WHITE_RANGE}
    SAMPLES = 120_000
    hal_w   = _halton_sequence(SAMPLES, 2)
    w_cdf   = _make_cdf(w_probs, WHITE_RANGE)
    for i in range(SAMPLES): m6_w[_sample_cdf(w_cdf, hal_w[i])] += 1
    m6_w = soft_floor(m6_w, 0.15)
    
    if SPECIAL_MAX > 0:
        ts = sum(combo_s.values()) + 1e-12
        s_probs = {n: combo_s[n] / ts for n in SPEC_RANGE}
        m6_s  = {n: 1e-9 for n in SPEC_RANGE}
        hal_s   = _halton_sequence(SAMPLES, 3)
        s_cdf   = _make_cdf(s_probs, SPEC_RANGE)
        for i in range(SAMPLES): m6_s[_sample_cdf(s_cdf, hal_s[i])] += 1
        m6_s = soft_floor(m6_s, 0.15)
    else:
        m6_s = {}

    prog(75, "Method 7 — Signature …")
    m7_w = {n: 1.0 for n in WHITE_RANGE}
    if HAS_NP and ERA3_DRAWS > 10:
        decay = 0.9997
        for idx, r in enumerate(era3_balls.tolist() if HAS_NP else era3_balls):
            w = decay ** (ERA3_DRAWS - 1 - idx)
            for b in r: m7_w[b] = m7_w.get(b, 1.0) + w
    m7_w = soft_floor(m7_w, 0.25)

    prog(83, "Synthesising with Bayesian weight update …")

    # ── Bayesian weight update ────────────────────────────────────────────────
    # Base (prior) weights for each white-ball method
    PRIOR_W = [0.15, 0.15, 0.08, 0.17, 0.20, 0.15, 0.10]
    METHOD_NAMES = ["Frequency", "Markov", "Spectral", "Gap",
                    "Neural", "MonteCarlo", "Decay"]

    if HAS_NP and ERA3_DRAWS > 60:
        w_weights = bayesian_weight_update(
            [w_freq, m2, m3_w, m4_w, m5_w, m6_w, m7_w],
            METHOD_NAMES,
            era3_rows,
            PRIOR_W,
            WHITE_RANGE,
            window=min(60, ERA3_DRAWS // 4),
        )
    else:
        w_weights = PRIOR_W

    final_w = arith_blend([w_freq, m2, m3_w, m4_w, m5_w, m6_w, m7_w],
                          w_weights, WHITE_RANGE)
    
    if SPECIAL_MAX > 0:
        final_s = arith_blend([s_freq, m3_s, m4_s, m5_s, m6_s],
                              [0.20, 0.10, 0.25, 0.30, 0.15], SPEC_RANGE)
    else:
        final_s = {}

    # ── Compute and log overall Brier Score ───────────────────────────────────
    if HAS_NP and ERA3_DRAWS > 10:
        f_sum = sum(final_w.values()) + 1e-12
        f_probs = {k: v / f_sum for k, v in final_w.items()}
        bs_recent = evaluate_method_brier(final_w, era3_rows, WHITE_RANGE, 50)
        bs_base   = brier_score_baseline(WHITE_COUNT, WHITE_MAX)
        print(f"[brier] Ensemble BS (last 50 draws): {bs_recent:.5f}  "
              f"(baseline random: {bs_base:.5f}  "
              f"improvement: {(bs_base - bs_recent) / bs_base * 100:.1f}%)")

    prog(90, "Selecting tickets …")
    sorted_w = sorted(WHITE_RANGE, key=lambda n: final_w[n], reverse=True)

    def get_band(n): return (n - 1) // 10

    tickets       = []
    used_sets     = []
    bands_covered = set()

    if HAS_NP and SPECIAL_MAX > 0:
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
            special = random.randint(1, SPECIAL_MAX) if SPECIAL_MAX > 0 else None

        tickets.append({"balls": sorted(chosen), "special": special})

    while len(tickets) < 5:
        tickets.append({
            "balls":   sorted(random.sample(list(WHITE_RANGE), WHITE_COUNT)),
            "special": random.randint(1, SPECIAL_MAX) if SPECIAL_MAX > 0 else None,
        })

    prog(100, "Done.")

    # ── Daily 3/4: position-aware straight prediction ────────────────────────
    # Each position is scored independently so the ticket represents the most
    # probable digit at each slot — this is the optimal "straight" play.
    # Viewers can also use it as a "box" play (any order) if they prefer.
    if game["white_count"] in [3, 4] and game["special_max"] == 0:
        if HAS_NP:
            # Build per-position probability distributions from 4 sources:
            #   1. Position frequency  — how often each digit appears at this slot
            #   2. Markov transition   — what tends to follow the last digit at this slot
            #   3. Spectral reversion  — pull toward positional mean
            #   4. Decay signature     — recency-weighted positional counts
            pos_probs = []  # pos_probs[pos] = np.array of shape (WHITE_MAX,) summing to 1
            for pos in range(WHITE_COUNT):
                col = era3_balls[:, pos].tolist()

                # 1. Positional frequency with Laplace smoothing
                cnt = Counter(col)
                freq_p = np.array(
                    [(cnt.get(n, 0) + 1.0) / (len(col) + WHITE_MAX) for n in WHITE_RANGE],
                    dtype=float)

                # 2. Markov: probability of each digit following the last seen at this pos
                trans = defaultdict(Counter)
                for i in range(len(col) - 1):
                    trans[col[i]][col[i + 1]] += 1
                last = col[-1]
                total = sum(trans[last].values())
                if total:
                    markov_p = np.array(
                        [(trans[last].get(n, 0) + 0.5) / (total + 0.5 * WHITE_MAX)
                         for n in WHITE_RANGE], dtype=float)
                else:
                    markov_p = freq_p.copy()

                # 3. Spectral: Gaussian centred on mean-reversion target
                col_arr = np.array(col, dtype=float)
                mu = col_arr.mean(); sig = col_arr.std() + 1e-6
                tgt = max(WHITE_RANGE.start, min(WHITE_MAX - 1,
                          mu + 0.2 * (mu - col_arr[-1])))
                spec_p = np.array(
                    [math.exp(-0.5 * ((n - tgt) / (sig * 3.0)) ** 2) for n in WHITE_RANGE],
                    dtype=float)

                # 4. Recency decay: weight recent draws more heavily
                decay = 0.9997
                decay_cnt = np.zeros(WHITE_MAX, dtype=float)
                for idx, val in enumerate(col):
                    w = decay ** (len(col) - 1 - idx)
                    decay_cnt[val - WHITE_RANGE.start] += w
                decay_p = decay_cnt + 1e-9

                # Blend the four sources
                blend = 0.30 * freq_p + 0.35 * markov_p + 0.20 * spec_p + 0.15 * decay_p
                blend /= blend.sum()
                pos_probs.append(blend)

            # Generate 5 ordered tickets by sampling each position independently
            digits = list(WHITE_RANGE)
            daily_tickets = []
            used_daily = []
            d_attempt = 0
            while len(daily_tickets) < 5 and d_attempt < 400:
                d_attempt += 1
                # Sample each position independently (WITH replacement within a ticket
                # is implicit — each position draws from the full 0-9 pool separately)
                combo = [int(np.random.choice(digits, p=pos_probs[p]))
                         for p in range(WHITE_COUNT)]
                if combo not in used_daily:
                    used_daily.append(combo)
                    # Store in draw order — do NOT sort
                    daily_tickets.append({"balls": combo, "special": None})
        else:
            # Fallback (no numpy): use overall final_w, sample each position
            digits = list(WHITE_RANGE)
            wts = [final_w.get(n, 1.0) for n in digits]
            daily_tickets = []
            used_daily = []
            for _ in range(5):
                combo = [random.choices(digits, weights=wts, k=1)[0]
                         for _ in range(WHITE_COUNT)]
                if combo not in used_daily:
                    used_daily.append(combo)
                daily_tickets.append({"balls": combo, "special": None})
        return daily_tickets

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
    now       = datetime.now()
    today     = now.date()
    draw_days = game["draw_days"]
    # If today is a draw day and it's before 11 PM (draws are ~11 PM ET),
    # use today's drawing
    if today.weekday() in draw_days and now.hour < 23:
        return datetime(today.year, today.month, today.day).strftime("%a, %b %d, %Y")
    # Otherwise find the next draw day
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


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL VALIDATION FRAMEWORK
#  Null hypothesis · Log-loss · Calibration · Backtesting · Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

import math
import random as _random


# ── Null hypothesis (uniform) ─────────────────────────────────────────────────

def null_hypothesis_probs(number_range) -> dict:
    """Uniform probability for every number — the baseline to beat."""
    nums = list(number_range)
    p    = 1.0 / len(nums)
    return {n: p for n in nums}


def null_hypothesis_brier(white_count: int, white_max: int) -> float:
    """Brier score of the uniform model (analytical)."""
    return brier_score_baseline(white_count, white_max)


# ── Log-loss ──────────────────────────────────────────────────────────────────

def log_loss_score(prob_dict: dict, actual_numbers: list,
                   number_range, eps: float = 1e-9) -> float:
    """
    Average log-loss across all numbers in range.
    For each number i:  loss_i = -(o_i * log(p_i) + (1-o_i) * log(1-p_i))
    Lower is better.  Null model baseline ≈ -log(k/N) for k drawn from N.
    """
    actual_set = set(actual_numbers)
    total = 0.0
    n     = 0
    for i in number_range:
        p  = max(eps, min(1 - eps, prob_dict.get(i, eps)))
        o  = 1.0 if i in actual_set else 0.0
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
        n     += 1
    return total / n if n > 0 else float('inf')


def null_log_loss(white_count: int, white_max: int) -> float:
    """Log-loss of the uniform model (analytical)."""
    p   = white_count / white_max
    eps = 1e-9
    p   = max(eps, min(1 - eps, p))
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


# ── Calibration curve ────────────────────────────────────────────────────────

def calibration_data(prob_dict: dict, draw_rows: list,
                     number_range, n_bins: int = 10) -> list:
    """
    Build calibration data: for each probability bin, compute the actual
    hit frequency.  Well-calibrated model: predicted 10% → ~10% actual.

    Returns list of dicts: {bin_low, bin_high, pred_mean, actual_freq, count}
    """
    bin_edges  = [i / n_bins for i in range(n_bins + 1)]
    bin_preds  = [[] for _ in range(n_bins)]
    bin_actual = [[] for _ in range(n_bins)]

    for row in draw_rows:
        actual_set = set(row["balls"])
        for num in number_range:
            p   = prob_dict.get(num, 0.0)
            o   = 1 if num in actual_set else 0
            b   = min(int(p * n_bins), n_bins - 1)
            bin_preds[b].append(p)
            bin_actual[b].append(o)

    result = []
    for i in range(n_bins):
        if not bin_preds[i]:
            continue
        result.append({
            "bin_low":     round(bin_edges[i], 3),
            "bin_high":    round(bin_edges[i + 1], 3),
            "pred_mean":   round(sum(bin_preds[i]) / len(bin_preds[i]), 4),
            "actual_freq": round(sum(bin_actual[i]) / len(bin_actual[i]), 4),
            "count":       len(bin_preds[i]),
        })
    return result


def calibration_error(calib_data: list) -> float:
    """
    Expected Calibration Error (ECE): weighted mean |pred - actual|.
    Lower is better; 0 = perfect calibration.
    """
    total_count = sum(b["count"] for b in calib_data)
    if total_count == 0:
        return 1.0
    return sum(
        b["count"] / total_count * abs(b["pred_mean"] - b["actual_freq"])
        for b in calib_data
    )


# ── Backtesting framework ────────────────────────────────────────────────────

def _simple_frequency_probs(rows: list, number_range,
                              alpha: float = 2.0) -> dict:
    """Fast frequency model used inside backtesting (avoids full 7-method run)."""
    from collections import Counter
    counts = Counter(b for r in rows for b in r["balls"])
    n      = len(rows) * 5
    return soft_floor(
        {num: (counts.get(num, 0) + alpha) / (n + alpha * len(list(number_range)))
         for num in number_range},
        0.3
    )


def backtest(rows: list, game: dict,
             train_size: int = 500,
             max_windows: int = 200) -> dict:
    """
    Rolling-window backtest.

    Train on draws[i : i+train_size], predict draw[i+train_size],
    slide forward by 1.  Uses the fast frequency model for speed
    (full 7-method per window would take hours).

    Returns aggregate metrics + per-window results.
    """
    WHITE_MAX   = game["white_max"]
    WHITE_COUNT = game["white_count"]
    WHITE_RANGE = range(1, WHITE_MAX + 1)
    NULL_PROBS  = null_hypothesis_probs(WHITE_RANGE)

    n       = len(rows)
    if n < train_size + 10:
        return {"error": f"Need at least {train_size + 10} rows, have {n}"}

    # Limit windows for performance
    total_windows = min(n - train_size, max_windows)
    step          = max(1, (n - train_size) // max_windows)

    results = []
    for i in range(0, total_windows * step, step):
        if i + train_size >= n:
            break
        train = rows[i : i + train_size]
        test  = rows[i + train_size]

        probs      = _simple_frequency_probs(train, WHITE_RANGE)
        bs_model   = brier_score(probs, test["balls"], WHITE_RANGE)
        bs_null    = brier_score(NULL_PROBS, test["balls"], WHITE_RANGE)
        ll_model   = log_loss_score(probs, test["balls"], WHITE_RANGE)
        ll_null    = log_loss_score(NULL_PROBS, test["balls"], WHITE_RANGE)

        # Hit rate: how many of top-k predicted numbers were actually drawn
        sorted_nums = sorted(WHITE_RANGE, key=lambda x: probs[x], reverse=True)
        top_k       = set(sorted_nums[:WHITE_COUNT * 3])   # top 15 / 21 candidates
        hits        = len(top_k & set(test["balls"]))

        results.append({
            "window_start": i,
            "draw_date":    test["date"],
            "brier_model":  round(bs_model, 6),
            "brier_null":   round(bs_null, 6),
            "logloss_model":round(ll_model, 6),
            "logloss_null": round(ll_null, 6),
            "hits_top15":   hits,
        })

    if not results:
        return {"error": "No windows completed"}

    bs_m  = [r["brier_model"]   for r in results]
    bs_n  = [r["brier_null"]    for r in results]
    ll_m  = [r["logloss_model"] for r in results]
    ll_n  = [r["logloss_null"]  for r in results]
    hits  = [r["hits_top15"]    for r in results]

    n_res = len(results)
    summary = {
        "windows":             n_res,
        "train_size":          train_size,
        "avg_brier_model":     round(sum(bs_m) / n_res, 6),
        "avg_brier_null":      round(sum(bs_n) / n_res, 6),
        "brier_improvement":   round((sum(bs_n) - sum(bs_m)) / sum(bs_n) * 100, 2),
        "avg_logloss_model":   round(sum(ll_m) / n_res, 6),
        "avg_logloss_null":    round(sum(ll_n) / n_res, 6),
        "logloss_improvement": round((sum(ll_n) - sum(ll_m)) / sum(ll_n) * 100, 2),
        "avg_hits_top15":      round(sum(hits) / n_res, 3),
        "expected_hits_random":round(WHITE_COUNT * 3 * WHITE_COUNT / WHITE_MAX, 3),
        "windows_model_beat_null": sum(
            1 for b_m, b_n in zip(bs_m, bs_n) if b_m < b_n
        ),
        "win_rate_pct": round(
            sum(1 for b_m, b_n in zip(bs_m, bs_n) if b_m < b_n) / n_res * 100, 1
        ),
    }
    return {"summary": summary, "windows": results[-20:]}  # last 20 windows


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_confidence_intervals(prob_dict: dict, draw_rows: list,
                                    number_range,
                                    n_bootstrap: int = 500,
                                    ci_level: float = 0.95) -> dict:
    """
    Bootstrap CI for Brier Score and Log-Loss.

    Resample draw_rows with replacement n_bootstrap times,
    recompute metrics each time, then take percentile intervals.

    Returns: {brier: {mean, lower, upper}, logloss: {mean, lower, upper}}
    """
    if not draw_rows or not HAS_NP:
        return {}

    brier_scores  = []
    logloss_scores = []

    for _ in range(n_bootstrap):
        sample = [draw_rows[_random.randint(0, len(draw_rows) - 1)]
                  for _ in range(len(draw_rows))]
        bs_vals = [brier_score(prob_dict, r["balls"], number_range)
                   for r in sample]
        ll_vals = [log_loss_score(prob_dict, r["balls"], number_range)
                   for r in sample]
        brier_scores.append(sum(bs_vals) / len(bs_vals))
        logloss_scores.append(sum(ll_vals) / len(ll_vals))

    alpha   = (1 - ci_level) / 2
    lo_idx  = int(alpha * n_bootstrap)
    hi_idx  = int((1 - alpha) * n_bootstrap)

    brier_sorted  = sorted(brier_scores)
    ll_sorted     = sorted(logloss_scores)

    return {
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "brier": {
            "mean":  round(sum(brier_scores) / n_bootstrap, 6),
            "lower": round(brier_sorted[lo_idx], 6),
            "upper": round(brier_sorted[hi_idx], 6),
        },
        "logloss": {
            "mean":  round(sum(logloss_scores) / n_bootstrap, 6),
            "lower": round(ll_sorted[lo_idx], 6),
            "upper": round(ll_sorted[hi_idx], 6),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION A — TEST FOR NON-RANDOMNESS
#  Chi-square · Entropy · Serial correlation · Runs test
# ══════════════════════════════════════════════════════════════════════════════

def chi_square_goodness_of_fit(rows: list, number_range) -> dict:
    """
    Chi-square test: do observed frequencies deviate from uniform?
    H0: every number equally likely (perfect randomness).
    Low p-value → evidence of non-randomness (but not which direction).

    χ² = Σ (O_i - E_i)² / E_i
    """
    from collections import Counter
    nums        = list(number_range)
    N           = len(nums)
    total_draws = sum(len(r["balls"]) for r in rows)
    expected    = total_draws / N          # uniform expectation per number
    observed    = Counter(b for r in rows for b in r["balls"])

    chi2 = sum(
        (observed.get(n, 0) - expected) ** 2 / expected
        for n in nums
    )
    df = N - 1

    # p-value via regularised incomplete gamma (scipy if available, else approx)
    p_value = None
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1 - chi2_dist.cdf(chi2, df))
    except Exception:
        # Wilson-Hilferty normal approximation
        try:
            z = ((chi2 / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
            # Φ complement (rough)
            p_value = max(0.0, min(1.0, 0.5 * math.erfc(z / math.sqrt(2))))
        except Exception:
            p_value = None

    # Most over/under-represented numbers
    deviations = sorted(
        [(n, observed.get(n,0), round((observed.get(n,0)-expected)/expected*100,1))
         for n in nums],
        key=lambda x: abs(x[2]), reverse=True
    )

    return {
        "chi2":            round(chi2, 4),
        "df":              df,
        "p_value":         round(p_value, 6) if p_value is not None else None,
        "expected_per_num":round(expected, 2),
        "reject_h0_5pct":  (p_value < 0.05) if p_value is not None else None,
        "interpretation":  (
            "Significant deviation from uniform — non-randomness detected"
            if p_value is not None and p_value < 0.05
            else "No significant deviation — consistent with randomness"
        ),
        "top_overdue":     deviations[:5],    # most under-represented
        "top_frequent":    deviations[-5:],   # most over-represented
    }


def entropy_analysis(rows: list, number_range) -> dict:
    """
    Shannon entropy of the observed frequency distribution.
    H = -Σ p_i * log2(p_i)

    Max entropy = log2(N) for uniform distribution.
    Lower entropy → more skewed = more structure (potentially exploitable).
    Entropy ratio close to 1.0 = very random.
    """
    from collections import Counter
    nums  = list(number_range)
    N     = len(nums)
    total = sum(len(r["balls"]) for r in rows) or 1
    counts = Counter(b for r in rows for b in r["balls"])

    h = 0.0
    for n in nums:
        p = counts.get(n, 0) / total
        if p > 0:
            h -= p * math.log2(p)

    max_h   = math.log2(N)
    ratio   = h / max_h if max_h > 0 else 1.0

    return {
        "entropy":       round(h, 6),
        "max_entropy":   round(max_h, 6),
        "entropy_ratio": round(ratio, 6),
        "interpretation": (
            "Near-maximum entropy — draw system appears well-randomised"
            if ratio > 0.98
            else "Below-maximum entropy — mild structure detected; may be exploitable"
            if ratio > 0.95
            else "Notable entropy deficit — structure present in historical draws"
        ),
    }


def serial_correlation(rows: list, lag: int = 1) -> dict:
    """
    Serial (autocorrelation) of the draw sum series.
    Tests whether consecutive draw sums are correlated (non-independent).
    Near-zero correlation → consistent with independence.
    """
    if not HAS_NP or len(rows) < lag + 20:
        return {"error": "Insufficient data"}

    sums = np.array([sum(r["balls"]) for r in rows], dtype=float)
    n    = len(sums)
    mu   = sums.mean()
    var  = ((sums - mu) ** 2).mean()
    if var == 0:
        return {"correlation": 0.0}

    # Pearson correlation between series and lagged series
    cov  = ((sums[lag:] - mu) * (sums[:-lag] - mu)).mean()
    corr = cov / var

    # Ljung-Box statistic for lag 1
    q  = n * (n + 2) * (corr ** 2) / (n - lag)
    p_lb = None
    try:
        from scipy.stats import chi2 as chi2_dist
        p_lb = float(1 - chi2_dist.cdf(q, df=lag))
    except Exception:
        pass

    return {
        "lag":            lag,
        "correlation":    round(float(corr), 6),
        "ljung_box_q":    round(float(q), 4),
        "p_value":        round(p_lb, 6) if p_lb is not None else None,
        "interpretation": (
            "Significant serial correlation — draws may not be fully independent"
            if p_lb is not None and p_lb < 0.05
            else "No significant serial correlation — draws appear independent"
        ),
    }


def runs_test(rows: list) -> dict:
    """
    Runs test on the draw-sum series (above/below median).
    Tests for too many or too few runs — detecting non-random patterns.
    """
    if len(rows) < 20:
        return {"error": "Insufficient data"}

    sums   = [sum(r["balls"]) for r in rows]
    median = sorted(sums)[len(sums) // 2]
    signs  = [1 if s >= median else -1 for s in sums]

    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1

    n1 = signs.count(1)
    n2 = signs.count(-1)
    n  = n1 + n2

    if n1 == 0 or n2 == 0:
        return {"runs": runs, "interpretation": "All values on same side of median"}

    # Expected runs and variance under H0
    mu_r  = 2 * n1 * n2 / n + 1
    var_r = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z     = (runs - mu_r) / math.sqrt(var_r) if var_r > 0 else 0.0

    p_val = None
    try:
        p_val = float(2 * (1 - 0.5 * math.erfc(-abs(z) / math.sqrt(2))))
    except Exception:
        pass

    return {
        "runs":          runs,
        "expected_runs": round(mu_r, 2),
        "z_score":       round(z, 4),
        "p_value":       round(p_val, 6) if p_val is not None else None,
        "interpretation": (
            "Significant non-randomness in run structure"
            if p_val is not None and p_val < 0.05
            else "Run structure consistent with randomness"
        ),
    }


def section_a_randomness_tests(rows: list, game: dict) -> dict:
    """
    Run all Section A tests. Returns a composite non-randomness score
    and individual test results.
    """
    WHITE_RANGE = range(1, game["white_max"] + 1)

    chi    = chi_square_goodness_of_fit(rows, WHITE_RANGE)
    ent    = entropy_analysis(rows, WHITE_RANGE)
    serial = serial_correlation(rows, lag=1)
    runs   = runs_test(rows)

    # Composite non-randomness score 0-100
    # (higher = more evidence of non-randomness = more potentially exploitable)
    score = 0.0
    if chi.get("reject_h0_5pct"):         score += 30
    if ent.get("entropy_ratio", 1) < 0.98: score += 20 * (1 - ent.get("entropy_ratio",1)/0.98)
    if serial.get("p_value") is not None and serial["p_value"] < 0.05: score += 25
    if runs.get("p_value") is not None    and runs["p_value"]   < 0.05: score += 25

    score = min(100, score)

    return {
        "non_randomness_score": round(score, 1),
        "recommendation": (
            "Strong evidence of structure — full model ensemble justified"
            if score >= 40
            else "Mild structure detected — model may add modest value over random"
            if score >= 15
            else "No detectable structure — draw system appears truly random; "
                 "predictions should be treated as educated guesses only"
        ),
        "chi_square":         chi,
        "entropy":            ent,
        "serial_correlation": serial,
        "runs_test":          runs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION B — BAYESIAN HIERARCHICAL MODEL (Dirichlet prior)
#  More academically rigorous than raw frequency counts.
# ══════════════════════════════════════════════════════════════════════════════

def dirichlet_bayesian_probs(rows: list, number_range,
                              alpha: float = 1.0) -> dict:
    """
    Bayesian frequency model with Dirichlet(α) prior.
    Posterior mean: p_i = (count_i + α) / (total + α * N)

    α = 1.0  → uniform (Laplace smoothing)
    α = 0.5  → Jeffreys prior (more conservative)
    α → 0    → maximum likelihood (no smoothing)

    More rigorous than raw counts: uncertainty is modelled, not ignored.
    """
    from collections import Counter
    nums   = list(number_range)
    N      = len(nums)
    counts = Counter(b for r in rows for b in r["balls"])
    total  = sum(counts.values())

    probs = {
        n: (counts.get(n, 0) + alpha) / (total + alpha * N)
        for n in nums
    }

    # Posterior concentration: higher = more confident
    alpha_post = {n: counts.get(n, 0) + alpha for n in nums}
    alpha_total = sum(alpha_post.values())

    # Posterior variance for each number
    variances = {
        n: (alpha_post[n] * (alpha_total - alpha_post[n]))
           / (alpha_total ** 2 * (alpha_total + 1))
        for n in nums
    }

    return {
        "probs":      probs,
        "variances":  variances,
        "alpha_prior": alpha,
        "n_draws":    len(rows),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHT OPTIMISATION
#  Learn weights from historical validation rather than setting subjectively.
# ══════════════════════════════════════════════════════════════════════════════

def optimize_weights(method_scores_list: list, rows: list,
                     number_range,
                     method_names: list,
                     window: int = 100,
                     n_grid: int = 5) -> list:
    """
    Grid-search weight optimisation.
    Minimise average Brier Score over the last `window` draws.

    For speed uses a coarse grid then refines around the best.
    Falls back to Bayesian posterior weights if grid search fails.

    w* = argmin_{w} (1/T) Σ_t BS(Σ_k w_k * score_k^t, actual_t)
    """
    if not HAS_NP or len(rows) < window + 10 or len(method_scores_list) < 2:
        # Fallback: uniform weights
        n = len(method_scores_list)
        return [1.0 / n] * n

    test_rows = rows[-window:]
    nums      = list(number_range)
    K         = len(method_scores_list)

    # Normalise each method's scores to probabilities once
    norm_probs = []
    for scores in method_scores_list:
        s = sum(scores.values()) + 1e-12
        norm_probs.append({n: scores.get(n, 0) / s for n in nums})

    best_bs   = float('inf')
    best_w    = [1.0 / K] * K

    # Coarse grid: each weight in {0, 0.25, 0.5, 0.75, 1.0}, normalised
    grid_vals = [i / (n_grid - 1) for i in range(n_grid)]

    import itertools
    # Limit combinations for performance: use random sampling for K > 4
    if K <= 4:
        combos = list(itertools.product(grid_vals, repeat=K))
    else:
        _rng = _random.Random(42)
        combos = [
            [_rng.choice(grid_vals) for _ in range(K)]
            for _ in range(2000)
        ]
        combos.append([1.0/K]*K)   # always include uniform

    for combo in combos:
        total = sum(combo)
        if total < 1e-9:
            continue
        w = [c / total for c in combo]

        # Blend and score
        bs_total = 0.0
        for row in test_rows:
            blended = {n: sum(w[k] * norm_probs[k][n] for k in range(K)) for n in nums}
            bs_total += brier_score(blended, row["balls"], number_range)
        avg_bs = bs_total / len(test_rows)

        if avg_bs < best_bs:
            best_bs = avg_bs
            best_w  = w

    print(f"[weight-opt] Best Brier: {best_bs:.6f}  weights: "
          + " ".join(f"{n}={w:.3f}" for n,w in zip(method_names, best_w)))

    return best_w


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION C — COMPARE TO CHANCE
# ══════════════════════════════════════════════════════════════════════════════

def compare_to_chance(prob_dict: dict, rows: list,
                      game: dict, window: int = 100) -> dict:
    """
    Head-to-head comparison: ensemble model vs null hypothesis (uniform).
    Reports whether the model beats random on Brier score, log-loss,
    and hit rate over the last `window` draws.
    """
    WHITE_RANGE = range(1, game["white_max"] + 1)
    WHITE_COUNT = game["white_count"]
    WHITE_MAX   = game["white_max"]

    test_rows  = rows[-window:] if len(rows) >= window else rows
    if not test_rows:
        return {"error": "No data"}

    null_probs = null_hypothesis_probs(WHITE_RANGE)
    # Normalise model probs
    s = sum(prob_dict.values()) + 1e-12
    model_probs = {k: v/s for k,v in prob_dict.items()}

    bs_model_vals, bs_null_vals   = [], []
    ll_model_vals, ll_null_vals   = [], []
    hits_model, hits_null         = [], []
    TOP_K = WHITE_COUNT * 3  # top 15/21 candidates

    sorted_model = sorted(WHITE_RANGE, key=lambda x: model_probs.get(x,0), reverse=True)
    sorted_null  = list(WHITE_RANGE)   # uniform = arbitrary order
    top_model    = set(sorted_model[:TOP_K])
    top_null     = set(sorted_null[:TOP_K])

    for row in test_rows:
        actual = row["balls"]
        bs_model_vals.append(brier_score(model_probs, actual, WHITE_RANGE))
        bs_null_vals.append( brier_score(null_probs,  actual, WHITE_RANGE))
        ll_model_vals.append(log_loss_score(model_probs, actual, WHITE_RANGE))
        ll_null_vals.append( log_loss_score(null_probs,  actual, WHITE_RANGE))
        hits_model.append(len(top_model & set(actual)))
        hits_null.append( len(top_null  & set(actual)))

    n   = len(test_rows)
    avg = lambda lst: sum(lst)/len(lst) if lst else 0

    model_wins_bs = sum(1 for m,nu in zip(bs_model_vals, bs_null_vals) if m < nu)
    model_wins_ll = sum(1 for m,nu in zip(ll_model_vals, ll_null_vals) if m < nu)

    return {
        "window":              n,
        "brier_model":         round(avg(bs_model_vals), 6),
        "brier_null":          round(avg(bs_null_vals),  6),
        "brier_improvement":   round((avg(bs_null_vals)-avg(bs_model_vals))/avg(bs_null_vals)*100, 2),
        "logloss_model":       round(avg(ll_model_vals), 6),
        "logloss_null":        round(avg(ll_null_vals),  6),
        "logloss_improvement": round((avg(ll_null_vals)-avg(ll_model_vals))/avg(ll_null_vals)*100, 2),
        "avg_hits_model":      round(avg(hits_model), 3),
        "avg_hits_null":       round(avg(hits_null),  3),
        "expected_hits_chance":round(TOP_K * WHITE_COUNT / WHITE_MAX, 3),
        "model_beats_null_brier_pct":  round(model_wins_bs / n * 100, 1),
        "model_beats_null_logloss_pct":round(model_wins_ll / n * 100, 1),
        "verdict": (
            f"Model beats random on {model_wins_bs/n*100:.0f}% of draws (Brier). "
            + ("Meaningful edge detected." if model_wins_bs/n > 0.55
               else "No reliable edge over random baseline.")
        ),
    }
