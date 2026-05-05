"""
STABLE ENGINE INTERFACE CONTRACT
--------------------------------
This file defines a STRICT, STABLE interface between your API (main.py)
and the prediction/scraping engine.

DO NOT change function names or signatures below unless you also update main.py.

REQUIRED PUBLIC FUNCTIONS:
- GAMES (dict)
- load_draws(filepath)
- save_draws(filepath, rows)
- scrape_game(game_key, year)
- analyze_and_predict(rows, game)
- full_backtest(rows, game)

Everything else is internal and safe to modify.
"""

import math
import random
import csv
import os
import requests
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime

# ============================================================
# GAME DEFINITIONS (API DEPENDS ON THIS)
# ============================================================

GAMES = {
    "daily3_evening": {
        "name": "Daily 3 Evening",
        "white_max": 9,
        "white_count": 3,
        "special_max": 0,
        "special_count": 0
    },
    "daily3_midday": {
        "name": "Daily 3 Midday",
        "white_max": 9,
        "white_count": 3,
        "special_max": 0,
        "special_count": 0
    },
    "daily4": {
        "name": "Daily 4",
        "white_max": 9,
        "white_count": 4,
        "special_max": 0,
        "special_count": 0
    }
}

# ============================================================
# UTILITIES (INTERNAL)
# ============================================================

def _parse_date(text):
    for fmt in ("%a, %b %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except:
            continue
    return None


def _normalize(d):
    s = sum(d.values()) + 1e-12
    return {k: v / s for k, v in d.items()}

# ============================================================
# DATA IO (STABLE)
# ============================================================

def load_draws(filepath):
    if not os.path.exists(filepath):
        return []

    rows = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            try:
                nums = list(map(int, row[1:]))
                rows.append({"balls": nums})
            except:
                continue

    return rows


def save_draws(filepath, rows):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "n1", "n2", "n3", "n4"])

        for i, r in enumerate(rows):
            writer.writerow([i] + r["balls"])

# ============================================================
# SCRAPER (INTERNAL CORE)
# ============================================================

def _scrape_lottery_net(url, game):
    res = requests.get(url, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")

    results = []

    for row in soup.select("table tbody tr"):
        try:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            dt = _parse_date(cols[0].get_text(strip=True))
            if not dt:
                continue

            nums = []

            for li in cols[-1].find_all("li"):
                if li.text.strip().isdigit():
                    nums.append(int(li.text.strip()))

            if not nums:
                for el in cols[-1].find_all(["span", "div"]):
                    txt = el.text.strip()
                    if txt.isdigit():
                        nums.append(int(txt))

            if not nums:
                raw = cols[-1].get_text(" ", strip=True)
                nums = [int(x) for x in raw.split() if x.isdigit()]

            if len(nums) != game["white_count"]:
                continue

            results.append({
                "date_str": dt.strftime("%a, %b %d, %Y"),
                "dt": dt,
                "balls": nums,
                "special": None
            })

        except:
            continue

    return results

# ============================================================
# PUBLIC SCRAPER (STABLE API FUNCTION)
# ============================================================

def scrape_game(game_key, year):
    if game_key not in GAMES:
        raise ValueError(f"Unknown game: {game_key}")

    if game_key == "daily3_evening":
        url = f"https://www.lottery.net/california/daily-3-evening/numbers/{year}"
    elif game_key == "daily3_midday":
        url = f"https://www.lottery.net/california/daily-3-midday/numbers/{year}"
    elif game_key == "daily4":
        url = f"https://www.lottery.net/california/daily-4/numbers/{year}"
    else:
        raise ValueError("Unsupported game")

    rows = _scrape_lottery_net(url, GAMES[game_key])
    rows.sort(key=lambda x: x["dt"])
    return rows

# ============================================================
# MODELS (INTERNAL)
# ============================================================

def _bayesian(rows, rng):
    c = Counter()
    for r in rows:
        c.update(r["balls"])
    return _normalize({n: c[n] + 1 for n in rng})


def _gap(rows, rng):
    last = {n: -1 for n in rng}
    for i, r in enumerate(rows):
        for b in r["balls"]:
            last[b] = i
    return _normalize({n: len(rows) - last[n] for n in rng})


def _decay(rows, rng, d=0.98):
    w = Counter()
    weight = 1
    for r in reversed(rows):
        for b in r["balls"]:
            w[b] += weight
        weight *= d
    return _normalize(w)

# ============================================================
# PREDICTION (STABLE API FUNCTION)
# ============================================================

def analyze_and_predict(rows, game):
    rng = list(range(0, game["white_max"] + 1))

    m1 = _bayesian(rows, rng)
    m2 = _gap(rows, rng)
    m3 = _decay(rows, rng)

    return {
        n: 0.4*m1[n] + 0.3*m2[n] + 0.3*m3[n]
        for n in rng
    }

# ============================================================
# BACKTEST (STABLE API FUNCTION)
# ============================================================

def full_backtest(rows, game, train_size=200):
    rng = list(range(0, game["white_max"] + 1))

    scores = []

    for i in range(len(rows) - train_size - 1):
        train = rows[i:i+train_size]
        test = rows[i+train_size]

        probs = analyze_and_predict(train, game)

        score = sum(
            (probs.get(n, 0) - (1 if n in test["balls"] else 0))**2
            for n in rng
        )

        scores.append(score)

    return {
        "avg_brier": sum(scores)/len(scores) if scores else None
    }
