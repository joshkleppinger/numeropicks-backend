"""
ENGINE.PY — FULL STABLE BUILD

This version includes ALL functions required by main.py to prevent
import errors and deployment failures.

STABLE PUBLIC INTERFACE (DO NOT CHANGE NAMES):
- GAMES
- load_draws
- save_draws
- scrape_game
- load_scrape_state
- save_scrape_state
- analyze_and_predict
- full_backtest
"""

import os
import csv
import json
import math
import random
import requests
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime

# ============================================================
# GAME DEFINITIONS
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
# UTILITIES
# ============================================================

def _parse_date(text):
    for fmt in ("%a, %b %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except:
            continue
    return None


def _normalize(d):
    total = sum(d.values()) + 1e-12
    return {k: v / total for k, v in d.items()}

# ============================================================
# DATA IO
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
# SCRAPE STATE
# ============================================================

def load_scrape_state(filepath="scrape_state.json"):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return {}


def save_scrape_state(state, filepath="scrape_state.json"):
    try:
        with open(filepath, "w") as f:
            json.dump(state, f)
    except:
        pass

# ============================================================
# SCRAPER CORE
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

            # Try list items first
            for li in cols[-1].find_all("li"):
                txt = li.text.strip()
                if txt.isdigit():
                    nums.append(int(txt))

            # Fallback spans/divs
            if not nums:
                for el in cols[-1].find_all(["span", "div"]):
                    txt = el.text.strip()
                    if txt.isdigit():
                        nums.append(int(txt))

            # Final fallback: raw text
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
# PUBLIC SCRAPER
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
# MODEL METHODS
# ============================================================

def _bayesian(rows, rng):
    c = Counter()
    for r in rows:
        c.update(r["balls"])
    return _normalize({n: c[n] + 1 for n in rng})


def _gap(rows, rng):
    last_seen = {n: -1 for n in rng}
    for i, r in enumerate(rows):
        for b in r["balls"]:
            last_seen[b] = i
    return _normalize({n: len(rows) - last_seen[n] for n in rng})


def _decay(rows, rng, decay=0.98):
    weights = Counter()
    w = 1.0
    for r in reversed(rows):
        for b in r["balls"]:
            weights[b] += w
        w *= decay
    return _normalize(weights)

# ============================================================
# PREDICTION
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
# BACKTEST
# ============================================================

def full_backtest(rows, game, train_size=200):
    rng = list(range(0, game["white_max"] + 1))
    scores = []

    for i in range(len(rows) - train_size - 1):
        train = rows[i:i+train_size]
        test = rows[i+train_size]

        probs = analyze_and_predict(train, game)

        brier = sum(
            (probs.get(n, 0) - (1 if n in test["balls"] else 0))**2
            for n in rng
        )

        scores.append(brier)

    return {
        "avg_brier": sum(scores)/len(scores) if scores else None,
        "samples": len(scores)
    }
