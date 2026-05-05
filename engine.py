import math
import random
import csv
import os
import requests
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime

# ─────────────────────────────────────────────
# GAME DEFINITIONS
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# DATE PARSER
# ─────────────────────────────────────────────

def parse_date(text):
    try:
        return datetime.strptime(text.strip(), "%a, %b %d, %Y")
    except:
        try:
            return datetime.strptime(text.strip(), "%b %d, %Y")
        except:
            return None

# ─────────────────────────────────────────────
# SCRAPER (FIXED)
# ─────────────────────────────────────────────

def scrape_lottery_net(url, game):
    print(f"Fetching {url} ...")

    res = requests.get(url, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")

    results = []

    for row in soup.select("table tbody tr"):
        try:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            # Parse date
            date_text = cols[0].get_text(strip=True)
            dt = parse_date(date_text)
            if not dt:
                continue

            # Extract numbers (robust)
            nums = []

            # Try <li>
            for li in cols[-1].find_all("li"):
                txt = li.get_text(strip=True)
                if txt.isdigit():
                    nums.append(int(txt))

            # Fallback: spans/divs
            if not nums:
                for el in cols[-1].find_all(["span", "div"]):
                    txt = el.get_text(strip=True)
                    if txt.isdigit():
                        nums.append(int(txt))

            # Final fallback: raw text split
            if not nums:
                raw = cols[-1].get_text(" ", strip=True)
                for part in raw.split():
                    if part.isdigit():
                        nums.append(int(part))

            if len(nums) != game["white_count"]:
                continue

            if not all(0 <= n <= game["white_max"] for n in nums):
                continue

            results.append({
                "date_str": dt.strftime("%a, %b %d, %Y"),
                "dt": dt,
                "balls": nums,
                "special": None
            })

        except Exception:
            continue

    print(f"  -> Parsed {len(results)} draws")
    return results

# ─────────────────────────────────────────────
# DATA IO
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def normalize(d):
    s = sum(d.values()) + 1e-12
    return {k: v / s for k, v in d.items()}

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

def bayesian_frequency(rows, number_range):
    counts = Counter()
    for r in rows:
        counts.update(r["balls"])
    return normalize({n: counts[n] + 1 for n in number_range})


def gap_analysis(rows, number_range):
    last_seen = {n: -1 for n in number_range}
    for i, r in enumerate(rows):
        for b in r["balls"]:
            last_seen[b] = i
    gaps = {n: len(rows) - last_seen[n] for n in number_range}
    return normalize(gaps)


def decay_model(rows, number_range, decay=0.98):
    weights = Counter()
    w = 1.0
    for r in reversed(rows):
        for b in r["balls"]:
            weights[b] += w
        w *= decay
    return normalize(weights)


def random_model(number_range):
    return {n: 1 / len(number_range) for n in number_range}

# ─────────────────────────────────────────────
# MONTE CARLO (SMOOTHING)
# ─────────────────────────────────────────────

def monte_carlo_smoothing(probs, number_range, k, n_sim=20000):
    counts = Counter()
    numbers = list(number_range)
    weights = [probs[n] for n in numbers]

    for _ in range(n_sim):
        picks = random.choices(numbers, weights=weights, k=k)
        counts.update(picks)

    return normalize(counts)

# ─────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────

def analyze_and_predict(rows, game):
    number_range = list(range(0, game["white_max"] + 1))

    m1 = bayesian_frequency(rows, number_range)
    m2 = gap_analysis(rows, number_range)
    m3 = decay_model(rows, number_range)

    blended = {
        n: 0.4 * m1[n] + 0.3 * m2[n] + 0.3 * m3[n]
        for n in number_range
    }

    probs = monte_carlo_smoothing(
        blended,
        number_range,
        game["white_count"]
    )

    return probs

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def brier_score(probs, actual, number_range):
    actual_set = set(actual)
    return sum(
        (probs.get(n, 0) - (1 if n in actual_set else 0)) ** 2
        for n in number_range
    )


def calibration_error(probs, rows, number_range):
    error = 0
    count = 0

    for n in number_range:
        p = probs.get(n, 0)
        actual_freq = sum(1 for r in rows if n in r["balls"]) / len(rows)
        error += abs(p - actual_freq)
        count += 1

    return error / (count + 1e-12)

# ─────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────

def full_backtest(rows, game, train_size=200, windows=20):
    number_range = list(range(0, game["white_max"] + 1))

    brier_model = []
    brier_null = []
    calib = []

    for i in range(min(windows, len(rows) - train_size - 1)):
        train = rows[i:i + train_size]
        test = rows[i + train_size]

        probs = analyze_and_predict(train, game)
        null = random_model(number_range)

        brier_model.append(
            brier_score(probs, test["balls"], number_range)
        )
        brier_null.append(
            brier_score(null, test["balls"], number_range)
        )

        calib.append(
            calibration_error(probs, train, number_range)
        )

    return {
        "avg_model_brier": sum(brier_model) / len(brier_model),
        "avg_null_brier": sum(brier_null) / len(brier_null),
        "calibration_error": sum(calib) / len(calib)
    }
