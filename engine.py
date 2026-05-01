# FULL LOTTERY PREDICTION ENGINE (MERGED VERSION)
# Includes:
# - Original 7-method ensemble (simplified placeholders where needed)
# - Monte Carlo (reframed as smoothing)
# - Full-engine backtesting
# - Baseline comparison
# - Calibration error
# - Paired t-test
# - Bootstrap significance test

import math
import random
from collections import Counter

# ─────────────────────────────────────────────
# GAME DEFINITIONS (REQUIRED FOR API)
# ─────────────────────────────────────────────

GAMES = {
    "powerball": {
        "name": "Powerball",
        "white_max": 69,
        "white_count": 5,
        "special_max": 26,
        "special_count": 1
    },
    "mega_millions": {
        "name": "Mega Millions",
        "white_max": 70,
        "white_count": 5,
        "special_max": 25,
        "special_count": 1
    },
    "superlotto": {
        "name": "SuperLotto Plus",
        "white_max": 47,
        "white_count": 5,
        "special_max": 27,
        "special_count": 1
    }

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def normalize(d):
    s = sum(d.values()) + 1e-12
    return {k: v/s for k,v in d.items()}

# ─────────────────────────────────────────────
# CORE METHODS (SIMPLIFIED REPRESENTATIONS)
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
    return {n: 1/len(number_range) for n in number_range}

# ─────────────────────────────────────────────
# MONTE CARLO (SMOOTHING)
# ─────────────────────────────────────────────

def monte_carlo_smoothing(probs, number_range, n_sim=50000):
    counts = Counter()
    numbers = list(number_range)
    weights = [probs[n] for n in numbers]

    for _ in range(n_sim):
        pick = random.choices(numbers, weights=weights, k=5)
        counts.update(pick)

    return normalize(counts)

# ─────────────────────────────────────────────
# ENSEMBLE ENGINE
# ─────────────────────────────────────────────

def analyze_and_predict(rows, number_range):
    m1 = bayesian_frequency(rows, number_range)
    m2 = gap_analysis(rows, number_range)
    m3 = decay_model(rows, number_range)

    # Blend (weights can be optimized later)
    blended = {n: 0.4*m1[n] + 0.3*m2[n] + 0.3*m3[n] for n in number_range}

    # Monte Carlo smoothing
    smoothed = monte_carlo_smoothing(blended, number_range)
    return smoothed

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def brier_score(probs, actual, number_range):
    actual_set = set(actual)
    return sum((probs.get(n,0) - (1 if n in actual_set else 0))**2 for n in number_range)


def calibration_error(probs, rows, number_range, bins=10):
    bucket = [[] for _ in range(bins)]
    for n in number_range:
        p = probs.get(n, 0)
        idx = min(int(p * bins), bins-1)
        actual_freq = sum(1 for r in rows if n in r["balls"]) / len(rows)
        bucket[idx].append((p, actual_freq))
    err = sum(abs(p-a) for b in bucket for p,a in b)
    count = sum(len(b) for b in bucket)
    return err / (count + 1e-12)

# ─────────────────────────────────────────────
# STAT TESTS
# ─────────────────────────────────────────────

def paired_t_test(a, b):
    n = len(a)
    diffs = [x-y for x,y in zip(a,b)]
    mean = sum(diffs)/n
    var = sum((d-mean)**2 for d in diffs)/(n-1)
    se = math.sqrt(var/n) if var>0 else 1e-9
    t = mean/se
    p = 2*(1-0.5*math.erfc(-abs(t)/math.sqrt(2)))
    return {"t": t, "p": p}


def bootstrap_test(a, b, n_boot=1000):
    n = len(a)
    better = 0
    for _ in range(n_boot):
        idx = [random.randint(0,n-1) for _ in range(n)]
        if sum(a[i] for i in idx)/n < sum(b[i] for i in idx)/n:
            better += 1
    return {"prob_model_better": better/n_boot}

# ─────────────────────────────────────────────
# BACKTEST (FULL ENGINE)
# ─────────────────────────────────────────────

def full_backtest(rows, number_range, train_size=200, windows=20):
    brier_model, brier_null, calib = [], [], []

    for i in range(min(windows, len(rows)-train_size-1)):
        train = rows[i:i+train_size]
        test = rows[i+train_size]

        probs = analyze_and_predict(train, number_range)
        null = random_model(number_range)

        brier_model.append(brier_score(probs, test["balls"], number_range))
        brier_null.append(brier_score(null, test["balls"], number_range))
        calib.append(calibration_error(probs, train, number_range))

    return {
        "avg_model_brier": sum(brier_model)/len(brier_model),
        "avg_null_brier": sum(brier_null)/len(brier_null),
        "calibration_error": sum(calib)/len(calib),
        "t_test": paired_t_test(brier_model, brier_null),
        "bootstrap": bootstrap_test(brier_model, brier_null)
    }

# ─────────────────────────────────────────────
# EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data = [{"balls": random.sample(range(1,70),5)} for _ in range(1000)]
    nums = list(range(1,70))

    print(full_backtest(data, nums))
