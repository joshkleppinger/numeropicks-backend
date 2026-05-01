# NOTE: This is a PATCHED version of your engine with the requested additions:
# - Full engine backtesting (optional, expensive)
# - Calibration error added to backtest output
# - Paired t-test on Brier scores
# - Bootstrap significance test
# - Commentary on Monte Carlo retained but not removed

# Only NEW / MODIFIED sections are shown for clarity. Integrate into your file.

import math
import random

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def paired_t_test(sample_a, sample_b):
    """
    Paired t-test for two dependent samples (e.g., model vs baseline Brier scores).
    Returns t-statistic and approximate p-value.
    """
    n = len(sample_a)
    if n < 5:
        return {"error": "Not enough samples"}

    diffs = [a - b for a, b in zip(sample_a, sample_b)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_err = math.sqrt(var / n) if var > 0 else 1e-9
    t_stat = mean_diff / std_err

    # Normal approximation
    p_value = 2 * (1 - 0.5 * math.erfc(-abs(t_stat) / math.sqrt(2)))

    return {
        "t_stat": round(t_stat, 6),
        "p_value": round(p_value, 6),
        "mean_difference": round(mean_diff, 6),
        "significant_5pct": p_value < 0.05
    }


def bootstrap_significance(sample_a, sample_b, n_bootstrap=1000):
    """
    Bootstrap test: probability that model beats baseline.
    """
    if len(sample_a) != len(sample_b) or len(sample_a) < 10:
        return {"error": "Invalid samples"}

    n = len(sample_a)
    better = 0

    for _ in range(n_bootstrap):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        a = sum(sample_a[i] for i in idx) / n
        b = sum(sample_b[i] for i in idx) / n
        if a < b:  # lower Brier is better
            better += 1

    prob = better / n_bootstrap

    return {
        "prob_model_beats_baseline": round(prob, 4),
        "significant_5pct": prob > 0.95
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL ENGINE BACKTEST (EXPENSIVE)
# ─────────────────────────────────────────────────────────────────────────────

def full_engine_backtest(rows, game, train_size=500, max_windows=50):
    """
    TRUE backtest using full 7-method engine.
    VERY SLOW. Use for final validation only.
    """

    from copy import deepcopy

    WHITE_RANGE = range(1, game["white_max"] + 1)

    brier_model = []
    brier_null = []
    calib_errors = []

    for i in range(min(max_windows, len(rows) - train_size - 1)):
        train = rows[i:i + train_size]
        test = rows[i + train_size]

        # Run FULL engine
        tickets = analyze_and_predict(train, game)

        # Convert tickets → probability approximation
        counts = {}
        for t in tickets:
            for b in t["balls"]:
                counts[b] = counts.get(b, 0) + 1

        total = sum(counts.values()) + 1e-12
        probs = {n: counts.get(n, 0) / total for n in WHITE_RANGE}

        null_probs = {n: 1/len(WHITE_RANGE) for n in WHITE_RANGE}

        bs_model = brier_score(probs, test["balls"], WHITE_RANGE)
        bs_null = brier_score(null_probs, test["balls"], WHITE_RANGE)

        brier_model.append(bs_model)
        brier_null.append(bs_null)

        # Calibration error
        calib = calibration_data(probs, train, WHITE_RANGE)
        calib_err = calibration_error(calib)
        calib_errors.append(calib_err)

    # Statistical tests
    ttest = paired_t_test(brier_model, brier_null)
    boot  = bootstrap_significance(brier_model, brier_null)

    return {
        "windows": len(brier_model),
        "avg_brier_model": sum(brier_model) / len(brier_model),
        "avg_brier_null": sum(brier_null) / len(brier_null),
        "avg_calibration_error": sum(calib_errors) / len(calib_errors),
        "t_test": ttest,
        "bootstrap": boot
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODIFY EXISTING BACKTEST (ADD CALIBRATION + SIGNIFICANCE)
# ─────────────────────────────────────────────────────────────────────────────

def enhanced_backtest(rows, game, train_size=500, max_windows=200):
    base = backtest(rows, game, train_size, max_windows)

    if "error" in base:
        return base

    windows = base["windows"]

    bs_model = [w["brier_model"] for w in windows]
    bs_null  = [w["brier_null"] for w in windows]

    # Add statistical tests
    base["summary"]["t_test"] = paired_t_test(bs_model, bs_null)
    base["summary"]["bootstrap"] = bootstrap_significance(bs_model, bs_null)

    # Add calibration error (global)
    WHITE_RANGE = range(1, game["white_max"] + 1)
    probs = _simple_frequency_probs(rows[-train_size:], WHITE_RANGE)
    calib = calibration_data(probs, rows[-train_size:], WHITE_RANGE)
    base["summary"]["calibration_error"] = calibration_error(calib)

    return base



