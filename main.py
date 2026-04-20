"""
NumeroPicks — FastAPI Backend
Serves prediction, scraping, history, and accuracy data for the web app.
"""

import threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import (
    GAMES, load_draws, save_draws, scrape_game, load_scrape_state,
    analyze_and_predict, save_predictions, compare_predictions,
    compute_accuracy, next_draw_date,
)

app = FastAPI(title="NumeroPicks API", version="1.0.0")

# ── CORS: allow the React frontend (any origin in dev, locked down in prod) ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to ["https://numeropicks.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory draw cache (loaded once on startup, updated after scrapes) ──────
_draw_cache: dict[str, list] = {}
_analyze_lock = threading.Lock()

# ── Progress tracking for long-running analysis ───────────────────────────────
_analysis_progress: dict[str, dict] = {}


def _get_draws(game_key: str) -> list:
    if game_key not in _draw_cache:
        _draw_cache[game_key] = load_draws(GAMES[game_key])
    return _draw_cache[game_key]


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "NumeroPicks API is running 🎱"}


@app.get("/games")
def list_games():
    """Return metadata for all supported games."""
    return {
        key: {
            "key":          g["key"],
            "display_name": g["display_name"],
            "white_max":    g["white_max"],
            "special_max":  g["special_max"],
            "special_name": g["special_name"],
            "next_draw":    next_draw_date(g),
            "row_count":    len(_get_draws(key)),
        }
        for key, g in GAMES.items()
    }


@app.get("/history/{game_key}")
def get_history(game_key: str, limit: int = 20):
    """Return the most recent draws for a game."""
    if game_key not in GAMES:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_key}")
    rows = _get_draws(game_key)
    return {
        "game":       game_key,
        "total_rows": len(rows),
        "draws":      rows[-limit:][::-1],   # most recent first
    }


@app.post("/scrape/{game_key}")
def scrape(game_key: str, background_tasks: BackgroundTasks):
    """Kick off a background scrape for one game."""
    if game_key not in GAMES:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_key}")

    def _run():
        rows = _get_draws(game_key)
        added, msg = scrape_game(GAMES[game_key], rows)
        _draw_cache[game_key] = rows
        print(f"[scrape/{game_key}] {msg}")

    background_tasks.add_task(_run)
    return {"status": "scrape started", "game": game_key}


@app.post("/scrape-all")
def scrape_all(background_tasks: BackgroundTasks):
    """Scrape all three games in the background."""
    def _run():
        for key, game in GAMES.items():
            rows = _get_draws(key)
            added, msg = scrape_game(game, rows)
            _draw_cache[key] = rows
            print(f"[scrape-all/{key}] {msg}")

    background_tasks.add_task(_run)
    return {"status": "scraping all games in background"}


@app.get("/scrape-status")
def scrape_status():
    """Return the last scrape timestamp and staleness flag."""
    ls = load_scrape_state()
    if ls is None:
        return {"last_scrape": None, "stale": True, "days_ago": None}
    days_ago = (datetime.now() - ls).days
    return {
        "last_scrape": ls.isoformat(),
        "days_ago":    days_ago,
        "stale":       days_ago >= 3,
    }


@app.get("/predict/{game_key}")
def predict(game_key: str, save: bool = True):
    """
    Run the full 7-method analysis and return 5 predicted tickets.
    Pass ?save=false to skip writing to the predictions CSV.
    This is synchronous — analysis can take 10-30s.
    """
    if game_key not in GAMES:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_key}")

    game = GAMES[game_key]
    rows = _get_draws(game_key)
    if not rows:
        raise HTTPException(status_code=400,
                            detail=f"No draw history for {game_key}. Run /scrape/{game_key} first.")

    with _analyze_lock:
        tickets = analyze_and_predict(rows, game)

    nd = next_draw_date(game)

    if save and tickets:
        save_predictions(game, tickets, nd)

    return {
        "game":           game_key,
        "next_draw":      nd,
        "tickets":        tickets,
        "special_name":   game["special_name"],
    }


@app.get("/accuracy/{game_key}")
def accuracy(game_key: str):
    """Return prediction accuracy stats and recent evaluated rounds."""
    if game_key not in GAMES:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_key}")

    game = GAMES[game_key]
    rows = _get_draws(game_key)
    data = compare_predictions(game, rows)

    evaluated = data["evaluated"]
    pending   = data["pending"]

    summary = None
    if evaluated:
        total  = len(evaluated)
        avg_w  = sum(r["white_matches"] for r in evaluated) / total
        sp_hit = sum(1 for r in evaluated if r["sp_match"]) / total * 100
        best_w = max(r["white_matches"] for r in evaluated)
        last   = evaluated[-1]
        summary = {
            "total_rounds":      total,
            "avg_white_matches": round(avg_w, 2),
            "special_hit_rate":  round(sp_hit, 1),
            "best_white_match":  best_w,
            "last_score":        round(last["score"] * 100),
            "last_draw_date":    last["target_draw_date"],
        }

    return {
        "game":      game_key,
        "summary":   summary,
        "evaluated": evaluated[-10:],    # last 10 rounds
        "pending":   pending,
    }


@app.get("/next-draw/{game_key}")
def get_next_draw(game_key: str):
    """Return the next draw date with a friendly formatted string."""
    if game_key not in GAMES:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_key}")
    game = GAMES[game_key]
    nd   = next_draw_date(game)
    # Build ordinal
    from datetime import datetime as dt
    d       = dt.strptime(nd, "%a, %b %d, %Y")
    day_num = d.day
    sfx     = "th" if 11 <= day_num <= 13 else {1:"st",2:"nd",3:"rd"}.get(day_num%10,"th")
    friendly = f"These numbers are for the {d.strftime('%A')}, {d.strftime('%B')} {day_num}{sfx} drawing"
    return {"date_str": nd, "friendly": friendly}


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    """Pre-load all draw histories into cache on startup."""
    for key in GAMES:
        rows = load_draws(GAMES[key])
        _draw_cache[key] = rows
        print(f"[startup] {key}: {len(rows):,} draws loaded")
