"""
NumeroPicks — Supabase/PostgreSQL persistence layer.
Stores predictions permanently so accuracy tracking survives Render restarts.
Falls back gracefully to CSV if no DATABASE_URL is set.
"""

import os
import json
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# ── Connection pool ────────────────────────────────────────────────────────────
_conn = None

def get_conn():
    global _conn
    if not DATABASE_URL:
        return None
    try:
        import psycopg2
        if _conn is None or _conn.closed:
            _conn = psycopg2.connect(DATABASE_URL, sslmode="require")
            _conn.autocommit = True
        return _conn
    except Exception as e:
        print(f"[db] connection error: {e}")
        return None


def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    if not conn:
        print("[db] No DATABASE_URL — skipping DB init, using CSV fallback")
        return False
    try:
        with conn.cursor() as cur:
            # Predictions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id              SERIAL PRIMARY KEY,
                    game            VARCHAR(30) NOT NULL,
                    prediction_date DATE        NOT NULL,
                    target_draw_date DATE       NOT NULL,
                    balls           JSONB       NOT NULL,
                    special         INTEGER     NOT NULL,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            # Scrape state table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scrape_state (
                    id          SERIAL PRIMARY KEY,
                    last_scrape TIMESTAMPTZ NOT NULL,
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            # Draw history cache table (so we don't re-scrape every restart)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS draw_history (
                    id          SERIAL PRIMARY KEY,
                    game        VARCHAR(30)  NOT NULL,
                    draw_date   VARCHAR(40)  NOT NULL,
                    balls       JSONB        NOT NULL,
                    special     INTEGER      NOT NULL,
                    UNIQUE(game, draw_date)
                );
            """)
        print("[db] Tables ready ✔")
        return True
    except Exception as e:
        print(f"[db] init error: {e}")
        return False


# ── Predictions ────────────────────────────────────────────────────────────────

def save_prediction_db(game: str, balls: list, special: int,
                        prediction_date: str, target_date: str) -> bool:
    conn = get_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions
                    (game, prediction_date, target_draw_date, balls, special)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                game,
                prediction_date,
                target_date,
                json.dumps(balls),
                int(special),
            ))
        return True
    except Exception as e:
        print(f"[db] save_prediction error: {e}")
        return False


def load_predictions_db(game: str) -> list:
    """Return all predictions for a game as list of dicts."""
    conn = get_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT prediction_date, target_draw_date, balls, special
                FROM predictions
                WHERE game = %s
                ORDER BY target_draw_date ASC
            """, (game,))
            rows = cur.fetchall()
        result = []
        for pred_date, tgt_date, balls_json, special in rows:
            result.append({
                "prediction_date":  str(pred_date),
                "target_draw_date": str(tgt_date),
                "pred_balls":       balls_json if isinstance(balls_json, list) else json.loads(balls_json),
                "pred_special":     int(special),
            })
        return result
    except Exception as e:
        print(f"[db] load_predictions error: {e}")
        return []


# ── Scrape state ───────────────────────────────────────────────────────────────

def save_scrape_state_db() -> bool:
    conn = get_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM scrape_state")
            cur.execute("INSERT INTO scrape_state (last_scrape) VALUES (NOW())")
        return True
    except Exception as e:
        print(f"[db] save_scrape_state error: {e}")
        return False


def load_scrape_state_db():
    """Return datetime of last scrape or None."""
    conn = get_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT last_scrape FROM scrape_state ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
        if row:
            return row[0].replace(tzinfo=None) if hasattr(row[0], 'replace') else None
        return None
    except Exception as e:
        print(f"[db] load_scrape_state error: {e}")
        return None


# ── Draw history cache ─────────────────────────────────────────────────────────

def save_draws_db(game: str, rows: list) -> int:
    """Upsert draw rows. Returns number of new rows inserted."""
    conn = get_conn()
    if not conn:
        return 0
    added = 0
    try:
        with conn.cursor() as cur:
            for r in rows:
                try:
                    cur.execute("""
                        INSERT INTO draw_history (game, draw_date, balls, special)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (game, draw_date) DO NOTHING
                    """, (
                        game,
                        r["date"],
                        json.dumps([int(b) for b in r["balls"]]),
                        int(r["special"]),
                    ))
                    if cur.rowcount > 0:
                        added += 1
                except Exception:
                    pass
        return added
    except Exception as e:
        print(f"[db] save_draws error: {e}")
        return 0


def load_draws_db(game: str) -> list:
    """Load all draw history for a game from DB."""
    conn = get_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT draw_date, balls, special
                FROM draw_history
                WHERE game = %s
                ORDER BY draw_date ASC
            """, (game,))
            rows = cur.fetchall()
        result = []
        for draw_date, balls_json, special in rows:
            balls = balls_json if isinstance(balls_json, list) else json.loads(balls_json)
            result.append({
                "date":    draw_date,
                "balls":   [int(b) for b in balls],
                "special": int(special),
            })
        return result
    except Exception as e:
        print(f"[db] load_draws error: {e}")
        return []
