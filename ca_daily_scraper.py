"""
ca_daily_scraper.py — California Daily 3 / Daily 4 historical data scraper.

Replaces the lottery.net scraper for these two games (lottery.net's Cloudflare
blocks Render's egress IPs). Uses m.lottostrategies.com as the primary data
source and falls back to lotteryextreme.com if needed. Both work cleanly on
cloud servers like Render, AWS, GCP, Heroku — they don't gate by IP reputation.

KEY FIX: The original engine deduplicated by *date alone*, which silently
discarded one of every two Daily 3 draws (midday + evening share the same date).
This module dedupes by (date, draw_type) so both daily draws are preserved.

Drop this file next to engine.py, then apply the small `engine.py` patch
documented in `INTEGRATION.md`.

Public API:
    scrape_daily(game_dict, existing_rows, log_fn=None) -> (added_count, message)
    parse_daily_html(html, expected_count) -> list[dict]
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Callable, List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

# Primary: m.lottostrategies.com (mobile site — clean HTML, no JS, no bot wall)
LS_RECENT = {
    "daily3_evening": "https://m.lottostrategies.com/CA/California/Daily-3-Evening/recent-results.htm",
    "daily3_midday":  "https://m.lottostrategies.com/CA/California/Daily-3-Midday/recent-results.htm",
    "daily4":         "https://m.lottostrategies.com/CA/California/Daily-4/recent-results.htm",
}

LS_MONTH = {
    "daily3_evening": "https://m.lottostrategies.com/CA/California/Daily-3-Evening/{month}-{year}/winning-numbers.htm",
    "daily3_midday":  "https://m.lottostrategies.com/CA/California/Daily-3-Midday/{month}-{year}/winning-numbers.htm",
    "daily4":         "https://m.lottostrategies.com/CA/California/Daily-4/{month}-{year}/winning-numbers.htm",
}

# Fallback: lotteryextreme.com (parses the same way the existing engine
# already knows about — engine._parse_lottery_net_page handles "lotteryextreme.com" branch)
LE_LATEST = {
    "daily3": "https://www.lotteryextreme.com/california/daily3-results",
    "daily4": "https://www.lotteryextreme.com/california/daily4-results",
}

# Recognised game labels in lottostrategies HTML <strong> blocks
GAME_LABELS = {
    "Daily 3 Evening": "Evening",
    "Daily 3 Midday":  "Midday",
    "Daily 4":         None,  # Daily 4 has no draw_type
}

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


# ──────────────────────────────────────────────────────────────────────────────
#  HTTP
# ──────────────────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent":      USER_AGENT,
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
    })
    return s


def _safe_get(session: requests.Session, url: str, timeout: int = 25) -> Optional[str]:
    """GET with sane defaults; return text or None on failure."""
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  PARSERS
# ──────────────────────────────────────────────────────────────────────────────

def parse_daily_html(html: str, expected_count: int) -> List[Dict]:
    """
    Parse a m.lottostrategies.com results page (recent-results.htm or
    Month-YYYY/winning-numbers.htm).

    Page structure (per draw):
        <strong>Daily 3 Evening</strong>          # game label
        <strong>Tue May 05, 2026</strong>         # date
        <table><tr><td>1</td><td>2</td><td>2</td></tr>…</table>

    Returns: list of dicts:
        {date_str, dt, balls, special=None, draw_type}

    `draw_type` is "Midday", "Evening", or None (for Daily 4).
    """
    soup = BeautifulSoup(html, "html.parser")
    bolds = soup.find_all(["strong", "b"])
    results: List[Dict] = []

    i = 0
    while i < len(bolds):
        text = bolds[i].get_text(strip=True)

        # First bold of a draw block is the game label, exact match
        if text not in GAME_LABELS:
            i += 1
            continue
        draw_type = GAME_LABELS[text]

        # Next bold should be the date "Tue May 05, 2026" (single-digit day OK)
        if i + 1 >= len(bolds):
            break

        date_text = bolds[i + 1].get_text(strip=True)
        dt = _parse_ls_date(date_text)
        if dt is None:
            # Not a real draw block (sidebar combined headers etc.) — skip the label only
            i += 1
            continue

        # First <table> after the date holds the numbers in row 1
        table = bolds[i + 1].find_next("table")
        if not table:
            i += 2
            continue

        first_tr = table.find("tr")
        if not first_tr:
            i += 2
            continue

        nums: List[int] = []
        for td in first_tr.find_all("td"):
            txt = td.get_text(strip=True)
            if txt.isdigit() and len(txt) == 1:  # single digits only — skips the "---" row
                nums.append(int(txt))

        if len(nums) != expected_count:
            i += 2
            continue
        if not all(0 <= n <= 9 for n in nums):
            i += 2
            continue

        results.append({
            "date_str":  dt.strftime("%a, %b %d, %Y"),
            "dt":        dt,
            "balls":     nums,            # ORDER MATTERS for Daily 3/4 — do NOT sort
            "special":   None,
            "draw_type": draw_type,
        })
        i += 2

    return results


def _parse_ls_date(s: str) -> Optional[datetime]:
    """Lotto Strategies date format: 'Tue May 05, 2026' (or 'Tue May 5, 2026')."""
    s = s.strip()
    for fmt in ("%a %b %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_lotteryextreme_html(html: str, expected_count: int) -> List[Dict]:
    """
    Parse lotteryextreme.com latest-draws page. Each daily date appears twice
    for Daily 3 (one row = midday, the other = evening). The HIGHER draw number
    on the same date is the evening draw; the LOWER is midday.

        <tr class='cy'>Sat, May 2  (05/02/2026) - #21018</tr>
        <tr class='c1'><td><ul class='displayball'><li>0</li><li>4</li><li>1</li></ul></td></tr>
    """
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict] = []

    for table in soup.find_all("table", class_="results3"):
        rows = table.find_all("tr")
        # Group rows into (date_row, numbers_row) pairs and remember draw # for ordering
        parsed_pairs: List[Tuple[datetime, int, List[int]]] = []
        i = 0
        while i < len(rows):
            cls = rows[i].get("class") or []
            if "cy" not in cls:
                i += 1
                continue

            txt = rows[i].get_text()
            m_date = re.search(r"\((\d{2})/(\d{2})/(\d{4})\)", txt)
            m_draw = re.search(r"#(\d+)", txt)
            if not (m_date and m_draw and i + 1 < len(rows)):
                i += 1
                continue
            mm, dd, yyyy = m_date.groups()
            try:
                dt = datetime(int(yyyy), int(mm), int(dd))
            except ValueError:
                i += 1
                continue
            draw_num = int(m_draw.group(1))

            ul = rows[i + 1].find("ul", class_="displayball")
            if not ul:
                i += 2
                continue
            nums = [int(li.get_text(strip=True))
                    for li in ul.find_all("li")
                    if li.get_text(strip=True).isdigit()]
            if len(nums) != expected_count or not all(0 <= n <= 9 for n in nums):
                i += 2
                continue
            parsed_pairs.append((dt, draw_num, nums))
            i += 2

        # For Daily 3: same date appears twice with different draw #s.
        # Higher # = evening, lower # = midday. For Daily 4: each date once.
        by_date: Dict[datetime.date, List[Tuple[int, List[int]]]] = {}
        for dt, dn, nums in parsed_pairs:
            by_date.setdefault(dt.date(), []).append((dn, nums))

        for d, items in by_date.items():
            items.sort(key=lambda x: x[0])  # ascending draw number
            dt = datetime(d.year, d.month, d.day)
            if expected_count == 3 and len(items) >= 2:
                # Two same-day draws → first is midday, second is evening
                results.append({
                    "date_str": dt.strftime("%a, %b %d, %Y"),
                    "dt": dt,
                    "balls": items[0][1],
                    "special": None,
                    "draw_type": "Midday",
                })
                results.append({
                    "date_str": dt.strftime("%a, %b %d, %Y"),
                    "dt": dt,
                    "balls": items[-1][1],
                    "special": None,
                    "draw_type": "Evening",
                })
            else:
                # Daily 4 (or Daily 3 with only one same-day entry)
                results.append({
                    "date_str": dt.strftime("%a, %b %d, %Y"),
                    "dt": dt,
                    "balls": items[-1][1],
                    "special": None,
                    "draw_type": "Evening" if expected_count == 3 else None,
                })

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  PARSE-DATE HELPER (matches engine.parse_date semantics)
# ──────────────────────────────────────────────────────────────────────────────

_DATE_FMTS = (
    "%a, %b %d, %Y", "%a, %b, %d, %Y", "%a, %b %d %Y",
    "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d",
)

def _parse_row_date(s: str) -> Optional[datetime]:
    s = s.strip().replace("  ", " ")
    s = re.sub(r"^\w+\s+(?=[A-Z])", "", s, count=1)  # strip "Monday " prefix if present
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def scrape_daily(game: Dict, existing_rows: List[Dict],
                 log_fn: Optional[Callable[[str], None]] = None) -> Tuple[int, str]:
    """
    Scrape Daily 3 (both midday + evening) or Daily 4 from m.lottostrategies.com.
    Falls back to lotteryextreme.com if the primary source fails.

    Mutates `existing_rows` in place by appending new dicts. Each row is:
        {"date": "Tue, May 05, 2026", "balls": [1, 2, 2], "special": None,
         "draw_type": "Evening" | "Midday" | None}

    Args:
        game: a GAMES[key] dict — must have "key", "white_count", "history_start"
        existing_rows: current list of rows (from CSV / DB)
        log_fn: optional logger (defaults to print)

    Returns: (number of new rows added, status message)
    """
    log = log_fn or (lambda m: print(f"[ca_daily_scraper] {m}"))
    key = game["key"]
    white_count = game["white_count"]

    if key not in ("daily3", "daily4"):
        return 0, f"scrape_daily called with unsupported game '{key}'"

    # Build set of (date, draw_type) keys we already have, so we know what to skip
    existing_keys, latest_dt = _index_existing(existing_rows, is_daily3=(key == "daily3"))
    log(f"Existing: {len(existing_rows):,} rows, latest = {latest_dt or 'none'}")

    # Decide what to fetch
    history_start = game.get("history_start", 2004 if key == "daily3" else 2010)
    today = datetime.now()
    if latest_dt is None:
        # Fresh install — full historical sweep
        backfill_to = datetime(history_start, 1, 1)
        log(f"First run — backfilling from {history_start}-01")
    else:
        # Incremental — only need recent + a small overlap month for safety
        backfill_to = latest_dt - timedelta(days=45)
        log(f"Incremental — backfilling to {backfill_to:%Y-%m}")

    all_parsed: List[Dict] = []
    session = _make_session()

    # ── Step 1: recent (last 30 days) — primary source ────────────────────────
    sub_keys = ["daily3_evening", "daily3_midday"] if key == "daily3" else ["daily4"]
    for sub in sub_keys:
        log(f"Fetching {sub} (recent) …")
        html = _safe_get(session, LS_RECENT[sub])
        if html:
            rows = parse_daily_html(html, white_count)
            log(f"  → {len(rows)} rows from primary")
            all_parsed.extend(rows)
        else:
            log(f"  → primary failed; trying fallback")
            # Fallback to lotteryextreme.com (only need to fetch once per game)
            fb = _safe_get(session, LE_LATEST[key])
            if fb:
                rows = parse_lotteryextreme_html(fb, white_count)
                log(f"  → {len(rows)} rows from fallback")
                all_parsed.extend(rows)
                break  # fallback returns BOTH midday + evening in one fetch
        time.sleep(0.6)

    # ── Step 2: month-by-month backfill (primary only) ────────────────────────
    cur = datetime(today.year, today.month, 1)
    backfill_floor = datetime(backfill_to.year, backfill_to.month, 1)
    consecutive_empty = 0

    while cur >= backfill_floor:
        any_data_this_month = False
        for sub in sub_keys:
            url = LS_MONTH[sub].format(month=MONTH_NAMES[cur.month - 1], year=cur.year)
            html = _safe_get(session, url)
            if html:
                rows = parse_daily_html(html, white_count)
                if rows:
                    any_data_this_month = True
                    all_parsed.extend(rows)
            time.sleep(0.4)

        if any_data_this_month:
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            # If we hit 3 empty months in a row during a fresh historical sweep,
            # we've gone past the start of the game's history — stop.
            if latest_dt is None and consecutive_empty >= 3:
                log(f"  3 empty months in a row at {cur:%Y-%m} — stopping backfill")
                break

        # Step back one month
        if cur.month == 1:
            cur = datetime(cur.year - 1, 12, 1)
        else:
            cur = datetime(cur.year, cur.month - 1, 1)

    # ── Step 3: dedup with existing data, preserving 2 daily draws ────────────
    new_rows: List[Dict] = []
    seen = set(existing_keys)
    for p in all_parsed:
        d = p["dt"].date()
        dtype = p.get("draw_type")
        # For daily3, key by (date, draw_type); for daily4, just (date, None)
        composite_key = (d, dtype) if key == "daily3" else (d, None)
        if composite_key in seen:
            continue
        seen.add(composite_key)
        new_rows.append({
            "date":      p["date_str"],
            "balls":     p["balls"],
            "special":   None,
            "draw_type": dtype,
        })

    # Sort and merge
    if new_rows:
        new_rows.sort(key=lambda r: (
            _parse_row_date(r["date"]) or datetime.min,
            0 if r.get("draw_type") == "Midday" else 1,  # midday before evening on the same date
        ))
        existing_rows.extend(new_rows)
        return len(new_rows), f"Added {len(new_rows)} new draw(s). Total: {len(existing_rows):,}"

    return 0, f"Up to date ({len(existing_rows):,} rows)"


def _index_existing(existing_rows: List[Dict],
                    is_daily3: bool) -> Tuple[set, Optional[datetime]]:
    """Build (set of keys we already have, latest datetime in the dataset)."""
    keys = set()
    latest: Optional[datetime] = None
    for r in existing_rows:
        dt = _parse_row_date(r["date"])
        if dt is None:
            continue
        if is_daily3:
            # If existing data has no draw_type yet, default to "Evening" (the only draw the
            # broken old scraper actually retained). The first run after upgrading will
            # therefore add the missing Midday draws.
            dtype = r.get("draw_type") or "Evening"
            keys.add((dt.date(), dtype))
        else:
            keys.add((dt.date(), None))
        if latest is None or dt > latest:
            latest = dt
    return keys, latest
