#!/usr/bin/env python3
"""
Generate mock brewery sales transaction data for RAG testing.
Outputs sample_docs/brewery_sales.csv (≤ 80 MB).
Run from the repo root: python generate_brewery_data.py
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
MAX_BYTES = 80 * 1024 * 1024
OUTPUT = Path(__file__).parent / "sample_docs" / "brewery_sales.csv"
START_DATE = date(2021, 5, 11)
END_DATE = date(2026, 5, 11)

# (name, style, abv, ibu, seasonal_tag, is_limited)
BEERS = [
    ("Golden Hour Lager",          "Lager",          4.5, 18, "Year-Round", False),
    ("Hop Cannon IPA",             "IPA",            6.8, 65, "Year-Round", False),
    ("Midnight Porter",            "Porter",         5.8, 30, "Year-Round", False),
    ("River Wheat",                "Wheat",          4.8, 15, "Year-Round", False),
    ("Summit Pale Ale",            "Pale Ale",       5.2, 38, "Year-Round", False),
    ("Amber Falls Amber Ale",      "Amber Ale",      5.5, 28, "Year-Round", False),
    ("Bloom Saison",               "Saison",         5.6, 22, "Spring",     False),
    ("Cherry Blossom Sour",        "Sour",           4.2, 10, "Spring",     False),
    ("Porch Swing Session IPA",    "Session IPA",    4.1, 40, "Summer",     False),
    ("Mango Tart Wheat",           "Wheat",          5.0, 12, "Summer",     False),
    ("Harvest Pumpkin Ale",        "Spiced Ale",     6.2, 20, "Fall",       False),
    ("Oktoberfest Märzen",         "Märzen",         5.9, 24, "Fall",       False),
    ("Winter Warmer Stout",        "Stout",          8.5, 35, "Winter",     False),
    ("Snowflake White IPA",        "White IPA",      6.0, 45, "Winter",     False),
    ("Barrel-Aged Imperial Stout", "Imperial Stout", 12.0, 40, "Year-Round", True),
    ("Triple Dry-Hopped DIPA",     "DIPA",           8.2, 85, "Year-Round", True),
]

# (name, price_lo, price_hi, weight)
SERVING_SIZES = [
    ("Half-pint", 4.0,  6.0,  0.15),
    ("Pint",      6.0,  10.0, 0.45),
    ("Flight",    12.0, 16.0, 0.10),
    ("Growler",   15.0, 22.0, 0.08),
    ("Crowler",   10.0, 15.0, 0.10),
    ("Can",       4.0,  8.0,  0.12),
]
SERVING_NAMES   = [s[0] for s in SERVING_SIZES]
SERVING_WEIGHTS = [s[3] for s in SERVING_SIZES]
SERVING_PRICES  = {s[0]: (s[1], s[2]) for s in SERVING_SIZES}

PAYMENT_METHODS = ["Cash", "Credit", "Debit", "Tab"]
PAYMENT_WEIGHTS = [0.20, 0.50, 0.20, 0.10]

LOCATIONS        = ["Taproom", "Patio", "Event", "To-Go"]
LOCATION_WEIGHTS = [0.60, 0.20, 0.15, 0.05]

EVENTS = [
    ("Trivia Night",  0.08),
    ("Live Music",    0.10),
    ("Beer Festival", 0.02),
    ("Tap Takeover",  0.03),
    ("Cask Night",    0.04),
    ("Brewery Tour",  0.05),
]

STAFF = [f"STAFF_{i:03d}" for i in range(1, 16)]

HEADERS = [
    "transaction_id", "date", "day_of_week", "time_of_day",
    "beer_name", "beer_style", "abv", "ibu", "serving_size",
    "unit_price", "quantity", "total_revenue",
    "payment_method", "location", "seasonal_tag", "is_limited_release",
    "event_name", "staff_id",
]


def get_season(d: date) -> str:
    m = d.month
    if m in (3, 4, 5):   return "Spring"
    if m in (6, 7, 8):   return "Summer"
    if m in (9, 10, 11): return "Fall"
    return "Winter"


def pick_event(d: date) -> str:
    dow = d.weekday()
    # Events are rare Mon-Thu, more likely Fri-Sun
    base_prob = {0: 0.03, 1: 0.03, 2: 0.04, 3: 0.05, 4: 0.20, 5: 0.30, 6: 0.20}[dow]
    if random.random() > base_prob:
        return ""
    r = random.random()
    cumulative = 0.0
    for name, prob in EVENTS:
        cumulative += prob
        if r < cumulative:
            return name
    return ""


def daily_transaction_count(d: date, has_event: bool) -> int:
    season_mult = {"Spring": 1.0, "Summer": 1.3, "Fall": 1.2, "Winter": 0.9}[get_season(d)]
    dow_mult    = {0: 1.0, 1: 1.0, 2: 1.1, 3: 1.1, 4: 1.8, 5: 2.5, 6: 2.0}[d.weekday()]
    count = int(150 * season_mult * dow_mult * random.uniform(0.85, 1.15))
    if has_event:
        count = int(count * random.uniform(1.3, 1.6))
    return max(10, count)


def available_beers(season: str) -> list:
    pool = []
    for beer in BEERS:
        _, _, _, _, tag, is_limited = beer
        if is_limited:
            if random.random() < 0.25:
                pool.append(beer)
        elif tag == "Year-Round" or tag == season:
            pool.append(beer)
    return pool or [BEERS[0]]


def random_time(d: date) -> str:
    if d.weekday() < 4:
        hours   = list(range(11, 23))
        weights = [1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1]
    else:
        hours   = list(range(11, 24))
        weights = [1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 1]
    hour = random.choices(hours, weights=weights, k=1)[0]
    return f"{hour:02d}:{random.randint(0, 59):02d}"


def generate() -> None:
    random.seed(SEED)
    OUTPUT.parent.mkdir(exist_ok=True)

    txn_id = 1
    rows_written = 0
    stopped_early = False

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

        current = START_DATE
        while current <= END_DATE:
            season = get_season(current)
            event  = pick_event(current)
            beers  = available_beers(season)
            count  = daily_transaction_count(current, bool(event))

            for _ in range(count):
                name, style, abv, ibu, tag, is_limited = random.choice(beers)
                serving = random.choices(SERVING_NAMES, weights=SERVING_WEIGHTS, k=1)[0]
                lo, hi  = SERVING_PRICES[serving]
                price_mod  = 1.0 + (abv - 5.0) * 0.05
                unit_price = round(random.uniform(lo, hi) * price_mod, 2)
                qty        = random.choices([1, 2, 3, 4], weights=[0.65, 0.25, 0.07, 0.03], k=1)[0]

                writer.writerow([
                    f"TXN-{txn_id:07d}",
                    current.isoformat(),
                    current.strftime("%A"),
                    random_time(current),
                    name, style, abv, ibu,
                    serving,
                    unit_price, qty, round(unit_price * qty, 2),
                    random.choices(PAYMENT_METHODS, weights=PAYMENT_WEIGHTS, k=1)[0],
                    random.choices(LOCATIONS,       weights=LOCATION_WEIGHTS, k=1)[0],
                    tag, is_limited,
                    event,
                    random.choice(STAFF),
                ])
                txn_id += 1
                rows_written += 1

            if rows_written % 5000 == 0:
                size = OUTPUT.stat().st_size
                print(f"\r  {rows_written:,} rows | {size / 1_048_576:.1f} MB ({size / MAX_BYTES * 100:.0f}%)   ", end="", flush=True)
                if size >= MAX_BYTES * 0.98:
                    stopped_early = True
                    break

            current += timedelta(days=1)

    size = OUTPUT.stat().st_size
    print(f"\r  {rows_written:,} rows | {size / 1_048_576:.1f} MB                          ")
    if stopped_early:
        print("  Stopped early at size limit.")
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    print(f"Generating brewery sales data ({START_DATE} → {END_DATE})...")
    generate()
    print("Done.")
