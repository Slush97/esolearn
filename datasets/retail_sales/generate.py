"""Generate synthetic retail sales data with seasonality and promotion effects."""
import csv
import numpy as np
from datetime import date, timedelta

np.random.seed(42)

OUT = "/home/esoc/esolearn/datasets/retail_sales/retail_sales.csv"

start = date(2023, 1, 1)
end = date(2024, 12, 31)
stores = [1, 2, 3, 4, 5]
categories = ["electronics", "clothing", "food", "home", "sports"]

# Base units and prices per category
cat_params = {
    "electronics": {"base_units": 15, "price_range": (49.99, 499.99), "price_mean": 149.99},
    "clothing":    {"base_units": 25, "price_range": (14.99, 129.99), "price_mean": 44.99},
    "food":        {"base_units": 50, "price_range": (2.99, 29.99),   "price_mean": 9.99},
    "home":        {"base_units": 12, "price_range": (19.99, 299.99), "price_mean": 69.99},
    "sports":      {"base_units": 10, "price_range": (14.99, 199.99), "price_mean": 54.99},
}

# US holidays (approximate)
holidays = set()
for yr in (2023, 2024):
    holidays.update([
        date(yr, 1, 1), date(yr, 1, 2),  # New Year
        date(yr, 7, 4),  # Independence Day
        date(yr, 11, 23), date(yr, 11, 24),  # Thanksgiving (approx)
        date(yr, 12, 24), date(yr, 12, 25), date(yr, 12, 26),  # Christmas
        date(yr, 12, 31),
    ])

# Seasonality multipliers by month
month_mult = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.05,
    7: 1.05, 8: 1.00, 9: 0.95, 10: 1.00, 11: 1.20, 12: 1.40,
}

rows = []
d = start
while d <= end:
    # Pick a random subset of store-category combos for this day (not all combos every day)
    n_combos = np.random.randint(5, 10)
    chosen_stores = np.random.choice(stores, size=n_combos, replace=True)
    chosen_cats = np.random.choice(categories, size=n_combos, replace=True)

    for store, cat in zip(chosen_stores, chosen_cats):
        dow = d.weekday()  # 0=Mon, 6=Sun
        month = d.month
        is_hol = int(d in holidays)
        promo = int(np.random.random() < 0.15)

        params = cat_params[cat]
        base = params["base_units"]

        # Seasonality
        seasonal = month_mult[month]
        # Weekend boost
        weekend_mult = 1.15 if dow >= 5 else 1.0
        # Holiday boost
        holiday_mult = 1.4 if is_hol else 1.0
        # Promotion boost
        promo_mult = 1.20 if promo else 1.0

        units = int(max(1, np.random.poisson(
            base * seasonal * weekend_mult * holiday_mult * promo_mult
        )))

        # Price with some variation
        price = round(np.clip(
            np.random.normal(params["price_mean"], params["price_mean"] * 0.15),
            params["price_range"][0], params["price_range"][1]
        ), 2)

        revenue = round(units * price, 2)

        rows.append([
            d.isoformat(), int(store), cat, units, price, revenue,
            promo, dow, month, is_hol,
        ])

    d += timedelta(days=1)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "date", "store_id", "product_category", "units_sold", "unit_price",
        "revenue", "promotion", "day_of_week", "month", "is_holiday",
    ])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
