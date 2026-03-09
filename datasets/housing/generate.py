"""Generate synthetic housing price data with realistic price modeling."""
import csv
import numpy as np

np.random.seed(42)

N = 1500
OUT = "/home/esoc/esolearn/datasets/housing/housing.csv"

neighborhoods = ["downtown", "suburban", "rural", "waterfront"]
nbhd_price_mult = {"downtown": 1.3, "suburban": 1.0, "rural": 0.7, "waterfront": 1.5}

rows = []
for i in range(N):
    hid = 200001 + i
    nbhd = np.random.choice(neighborhoods, p=[0.25, 0.40, 0.20, 0.15])

    # Square feet depends on neighborhood
    base_sqft = {"downtown": 1200, "suburban": 1800, "rural": 2200, "waterfront": 2000}
    sqft = int(np.clip(np.random.normal(base_sqft[nbhd], 500), 500, 6000))

    bedrooms = int(np.clip(round(sqft / 500), 1, 7))
    bathrooms = int(np.clip(round(bedrooms * 0.8 + np.random.normal(0, 0.5)), 1, 5))

    # Lot size in acres
    lot_mult = {"downtown": 0.1, "suburban": 0.3, "rural": 2.0, "waterfront": 0.5}
    lot_size = round(max(0.05, np.random.exponential(lot_mult[nbhd])), 2)

    year_built = int(np.clip(np.random.normal(1985, 20), 1920, 2024))
    garage_size = int(np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.45, 0.15]))
    condition = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.35, 0.30, 0.15]))
    has_pool = int(np.random.random() < (0.3 if nbhd in ("suburban", "waterfront") else 0.1))
    dist_school = round(max(0.1, np.random.exponential(3)), 1)
    dist_downtown = round(
        max(0.1, np.random.normal(
            {"downtown": 1, "suburban": 8, "rural": 20, "waterfront": 12}[nbhd], 3
        )), 1
    )

    # Price model
    price = (
        50000
        + 120 * sqft
        + 15000 * bedrooms
        + 10000 * bathrooms
        + 20000 * lot_size
        + 500 * max(0, year_built - 1950)
        + 12000 * garage_size
        + 8000 * condition
        + 25000 * has_pool
        - 2000 * dist_school
        - 1500 * dist_downtown
    )
    price *= nbhd_price_mult[nbhd]
    # Add noise (~10%)
    price *= np.random.normal(1.0, 0.10)
    price = int(max(50000, price))

    rows.append([
        hid, sqft, bedrooms, bathrooms, lot_size, year_built,
        garage_size, nbhd, condition, has_pool, dist_school,
        dist_downtown, price,
    ])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "house_id", "square_feet", "bedrooms", "bathrooms", "lot_size",
        "year_built", "garage_size", "neighborhood", "condition", "has_pool",
        "distance_to_school", "distance_to_downtown", "price",
    ])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
