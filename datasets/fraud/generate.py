"""Generate synthetic fraud transaction data with ~2% fraud rate."""
import csv
import numpy as np

np.random.seed(42)

N = 5000
N_FRAUD = 100
N_LEGIT = N - N_FRAUD
OUT = "/home/esoc/esolearn/datasets/fraud/fraud.csv"

rows = []

def gen_legit():
    amount = round(np.clip(np.random.lognormal(3.5, 1.0), 0.50, 5000), 2)
    hour = int(np.random.choice(24, p=_hour_probs_legit))
    dow = int(np.random.randint(0, 7))
    merchant = int(np.random.randint(1, 11))
    dist_home = round(max(0, np.random.exponential(5)), 1)
    dist_last = round(max(0, np.random.exponential(3)), 1)
    ratio_median = round(max(0.01, np.random.lognormal(0, 0.5)), 2)
    is_foreign = int(np.random.random() < 0.05)
    is_weekend = int(dow >= 5)
    velocity = int(np.clip(np.random.poisson(2), 0, 20))
    return [amount, hour, dow, merchant, dist_home, dist_last,
            ratio_median, is_foreign, is_weekend, velocity, 0]

def gen_fraud():
    amount = round(np.clip(np.random.lognormal(5.5, 1.2), 10, 25000), 2)
    hour = int(np.random.choice(24, p=_hour_probs_fraud))
    dow = int(np.random.randint(0, 7))
    merchant = int(np.random.choice([1, 2, 3, 7, 8, 9, 10], p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]))
    dist_home = round(max(0, np.random.exponential(25)), 1)
    dist_last = round(max(0, np.random.exponential(20)), 1)
    ratio_median = round(max(0.5, np.random.lognormal(1.5, 0.8)), 2)
    is_foreign = int(np.random.random() < 0.35)
    is_weekend = int(dow >= 5)
    velocity = int(np.clip(np.random.poisson(6), 0, 30))
    return [amount, hour, dow, merchant, dist_home, dist_last,
            ratio_median, is_foreign, is_weekend, velocity, 1]

# Hour distributions
_legit_weights = np.array([1,0.5,0.3,0.2,0.2,0.3,0.8,2,4,5,5,5,6,5,5,5,5,5,4,4,3,3,2,1.5], dtype=float)
_hour_probs_legit = _legit_weights / _legit_weights.sum()
_fraud_weights = np.array([3,3,4,4,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,3], dtype=float)
_hour_probs_fraud = _fraud_weights / _fraud_weights.sum()

for _ in range(N_LEGIT):
    rows.append(gen_legit())
for _ in range(N_FRAUD):
    rows.append(gen_fraud())

# Shuffle
np.random.shuffle(rows)

# Add transaction IDs
final = []
for i, r in enumerate(rows):
    final.append([300001 + i] + r)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "transaction_id", "amount", "hour_of_day", "day_of_week",
        "merchant_category", "distance_from_home",
        "distance_from_last_transaction", "ratio_to_median_amount",
        "is_foreign", "is_weekend", "velocity_last_hour", "is_fraud",
    ])
    w.writerows(final)

print(f"Wrote {len(final)} rows to {OUT}")
