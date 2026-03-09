"""Generate synthetic credit scoring data with realistic approval correlations."""
import csv
import numpy as np

np.random.seed(42)

N = 2000
OUT = "/home/esoc/esolearn/datasets/credit_scoring/credit_scoring.csv"

rows = []
for i in range(N):
    aid = 100001 + i
    age = int(np.clip(np.random.normal(40, 12), 21, 75))
    income = int(np.clip(np.random.lognormal(10.9, 0.5), 20000, 300000))
    employment_years = int(np.clip(np.random.exponential(6), 0, 40))
    debt_to_income = round(np.clip(np.random.beta(2, 5) * 0.8, 0.01, 0.75), 2)
    credit_history_months = int(np.clip(np.random.normal(120, 60), 6, 480))
    num_open_accounts = int(np.clip(np.random.poisson(5), 0, 25))
    num_delinquencies = int(np.clip(np.random.exponential(0.8), 0, 15))
    loan_amount = int(np.clip(np.random.lognormal(9.5, 0.7), 1000, 500000))
    loan_term = np.random.choice([12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360])
    interest_rate = round(np.clip(np.random.normal(7, 3), 2.5, 25.0), 1)

    # Approval logistic model
    logit = (
        -1.0
        + 0.8 * (income / 100000)
        - 3.0 * debt_to_income
        + 0.3 * (employment_years / 10)
        + 0.2 * (credit_history_months / 120)
        - 0.6 * num_delinquencies
        - 0.3 * (loan_amount / 100000)
        + 0.1 * (age / 40)
        - 0.2 * (interest_rate / 10)
    )
    prob = 1 / (1 + np.exp(-logit))
    approved = int(np.random.random() < prob)

    rows.append([
        aid, age, income, employment_years, debt_to_income,
        credit_history_months, num_open_accounts, num_delinquencies,
        loan_amount, loan_term, interest_rate, approved,
    ])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "applicant_id", "age", "income", "employment_years",
        "debt_to_income", "credit_history_months", "num_open_accounts",
        "num_delinquencies", "loan_amount", "loan_term", "interest_rate",
        "approved",
    ])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
