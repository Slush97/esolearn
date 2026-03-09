"""Generate messy customer survey data for data cleaning exercises."""
import csv
import random
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N = 500
OUT = "/home/esoc/esolearn/datasets/customer_survey/customer_survey.csv"

# Helpers
regions = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
categories = ["Electronics", "Clothing", "Home & Garden", "Food & Beverage", "Health & Beauty"]
referrals = ["Google", "Friend", "Social Media", "TV Ad", "Email", "In-Store", ""]
educations = ["High School", "Some College", "Bachelor's", "Master's", "PhD", "Associate's", ""]
gender_variants = ["M", "Male", "male", "m", "F", "Female", "female", "f", "Non-binary", "Prefer not to say", ""]
date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%B %d, %Y"]

comments_pool = [
    "Great product!", "Not satisfied with delivery time.", "Would recommend to friends.",
    "Too expensive for what you get.", "Excellent customer service.", "Product broke after 2 weeks.",
    "Love the quality!", "Meh, it's okay.", "Will buy again.", "Terrible experience.",
    "Fast shipping!", "Packaging was damaged.", "Better than expected.", "Not worth the price.",
    "Good value for money.", "Could be improved.", "Amazing!", "Disappointed.", "", "",
    "N/A", "no comment", "....", "5 stars!!!", "worst purchase ever", None,
]

rows = []
used_ids = list(range(1001, 1001 + N))
# Inject ~10 duplicate IDs
for _ in range(10):
    idx = random.randint(0, len(used_ids) - 1)
    used_ids.append(used_ids[idx])
random.shuffle(used_ids)
used_ids = used_ids[:N]

base_date = datetime(2024, 1, 1)

for i in range(N):
    cid = used_ids[i]

    # Age: mostly 18-80, with ~2% outliers (0, 150, -5, 999)
    if random.random() < 0.02:
        age = random.choice([0, -5, 150, 999, 200])
    elif random.random() < 0.05:
        age = ""  # missing
    else:
        age = int(np.random.normal(42, 14))
        age = max(18, min(85, age))

    # Gender: inconsistent entries
    gender = random.choice(gender_variants)

    # Income: realistic with some missing and outliers
    if random.random() < 0.05:
        income = ""
    elif random.random() < 0.02:
        income = random.choice([-5000, 0, 1000000, 999999])
    else:
        income = int(np.random.lognormal(10.8, 0.6))
        income = max(15000, min(250000, income))

    education = random.choice(educations)

    # Satisfaction score 1-10
    satisfaction = random.randint(1, 10)

    # Purchase frequency
    purchase_freq = max(0, int(np.random.poisson(4)))

    # Tenure months
    tenure = max(1, int(np.random.exponential(24)))

    region = random.choice(regions)
    category = random.choice(categories)

    # Rating 1-5 with some missing
    if random.random() < 0.05:
        rating = ""
    elif random.random() < 0.01:
        rating = random.choice([0, 6, -1, 99])  # invalid ratings
    else:
        rating = random.randint(1, 5)

    referral = random.choice(referrals)

    # Email valid: mostly True/1, some False/0, some messy
    email_choices = ["True", "False", "1", "0", "true", "false", "yes", "no", "Y", "N", ""]
    email_valid = random.choice(email_choices)

    # Response date: mixed formats
    days_offset = random.randint(0, 700)
    resp_date = base_date + timedelta(days=days_offset)
    fmt = random.choice(date_formats)
    response_date = resp_date.strftime(fmt)

    # Comments
    comment = random.choice(comments_pool)
    if comment is None:
        comment = ""

    rows.append([
        cid, age, gender, income, education, satisfaction,
        purchase_freq, tenure, region, category, rating,
        referral, email_valid, response_date, comment,
    ])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "customer_id", "age", "gender", "income", "education",
        "satisfaction_score", "purchase_frequency", "tenure_months",
        "region", "product_category", "rating", "referral_source",
        "email_valid", "response_date", "comments",
    ])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
