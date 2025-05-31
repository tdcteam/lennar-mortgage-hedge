import pandas as pd
import numpy as np

# Settings for your mock builder pipeline
n_loans = 500    # You can adjust this for larger/smaller pipelines
np.random.seed(42)

# Generate fields
notional = np.random.uniform(200000, 700000, size=n_loans)  # loan sizes $200k–$700k
duration = np.random.uniform(0.10, 0.20, size=n_loans)      # 0.10–0.20 yrs (1.2–2.4 months)
pull_through = np.random.uniform(0.90, 0.98, size=n_loans)  # 90%–98% pull-through

# Evenly spread origination dates across 8 months
date_rng = pd.date_range("2023-10-01", "2024-05-30", periods=n_loans)
np.random.shuffle(date_rng.values)
orig_date = date_rng.strftime('%Y-%m-%d')

# Assign coupons, normal around 6.75%
coupon = np.clip(np.random.normal(6.75, 0.4, n_loans), 6.0, 8.0).round(2)

# Build the DataFrame
df = pd.DataFrame({
    "notional": notional.round(2),
    "duration": duration.round(3),
    "pull_through": pull_through.round(3),
    "orig_date": orig_date,
    "coupon": coupon
})

# Optional: add strategy_type for future modeling
df["strategy_type"] = "builder_forward"

# Save the file
df.to_csv("builder_forward_backtest.csv", index=False)

print("Script is running...Generated 'builder_forward_backtest.csv' in your project folder.")
