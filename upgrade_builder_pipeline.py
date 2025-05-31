import pandas as pd
import numpy as np

# Load your uploaded builder_forward mock pipeline
df = pd.read_csv("2b631306-2d5d-43eb-8851-46247af902ed.csv")

# Generate random origination dates in the past 8 months
n = len(df)
date_rng = pd.date_range("2023-10-01", "2024-05-30", periods=n)
np.random.shuffle(date_rng.values)
df['orig_date'] = date_rng.strftime('%Y-%m-%d')

# Assign coupons with a normal distribution (mean 6.75%, std 0.4%)
np.random.seed(42)
df['coupon'] = np.clip(np.random.normal(loc=6.75, scale=0.4, size=n), 6.0, 8.0)

# Round coupons for realism
df['coupon'] = df['coupon'].round(2)

# Save the upgraded CSV
df.to_csv("builder_forward_backtest.csv", index=False)

print("CSV saved as 'builder_forward_backtest.csv' and ready for backtesting!")
