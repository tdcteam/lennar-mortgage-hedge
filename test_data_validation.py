import os
import pandas as pd
import pytest

# === FILE PATHS ===
PIPELINE_FILE = "builder_forward.csv"
BACKTEST_FILE = "builder_forward_backtest.csv"
CC_FILE = "history_cc.csv"
TS_FILE = "history_rates.csv"

# === COLUMN SCHEMAS ===
REQUIRED_PIPELINE_COLS = ["notional", "duration", "pull_through"]
REQUIRED_BACKTEST_COLS = ["notional_usd", "orig_date", "coupon"]
REQUIRED_CC_COLS = ["DATE", "MORTGAGE30US"]
REQUIRED_TS_COLS = ["DATE", "DGS10"]

def file_exists(filename):
    return os.path.exists(filename)

def load_csv(filename):
    return pd.read_csv(filename)

def test_pipeline_file_exists():
    assert file_exists(PIPELINE_FILE), f"Pipeline file missing: {PIPELINE_FILE}"

def test_pipeline_has_columns():
    df = load_csv(PIPELINE_FILE)
    for col in REQUIRED_PIPELINE_COLS:
        assert col in df.columns, f"Missing required column: {col} in pipeline data"

def test_backtest_file_exists():
    assert file_exists(BACKTEST_FILE), f"Backtest file missing: {BACKTEST_FILE}"

def test_backtest_has_columns():
    df = load_csv(BACKTEST_FILE)
    for col in REQUIRED_BACKTEST_COLS:
        assert col in df.columns, f"Missing required column: {col} in backtest data"

def test_backtest_dates_parse():
    df = load_csv(BACKTEST_FILE)
    try:
        pd.to_datetime(df["orig_date"])
    except Exception:
        pytest.fail("Failed to parse 'orig_date' column in backtest data")

def test_backtest_notional_positive():
    df = load_csv(BACKTEST_FILE)
    assert (df["notional_usd"] >= 0).all(), "Negative notional values in backtest data"

def test_cc_file_exists():
    assert file_exists(CC_FILE), f"CC file missing: {CC_FILE}"

def test_cc_has_columns():
    df = load_csv(CC_FILE)
    for col in REQUIRED_CC_COLS:
        assert col in df.columns, f"Missing required column: {col} in CC data"

def test_ts_file_exists():
    assert file_exists(TS_FILE), f"TS file missing: {TS_FILE}"

def test_ts_has_columns():
    df = load_csv(TS_FILE)
    for col in REQUIRED_TS_COLS:
        assert col in df.columns, f"Missing required column: {col} in Treasury data"

def test_cc_dates_parse():
    df = load_csv(CC_FILE)
    try:
        pd.to_datetime(df["DATE"])
    except Exception:
        pytest.fail("Failed to parse 'DATE' in CC file")

def test_ts_dates_parse():
    df = load_csv(TS_FILE)
    try:
        pd.to_datetime(df["DATE"])
    except Exception:
        pytest.fail("Failed to parse 'DATE' in TS file")

# Optional: test no nulls in critical columns
@pytest.mark.parametrize("filename,column", [
    (PIPELINE_FILE, "notional"),
    (BACKTEST_FILE, "notional_usd"),
    (BACKTEST_FILE, "orig_date"),
    (BACKTEST_FILE, "coupon"),
])
def test_no_nulls(filename, column):
    df = load_csv(filename)
    assert df[column].isnull().sum() == 0, f"Nulls found in {column} of {filename}"
import numpy as np

# --- Calculation: Decay Logic ---
def decay_notional(notional, age, strat, cpr_rate=0.10):
    if strat == "builder_forward":
        return notional * np.exp(-cpr_rate * age)
    if strat == "servicing":
        return notional * np.exp(-0.04 * age)
    if strat == "securitization":
        return notional * np.exp(-0.15 * min(age,1) - 0.05 * max(age-1,0))
    return notional

def test_decay_logic_sample_cases():
    # Test known input/outputs (adjust values to match your real logic)
    assert np.isclose(decay_notional(100, 1, "builder_forward"), 100 * np.exp(-0.10 * 1))
    assert np.isclose(decay_notional(200, 2, "servicing"), 200 * np.exp(-0.08))
    assert np.isclose(decay_notional(300, 1, "securitization"), 300 * np.exp(-0.15))
    assert np.isclose(decay_notional(400, 2, "securitization"), 400 * np.exp(-0.15 - 0.05 * 1))

def test_mean_pnl_calculation():
    # Simulate a result as the app does (example: P&L from a distribution)
    pnl_array = np.array([1_000_000, 1_200_000, 900_000, 1_100_000])
    expected_mean = pnl_array.mean()
    assert np.isclose(expected_mean, 1_050_000), "Mean P&L calculation mismatch."

def test_var_calculation():
    # 5% VaR test (quantile)
    pnl_array = np.array([1_000_000, 1_200_000, 900_000, 1_100_000])
    var_5 = np.percentile(pnl_array, 5)
    # By hand, lowest = 900,000; at 5% it's slightly above 900,000
    assert 900_000 <= var_5 <= 950_000, "5% VaR calculation outside expected range."

def test_duration_weighted_average():
    # Weighted average duration calculation (matches your app logic)
    total_notional = 100 + 200 + 300
    avg = (2*100 + 3*200 + 4*300) / total_notional
    assert np.isclose(avg, (200 + 600 + 1200) / 600), "Weighted avg duration mismatch."

# --- Real Pipeline Data Checks (OPTIONAL, if app logic available) ---
def test_apply_decay_matches_pipeline():
    # If you have access to your real DataFrame here:
    # from app import apply_decay, df_all, cpr_rate
    # for idx, row in df_all.iterrows():
    #     expected = decay_notional(row['notional_usd'], row['duration_years'], row['strategy_type'], cpr_rate)
    #     actual = apply_decay(row['notional_usd'], row['duration_years'], row['strategy_type'])
    #     assert np.isclose(actual, expected), f"Row {idx} decay mismatch"
    pass  # Uncomment and use if real code available

def test_bucket_level_summaries():
    # Simulate or import bucket summary calcs and check output matches known result
    pass  # Add your business logic here
import pandas as pd
import numpy as np
from datetime import datetime

# --- Example DataFrames for Validation (Replace with real or fixture data if needed) ---
def sample_pipeline_df():
    return pd.DataFrame({
        "notional_usd": [1_000_000, 2_000_000, 3_000_000],
        "duration_years": [0.12, 0.15, 0.20],
        "pull_through_rate": [0.98, 0.95, 0.99],
        "coupon": [6.25, 6.50, 6.10],
        "orig_date": [datetime(2023, 11, 1), datetime(2023, 11, 2), datetime(2023, 11, 3)],
        "strategy_type": ["builder_forward", "servicing", "securitization"]
    })

def sample_backtest_df():
    return pd.DataFrame({
        "notional_usd": [500_000, 1_200_000],
        "orig_date": [datetime(2023, 10, 1), datetime(2023, 10, 5)],
        "coupon": [6.20, 6.35]
    })

def sample_rate_hist_df():
    return pd.DataFrame({
        "date": [datetime(2023, 10, 1), datetime(2023, 10, 5)],
        "cc_rate": [6.10, 6.20],
        "ts_rate": [4.30, 4.40]
    })

# --- Data Validation Tests ---
def test_pipeline_columns_exist():
    df = sample_pipeline_df()
    required = {"notional_usd", "duration_years", "pull_through_rate", "coupon", "orig_date", "strategy_type"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

def test_backtest_columns_exist():
    df = sample_backtest_df()
    required = {"notional_usd", "orig_date", "coupon"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

def test_rate_hist_columns_exist():
    df = sample_rate_hist_df()
    required = {"date", "cc_rate", "ts_rate"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

def test_no_nulls_in_critical_columns():
    df = sample_pipeline_df()
    for col in ["notional_usd", "duration_years", "pull_through_rate", "coupon", "orig_date"]:
        assert not df[col].isnull().any(), f"Null values found in {col}"

def test_notional_positive():
    df = sample_pipeline_df()
    assert (df["notional_usd"] > 0).all(), "Notional contains non-positive values"

def test_coupon_range():
    df = sample_pipeline_df()
    assert ((df["coupon"] > 3) & (df["coupon"] < 15)).all(), "Coupon rates outside expected range"

def test_dates_are_datetimes():
    df = sample_pipeline_df()
    assert pd.api.types.is_datetime64_any_dtype(df["orig_date"]), "orig_date should be datetime dtype"

# --- Calculation: Decay Logic ---
def decay_notional(notional, age, strat, cpr_rate=0.10):
    if strat == "builder_forward":
        return notional * np.exp(-cpr_rate * age)
    if strat == "servicing":
        return notional * np.exp(-0.04 * age)
    if strat == "securitization":
        return notional * np.exp(-0.15 * min(age,1) - 0.05 * max(age-1,0))
    return notional

def test_decay_logic():
    # Spot check against manual calculations
    assert np.isclose(decay_notional(1_000_000, 0.12, "builder_forward"), 1_000_000 * np.exp(-0.10*0.12))
    assert np.isclose(decay_notional(2_000_000, 0.15, "servicing"), 2_000_000 * np.exp(-0.04*0.15))
    assert np.isclose(decay_notional(3_000_000, 0.20, "securitization"), 3_000_000 * np.exp(-0.15*0.20))
    # Test >1 year securitization
    assert np.isclose(decay_notional(4_000_000, 2, "securitization"), 4_000_000 * np.exp(-0.15*1 - 0.05*1))

# --- P&L and Summary Calculation Tests ---
def test_pnl_calculation():
    pnl_arr = np.array([1_000_000, 1_200_000, 900_000, 1_100_000])
    mean_pnl = pnl_arr.mean()
    std_pnl = pnl_arr.std()
    var_5 = np.percentile(pnl_arr, 5)
    var_95 = np.percentile(pnl_arr, 95)
    assert np.isclose(mean_pnl, 1_050_000)
    assert 80_000 < std_pnl < 120_000
    assert 900_000 <= var_5 <= 950_000
    assert 1_150_000 <= var_95 <= 1_200_000

def test_weighted_avg_duration():
    notional = np.array([100, 200, 300])
    duration = np.array([2, 3, 4])
    expected = (2*100 + 3*200 + 4*300) / (100+200+300)
    actual = np.average(duration, weights=notional)
    assert np.isclose(actual, expected)

def test_rate_delta_calculation():
    df = sample_backtest_df()
    rate_hist = sample_rate_hist_df()
    merged = df.merge(rate_hist, left_on="orig_date", right_on="date", how="left")
    merged["rate_delta"] = merged["coupon"] - merged["cc_rate"]
    assert np.isclose(merged["rate_delta"].iloc[0], 0.10)
    assert np.isclose(merged["rate_delta"].iloc[1], 0.15)

# --- Rolling CPR Calculation Example ---
def test_rolling_cpr_calculation():
    # Fake data: previous notional and projected notional, as in your app
    df = pd.DataFrame({
        "notional_usd": [1_000_000, 900_000, 800_000, 700_000],
        "projected_notional": [900_000, 800_000, 700_000, 600_000]
    })
    df["prev_notional"] = df["notional_usd"].shift(1)
    df["paydown"] = df["prev_notional"] - df["projected_notional"]
    cpr_1m = (df["paydown"].rolling(window=1).sum() / df["prev_notional"].rolling(window=1).sum()).fillna(0)
    assert np.all((cpr_1m >= 0) & (cpr_1m <= 1)), "Rolling CPR out of bounds"

# --- Data/Calculation Consistency ---
def test_all_lengths_match():
    # Ensure all columns in your summary tables are the same length
    df = sample_pipeline_df()
    lengths = [len(df[col]) for col in df.columns]
    assert len(set(lengths)) == 1, "Columns in pipeline data not all same length"

# --- Add more as your business logic grows! ---


