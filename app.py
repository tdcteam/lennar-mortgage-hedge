import os
import json
import subprocess
import tempfile
from datetime import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- AI Commentary Setup ---
try:
    import openai
    client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
    def get_ai_commentary(prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": (
                        "You are a world-class mortgage pipeline analytics and capital markets advisor. "
                        "Provide clear, concise, and actionable commentary for Doug and the executive team. "
                        "Highlight what matters and relate insights to Lennar’s business context."
                    )},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI commentary not available: {str(e)}"
except Exception as e:
    def get_ai_commentary(prompt):
        return f"AI commentary not available: {str(e)}"

from hedge_model import simulate_hedge_pnl

st.set_page_config(page_title="Mortgage Hedge Simulator", layout="wide")

# Ensure working directory for relative paths
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except:
    pass

# ----------- REQUIRED COLUMN MAPS -----------
REQUIRED_COLUMNS = {
    "pipeline": ["notional", "duration", "pull_through", "coupon", "orig_date"],
    "backtest": ["notional_usd", "coupon", "orig_date"],
    "history_cc": ["DATE", "MORTGAGE30US"],
    "history_rates": ["DATE", "DGS10"]
}

def check_csv_columns(file_path, expected_cols, section_name):
    try:
        df = pd.read_csv(file_path)
        for c in expected_cols:
            if c not in df.columns:
                st.error(f"Missing column '{c}' in {section_name} file!")
                return False
        return True
    except Exception as e:
        st.error(f"Failed to load {section_name} file: {e}")
        return False

# ----------- SIDEBAR: Data Source & Upload Controls -----------
st.sidebar.title("📡 Data Source Controls")
st.sidebar.caption(
    "Upload new files below to override defaults (live/repo files are used unless you upload replacements)."
)

def file_selector(label, default_path, key, file_types):
    up = st.sidebar.file_uploader(label, type=file_types, key=key)
    if up:
        temp_path = f"uploaded_{key}.csv"
        with open(temp_path, "wb") as f:
            f.write(up.read())
        return temp_path
    return default_path

# Default file paths
STRATEGY_FILES = {
    "builder_forward": "builder_forward.csv",
    "servicing":       "servicing.csv",
    "securitization":  "securitization.csv"
}

# Allow upload/replace of pipeline CSVs
pipeline_paths = {}
for strat, path in STRATEGY_FILES.items():
    pipeline_paths[strat] = file_selector(
        f"Upload {strat.replace('_', ' ').title()} CSV", path, f"file_{strat}", ["csv"]
    )

# Backtesting files (for backtesting tab)
backtest_pipe = file_selector("Upload Builder Forward Backtest CSV", "builder_forward_backtest.csv", "backtest_pipe", ["csv"])
backtest_cc = file_selector("Upload Historical CC Rate CSV", "history_cc.csv", "backtest_cc", ["csv"])
backtest_ts = file_selector("Upload Historical Treasury CSV", "history_rates.csv", "backtest_ts", ["csv"])

st.sidebar.markdown("---")

# ----------- Load & tag dataframes -----------
dfs = []
for strat, path in pipeline_paths.items():
    if not os.path.exists(path):
        st.sidebar.error(f"Missing file: {path}")
        st.stop()
    df = pd.read_csv(path).rename(columns={
        "notional": "notional_usd",
        "duration": "duration_years",
        "pull_through": "pull_through_rate"
    })
    df["strategy_type"] = strat
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# ----------- Decay logic -----------
def apply_decay(n, age, strat):
    if strat == "builder_forward":
        return n * np.exp(-cpr_rate * age)
    if strat == "servicing":
        return n * np.exp(-0.04 * age)
    if strat == "securitization":
        return n * np.exp(-0.15 * min(age,1) - 0.05 * max(age-1,0))
    return n

COLORS = {"navy": "#003366", "gold": "#FFCC00"}

# ----------- AG-Grid Premium Table Setup -----------
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

def premium_table(df, key):
    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination()
        gb.configure_default_column(editable=False, groupable=True)
        gb.configure_side_bar()
        grid_options = gb.build()
        AgGrid(df, gridOptions=grid_options, theme="balham", enable_enterprise_modules=False, key=key)
    else:
        st.dataframe(df)

# ----------- Tabs -----------
tab_sim, tab_scen, tab_back, tab_upload, tab_validate = st.tabs([
    "🔧 Simulator", 
    "💾 Scenarios", 
    "📊 Backtesting",
    "📥 Data Upload",
    "✅ Validation"
])

# ----------- SIMULATOR TAB -----------
with tab_sim:
    st.markdown("## 🏦 Mortgage Hedge Simulator")
    st.caption("A next-generation, AI-powered pipeline risk & hedge analytics platform, built custom for Lennar.")

    with st.expander("ℹ️ What is this tool?"):
        st.write(
            "This platform provides best-in-class analytics, simulation, and reporting for mortgage pipeline risk, "
            "hedging, and capital markets management. All metrics, charts, and AI commentary are tailored for Lennar's operations."
        )

    st.sidebar.header("🔧 Hedge Simulation Controls")
    coverage       = st.sidebar.slider("Coverage Ratio", 0.5, 1.0, 0.9, 0.05)
    rate_vol       = st.sidebar.slider("Rate Volatility (σ)", 0.001, 0.02, 0.005, 0.001)
    cpr_rate       = st.sidebar.slider("Builder CPR Rate", 0.0, 0.5, 0.10, 0.01)
    model_choice   = st.sidebar.radio("Rate Model", ["Normal Shocks", "Hull-White"])
    sim_count      = st.sidebar.number_input("Number of Simulations", 1000, 50000, 10000, step=1000)
    duration_range = st.sidebar.slider(
        "Duration Range (yrs)", 0.05, 0.25, (0.10, 0.20), step=0.01,
        help="Builder-forward loans are typically hedged for 2–13 weeks."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Buckets")
    available_strats = ["builder_forward"]
    future_strats = ["servicing", "securitization"]
    selected_strats  = st.sidebar.multiselect(
        "Active Buckets",
        available_strats,
        default=available_strats,
        help="Other buckets will be available as Lennar expands its hedging strategies."
    )
    for b in future_strats:
        st.sidebar.checkbox(
            f"{b.replace('_',' ').title()} (coming soon)",
            value=False, key=f"{b}_disabled", disabled=True, help="Activate when Lennar retains servicing or securitizes loans.")

    # Scenario Management
    SCEN_FILE = "scenarios.json"
    st.sidebar.header("💾 Scenario Management")
    if os.path.exists(SCEN_FILE):
        with open(SCEN_FILE) as f:
            scenarios = json.load(f)
    else:
        scenarios = []

    save_name = st.sidebar.text_input("Save Scenario As", f"Scenario {len(scenarios)+1}")
    if st.sidebar.button("Save Scenario"):
        scenarios.append({
            "name": save_name,
            "coverage": coverage,
            "rate_vol": rate_vol,
            "cpr_rate": cpr_rate,
            "model": model_choice,
            "sim_count": sim_count,
            "duration_range": duration_range,
            "strategies": selected_strats
        })
        with open(SCEN_FILE, "w") as f:
            json.dump(scenarios, f, indent=2)
        subprocess.run(["git", "add", SCEN_FILE], check=False)
        subprocess.run(["git", "commit", "-m", f"Save scenario {save_name}"], check=False)
        st.sidebar.success(f"Saved '{save_name}'")

    if st.sidebar.button("Clear All Scenarios"):
        scenarios.clear()
        with open(SCEN_FILE, "w") as f:
            json.dump(scenarios, f)
        st.sidebar.success("All scenarios cleared!")

    # Pipeline Totals by Bucket (PREMIUM UI)
    st.sidebar.header("📊 Pipeline Totals by Bucket")
    pre = df_all[df_all.strategy_type.isin(selected_strats)].groupby("strategy_type")["notional_usd"].sum()/1e6
    df_all["decayed_notional"] = df_all.apply(
        lambda r: apply_decay(r["notional_usd"], r["duration_years"], r["strategy_type"]), axis=1
    )
    post = df_all[df_all.strategy_type.isin(selected_strats)].groupby("strategy_type")["decayed_notional"].sum()/1e6

    for strat in selected_strats:
        st.sidebar.write(
            f"**{strat.replace('_',' ').title()}**  ➔  "
            f"Pre-CPR ${pre.get(strat,0):,.1f}M → Post-CPR ${post.get(strat,0):,.1f}M"
        )
        avg_dur = df_all[df_all.strategy_type==strat]["duration_years"].mean()
        decayed_dur = np.average(
            df_all[df_all.strategy_type==strat]["duration_years"],
            weights=df_all[df_all.strategy_type==strat]["decayed_notional"] + 1e-9
        )
        st.sidebar.write(
            f"• Avg Pipeline Duration: {avg_dur:.2f} yrs (Post-decay: ~{decayed_dur:.2f} yrs)"
        )

    def hull_white_vol(a,sigma,T):
        return sigma*np.sqrt((1-np.exp(-2*a*T))/(2*a))

    def run_engine(csv_file, cov, vol, n_sims, model):
        if model=="Hull-White":
            df_dur = pd.read_csv(csv_file)
            T = df_dur["duration_years"].mean()
            vol_hw = hull_white_vol(0.03,0.01,T)
            return simulate_hedge_pnl(csv_file, coverage=cov, rate_vol=vol_hw, n_sims=n_sims)
        return simulate_hedge_pnl(csv_file, coverage=cov, rate_vol=vol, n_sims=n_sims)

    def multi_engine(df_all, cov, vol, n_sims, model, strategies):
        summ = {}
        res  = {}
        for strat in strategies:
            df_t = df_all[df_all.strategy_type==strat].copy()
            df_t["notional_usd"] = df_t.apply(
                lambda r: apply_decay(r["notional_usd"], r["duration_years"], strat), axis=1
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            df_t[["notional_usd","duration_years","pull_through_rate"]].to_csv(tmp.name,index=False)
            s_i,r_i = run_engine(tmp.name,cov,vol,n_sims,model)
            summ[strat] = s_i
            res[strat]  = r_i
        pnl_arrs = [res[s]["pnl"] for s in strategies]
        total_pnl = np.sum(np.vstack(pnl_arrs),axis=0)
        tot_not  = sum(summ[s]["total_notional"] for s in strategies)
        avg_dur  = sum(summ[s]["avg_duration"]*summ[s]["total_notional"] for s in strategies)/tot_not
        summary_total = {
            "coverage":      cov,
            "total_notional":tot_not,
            "avg_duration":  avg_dur,
            "mean_pnl":      total_pnl.mean(),
            "std_pnl":       total_pnl.std(),
            "p5_pnl":        np.percentile(total_pnl,5),
            "p95_pnl":       np.percentile(total_pnl,95)
        }
        return summary_total,{"pnl":total_pnl},summ,res

    summary,results,bucket_summ,bucket_res = multi_engine(
        df_all,coverage,rate_vol,sim_count,model_choice,selected_strats
    )

    st.markdown("#### Key Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns([1,1,1,1])
    kpi1.metric("Coverage",      f"{coverage:.0%}", help="Percent of pipeline covered by hedge instruments.")
    kpi2.metric("Mean P&L ($M)",  f"{summary['mean_pnl']/1e6:,.2f}", help="Expected profit/loss of hedge.")
    kpi3.metric("5% VaR ($M)",    f"{summary['p5_pnl']/1e6:,.2f}", help="Worst-case (5th percentile) loss.")
    kpi4.metric("Builder CPR",    f"{cpr_rate:.1%}", help="Conditional prepayment rate for builder-forward loans.")

    st.markdown("#### 📋 Overall Summary")
    df_overall = pd.DataFrame({
        "Metric":["Total Hedge Notional ($M)","Avg Duration (yrs)",
                  "Mean P&L ($M)","STD P&L ($M)","5% VaR ($M)","95% VaR ($M)"],
        "Value":[
            round(summary["total_notional"]/1e6,2),
            round(summary["avg_duration"],2),
            round(summary["mean_pnl"]/1e6,2),
            round(summary["std_pnl"]/1e6,2),
            round(summary["p5_pnl"]/1e6,2),
            round(summary["p95_pnl"]/1e6,2)
        ]
    })
    premium_table(df_overall, key="overall_summary")

    excel_buffer = BytesIO()
    df_overall.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)
    st.download_button(
        "Export Results (Excel)",
        data=excel_buffer,
        file_name="simulation_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("#### 📂 Bucket-level Summaries")
    rows = {
        "Notional ($M)":lambda s:s["total_notional"]/1e6,
        "Duration (yrs)":lambda s:s["avg_duration"],
        "Mean P&L":      lambda s:s["mean_pnl"]/1e6,
        "STD P&L":       lambda s:s["std_pnl"]/1e6,
        "5% VaR":        lambda s:s["p5_pnl"]/1e6,
        "95% VaR":       lambda s:s["p95_pnl"]/1e6
    }
    df_buckets = pd.DataFrame({
        strat:[fn(bucket_summ[strat]) for fn in rows.values()]
        for strat in selected_strats
    },index=list(rows.keys()))
    premium_table(df_buckets, key="bucket_summary")

    st.markdown("#### 📈 Total P&L Distribution")
    st.caption("Shows the full range of simulated outcomes—where your pipeline P&L might land under current assumptions.")
    fig,ax=plt.subplots(figsize=(8,4))
    data = results["pnl"]/1e6
    ax.hist(data,bins=50,facecolor=COLORS["navy"],edgecolor="black",alpha=0.8)
    m=data.mean();v=np.percentile(data,5)
    ax.axvline(m,color=COLORS["gold"],linestyle="--",label=f"Mean={m:.2f}")
    ax.axvline(v,color="red",linestyle="--",label=f"5% VaR={v:.2f}")
    ax.set_xlabel("P&L ($M)");ax.set_ylabel("Frequency")
    ax.legend();ax.grid(axis="y",linestyle="--",alpha=0.5)
    st.pyplot(fig,use_container_width=True)

    st.markdown("#### 📊 P&L Cumulative Distribution (CDF)")
    st.caption("How likely is it that P&L will be better than a threshold? This graph makes risk explicit.")
    fig,ax=plt.subplots(figsize=(8,4))
    sd=np.sort(data);p=np.linspace(0,100,len(sd))
    ax.plot(sd,p,color=COLORS["navy"],linewidth=2)
    ax.axhline(5,color="red",linestyle="--",label="5% VaR")
    ax.set_xlabel("P&L ($M)");ax.set_ylabel("Cumulative %")
    ax.legend();ax.grid(linestyle="--",alpha=0.5)
    st.pyplot(fig,use_container_width=True)

    st.markdown("#### 📦 P&L Boxplot")
    st.caption("Visual summary of distribution, median, outliers—good for quick risk benchmarking.")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.boxplot(data,vert=False,patch_artist=True,
               boxprops=dict(facecolor=COLORS["navy"],color=COLORS["navy"]),
               medianprops=dict(color=COLORS["gold"],linewidth=2))
    ax.set_xlabel("P&L ($M)");ax.grid(axis="x",linestyle="--",alpha=0.5)
    st.pyplot(fig,use_container_width=True)

    st.markdown("#### 🔖 Bucket P&L Overlays")
    st.caption("Compare risk/return by strategy bucket (servicing, builder, securitization). Great for what-if analysis.")
    fig,ax=plt.subplots(figsize=(8,4))
    for strat in selected_strats:
        ax.hist(bucket_res[strat]["pnl"]/1e6,bins=40,alpha=0.4,label=strat)
    ax.set_xlabel("P&L ($M)");ax.set_ylabel("Frequency")
    ax.legend();ax.grid(linestyle="--",alpha=0.5)
    st.pyplot(fig,use_container_width=True)

    ai_prompt = f"""Summary Table:
{df_overall}
Bucket Summaries:
{df_buckets}
Notable model parameters: Coverage={coverage:.0%}, CPR={cpr_rate:.2%}, Model={model_choice}, Simulations={sim_count}
"""
    st.markdown("#### 🤖 AI-Assisted Commentary")
    ai_text = get_ai_commentary(ai_prompt)
    st.info(ai_text)

# ----------- SCENARIOS TAB -----------
with tab_scen:
    st.markdown("## 💾 Saved Scenarios")
    st.caption("Save, load, and compare scenario results—see how risk changes under different assumptions.")
    for sc in scenarios:
        cols=st.columns([4,1,1])
        cols[0].write(f"• {sc['name']}")
        if cols[1].button("▶ Load",key=f"load_{sc['name']}"):
            st.session_state.update({
                "coverage":sc["coverage"],
                "rate_vol":sc["rate_vol"],
                "cpr_rate":sc["cpr_rate"],
                "model_choice":sc["model"],
                "sim_count":sc["sim_count"],
                "duration_range":tuple(sc["duration_range"]),
                "selected_strats":sc["strategies"]
            })
        if cols[2].button("❌ Delete",key=f"del_{sc['name']}"):
            scenarios=[s for s in scenarios if s["name"]!=sc["name"]]
            with open(SCEN_FILE,"w") as f: json.dump(scenarios,f,indent=2)
            st.warning(f"Deleted '{sc['name']}'")

    st.markdown("---")
    st.markdown("### 🔍 Compare P&L Histograms")
    names=[s["name"] for s in scenarios]
    sel=st.multiselect("Pick two",names,default=names[:2],key="cmp")
    if len(sel)==2:
        a=next(s for s in scenarios if s["name"]==sel[0])
        b=next(s for s in scenarios if s["name"]==sel[1])
        _,r1,_,_ = multi_engine(df_all,a["coverage"],a["rate_vol"],a["sim_count"],a["model"],a["strategies"])
        _,r2,_,_ = multi_engine(df_all,b["coverage"],b["rate_vol"],b["sim_count"],b["model"],b["strategies"])
        fig,ax=plt.subplots(figsize=(8,4))
        ax.hist(r1["pnl"]/1e6,bins=40,alpha=0.5,label=sel[0])
        ax.hist(r2["pnl"]/1e6,bins=40,alpha=0.5,label=sel[1])
        ax.set_xlabel("P&L ($M)");ax.set_ylabel("Frequency")
        ax.legend();ax.grid(linestyle="--",alpha=0.5)
        st.pyplot(fig,use_container_width=True)
    else:
        st.info("Select exactly two scenarios to compare.")

    scenario_comment_prompt = f"""Saved Scenarios:
{json.dumps(scenarios, indent=2)}
Explain how scenario management and comparison can be leveraged by Lennar to improve their risk and pipeline management.
"""
    st.markdown("#### 🤖 AI-Assisted Commentary")
    scenario_ai_text = get_ai_commentary(scenario_comment_prompt)
    st.info(scenario_ai_text)

# ----------- BACKTESTING TAB -----------
with tab_back:
    st.markdown("## 📊 Backtesting Engine: Pipeline Realization and Hedge Performance")
    st.caption("Analyze historical realized P&L, notional decay, and rate trends. Supports both live and uploaded data.")

    pipe_file = backtest_pipe
    cc_file = backtest_cc
    ts_file = backtest_ts

    missing = []
    for f in [pipe_file, cc_file, ts_file]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        st.warning(f"Missing required files for backtesting: {', '.join(missing)}. "
                   "Please upload all three: 'builder_forward_backtest.csv', 'history_cc.csv', and 'history_rates.csv'.")
        st.stop()

    df = pd.read_csv(pipe_file, parse_dates=["orig_date"])
    hist_cc = pd.read_csv(cc_file, parse_dates=["DATE"]).rename(columns={"DATE": "date", "MORTGAGE30US": "cc_rate"})
    hist_ts = pd.read_csv(ts_file, parse_dates=["DATE"]).rename(columns={"DATE": "date", "DGS10": "ts_rate"})
    rate_hist = pd.merge(hist_cc, hist_ts, on="date", how="inner").dropna()
    rate_hist["cc_rate"] = pd.to_numeric(rate_hist["cc_rate"], errors="coerce")
    rate_hist["ts_rate"] = pd.to_numeric(rate_hist["ts_rate"], errors="coerce")
    rate_hist = rate_hist.dropna()

    # ---- Decay Slider ----
    st.markdown("### Select Decay Rate for Backtest")
    user_decay_rate = st.slider(
        "Select Decay Rate (annualized, e.g. 0.10 = 10%)", 0.01, 0.50, 0.10, 0.01,
        help="Decay rate determines speed of paydown. 10% typical for pipeline run-off."
    )

    st.markdown("### Select Backtest Window")
    min_date, max_date = rate_hist["date"].min(), rate_hist["date"].max()
    default_start = max(min_date, pd.to_datetime("2023-10-01"))
    default_end = max_date
    d1, d2 = st.date_input(
        "Backtest Window:",
        [default_start, default_end],
        min_value=min_date, max_value=max_date
    )
    d1, d2 = pd.to_datetime(d1), pd.to_datetime(d2)  # Ensure datetime64 for comparison
    window = (rate_hist["date"] >= d1) & (rate_hist["date"] <= d2)
    rate_hist_win = rate_hist.loc[window]

    df_bt = df[(df["orig_date"] >= d1) & (df["orig_date"] <= d2)].copy()

    st.markdown("### Backtest Pipeline Sample")
    st.caption("Sample of loans originated in the backtest window. Columns: notional, origination date, coupon, etc.")
    premium_table(df_bt.head(25), key="bt_sample")

    df_bt = df_bt.merge(rate_hist[["date", "cc_rate"]], left_on="orig_date", right_on="date", how="left", suffixes=('', '_orig'))
    df_bt["rate_delta"] = df_bt["coupon"] - df_bt["cc_rate"]

    dt_months = (d2 - d1).days // 30
    df_bt["projected_notional"] = df_bt["notional_usd"] * np.exp(-user_decay_rate * dt_months/12)

    st.markdown("### 📈 Realized Rate Spread (Time Series)")
    st.caption("Shows how the realized pipeline coupon vs benchmark (30yr CC) evolves over the window.")
    time_df = df_bt.groupby("orig_date").agg({"rate_delta": "mean"}).sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_df.index, time_df["rate_delta"], marker="o", color="#003366")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("Coupon - 30yr CC (%)")
    ax.set_xlabel("Origination Date")
    ax.set_title("Average Rate Spread by Origination Date")
    st.pyplot(fig, use_container_width=True)

    st.markdown("### 📉 Notional Decay Distribution")
    st.caption("Histogram of projected notional after decay, based on actual pipeline performance over the window.")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(df_bt["projected_notional"]/1e6, bins=40, alpha=0.7, color="#003366", label="Projected Notional ($M)")
    ax.set_xlabel("Projected Notional ($M)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig, use_container_width=True)

    st.markdown("### 🚀 Realized Prepayment Speeds (1M, 3M Rolling CPR)")
    st.caption("Tracks the speed of paydowns in the pipeline. Useful for tracking market cycles or seasonal effects.")
    df_bt = df_bt.sort_values("orig_date")
    df_bt["prev_notional"] = df_bt["notional_usd"].shift(1)
    df_bt["paydown"] = df_bt["prev_notional"] - df_bt["projected_notional"]
    cpr_1m = (df_bt["paydown"].rolling(window=1).sum() / df_bt["prev_notional"].rolling(window=1).sum()).fillna(0)
    cpr_3m = (df_bt["paydown"].rolling(window=3).sum() / df_bt["prev_notional"].rolling(window=3).sum()).fillna(0)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_bt["orig_date"], cpr_1m * 100, label="1M CPR", color="#FFCC00")
    ax.plot(df_bt["orig_date"], cpr_3m * 100, label="3M CPR", color="#003366", linestyle="--")
    ax.set_ylabel("CPR (%)")
    ax.set_xlabel("Origination Date")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("### ⬇️ Export Backtest Data")
    excel_buffer_bt = BytesIO()
    df_bt.to_excel(excel_buffer_bt, index=False, engine="openpyxl")
    excel_buffer_bt.seek(0)
    st.download_button(
        "Export Backtest Sample (Excel)",
        data=excel_buffer_bt,
        file_name="backtest_sample.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    backtest_prompt = f"""
    Backtest Window: {d1.date()} to {d2.date()}
    Pipeline Count: {len(df_bt)}
    Total Projected Notional ($M): {round(df_bt['projected_notional'].sum()/1e6, 2)}
    Mean Coupon-CC Spread (%): {round(df_bt['rate_delta'].mean(), 3)}
    Provide an executive summary and what stands out about pipeline realization vs historical market rates. Suggest action items for the Lennar team.
    """
    st.markdown("#### 🤖 AI-Assisted Commentary")
    backtest_commentary = get_ai_commentary(backtest_prompt)
    st.info(backtest_commentary)

    st.markdown("---")
    st.caption("All analytics, charts, and commentary above are fully reproducible using your own live data or uploads. For questions or further enhancements, contact the platform admin.")

# ----------- DATA UPLOAD TAB -----------
with tab_upload:
    st.markdown("## 📥 Real-Time & Bulk Data Upload")
    st.caption("Add or update pipeline data instantly. Upload a new CSV, preview results, and load into the platform in real-time.")

    upload_type = st.radio("What would you like to upload?", [
        "Pipeline Data (CSV)",
        "Backtest Pipeline Data (CSV)",
        "Historical CC Rates (CSV)",
        "Historical Treasury Rates (CSV)"
    ])

    # Set up what we're uploading
    if upload_type == "Pipeline Data (CSV)":
        expected_cols = REQUIRED_COLUMNS["pipeline"]
        default_file = "builder_forward.csv"
        label = "Pipeline"
    elif upload_type == "Backtest Pipeline Data (CSV)":
        expected_cols = REQUIRED_COLUMNS["backtest"]
        default_file = "builder_forward_backtest.csv"
        label = "Backtest"
    elif upload_type == "Historical CC Rates (CSV)":
        expected_cols = REQUIRED_COLUMNS["history_cc"]
        default_file = "history_cc.csv"
        label = "Historical CC"
    elif upload_type == "Historical Treasury Rates (CSV)":
        expected_cols = REQUIRED_COLUMNS["history_rates"]
        default_file = "history_rates.csv"
        label = "Historical Treasury"
    else:
        st.warning("Choose a data type to upload.")
        st.stop()

    st.markdown("### 1️⃣ Upload File")
    up = st.file_uploader(f"Upload {label} CSV", type=["csv"], key=f"upload_{label.lower()}")

    if up:
        tmp_path = f"uploaded_{label.lower()}_realtime.csv"
        with open(tmp_path, "wb") as f:
            f.write(up.read())
        if not check_csv_columns(tmp_path, expected_cols, f"{label} Upload"):
            st.error(f"❌ Uploaded file is missing columns: {expected_cols}. Please check your file.")
            st.stop()
        df_preview = pd.read_csv(tmp_path)
        st.markdown("### 2️⃣ Preview Data (First 20 Rows)")
        st.dataframe(df_preview.head(20))
        if st.button(f"🚀 Load Into Platform ({label})"):
            if os.path.exists(default_file):
                os.rename(default_file, default_file + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.replace(tmp_path, default_file)
            st.success(f"✅ {label} file replaced! The platform will now use this data.")
            st.info("**To reload this data immediately, refresh the app.**")
        else:
            st.info("Nothing has been loaded yet. Upload and preview your data, then click Load Into Platform.")
    else:
        st.info("Upload a CSV file to get started.")

    st.markdown("---")
    st.markdown("### 🔗 **(Coming Soon)**: Live Data Feed Integration")
    st.caption("Connect directly to cloud, SFTP, or API feeds for true real-time automation. Ask us about custom connectors!")

    st.markdown("### 📒 Audit Trail")
    for fname in [default_file]:
        bak_files = [f for f in os.listdir('.') if f.startswith(fname+'.bak_')]
        if bak_files:
            st.write(f"Backups found for {fname}:")
            for f in sorted(bak_files, reverse=True):
                st.code(f)

# ----------- VALIDATION TAB (Example: simple tests, future expansion) -----------
with tab_validate:
    st.markdown("## ✅ Best-in-Class Data Validation & Unit Tests")
    st.caption("Automated schema, sanity, and consistency checks for your core data files. Click any warning for more details.")

    # --- Define checks per file type ---
    FILE_TESTS = {
        "Pipeline Data": {
            "path": "builder_forward.csv",
            "required": ["notional", "duration", "pull_through", "coupon", "orig_date"],  # Edit as needed!
            "examples": {"notional": "positive number", "duration": "0 < x < 1", "pull_through": "0 < x <= 1", "orig_date": "valid date", "coupon": "reasonable interest rate"}
        },
        "Backtest Pipeline Data": {
            "path": "builder_forward_backtest.csv",
            "required": ["notional_usd", "coupon", "orig_date"],
            "examples": {"notional_usd": "positive number", "coupon": "reasonable interest rate", "orig_date": "valid date"}
        },
        "Historical CC Rates": {
            "path": "history_cc.csv",
            "required": ["DATE", "MORTGAGE30US"],
            "examples": {"DATE": "date", "MORTGAGE30US": "reasonable interest rate"}
        },
        "Historical Treasury Rates": {
            "path": "history_rates.csv",
            "required": ["DATE", "DGS10"],
            "examples": {"DATE": "date", "DGS10": "reasonable interest rate"}
        }
    }

    result_summary = []

    for label, test in FILE_TESTS.items():
        path = test["path"]
        required = test["required"]
        examples = test.get("examples", {})
        st.markdown(f"### {label}")

        if not os.path.exists(path):
            st.error(f"❌ Missing file: `{path}`")
            result_summary.append((label, "red", "Missing"))
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            st.error(f"❌ Failed to load `{path}`: {e}")
            result_summary.append((label, "red", "Cannot load"))
            continue

        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            result_summary.append((label, "red", f"Missing columns: {missing}"))
            continue
        else:
            st.success(f"✔️ All required columns present.")

        # --- Sanity checks ---
        sanity_issues = []
        # 1. Check for nulls
        nulls = df[required].isnull().sum()
        if nulls.any():
            details = nulls[nulls > 0].to_dict()
            st.warning(f"⚠️ Null values in columns: {details}")
            sanity_issues.append(f"Nulls: {details}")

        # 2. Check datatypes & values (custom logic by file)
        try:
            if "notional" in df.columns:
                neg_not = df[df["notional"] <= 0]
                if not neg_not.empty:
                    st.warning(f"⚠️ Notional <= 0 for {len(neg_not)} rows.")
                    sanity_issues.append(f"Negative/zero notional ({len(neg_not)})")
            if "notional_usd" in df.columns:
                neg_not = df[df["notional_usd"] <= 0]
                if not neg_not.empty:
                    st.warning(f"⚠️ notional_usd <= 0 for {len(neg_not)} rows.")
                    sanity_issues.append(f"Negative/zero notional_usd ({len(neg_not)})")
            if "duration" in df.columns:
                bad = df[(df["duration"] <= 0) | (df["duration"] > 1)]
                if not bad.empty:
                    st.warning(f"⚠️ Unusual duration (should be 0 < x < 1 year): {len(bad)} rows.")
                    sanity_issues.append(f"Out-of-bounds duration ({len(bad)})")
            if "duration_years" in df.columns:
                bad = df[(df["duration_years"] <= 0) | (df["duration_years"] > 1)]
                if not bad.empty:
                    st.warning(f"⚠️ Unusual duration_years (should be 0 < x < 1 year): {len(bad)} rows.")
                    sanity_issues.append(f"Out-of-bounds duration_years ({len(bad)})")
            if "pull_through" in df.columns:
                bad = df[(df["pull_through"] < 0) | (df["pull_through"] > 1)]
                if not bad.empty:
                    st.warning(f"⚠️ pull_through not in [0,1]: {len(bad)} rows.")
                    sanity_issues.append(f"pull_through out of range ({len(bad)})")
            if "pull_through_rate" in df.columns:
                bad = df[(df["pull_through_rate"] < 0) | (df["pull_through_rate"] > 1)]
                if not bad.empty:
                    st.warning(f"⚠️ pull_through_rate not in [0,1]: {len(bad)} rows.")
                    sanity_issues.append(f"pull_through_rate out of range ({len(bad)})")
            if "coupon" in df.columns:
                bad = df[(df["coupon"] < 0.01) | (df["coupon"] > 0.15)]
                if not bad.empty:
                    st.warning(f"⚠️ Unusual coupon rate (<1% or >15%): {len(bad)} rows.")
                    sanity_issues.append(f"coupon unusual ({len(bad)})")
            if "MORTGAGE30US" in df.columns:
                bad = df[(df["MORTGAGE30US"] < 0.01) | (df["MORTGAGE30US"] > 0.15)]
                if not bad.empty:
                    st.warning(f"⚠️ MORTGAGE30US out-of-bounds: {len(bad)} rows.")
                    sanity_issues.append(f"MORTGAGE30US unusual ({len(bad)})")
            if "DGS10" in df.columns:
                bad = df[(df["DGS10"] < 0.01) | (df["DGS10"] > 0.15)]
                if not bad.empty:
                    st.warning(f"⚠️ DGS10 out-of-bounds: {len(bad)} rows.")
                    sanity_issues.append(f"DGS10 unusual ({len(bad)})")
            # Date checks
            date_cols = [c for c in df.columns if "date" in c.lower()]
            for c in date_cols:
                try:
                    _ = pd.to_datetime(df[c])
                except Exception as e:
                    st.warning(f"⚠️ Some values in {c} are not valid dates.")
                    sanity_issues.append(f"{c} invalid dates")
        except Exception as e:
            st.error(f"Sanity checks error: {e}")
            sanity_issues.append(str(e))

        # --- Feedback summary (plain info, no HTML/unsafe_allow_html) ---
        if sanity_issues:
            st.info(f"Validation Warnings: {sanity_issues}")
            result_summary.append((label, "yellow", sanity_issues))
        else:
            st.success("No major data issues found! 🎉")
            result_summary.append((label, "green", "Clean"))

        st.caption("Sample Data:")
        st.dataframe(df.head(8))

    # --- Summary Dashboard ---
    st.markdown("---")
    st.markdown("### 🔎 Validation Summary")
    cols = st.columns([2,1,5])
    cols[0].markdown("**File**")
    cols[1].markdown("**Status**")
    cols[2].markdown("**Notes**")
    status_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    for label, stat, note in result_summary:
        cols[0].write(label)
        cols[1].write(f"{status_emoji.get(stat, '')} {stat.upper()}")
        cols[2].write(str(note))

    # --- AI Commentary on Data Health ---
    health_report = "\n".join([f"{lbl}: {stat} - {note}" for lbl, stat, note in result_summary])
    ai_health_prompt = f"""
    Here is a summary of the data validation for mortgage analytics:
    {health_report}
    Give a 2-3 sentence summary for Doug and Lennar on data readiness and any issues to prioritize.
    """
    st.markdown("#### 🤖 AI Commentary on Data Health")
    ai_health = get_ai_commentary(ai_health_prompt)
    st.info(ai_health)

    st.markdown("---")
    st.caption("Want even deeper checks or auto-remediation? Contact the platform team for custom data auditing!")

