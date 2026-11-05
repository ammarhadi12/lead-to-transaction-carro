import pandas as pd
import numpy as np

# ---------------------------------------------------------
# TASK 1: DATA CLEANING & PIPELINE HYGIENE
# ---------------------------------------------------------
leads = pd.read_csv("Dataset - sourcing_leads.csv")
tx = pd.read_csv("Dataset - transactions.csv")

print(f"Raw counts -> Leads: {len(leads)}, Transactions: {len(tx)}")

# 1. Remove duplicate lead_ids in transactions (theres 5 duplicate rows at bottom of file)
dups = tx[tx.duplicated(subset=["lead_id"], keep=False)]
print(f"Found {len(tx) - tx['lead_id'].nunique()} duplicate transaction rows:")
print(dups[["transaction_id", "lead_id", "handover_date", "buy_price"]])

tx_clean = tx.drop_duplicates(subset=["lead_id"], keep="first").copy()

# 2. Parse dates and extract lead cohort month (July = 7, August = 8)
leads["created_at"] = pd.to_datetime(leads["created_at"])
leads["month"] = leads["created_at"].dt.month
leads["car_age"] = 2026 - leads["manufacture_year"]

# 3. Merge datasets on lead_id
df = leads.merge(tx_clean, on="lead_id", how="left")

# 4. Fix buy_price == 0 (3 rows) by imputing with quoted_indicative_price
zero_buy = df["buy_price"] == 0
print(f"\nImputing {zero_buy.sum()} rows where buy_price == 0 with quoted_indicative_price.")
df.loc[zero_buy, "buy_price"] = df.loc[zero_buy, "quoted_indicative_price"]

# 5. Handle negative days_to_sell (4 rows) -> Set to NaN so we don't lose valid GP rows
neg_days = df["days_to_sell"] < 0
print(f"Setting {neg_days.sum()} negative days_to_sell values to NaN.")
df.loc[neg_days, "days_to_sell"] = np.nan


# ---------------------------------------------------------
# TASK 2A: FUNNEL CONVERSION ANALYSIS
# ---------------------------------------------------------
print("\n--- 1. OVERALL MONTHLY FUNNEL CONVERSION ---")
overall_funnel = pd.crosstab(df["month"], df["inspection_status"], normalize="index") * 100
overall_funnel["total_leads"] = df.groupby("month")["lead_id"].count()
print(overall_funnel.round(2))

print("\n--- 2. CONVERSION & LEAD VOLUME BY CHANNEL (SIMPSON'S PARADOX) ---")
channel_funnel = df.groupby(["month", "channel"]).agg(
    leads=("lead_id", "count"),
    completed_pct=("inspection_status", lambda x: (x == "Completed").mean() * 100),
    noshow_pct=("inspection_status", lambda x: (x == "No Show").mean() * 100),
    cancelled_pct=("inspection_status", lambda x: (x == "Cancelled").mean() * 100),
)
channel_funnel["lead_share_pct"] = (
    channel_funnel.groupby(level=0)["leads"].transform(lambda x: x / x.sum() * 100)
)
print(channel_funnel.round(2))


# ---------------------------------------------------------
# TASK 2B: UNIT ECONOMICS & MARGIN DECAY
# ---------------------------------------------------------
# Filter to completed transactions only
tx_df = df[df["transaction_id"].notna()].copy()

# Calculate metrics
tx_df["spread"] = tx_df["sell_price"] - tx_df["buy_price"]
tx_df["markup_pct"] = (tx_df["spread"] / tx_df["buy_price"]) * 100
tx_df["gp"] = tx_df["sell_price"] - tx_df["buy_price"] - tx_df["refurb_cost"]
tx_df["gp_margin_pct"] = (tx_df["gp"] / tx_df["sell_price"]) * 100
tx_df["cohort"] = np.where(tx_df["manufacture_year"] <= 2018, "2014-2018 (Older)", "2019-2024 (Newer)")

print("\n--- 3. OVERALL MONTHLY UNIT ECONOMICS ---")
monthly_econ = tx_df.groupby("month").agg(
    cars_sold=("transaction_id", "count"),
    avg_buy=("buy_price", "mean"),
    avg_sell=("sell_price", "mean"),
    avg_spread=("spread", "mean"),
    avg_refurb=("refurb_cost", "mean"),
    avg_gp=("gp", "mean"),
    avg_gp_margin_pct=("gp_margin_pct", "mean"),
    avg_days_to_sell=("days_to_sell", "mean"),
)
print(monthly_econ.round(2))

print("\n--- 4. COHORT BREAKDOWN: OLDER (2014-2018) VS NEWER (2019-2024) ---")
cohort_econ = tx_df.groupby(["month", "cohort"]).agg(
    cars_sold=("transaction_id", "count"),
    avg_mileage=("mileage_km", "mean"),
    avg_buy=("buy_price", "mean"),
    avg_sell=("sell_price", "mean"),
    avg_spread=("spread", "mean"),
    avg_markup_pct=("markup_pct", "mean"),
    avg_refurb=("refurb_cost", "mean"),
    avg_gp=("gp", "mean"),
    avg_gp_margin_pct=("gp_margin_pct", "mean"),
    avg_days_to_sell=("days_to_sell", "mean"),
)
print(cohort_econ.round(2))

print("\n--- 5. AUGUST PERFORMANCE BY COHORT & MAKE ---")
aug_make = tx_df[tx_df["month"] == 8].groupby(["cohort", "make"]).agg(
    cars_sold=("transaction_id", "count"),
    avg_spread=("spread", "mean"),
    avg_refurb=("refurb_cost", "mean"),
    avg_gp=("gp", "mean"),
    avg_gp_margin_pct=("gp_margin_pct", "mean"),
    unprofitable_pct=("gp", lambda x: (x < 0).mean() * 100),
)
print(aug_make.round(2))