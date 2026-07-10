"""
IPO first-5-day overnight vs. intraday return decomposition.
Uses the same symbol list and data source as post_ipo_neglect_test.py.
Overnight: (Open[t] - Close[t-1]) / Close[t-1]
Intraday:  (Close[t] - Open[t])   / Open[t]
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR = "data"
SYMBOLS_FILE = "config/symbols.txt"
XU100_PATH = "data_market/XU100.csv"
FIRST_N_DAYS = 5

xu100_df = pd.read_csv(XU100_PATH, parse_dates=["Date"])
xu100_df = xu100_df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
xu100_close = xu100_df.set_index("Date")["Close"]
xu100_ret = xu100_close.pct_change()  # close-to-close XU100 daily return

with open(SYMBOLS_FILE) as f:
    symbols = [l.strip() for l in f if l.strip()]

EXCLUDE = {"USDTRY", "USDTRY=X", "ISKUR"}
records = []

for sym in symbols:
    if sym in EXCLUDE:
        continue
    path = os.path.join(DATA_DIR, f"{sym}.csv")
    if not os.path.exists(path):
        continue
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        continue
    df = df.sort_values("Date").reset_index(drop=True)
    needed = {"Date", "Open", "Close"}
    if not needed.issubset(df.columns):
        continue
    df = df.dropna(subset=["Open", "Close"])
    if len(df) < FIRST_N_DAYS + 1:
        continue

    # IPO date = first valid row (index 0); include day 0 (intraday only) + days 1-4
    # Day 0: no overnight (no previous close), intraday = Open→Close
    row0 = df.iloc[0]
    if row0["Open"] != 0 and pd.notna(row0["Open"]) and pd.notna(row0["Close"]):
        intraday0 = (row0["Close"] - row0["Open"]) / row0["Open"]
        xu100_d0  = xu100_ret.get(row0["Date"], np.nan)
        total_exc0 = intraday0 - xu100_d0 if pd.notna(xu100_d0) else np.nan
        records.append({
            "symbol": sym,
            "date": row0["Date"],
            "day": 0,
            "overnight_bps": 0.0,
            "intraday_bps":  intraday0 * 10000,
            "total_raw_bps": intraday0 * 10000,
            "total_excess_bps": total_exc0 * 10000 if pd.notna(total_exc0) else np.nan,
        })

    for i in range(1, FIRST_N_DAYS):  # days 1-4 (together with day 0 = 5 days)
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if prev["Close"] == 0 or row["Open"] == 0:
            continue
        overnight = (row["Open"] - prev["Close"]) / prev["Close"]
        intraday  = (row["Close"] - row["Open"]) / row["Open"]
        total_raw = (row["Close"] - prev["Close"]) / prev["Close"]
        xu100_on_date = xu100_ret.get(row["Date"], np.nan)
        total_excess = total_raw - xu100_on_date if pd.notna(xu100_on_date) else np.nan
        records.append({
            "symbol": sym,
            "date": row["Date"],
            "day": i,
            "overnight_bps": overnight * 10000,
            "intraday_bps":  intraday  * 10000,
            "total_raw_bps": total_raw * 10000,
            "total_excess_bps": total_excess * 10000 if pd.notna(total_excess) else np.nan,
        })

df_all = pd.DataFrame(records)
print(f"Gözlem sayısı (sembol x gün): {len(df_all)}")
print(f"Sembol sayısı: {df_all['symbol'].nunique()}")

means = df_all[["overnight_bps", "intraday_bps", "total_raw_bps", "total_excess_bps"]].mean()
print(f"\nOrtalamalar (bps/gün, ilk {FIRST_N_DAYS} gün):")
print(f"  Overnight (raw):        {means['overnight_bps']:+.2f} bps")
print(f"  Intraday  (raw):        {means['intraday_bps']:+.2f} bps")
print(f"  Total raw:              {means['total_raw_bps']:+.2f} bps")
print(f"  Total XU100-excess:     {means['total_excess_bps']:+.2f} bps")
print(f"  (Referans: post_ipo_neglect_test = 386.38 bps)")
print(f"  Overnight + Intraday additive check: {means['overnight_bps'] + means['intraday_bps']:+.2f} bps")

out_csv = "data/carry_trade/ipo_overnight_intraday_decomposition.csv"
df_all.to_csv(out_csv, index=False)
print(f"\nKaydedildi: {out_csv}")

# --- Grafik ---
labels = ["Total (5-day avg)", "Overnight Component", "Intraday Component"]
values = [means["total_raw_bps"], means["overnight_bps"], means["intraday_bps"]]
colors = ["#3498db", "#2ecc71" if means["overnight_bps"] >= 0 else "#e74c3c",
          "#2ecc71" if means["intraday_bps"] >= 0 else "#e74c3c"]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
ax.axhline(0, color="black", linewidth=1.2)

for bar, val in zip(bars, values):
    offset = 8 if val >= 0 else -15
    va = "bottom" if val >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
            f"{val:+.1f} bps", ha="center", va=va, fontsize=11, fontweight="bold")

ax.set_title("IPO First-Day Effect: Overnight vs. Intraday Decomposition",
             fontsize=12, fontweight="bold", pad=14)
ax.set_ylabel("Return (basis points)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(0.99, 0.01,
         "Not: Overnight gap market open'dan önce oluşur — normal seans emirleriyle yakalanamaz.",
         ha="right", va="bottom", fontsize=8, color="gray")

plt.tight_layout()
out_png = "visualizations/ipo_overnight_vs_intraday.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Grafik kaydedildi: {out_png}")
