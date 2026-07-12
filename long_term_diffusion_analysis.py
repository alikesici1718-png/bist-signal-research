"""
Long-term post-KAP-disclosure return analysis: 1, 3, 6, 12 months.
Same data sources as information_diffusion_speed_test.py.
Descriptive statistics only — no significance testing.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR   = "data"
KAP_FILE   = "kap_data/kap_financial_report_dates.csv"
XU100_PATH = "data_market/XU100.csv"
EXCLUDE    = {"USDTRY", "USDTRY=X", "ISKUR"}
MIN_ROWS   = 250
LOOKBACK_MONTHS = 3
# Target horizons in approximate trading days
HORIZONS   = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}

# ---------- XU100 ----------
xu100_close = (
    pd.read_csv(XU100_PATH, parse_dates=["Date"])
    .sort_values("Date")
    .set_index("Date")["Close"]
)

# ---------- PRICE PANEL ----------
price_data = {}
for f in glob.glob(os.path.join(DATA_DIR, "*.csv")):
    sym = os.path.splitext(os.path.basename(f))[0]
    if sym in EXCLUDE:
        continue
    try:
        df = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            continue
        if not {"Date", "Close", "Volume"}.issubset(df.columns):
            continue
        df["tl_vol"] = df["Close"] * df["Volume"]
        price_data[sym] = df.set_index("Date")
    except Exception:
        continue

print(f"Yüklenen sembol: {len(price_data)}")

# ---------- KAP DATES ----------
kap = pd.read_csv(KAP_FILE)
kap["publish_dt"] = pd.to_datetime(kap["publish_date"], dayfirst=True, errors="coerce")
kap = kap.dropna(subset=["publish_dt", "ticker"])
kap = kap[~kap["ticker"].isin(EXCLUDE)]
kap = kap.sort_values("publish_dt").drop_duplicates(
    subset=["ticker", "year", "period"], keep="first"
)
print(f"KAP olay sayısı: {len(kap)}, ticker: {kap['ticker'].nunique()}")

# ---------- TRADING DATE INDEX ----------
all_trading_dates = pd.to_datetime(sorted(
    set().union(*[set(df.index) for df in price_data.values()])
))

# ---------- EVENT ANALYSIS ----------
records = []

for _, row in kap.iterrows():
    sym = row["ticker"]
    if sym not in price_data:
        continue

    pdf = price_data[sym]
    pub_dt = row["publish_dt"]

    # Entry: first trading day on or after publication date
    future = all_trading_dates[all_trading_dates >= pub_dt]
    if len(future) == 0:
        continue
    entry_date = future[0]

    entry_price = pdf["Close"].get(entry_date, np.nan)
    if pd.isna(entry_price) or entry_price == 0:
        continue

    # Lookback liquidity: 3-month trailing avg TL volume before pub_dt
    lb_end   = pub_dt
    lb_start = pub_dt - pd.DateOffset(months=LOOKBACK_MONTHS)
    lb_mask  = (pdf.index >= lb_start) & (pdf.index < lb_end)
    avg_tl_vol = pdf.loc[lb_mask, "tl_vol"].mean()
    if pd.isna(avg_tl_vol) or avg_tl_vol <= 0:
        continue

    rec = {
        "ticker": sym,
        "publish_dt": pub_dt,
        "entry_date": entry_date,
        "avg_tl_vol": avg_tl_vol,
    }

    for label, n_days in HORIZONS.items():
        # Exit: nth trading day after entry
        fwd = all_trading_dates[all_trading_dates > entry_date]
        if len(fwd) < n_days:
            rec[f"excess_{label}"] = np.nan
            continue
        exit_date = fwd[n_days - 1]

        exit_price = pdf["Close"].get(exit_date, np.nan)
        if pd.isna(exit_price) or exit_price == 0:
            rec[f"excess_{label}"] = np.nan
            continue

        xu100_entry = xu100_close.get(entry_date, np.nan)
        xu100_exit  = xu100_close.get(exit_date, np.nan)
        if pd.isna(xu100_entry) or pd.isna(xu100_exit) or xu100_entry == 0:
            rec[f"excess_{label}"] = np.nan
            continue

        stock_ret  = (exit_price - entry_price) / entry_price
        xu100_ret_ = (xu100_exit - xu100_entry) / xu100_entry
        rec[f"excess_{label}"] = (stock_ret - xu100_ret_) * 10000  # bps

    records.append(rec)

df_all = pd.DataFrame(records)
print(f"\nToplam analiz edilebilir olay: {len(df_all)}")

# ---------- QUARTILE ASSIGNMENT (cross-sectional by year-month) ----------
df_all["ym"] = df_all["publish_dt"].dt.to_period("M")
quartile_parts = []
for ym, grp in df_all.groupby("ym"):
    try:
        q = pd.qcut(grp["avg_tl_vol"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        q = pd.Series(np.nan, index=grp.index)
    quartile_parts.append(q)
df_all["quartile"] = pd.concat(quartile_parts)
df_all = df_all.dropna(subset=["quartile"])
df_all["quartile"] = df_all["quartile"].astype(str)

# ---------- DESCRIPTIVE TABLE ----------
print("\n" + "=" * 72)
print("Kartil Bazında Uzun Vadeli Excess Return (XU100'e Göre, bps)")
print("Düşük likidite = Q1 (alt %25 TL hacim), Yüksek = Q4")
print("=" * 72)

rows = []
for q in ["Q1", "Q4"]:
    label = "Low (Q1)" if q == "Q1" else "High (Q4)"
    sub = df_all[df_all["quartile"] == q]
    for label_h, col in [("1m", "excess_1m"), ("3m", "excess_3m"),
                          ("6m", "excess_6m"), ("12m", "excess_12m")]:
        vals = sub[col].dropna()
        rows.append({
            "Kartil": label,
            "Pencere": label_h,
            "n": len(vals),
            "Ort_excess_bps": round(vals.mean(), 2) if len(vals) > 0 else np.nan,
            "Medyan_bps":     round(vals.median(), 2) if len(vals) > 0 else np.nan,
        })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))

# Save
out = "data/carry_trade/long_term_diffusion_results.csv"
df_all.to_csv(out, index=False)
result_df.to_csv("data/carry_trade/long_term_diffusion_summary.csv", index=False)
print(f"\nKaydedildi: {out}")
print("Kaydedildi: data/carry_trade/long_term_diffusion_summary.csv")
