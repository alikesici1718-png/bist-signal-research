"""
Long-term KAP diffusion analysis: outlier diagnostics, cleaning, Mann-Whitney tests.
Builds on long_term_diffusion_analysis.py results.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR   = "data"
KAP_FILE   = "kap_data/kap_financial_report_dates.csv"
XU100_PATH = "data_market/XU100.csv"
EXCLUDE    = {"USDTRY", "USDTRY=X", "ISKUR"}
MIN_ROWS   = 250
LOOKBACK_MONTHS = 3
HORIZONS   = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
OUTLIER_THRESHOLD_BPS = 50_000   # |excess| > 500% = likely data issue

# ---------- RE-RUN DATA PREP (same as long_term_diffusion_analysis.py) ----------
xu100_close = (
    pd.read_csv(XU100_PATH, parse_dates=["Date"])
    .sort_values("Date")
    .set_index("Date")["Close"]
)

price_data = {}
for f in glob.glob(os.path.join(DATA_DIR, "*.csv")):
    sym = os.path.splitext(os.path.basename(f))[0]
    if sym in EXCLUDE:
        continue
    try:
        df = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS or not {"Date","Close","Volume"}.issubset(df.columns):
            continue
        df["tl_vol"] = df["Close"] * df["Volume"]
        price_data[sym] = df.set_index("Date")
    except Exception:
        continue

kap = pd.read_csv(KAP_FILE)
kap["publish_dt"] = pd.to_datetime(kap["publish_date"], dayfirst=True, errors="coerce")
kap = kap.dropna(subset=["publish_dt","ticker"])
kap = kap[~kap["ticker"].isin(EXCLUDE)]
kap = kap.sort_values("publish_dt").drop_duplicates(
    subset=["ticker","year","period"], keep="first")

all_trading_dates = pd.to_datetime(sorted(
    set().union(*[set(df.index) for df in price_data.values()])
))

records = []
for _, row in kap.iterrows():
    sym = row["ticker"]
    if sym not in price_data:
        continue
    pdf = price_data[sym]
    pub_dt = row["publish_dt"]
    future = all_trading_dates[all_trading_dates >= pub_dt]
    if len(future) == 0:
        continue
    entry_date = future[0]
    entry_price = pdf["Close"].get(entry_date, np.nan)
    if pd.isna(entry_price) or entry_price == 0:
        continue
    lb_end   = pub_dt
    lb_start = pub_dt - pd.DateOffset(months=LOOKBACK_MONTHS)
    lb_mask  = (pdf.index >= lb_start) & (pdf.index < lb_end)
    avg_tl_vol = pdf.loc[lb_mask, "tl_vol"].mean()
    if pd.isna(avg_tl_vol) or avg_tl_vol <= 0:
        continue
    rec = {"ticker": sym, "publish_dt": pub_dt, "entry_date": entry_date,
           "avg_tl_vol": avg_tl_vol}
    for label, n_days in HORIZONS.items():
        fwd = all_trading_dates[all_trading_dates > entry_date]
        if len(fwd) < n_days:
            rec[f"excess_{label}"] = np.nan
            continue
        exit_date = fwd[n_days - 1]
        exit_price = pdf["Close"].get(exit_date, np.nan)
        xu100_entry = xu100_close.get(entry_date, np.nan)
        xu100_exit  = xu100_close.get(exit_date, np.nan)
        if any(pd.isna(x) or x == 0 for x in [exit_price, xu100_entry, xu100_exit]):
            rec[f"excess_{label}"] = np.nan
            continue
        stock_ret  = (exit_price - entry_price) / entry_price
        xu100_ret_ = (xu100_exit - xu100_entry) / xu100_entry
        rec[f"excess_{label}"] = (stock_ret - xu100_ret_) * 10000
    records.append(rec)

df_all = pd.DataFrame(records)

# Quartile assignment
df_all["ym"] = df_all["publish_dt"].dt.to_period("M")
parts = []
for ym, grp in df_all.groupby("ym"):
    try:
        q = pd.qcut(grp["avg_tl_vol"], 4, labels=["Q1","Q2","Q3","Q4"])
    except ValueError:
        q = pd.Series(np.nan, index=grp.index)
    parts.append(q)
df_all["quartile"] = pd.concat(parts)
df_all = df_all.dropna(subset=["quartile"])
df_all["quartile"] = df_all["quartile"].astype(str)

print(f"Toplam olay (kartil atandı): {len(df_all)}")

# ============================================================
# ADIM 1: OUTLIER TESPİTİ — 12m penceresinde
# ============================================================
print("\n" + "=" * 70)
print("ADIM 1 — Outlier Tespiti (|excess_12m| > 50,000 bps)")
print("=" * 70)

extreme = df_all[df_all["excess_12m"].abs() > OUTLIER_THRESHOLD_BPS].copy()
print(f"Aşırı uç gözlem sayısı (12m): {len(extreme)}")

# Her extreme semboldeki fiyat davranışını kontrol et
extreme_syms = extreme["ticker"].unique()
print(f"Etkilenen sembol sayısı: {len(extreme_syms)}")

diag_rows = []
for sym in sorted(extreme_syms):
    pdf = price_data[sym]
    closes = pdf["Close"].dropna()
    # Max günlük değişim (split indicator)
    daily_chg = closes.pct_change().abs()
    max_chg = daily_chg.max()
    max_chg_date = daily_chg.idxmax() if not daily_chg.empty else None
    n_events = len(extreme[extreme["ticker"] == sym])
    ex_vals = extreme.loc[extreme["ticker"] == sym, "excess_12m"].values
    diag_rows.append({
        "Sembol": sym,
        "N_aşırı_olay": n_events,
        "Max_günlük_değ_%": round(max_chg * 100, 1),
        "Max_değ_tarihi": str(max_chg_date)[:10] if max_chg_date else "—",
        "Excess_12m_min": round(ex_vals.min()),
        "Excess_12m_max": round(ex_vals.max()),
    })

diag_df = pd.DataFrame(diag_rows).sort_values("Max_günlük_değ_%", ascending=False)
print(diag_df.to_string(index=False))

# Heuristic: if max single-day move > 80% = likely split/corporate action artifact
SPLIT_THRESHOLD = 0.80
split_syms = set(
    diag_df.loc[diag_df["Max_günlük_değ_%"] / 100 > SPLIT_THRESHOLD, "Sembol"]
)
print(f"\nSplit/corporate action şüpheli (max günlük değişim >80%): {sorted(split_syms)}")

# ============================================================
# ADIM 2: FİLTRELEME
# ============================================================
print("\n" + "=" * 70)
print("ADIM 2 — Filtreleme")
print("=" * 70)

# Rule A: events where the 12m exit is in a split-artifact symbol → exclude
df_all["data_error"] = df_all["ticker"].isin(split_syms)

# Rule B: genuine extreme but real — flag, keep in dataset but note
df_all["genuine_extreme"] = (
    (df_all["excess_12m"].abs() > OUTLIER_THRESHOLD_BPS) &
    ~df_all["data_error"]
)

n_error   = df_all["data_error"].sum()
n_genuine = df_all["genuine_extreme"].sum()
print(f"Veri hatası (split/adjusted fiyat) nedeniyle çıkarılan: {n_error} gözlem")
print(f"Gerçek ama aşırı uç (flaglendi, silinmedi):             {n_genuine} gözlem")

df_clean = df_all[~df_all["data_error"]].copy()
print(f"Temizlenmiş veri seti: {len(df_clean)} gözlem ({len(df_all) - len(df_clean)} çıkarıldı)")

# ============================================================
# ADIM 3-4: MANN-WHITNEY U + BH-FDR
# ============================================================
print("\n" + "=" * 70)
print("ADIM 3-4 — Mann-Whitney U: Q1 vs Q4, her pencere (BH-FDR düzeltmeli)")
print("=" * 70)

def bh_fdr(p_values):
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n); ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    for i in range(n-2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i+1]])
    return q.clip(max=1.0)

mw_rows = []
for label in ["1m", "3m", "6m", "12m"]:
    col = f"excess_{label}"
    q1 = df_clean[(df_clean["quartile"] == "Q1")][col].dropna()
    q4 = df_clean[(df_clean["quartile"] == "Q4")][col].dropna()
    stat, p = stats.mannwhitneyu(q1, q4, alternative="two-sided")
    mw_rows.append({
        "Pencere": label,
        "n_Q1": len(q1),
        "Medyan_Q1_bps": round(q1.median(), 2),
        "n_Q4": len(q4),
        "Medyan_Q4_bps": round(q4.median(), 2),
        "MW_stat": round(stat),
        "p_value": round(p, 4),
    })

mw_df = pd.DataFrame(mw_rows)
mw_df["q_BH"] = bh_fdr(mw_df["p_value"].values).round(4)

print(mw_df.to_string(index=False))

print(f"\nNot: {n_error} gözlem veri hatası nedeniyle çıkarıldı, "
      f"{n_genuine} aşırı uç gözlem flaglendi ama dahil edildi.")
print(f"Parametreler: outlier eşiği={OUTLIER_THRESHOLD_BPS:,} bps, "
      f"split eşiği=%{SPLIT_THRESHOLD*100:.0f}, lookback={LOOKBACK_MONTHS}ay")
