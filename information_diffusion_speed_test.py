# ON-TAAHHUT: Dusuk likidite/kucuk piyasa degerine sahip hisselerde, KAP
# finansal rapor aciklamasi sonrasi fiyat tepkisinin ILK GUN'den daha az
# pay almasi (yavas bilgi difuzyonu), yuksek likidite hisselerde ise
# tepkinin cogunlukla ILK GUN icinde tamamlanmasi bekleniyor. Ayrica,
# dusuk likidite grubunda ilk-gun-sonrasi donemde AYNI YONDE devam eden
# bir "drift" olup olmadigi test edilecek (bilgi difuzyon gecikmesinin
# yakalanabilir olup olmadigi). Tek seferlik test.

"""
information_diffusion_speed_test.py

Veri: kap_data/kap_financial_report_dates.csv + data/*.csv + data_market/XU100.csv
Metodoloji: finansal rapor aciklamasi sonrasi 5 islem gunu tepki analizi.
Cikti: konsol tablolari
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
LOOKBACK_MONTHS = 3   # likidite hesabi icin gecmis pencere
WINDOW     = 5        # aciklamadan sonraki islem gunu sayisi
MIN_OBS    = 3        # bir olayda en az bu kadar gun verisi olmali
COST_LOW   = 20.0     # Midas+DUSUK round-trip bps
COST_HIGH  = 150.0    # AtaYatirim+YUKSEK round-trip bps
PLACEBO_SEED = 42


# ---------- BH-FDR ----------
def bh_fdr(p_values):
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return q.clip(max=1.0)


# ---------- FİYAT PANELİ ----------
xu100_df = pd.read_csv(XU100_PATH, parse_dates=["Date"])
xu100_close = xu100_df.sort_values("Date").set_index("Date")["Close"]
xu100_ret = xu100_close.pct_change()

price_data = {}   # sym -> DataFrame(Date, Close, Volume, tl_vol)
close_ret  = {}   # sym -> Series daily return

for f in glob.glob(os.path.join(DATA_DIR, "*.csv")):
    sym = os.path.splitext(os.path.basename(f))[0]
    if sym in EXCLUDE:
        continue
    try:
        df = pd.read_csv(f, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            continue
        if not {"Date", "Close", "Volume"}.issubset(df.columns):
            continue
        df["tl_vol"] = df["Close"] * df["Volume"]
        price_data[sym] = df.set_index("Date")
        close_ret[sym] = df.set_index("Date")["Close"].pct_change()
    except Exception:
        continue

print(f"Yuklenen sembol: {len(price_data)}")


# ---------- KAP TARİHLERİ ----------
kap = pd.read_csv(KAP_FILE)
kap["publish_dt"] = pd.to_datetime(kap["publish_date"], dayfirst=True, errors="coerce")
kap = kap.dropna(subset=["publish_dt", "ticker"])
kap = kap[~kap["ticker"].isin(EXCLUDE)]
# Her ticker+yil+period icin en erken bildirimi kullan (duplikat temizleme)
kap = kap.sort_values("publish_dt").drop_duplicates(
    subset=["ticker", "year", "period"], keep="first"
)
print(f"KAP olay sayisi (tekil): {len(kap)}, ticker: {kap['ticker'].nunique()}")


# ---------- ORTAK TRADİNG GÜN İNDEKSİ ----------
all_trading_dates = sorted(
    set().union(*[set(df.index) for df in price_data.values()])
)
all_trading_dates = pd.to_datetime(all_trading_dates)


def next_trading_day(dt, offset=1):
    """dt'den offset kadar sonraki trading gunu."""
    future = all_trading_dates[all_trading_dates > dt]
    if len(future) < offset:
        return None
    return future[offset - 1]


def trading_window(start_dt, n):
    """start_dt'den itibaren n trading gunu (start_dt dahil)."""
    future = all_trading_dates[all_trading_dates >= start_dt]
    if len(future) < n:
        return future.tolist()
    return future[:n].tolist()


# ---------- OLAY ANALİZİ ----------
records = []

for _, row in kap.iterrows():
    sym = row["ticker"]
    if sym not in price_data:
        continue

    pub_dt = row["publish_dt"]

    # t+1 open giris referansi: aciklamadan sonraki ilk trading gunu
    entry_dt = next_trading_day(pub_dt, offset=1)
    if entry_dt is None:
        continue

    # WINDOW gunluk trading penceresi (entry_dt dahil)
    win_days = trading_window(entry_dt, WINDOW)
    if len(win_days) < MIN_OBS:
        continue

    # Excess return: her gun icin stock - XU100
    ret_s = close_ret[sym]
    daily_exc = []
    for d in win_days:
        sr = ret_s.get(d, np.nan)
        xr = xu100_ret.get(d, np.nan)
        if pd.isna(sr) or pd.isna(xr):
            daily_exc.append(np.nan)
        else:
            daily_exc.append(sr - xr)

    daily_exc = np.array(daily_exc, dtype=float)
    valid = ~np.isnan(daily_exc)
    if valid.sum() < MIN_OBS:
        continue

    # Gun 1 payi: |day1| / sum(|day1|..|day5|)
    abs_exc = np.abs(daily_exc)
    total_abs = np.nansum(abs_exc)
    if total_abs == 0:
        continue
    day1_share = abs_exc[0] / total_abs if not np.isnan(abs_exc[0]) else np.nan

    # Gün 2-5 drift: day1 yonuyle ayni yonde devam eden excess
    day1_sign = np.sign(daily_exc[0]) if not np.isnan(daily_exc[0]) else np.nan
    drift_25 = np.nansum(daily_exc[1:]) * day1_sign  # pozitif = ayni yon devam

    # Likidite: pub_dt oncesi 3 aylik ortalama TL hacim
    lb_end = pub_dt
    lb_start = pub_dt - pd.DateOffset(months=LOOKBACK_MONTHS)
    vol_s = price_data[sym]["tl_vol"]
    lb_vol = vol_s.loc[(vol_s.index >= lb_start) & (vol_s.index < lb_end)]
    avg_tl_vol = lb_vol.mean() if len(lb_vol) > 5 else np.nan

    records.append({
        "ticker": sym,
        "publish_dt": pub_dt,
        "entry_dt": entry_dt,
        "day1_exc": daily_exc[0],
        "day25_exc_sum": np.nansum(daily_exc[1:]),
        "day1_share": day1_share,
        "drift_25_signed": drift_25,
        "avg_tl_vol": avg_tl_vol,
    })

df_ev = pd.DataFrame(records).dropna(subset=["day1_share", "avg_tl_vol"])
print(f"Analiz edilebilir olay: {len(df_ev)}")


# ---------- LİKİDİTE TERTİLİ ----------
df_ev["tertil"] = pd.qcut(
    df_ev["avg_tl_vol"], 3, labels=["Dusuk", "Orta", "Yuksek"]
)

# ---------- TABLO 1: GÜN 1 PAYI ----------
t1_rows = []
for t in ["Dusuk", "Orta", "Yuksek"]:
    sub = df_ev[df_ev["tertil"] == t]["day1_share"].dropna()
    stat, pv = stats.ttest_1samp(sub, popmean=sub.mean())  # vs kendi ortalamasi
    # Pairwise icin grup karsilastirmasi kullanacagiz; burada vs Yuksek
    t1_rows.append({"Tertil": t, "n": len(sub), "Ort_gun1_pay": sub.mean(),
                    "_data": sub})

# Pairwise t-test: Dusuk vs Yuksek, Dusuk vs Orta, Orta vs Yuksek
pw_pairs = [("Dusuk", "Yuksek"), ("Dusuk", "Orta"), ("Orta", "Yuksek")]
pw_rows = []
for a, b in pw_pairs:
    da = df_ev[df_ev["tertil"] == a]["day1_share"].dropna()
    db = df_ev[df_ev["tertil"] == b]["day1_share"].dropna()
    t_stat, pv = stats.ttest_ind(da, db, equal_var=False)
    pw_rows.append({"Karsilastirma": f"{a} vs {b}", "t_stat": t_stat, "p_value": pv,
                    f"Ort_{a}": da.mean(), f"Ort_{b}": db.mean()})

pw_df = pd.DataFrame(pw_rows)
pw_df["q_value_BH"] = bh_fdr(pw_df["p_value"].values)

# ---------- TABLO 2: DUSUK LİKİDİTE DRIFT (gün 2-5) ----------
low_drift = df_ev[df_ev["tertil"] == "Dusuk"]["drift_25_signed"].dropna() * 10000  # bps
t2_t, t2_p = stats.ttest_1samp(low_drift, 0)
t2_net_low  = low_drift.mean() - COST_LOW
t2_net_high = low_drift.mean() - COST_HIGH

# ---------- TABLO 3: PLACEBO ----------
rng = np.random.default_rng(PLACEBO_SEED)
all_pub_dates = df_ev["publish_dt"].values

pb_records = []
for _, row in df_ev.iterrows():
    sym = row["ticker"]
    fake_dt = pd.Timestamp(rng.choice(all_pub_dates))

    entry_dt = next_trading_day(fake_dt, offset=1)
    if entry_dt is None:
        continue
    win_days = trading_window(entry_dt, WINDOW)
    if len(win_days) < MIN_OBS:
        continue

    ret_s = close_ret[sym]
    daily_exc = []
    for d in win_days:
        sr = ret_s.get(d, np.nan)
        xr = xu100_ret.get(d, np.nan)
        daily_exc.append(sr - xr if not (pd.isna(sr) or pd.isna(xr)) else np.nan)

    daily_exc = np.array(daily_exc, dtype=float)
    if np.isnan(daily_exc[0]):
        continue
    total_abs = np.nansum(np.abs(daily_exc))
    if total_abs == 0:
        continue
    day1_share = abs(daily_exc[0]) / total_abs
    drift_25   = np.nansum(daily_exc[1:]) * np.sign(daily_exc[0])

    pb_records.append({
        "tertil": row["tertil"],
        "day1_share": day1_share,
        "drift_25_signed": drift_25,
    })

pb_df = pd.DataFrame(pb_records).dropna()

pb_pw_rows = []
for a, b in pw_pairs:
    da = pb_df[pb_df["tertil"] == a]["day1_share"].dropna()
    db = pb_df[pb_df["tertil"] == b]["day1_share"].dropna()
    if len(da) < 3 or len(db) < 3:
        pb_pw_rows.append({"Karsilastirma": f"{a} vs {b}",
                           "t_stat": np.nan, "p_value": np.nan})
        continue
    t_stat, pv = stats.ttest_ind(da, db, equal_var=False)
    pb_pw_rows.append({"Karsilastirma": f"{a} vs {b}", "t_stat": t_stat, "p_value": pv})

pb_pw_df = pd.DataFrame(pb_pw_rows)
pb_pw_df["q_value_BH"] = bh_fdr(pb_pw_df["p_value"].fillna(1).values)

pb_low_drift = pb_df[pb_df["tertil"] == "Dusuk"]["drift_25_signed"].dropna() * 10000
pb_t, pb_p   = stats.ttest_1samp(pb_low_drift, 0) if len(pb_low_drift) > 2 else (np.nan, np.nan)

# ---------- RAPOR ----------
print("\n" + "=" * 65)
print("TABLO 1 — Likidite tertillerine gore ortalama 'Gun 1 Payi'")
print("  (|day1 excess| / sum(|day1|..|day5|))")
print("=" * 65)
t1_out = pd.DataFrame([{
    "Tertil": r["Tertil"], "n": r["n"],
    "Ort_gun1_pay": r["Ort_gun1_pay"]
} for r in t1_rows])
print(t1_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n--- Pairwise t-test (Welch, BH-FDR) ---")
print(pw_df[["Karsilastirma", "t_stat", "p_value", "q_value_BH"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 65)
print("TABLO 2 — Dusuk likidite grubu: Gun 2-5 devam eden drift")
print("  (day1 yonuyle ayni yonde, excess return vs XU100, bps)")
print("=" * 65)
print(f"  n               : {len(low_drift)}")
print(f"  Brut drift      : {low_drift.mean():.2f} bps (gun 2-5 toplam)")
print(f"  t-stat          : {t2_t:.3f}")
print(f"  p-value         : {t2_p:.4f}")
print(f"  Net (LOW  {COST_LOW:.0f}bps) : {t2_net_low:.2f} bps")
print(f"  Net (HIGH {COST_HIGH:.0f}bps): {t2_net_high:.2f} bps")

print("\n" + "=" * 65)
print("TABLO 3 — PLACEBO (rastgele tarih, seed=42)")
print("=" * 65)
print("  Pairwise gun-1-payi karsilastirmasi:")
print(pb_pw_df[["Karsilastirma", "t_stat", "p_value", "q_value_BH"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\n  Dusuk likidite placebo drift: {pb_low_drift.mean():.2f} bps, "
      f"t={pb_t:.3f}, p={pb_p:.4f} (n={len(pb_low_drift)})")
