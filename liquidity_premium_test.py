# ON-TAAHHUT: Dusuk likidite (kartil Q1) hisselerin, cost-adjusted forward
# getirisinin diger kartillerden sistematik olarak daha yuksek olmasi
# bekleniyor (likidite risk primi). Beklenen buyukluk: mutevazi ama pozitif.
# Bu, teknik bir sinyal degil, saf likidite-bazli bir prim testidir.
# Tek seferlik test, sonuc ne olursa olsun kabul edilecek.

"""
liquidity_premium_test.py

BIST dusuk likidite primi testi.
Q1 (en dusuk hacim) hisselerin cost-adjusted getirisi diger kartillerden
anlamli sekilde yuksek mi? (Likidite risk primi hipotezi)

Cikti: konsol tablosu (tek seferlik test, no-edit)
"""

import glob
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------- PARAMETRELER ----------
DATA_DIR = "data"
MIN_ROWS = 250
EXCLUDE_NAMES = {"USDTRY", "USDTRY=X", "ISKUR"}
IPO_SEASONING_MONTHS = 3        # ilk N ay taramadan dislanir
LOOKBACK_MONTHS = 3             # likidite hesabi icin gecmis pencere
COMMISSION_BPS = 38.0           # round-trip komisyon (AtaYatirim)
MARKET_IMPACT_PER_1PCT = 10.0   # gunluk hacmin %1'i icin ek bps
POSITION_SIZE_TL = 10_000       # gercekci kucuk sermaye varsayimi (TL)
MIN_DAILY_VOL_TL = 5_000        # bu altindaki gunluk hacim = pozisyon acilamaz, gozlem disla
MIN_VOL_SHARE = 0.05            # pozisyon gunluk hacmin en az %5'i olmali (kapasite filtresi)
MIN_UNIQUE_OPEN = 3             # bir ay icinde unique Open < 3 => stale price, gozlem disla
PLACEBO_SEED = 42
# ----------------------------------

K_CS = 3 - 2 * np.sqrt(2)       # Corwin-Schultz sabiti


def load_symbol(path):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None
        if not {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def corwin_schultz_spread(high, low):
    """Gunluk CS spread tahmini (oran olarak). Negatifler 0'a kirpilir."""
    high = high.replace(0, np.nan)
    low = low.replace(0, np.nan)
    hl1 = np.log(high / low) ** 2
    high2 = high.rolling(2).max()
    low2 = low.rolling(2).min()
    hl2 = np.log(high2 / low2) ** 2
    beta = hl1 + hl1.shift(1)
    gamma = hl2
    sqrt_beta = np.sqrt(beta.clip(lower=0))
    sqrt_gamma = np.sqrt(gamma.clip(lower=0))
    alpha = (np.sqrt(2) - 1) * sqrt_beta / K_CS - sqrt_gamma / np.sqrt(K_CS)
    with np.errstate(over="ignore", invalid="ignore"):
        exp_a = np.exp(alpha)
        spread = 2 * (exp_a - 1) / (1 + exp_a)
    return spread.clip(lower=0, upper=0.5)


# ---------- XU100 YUKLE ----------
xu100_df = pd.read_csv("data_market/XU100.csv", parse_dates=["Date"])
xu100_df = xu100_df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
xu100_open = xu100_df.set_index("Date")["Open"]

# ---------- VERİ YUKLE ----------
files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
symbols = {}
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    if name in EXCLUDE_NAMES:
        continue
    df = load_symbol(f)
    if df is not None:
        symbols[name] = df

all_dates = pd.to_datetime(
    sorted(set().union(*[df["Date"].unique() for df in symbols.values()]))
)

open_p, close_p, vol_p, spread_p = {}, {}, {}, {}
for name, df in symbols.items():
    s = df.set_index("Date")
    open_p[name] = s["Open"].reindex(all_dates)
    close_p[name] = s["Close"].reindex(all_dates)
    vol_p[name] = s["Volume"].reindex(all_dates)
    spread_p[name] = corwin_schultz_spread(s["High"], s["Low"]).reindex(all_dates)

open_panel = pd.DataFrame(open_p)
close_panel = pd.DataFrame(close_p)
vol_panel = pd.DataFrame(vol_p)
spread_panel = pd.DataFrame(spread_p)

# TL hacim (Close * Volume)
tl_vol_panel = close_panel * vol_panel

# ---------- AYLIK KARTİL + FORWARD GETİRİ ----------
# Her ay için: gecmis 3 ay TL hacim ortalamasini hesapla, kartile ata,
# sonraki ay forward getirisi hesapla.

open_panel.index = pd.to_datetime(open_panel.index)

# Her sembol icin IPO seasoning: ilk 3 ayin tum verileri dislanir
ipo_cutoff = {}
for name, df in symbols.items():
    first_date = df["Date"].min()
    ipo_cutoff[name] = first_date + pd.DateOffset(months=IPO_SEASONING_MONTHS)

# Ay sonu indeksi olustur
months = pd.period_range(
    start=open_panel.index.min().to_period("M"),
    end=open_panel.index.max().to_period("M"),
    freq="M"
)

records = []
n_excl_vol = 0    # MIN_DAILY_VOL_TL veya MIN_VOL_SHARE nedeniyle dislanan
n_excl_stale = 0  # stale price nedeniyle dislanan

# Stale price kontrolu icin: her sembol-ay icin forward donemde unique Open sayisi
# Panel uzerinden onceden hesapla
def count_unique_open_per_month(panel, months_range):
    """Her (sembol, ay) icin o aydaki unique Open fiyat sayisini doner."""
    result = {}
    for m in months_range:
        mask = panel.index.to_period("M") == m
        sub = panel.loc[mask]
        result[str(m)] = sub.nunique(axis=0)
    return result

unique_open_by_month = count_unique_open_per_month(open_panel, months)

for i in range(LOOKBACK_MONTHS, len(months) - 1):
    current_month = months[i]
    next_month = months[i + 1]

    # lookback penceresi: son LOOKBACK_MONTHS ay
    lb_start = months[i - LOOKBACK_MONTHS].start_time
    lb_end = current_month.start_time  # bu ay dahil degil (look-ahead yok)

    # gecmis TL hacim
    lb_mask = (open_panel.index >= lb_start) & (open_panel.index < lb_end)
    lb_vol = tl_vol_panel.loc[lb_mask].mean(skipna=True)

    # forward donem: sonraki ay
    fwd_mask = open_panel.index.to_period("M") == next_month

    if fwd_mask.sum() < 2:
        continue

    fwd_opens = open_panel.loc[fwd_mask]
    entry_open = fwd_opens.iloc[0]
    exit_open = fwd_opens.iloc[-1]

    fwd_ret = (exit_open - entry_open) / entry_open

    # XU100 ayni donem forward getirisi (entry ve exit tarihleriyle eslestir)
    fwd_entry_date = fwd_opens.index[0]
    fwd_exit_date = fwd_opens.index[-1]
    xu100_entry = xu100_open.get(fwd_entry_date, np.nan)
    xu100_exit = xu100_open.get(fwd_exit_date, np.nan)
    if pd.notna(xu100_entry) and pd.notna(xu100_exit) and xu100_entry != 0:
        xu100_fwd = (xu100_exit - xu100_entry) / xu100_entry
    else:
        xu100_fwd = np.nan

    # CS spread tahmini: bu aydaki medyan spread
    cur_mask = open_panel.index.to_period("M") == current_month
    cs_spread = spread_panel.loc[cur_mask].median(skipna=True)

    # forward ay unique Open sayisi (stale price filtresi icin)
    unique_open_fwd = unique_open_by_month.get(str(next_month), pd.Series(dtype=float))

    for name in lb_vol.index:
        if pd.isna(lb_vol[name]):
            continue
        if pd.isna(fwd_ret.get(name, np.nan)):
            continue
        if pd.isna(xu100_fwd):
            continue

        # IPO seasoning filtresi
        entry_date = fwd_opens.index[0]
        if entry_date < ipo_cutoff.get(name, pd.Timestamp.min):
            continue

        daily_tl_vol = lb_vol[name]

        # Likidite/kapasite filtresi: mutlak alt esik VEYA pozisyon paya gore buyuk
        vol_share = POSITION_SIZE_TL / daily_tl_vol if daily_tl_vol > 0 else np.inf
        if daily_tl_vol < MIN_DAILY_VOL_TL or vol_share > (1.0 / MIN_VOL_SHARE):
            n_excl_vol += 1
            continue

        # Stale price filtresi: forward aydaki unique Open < MIN_UNIQUE_OPEN
        uopen = unique_open_fwd.get(name, np.nan)
        if pd.isna(uopen) or uopen < MIN_UNIQUE_OPEN:
            n_excl_stale += 1
            continue

        gross = fwd_ret[name]
        excess = gross - xu100_fwd  # market-adjusted gross return
        sp = cs_spread.get(name, np.nan)
        if pd.isna(sp):
            sp = 0.0

        impact_frac = vol_share  # POSITION_SIZE_TL / daily_tl_vol
        market_impact_bps = impact_frac * 100 * MARKET_IMPACT_PER_1PCT

        total_cost = sp + (COMMISSION_BPS / 10000.0) + (market_impact_bps / 10000.0)
        net_excess_bps = (excess - total_cost) * 10000

        records.append({
            "month": str(current_month),
            "symbol": name,
            "tl_vol_lb": daily_tl_vol,
            "gross_bps": gross * 10000,
            "xu100_bps": xu100_fwd * 10000,
            "excess_bps": excess * 10000,
            "cs_spread_bps": sp * 10000,
            "market_impact_bps": market_impact_bps,
            "total_cost_bps": total_cost * 10000,
            "net_bps": net_excess_bps,
        })

df_all = pd.DataFrame(records)
print(f"Toplam gozlem (filtre sonrasi): {len(df_all)} (sembol-ay ciftleri)")
print(f"  Dislanan — likidite/kapasite filtresi: {n_excl_vol}")
print(f"  Dislanan — stale price filtresi:       {n_excl_stale}")

# ---------- KARTİL ATAMASI ----------
# Her ay icin kartil: o aydaki tl_vol_lb dagilimina gore (cross-sectional)
quartiles = []
for month, grp in df_all.groupby("month"):
    try:
        q = pd.qcut(grp["tl_vol_lb"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        q = pd.Series(np.nan, index=grp.index)
    quartiles.append(q)
df_all["quartile"] = pd.concat(quartiles)
df_all = df_all.dropna(subset=["quartile"])
df_all["quartile"] = df_all["quartile"].astype(str)

# ---------- BH-FDR ----------
def bh_fdr(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR duzeltmesi. q-value doner."""
    p = np.array(p_values)
    n = len(p)
    ranks = np.argsort(p) + 1
    q = np.empty(n)
    q[np.argsort(p)] = p * n / ranks
    # monotonluk
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    return q.clip(max=1.0)


# ---------- TABLO 1: KARTİL SONUÇLARI ----------
quartile_stats = []
for q in ["Q1", "Q2", "Q3", "Q4"]:
    sub = df_all[df_all["quartile"] == q]["net_bps"].dropna()
    if len(sub) < 5:
        continue
    t, p = stats.ttest_1samp(sub, 0)
    quartile_stats.append({
        "Kartil": q,
        "n": len(sub),
        "Ort_net_bps": sub.mean(),
        "t_stat": t,
        "p_value": p,
    })

qs_df = pd.DataFrame(quartile_stats)
qs_df["q_value_BH"] = bh_fdr(qs_df["p_value"].values)

# ---------- TABLO 2: Q1 vs diger karsilastirmalar (t-test, BH-FDR) ----------
q1_data = df_all[df_all["quartile"] == "Q1"]["net_bps"].dropna()
comparisons = []
for q in ["Q2", "Q3", "Q4"]:
    other = df_all[df_all["quartile"] == q]["net_bps"].dropna()
    t, p = stats.ttest_ind(q1_data, other, equal_var=False)
    comparisons.append({"Karsilastirma": f"Q1 vs {q}", "t_stat": t, "p_value": p})

comp_df = pd.DataFrame(comparisons)
comp_df["q_value_BH"] = bh_fdr(comp_df["p_value"].values)

# ---------- TABLO 3: PLACEBO ----------
rng = np.random.default_rng(PLACEBO_SEED)


placebo_qs = []
for month, grp in df_all.groupby("month"):
    n = len(grp)
    labels = ["Q1", "Q2", "Q3", "Q4"]
    pq = pd.Series([labels[i % 4] for i in rng.permutation(n)], index=grp.index)
    placebo_qs.append(pq)
df_all["placebo_q"] = pd.concat(placebo_qs)

placebo_stats = []
q1p_data = df_all[df_all["placebo_q"] == "Q1"]["net_bps"].dropna()
for q in ["Q2", "Q3", "Q4"]:
    other = df_all[df_all["placebo_q"] == q]["net_bps"].dropna()
    t, p = stats.ttest_ind(q1p_data, other, equal_var=False)
    placebo_stats.append({"Placebo": f"Q1 vs {q}", "t_stat": t, "p_value": p})

placebo_df = pd.DataFrame(placebo_stats)
placebo_df["q_value_BH"] = bh_fdr(placebo_df["p_value"].values)

# ---------- RAPOR ----------
print("\n" + "=" * 65)
print("TABLO 1 — Her kartil: cost-adjusted EXCESS getiri vs XU100 (t vs 0)")
print("=" * 65)
print(qs_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n" + "=" * 65)
print("TABLO 2 — Q1 vs diger kartiller, excess return (Welch t-test, BH-FDR)")
print("=" * 65)
print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n" + "=" * 65)
print("TABLO 3 — PLACEBO (rastgele kartil atamasi, seed=42)")
print("=" * 65)
print(placebo_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print(f"\nParametreler: commission={COMMISSION_BPS}bps, market_impact={MARKET_IMPACT_PER_1PCT}bps per 1% daily vol, position={POSITION_SIZE_TL:,}TL, min_vol={MIN_DAILY_VOL_TL:,}TL, min_vol_share={MIN_VOL_SHARE*100:.0f}%, min_unique_open={MIN_UNIQUE_OPEN}")
