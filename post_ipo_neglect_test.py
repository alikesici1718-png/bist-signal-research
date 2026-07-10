# ON-TAAHHUT: Halka arz sonrasi 12-36 ay arasi "ihmal donemi" (yuksek ilgi
# donemi sonrasi, likidite/takip azaldigi donem) tanimlaniyor. Bu donemde
# XU100'e gore excess return'un sistematik olarak farkli (pozitif ya da
# negatif, yon onceden belirtilmiyor) olup olmadigi test ediliyor.
# Bu tek seferlik bir testtir, sonuc ne olursa olsun kabul edilecek.

"""
post_ipo_neglect_test.py

IPO sonrasi "ihmal donemi" (12-36 ay) etkisi testi.
Gozlem birimi: gunluk close-to-close XU100 excess return.
Donem tanimlari:
  Yeni   : 0-6 ay (listelenmeden itibaren)
  Gecis  : 6-12 ay (analiz disi)
  Ihmal  : 12-36 ay
  Olgun  : 36+ ay
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

DATA_DIR = "data"
SYMBOLS_FILE = "config/symbols.txt"
XU100_PATH = "data_market/XU100.csv"
MIN_ROWS = 60          # cok kisa veri setlerini atla
PLACEBO_SEED = 42

# Donem sinirlari (ay cinsinden, listelenmeden itibaren)
YENI_END   = 6
IHMAL_START = 12
IHMAL_END   = 36
OLGUN_START = 36


# ---------- XU100 YUKLE ----------
xu100_df = pd.read_csv(XU100_PATH, parse_dates=["Date"])
xu100_df = xu100_df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
xu100_close = xu100_df.set_index("Date")["Close"]
xu100_ret = xu100_close.pct_change()  # gunluk close-to-close getiri


# ---------- SEMBOLLER ----------
with open(SYMBOLS_FILE, "r") as f:
    all_symbols = [line.strip() for line in f if line.strip()]

EXCLUDE = {"USDTRY", "USDTRY=X", "ISKUR"}

records = []  # (symbol, date, age_months, excess_ret)

for sym in all_symbols:
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
    if len(df) < MIN_ROWS:
        continue
    if "Close" not in df.columns:
        continue

    # Adim 1: ilk islem gunu = ilk NaN olmayan Close
    valid = df[df["Close"].notna()]
    if valid.empty:
        continue
    ipo_date = valid["Date"].iloc[0]

    # Gunluk close-to-close getiri
    s = df.set_index("Date")["Close"].sort_index()
    stock_ret = s.pct_change()

    # XU100 ile hizala
    common_dates = stock_ret.index.intersection(xu100_ret.index)
    if len(common_dates) < MIN_ROWS:
        continue

    sr = stock_ret.loc[common_dates].dropna()
    xr = xu100_ret.loc[sr.index].dropna()
    common = sr.index.intersection(xr.index)
    sr = sr.loc[common]
    xr = xr.loc[common]

    excess = sr - xr

    # Adim 2: her gun icin "yasini" hesapla
    # trading_day: semboldeki kac. islem gunu (0-indexed, ipo_date=0)
    stock_dates = sorted(df[df["Close"].notna()]["Date"].tolist())
    ipo_idx = stock_dates.index(ipo_date) if ipo_date in stock_dates else 0
    trading_day_map = {d: i - ipo_idx for i, d in enumerate(stock_dates)}

    for dt, exc in excess.items():
        if pd.isna(exc):
            continue
        age_months = (dt.year - ipo_date.year) * 12 + (dt.month - ipo_date.month)
        if age_months < 0:
            age_months = 0
        trading_day = trading_day_map.get(dt, -1)
        records.append({
            "symbol": sym,
            "date": dt,
            "age_months": age_months,
            "trading_day": trading_day,
            "excess_ret": exc,
            "ipo_date": ipo_date,
        })

df_all = pd.DataFrame(records)
print(f"Toplam gunluk gozlem: {len(df_all)}, sembol sayisi: {df_all['symbol'].nunique()}")

# ---------- DONEM ATAMASI ----------
def assign_period(age):
    if age <= YENI_END:
        return "Yeni"
    elif YENI_END < age < IHMAL_START:
        return "Gecis"
    elif IHMAL_START <= age <= IHMAL_END:
        return "Ihmal"
    else:
        return "Olgun"

df_all["period"] = df_all["age_months"].apply(assign_period)
df_analiz = df_all[df_all["period"] != "Gecis"].copy()
print(f"Analiz gozlemleri (Gecis disinda): {len(df_analiz)}")
print(df_analiz["period"].value_counts().to_string())


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


# ---------- ADIM 3-4: GERCEK TEST ----------
period_order = ["Yeni", "Ihmal", "Olgun"]

def period_stats(df_sub):
    rows = []
    for p in period_order:
        sub = df_sub[df_sub["period"] == p]["excess_ret"].dropna()
        if len(sub) < 10:
            rows.append({"Donem": p, "n": len(sub), "Ort_bps/gun": np.nan,
                         "t_stat": np.nan, "p_value": np.nan})
            continue
        bps = sub.mean() * 10000
        t, pv = stats.ttest_1samp(sub, 0)
        rows.append({"Donem": p, "n": len(sub), "Ort_bps/gun": bps,
                     "t_stat": t, "p_value": pv})
    df_s = pd.DataFrame(rows)
    df_s["q_value_BH"] = bh_fdr(df_s["p_value"].fillna(1).values)
    return df_s

tablo1 = period_stats(df_analiz)

# Pairwise karsilastirma (3 cift, BH-FDR)
pairs = [("Yeni", "Ihmal"), ("Yeni", "Olgun"), ("Ihmal", "Olgun")]
comp_rows = []
for p1, p2 in pairs:
    d1 = df_analiz[df_analiz["period"] == p1]["excess_ret"].dropna()
    d2 = df_analiz[df_analiz["period"] == p2]["excess_ret"].dropna()
    if len(d1) < 5 or len(d2) < 5:
        comp_rows.append({"Karsilastirma": f"{p1} vs {p2}", "t_stat": np.nan, "p_value": np.nan})
        continue
    t, pv = stats.ttest_ind(d1, d2, equal_var=False)
    comp_rows.append({"Karsilastirma": f"{p1} vs {p2}", "t_stat": t, "p_value": pv})
tablo2 = pd.DataFrame(comp_rows)
tablo2["q_value_BH"] = bh_fdr(tablo2["p_value"].fillna(1).values)

# ---------- ADIM 5: CALENDAR-TIME FIXED EFFECTS REGRESYONU ----------
# excess_return ~ C(period, Treatment("Olgun")) + C(year_month)
# Olgun referans kategori; Yeni ve Ihmal katsayilari, ayni takvim ayindaki
# Olgun gozlemlere gore fazlalik/eksiklik anlamina gelir.

df_reg = df_analiz.copy().reset_index(drop=True)
df_reg["ym"] = df_reg["date"].dt.to_period("M").astype(str)
df_reg["excess_ret_bps"] = df_reg["excess_ret"] * 10000

# Orneklem: her sembol-ay'dan max 5 gun (bellek tasarrufu)
rng = np.random.default_rng(PLACEBO_SEED)
idx_sample = []
for (sym, ym_val), grp in df_reg.groupby(["symbol", "ym"]):
    chosen = grp.index.tolist()
    if len(chosen) > 5:
        chosen = rng.choice(chosen, 5, replace=False).tolist()
    idx_sample.extend(chosen)
df_reg_sample = df_reg.loc[idx_sample].reset_index(drop=True)
print(f"\nRegresyon orneklem boyutu: {len(df_reg_sample):,} gozlem")

# Design matrix: period dummies (ref=Olgun) + ym dummies (ref=ilk ay)
period_dummies = pd.get_dummies(df_reg_sample["period"], drop_first=False).drop(columns=["Olgun"])
ym_dummies = pd.get_dummies(df_reg_sample["ym"], drop_first=True)  # ilk ay referans
X = pd.concat([pd.Series(1.0, index=df_reg_sample.index, name="const"),
               period_dummies.astype(float),
               ym_dummies.astype(float)], axis=1)
y = df_reg_sample["excess_ret_bps"].values

model = sm.OLS(y, X).fit(cov_type="HC3")

coef_rows = []
for label in ["Yeni", "Ihmal"]:
    coef = model.params.get(label, np.nan)
    tval = model.tvalues.get(label, np.nan)
    pval = model.pvalues.get(label, np.nan)
    coef_rows.append({"Donem (vs Olgun)": label, "Katsayi_bps": coef,
                      "t_stat": tval, "p_value": pval})
tablo3 = pd.DataFrame(coef_rows)
tablo3["q_value_BH"] = bh_fdr(tablo3["p_value"].fillna(1).values)

# ---------- PLACEBO: Donem etiketlerini sembol bazinda karistir ----------
# Her sembolun gercek IPO tarihi ve excess return serisi korunur;
# sadece hangi sembolun "Yeni"/"Ihmal"/"Olgun" olarak siniflandirildigi
# kardirilir (sembol bazinda donem etiket permutasyonu).
sym_period_map = df_analiz.groupby("symbol")["period"].agg(
    lambda x: x.mode()[0] if not x.empty else "Olgun"
)
shuffled_periods = sym_period_map.values.copy()
rng.shuffle(shuffled_periods)
sym_placebo_period = dict(zip(sym_period_map.index, shuffled_periods))

df_placebo = df_analiz.copy().reset_index(drop=True)
df_placebo["period_placebo"] = df_placebo["symbol"].map(sym_placebo_period)
df_placebo = df_placebo[df_placebo["period_placebo"] != "Gecis"]

placebo_comp_rows = []
for p1, p2 in pairs:
    d1 = df_placebo[df_placebo["period_placebo"] == p1]["excess_ret"].dropna()
    d2 = df_placebo[df_placebo["period_placebo"] == p2]["excess_ret"].dropna()
    if len(d1) < 5 or len(d2) < 5:
        placebo_comp_rows.append({"Placebo": f"{p1} vs {p2}", "t_stat": np.nan, "p_value": np.nan})
        continue
    t, pv = stats.ttest_ind(d1, d2, equal_var=False)
    placebo_comp_rows.append({"Placebo": f"{p1} vs {p2}", "t_stat": t, "p_value": pv})
tablo4 = pd.DataFrame(placebo_comp_rows)
tablo4["q_value_BH"] = bh_fdr(tablo4["p_value"].fillna(1).values)

# ---------- RAPOR ----------
print("\n" + "=" * 65)
print("TABLO 1 — Her donem: ortalama gunluk excess return vs XU100 (t vs 0)")
print("=" * 65)
print(tablo1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 65)
print("TABLO 2 — Pairwise karsilastirma, ham (Welch t-test, BH-FDR)")
print("=" * 65)
print(tablo2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 65)
print("TABLO 3 — Calendar-time fixed effects regresyonu (referans: Olgun)")
print("  excess_ret_bps ~ period_dummies + ym_dummies, HC3 SE")
print("=" * 65)
print(tablo3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 65)
print("TABLO 4 — PLACEBO: sembol bazinda donem etiketi permutasyonu (seed=42)")
print("=" * 65)
print(tablo4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ================================================================
# EK TEST 1 — "Yeni" donemi: ilk 5 islem gunu vs geri kalan
# ================================================================
df_yeni = df_all[df_all["period"] == "Yeni"].copy().reset_index(drop=True)
df_yeni["yeni_sub"] = df_yeni["trading_day"].apply(
    lambda d: "Ilk5gun" if 0 <= d <= 4 else "Sonrasi"
)
df_yeni["ym"] = df_yeni["date"].dt.to_period("M").astype(str)
df_yeni["excess_ret_bps"] = df_yeni["excess_ret"] * 10000

# Calendar-time FE regresyonu: referans = "Sonrasi"
period_dummies_t1 = pd.get_dummies(df_yeni["yeni_sub"], drop_first=False).drop(columns=["Sonrasi"], errors="ignore")
ym_dummies_t1 = pd.get_dummies(df_yeni["ym"], drop_first=True)
X_t1 = pd.concat([pd.Series(1.0, index=df_yeni.index, name="const"),
                  period_dummies_t1.astype(float),
                  ym_dummies_t1.astype(float)], axis=1)
y_t1 = df_yeni["excess_ret_bps"].values
model_t1 = sm.OLS(y_t1, X_t1).fit(cov_type="HC3")

test1_rows = []
for label in ["Ilk5gun"]:
    coef = model_t1.params.get(label, np.nan)
    tval = model_t1.tvalues.get(label, np.nan)
    pval = model_t1.pvalues.get(label, np.nan)
    n_sub = (df_yeni["yeni_sub"] == label).sum()
    n_ref = (df_yeni["yeni_sub"] == "Sonrasi").sum()
    test1_rows.append({"Alt_donem": f"Ilk5gun (n={n_sub:,}) vs Sonrasi (n={n_ref:,})",
                       "Katsayi_bps": coef, "t_stat": tval, "p_value": pval})
tablo_t1 = pd.DataFrame(test1_rows)

# Ham karsilastirma da ekle (basit t-test)
d_ilk = df_yeni[df_yeni["yeni_sub"] == "Ilk5gun"]["excess_ret_bps"].dropna()
d_son = df_yeni[df_yeni["yeni_sub"] == "Sonrasi"]["excess_ret_bps"].dropna()
t_ham, p_ham = stats.ttest_ind(d_ilk, d_son, equal_var=False)

print("\n" + "=" * 65)
print("TEST 1 — 'Yeni' alt-donem: Ilk 5 islem gunu vs kalan")
print("  Calendar-time FE regresyonu (referans: Sonrasi), HC3 SE")
print("=" * 65)
print(tablo_t1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\n  Ham ortalamalar (calendar-time FE olmadan):")
print(f"    Ilk5gun : {d_ilk.mean():.2f} bps/gun (n={len(d_ilk):,})")
print(f"    Sonrasi : {d_son.mean():.2f} bps/gun (n={len(d_son):,})")
print(f"    Fark t-test: t={t_ham:.3f}, p={p_ham:.4f}")


# ================================================================
# EK TEST 2 — Veri kalitesi: stale-price filtresi "Yeni" donem
# ================================================================
# Sembol-ay bazinda unique Open fiyat sayisi (forward ay = o ayin gunleri)
# df_all icin Open panel gerekmez; dogrudan data/*.csv'den hesapla
# Zaten df_yeni var; unique Open proxy olarak Close unique sayisini kullan
# (Open kolonu df_all'da yok, price proxy: excess_ret == 0 olan ard-ardina gunler
#  stale gosterge olarak kullanilabilir, ama daha temiz: CSV'den unique Open al)

stale_key_set = set()  # (symbol, ym_str) -> stale mi?
MIN_UNIQUE_OPEN_T2 = 3
for sym in df_yeni["symbol"].unique():
    path = os.path.join(DATA_DIR, f"{sym}.csv")
    if not os.path.exists(path):
        continue
    try:
        df_tmp = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        continue
    if "Open" not in df_tmp.columns:
        continue
    df_tmp["ym"] = df_tmp["Date"].dt.to_period("M").astype(str)
    for ym_val, grp in df_tmp.groupby("ym"):
        if grp["Open"].nunique() < MIN_UNIQUE_OPEN_T2:
            stale_key_set.add((sym, ym_val))

df_yeni["ym"] = df_yeni["date"].dt.to_period("M").astype(str)
df_yeni["is_stale"] = df_yeni.apply(
    lambda r: (r["symbol"], r["ym"]) in stale_key_set, axis=1
)
n_stale = df_yeni["is_stale"].sum()
df_yeni_clean = df_yeni[~df_yeni["is_stale"]].reset_index(drop=True)

# FE regresyonu filtre sonrasi
period_dummies_t2 = pd.get_dummies(df_yeni_clean["yeni_sub"], drop_first=False).drop(columns=["Sonrasi"], errors="ignore")
ym_dummies_t2 = pd.get_dummies(df_yeni_clean["ym"], drop_first=True)
X_t2 = pd.concat([pd.Series(1.0, index=df_yeni_clean.index, name="const"),
                  period_dummies_t2.astype(float),
                  ym_dummies_t2.astype(float)], axis=1)
y_t2 = df_yeni_clean["excess_ret_bps"].values
model_t2 = sm.OLS(y_t2, X_t2).fit(cov_type="HC3")

print("\n" + "=" * 65)
print("TEST 2 — Stale-price filtresi etkisi (unique Open < 3 olan sembol-ay disla)")
print("=" * 65)
print(f"  'Yeni' donem toplam gozlem  : {len(df_yeni):,}")
print(f"  Stale olarak dislanan       : {n_stale:,} ({100*n_stale/len(df_yeni):.1f}%)")
print(f"  Kalan gozlem                : {len(df_yeni_clean):,}")
print()

# Onceki model (T1 Sonrasi referanssiz, tum Yeni vs 0)
t_pre, p_pre = stats.ttest_1samp(df_yeni["excess_ret_bps"].dropna(), 0)
t_post, p_post = stats.ttest_1samp(df_yeni_clean["excess_ret_bps"].dropna(), 0)

tablo_t2 = pd.DataFrame([
    {"Durum": "Filtre oncesi (tum Yeni)", "n": len(df_yeni),
     "Ort_bps/gun": df_yeni["excess_ret_bps"].mean(), "t_stat": t_pre, "p_value": p_pre},
    {"Durum": "Filtre sonrasi (stale cikarildi)", "n": len(df_yeni_clean),
     "Ort_bps/gun": df_yeni_clean["excess_ret_bps"].mean(), "t_stat": t_post, "p_value": p_post},
])
print(tablo_t2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

t2_coef = model_t2.params.get("Ilk5gun", np.nan)
t2_tval = model_t2.tvalues.get("Ilk5gun", np.nan)
t2_pval = model_t2.pvalues.get("Ilk5gun", np.nan)
print(f"\n  Ilk5gun vs Sonrasi (FE, stale-filtreli): katsayi={t2_coef:.4f} bps, t={t2_tval:.4f}, p={t2_pval:.4f}")


# ================================================================
# EK TEST 3 — Maliyet-dahil net sonuc
# ================================================================
# Bulgu: +35.98 bps/gun (calendar-time FE, referans Olgun)
# Bu rakam, 6 aylik "Yeni" donemi boyunca tutulacak TEK POZISYON
# varsayimiyla yorumlanmalidir (buy-and-hold), cunku gozlem birimi
# gunluk close-to-close getiridir ama pozisyon her gun acilip
# kapatilmiyor -- bir hisseye "Yeni" doneminde girilip 6 ay tutulur.
#
# Maliyet yapisi (tek seferlik gir-cik):
#   Midas+DUSUK  : 20 bps round-trip (tek yonlu spread ~10bps, komisyon ~10bps)
#   AtaYatirim+YUKSEK: 150 bps round-trip
# 6 aylik donemde kumulatif gunluk excess: avg_bps/gun * trading_days
#
# "Yeni" donem ham ortalamasi: 7.74 bps/gun (tablo1), FE katsayisi vs Olgun: +35.98 bps/gun
# (FE katsayisi "Olgun'a gore fazi" gosteriyor, toplam Yeni getirisi degil)
# Mutlak Yeni donem getirisi olarak Tablo1 degerini (7.74 bps/gun) kullan.

avg_yeni_bps_day = df_all[df_all["period"] == "Yeni"]["excess_ret"].mean() * 10000
avg_yeni_ilk5 = d_ilk.mean()
avg_yeni_sonrasi = d_son.mean()

# 6 ay yaklasik 126 islem gunu (BIST)
TRADING_DAYS_6M = 126
kum_yeni = avg_yeni_bps_day * TRADING_DAYS_6M

COST_LOW_BPS  = 20.0   # Midas+DUSUK round-trip
COST_HIGH_BPS = 150.0  # AtaYatirim+YUKSEK round-trip

net_low  = kum_yeni - COST_LOW_BPS
net_high = kum_yeni - COST_HIGH_BPS

print("\n" + "=" * 65)
print("TEST 3 — Maliyet-dahil net sonuc")
print("  Varsayim: 'Yeni' doneminde tek seferlik gir-cik (buy-and-hold 6 ay)")
print(f"  Ortalama gunluk excess return (Yeni donem, vs XU100): {avg_yeni_bps_day:.2f} bps/gun")
print(f"  Kumulatif (~{TRADING_DAYS_6M} islem gunu x {avg_yeni_bps_day:.2f} bps): {kum_yeni:.1f} bps (~{kum_yeni/100:.1f}%)")
print("=" * 65)

tablo_t3 = pd.DataFrame([
    {"Maliyet_Senaryosu": f"Midas+DUSUK ({COST_LOW_BPS:.0f} bps rt)",
     "Kumulatif_gross_bps": kum_yeni, "Maliyet_bps": COST_LOW_BPS, "Net_bps": net_low},
    {"Maliyet_Senaryosu": f"AtaYatirim+YUKSEK ({COST_HIGH_BPS:.0f} bps rt)",
     "Kumulatif_gross_bps": kum_yeni, "Maliyet_bps": COST_HIGH_BPS, "Net_bps": net_high},
])
print(tablo_t3.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
print(f"\n  NOT: +35.98 bps/gun (Tablo3 FE katsayisi) Olgun'A-GORE FAZLALIK'tir,")
print(f"  mutlak getiri degil. Mutlak gunluk Yeni donem ortalamasi: {avg_yeni_bps_day:.2f} bps/gun kullanildi.")
print(f"\n  Ilk 5 gun vs kalan Yeni donem karsilastirmasi:")
print(f"    Ilk 5 gun  : {avg_yeni_ilk5:.2f} bps/gun")
print(f"    6. gun+    : {avg_yeni_sonrasi:.2f} bps/gun")
