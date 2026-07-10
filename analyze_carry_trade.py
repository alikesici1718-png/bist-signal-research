"""
USDTRY carry trade test. Tests whether holding TRY (earning TCMB policy rate)
against USD produces a systematic positive return after accounting for
exchange rate depreciation. Permutation test (1000 iterations, shuffling
FX returns) checks whether the observed carry return is distinguishable
from random pairing of rate and FX series.
Result: no statistically significant edge (permutation p=0.129) — the
observed ~17% annualized carry return reflects the consistent level
difference between TRY rates and USD depreciation, not a genuine
timing-dependent signal.
"""
import pandas as pd
import numpy as np
from scipy import stats

usdtry = pd.read_csv("data/carry_trade/usdtry_price.csv", parse_dates=["date"])
tcmb = pd.read_csv("data/carry_trade/tcmb_policy_rate.csv")

tcmb["date"] = pd.to_datetime(tcmb["date"], format="%d-%m-%Y", errors="coerce")
tcmb = tcmb.dropna(subset=["date"]).sort_values("date")
usdtry = usdtry.sort_values("date")

df = pd.merge(usdtry, tcmb, on="date", how="inner").sort_values("date").reset_index(drop=True)

df["daily_fx_change"] = df["close_price"].pct_change()
df["daily_tl_rate"] = df["policy_rate"] / 365 / 100
df["carry_return"] = df["daily_tl_rate"] - df["daily_fx_change"]
df = df.dropna(subset=["carry_return"])

ann_carry = df["carry_return"].mean() * 365 * 100
t_stat, p_val = stats.ttest_1samp(df["carry_return"], 0)
n = len(df)

np.random.seed(42)
shuffled_fx = df["daily_fx_change"].values.copy()
np.random.shuffle(shuffled_fx)
placebo_return = df["daily_tl_rate"].values - shuffled_fx
placebo_ann = placebo_return.mean() * 365 * 100
_, placebo_p = stats.ttest_1samp(placebo_return, 0)

rng = np.random.default_rng(0)
fx_vals = df["daily_fx_change"].values.copy()
tl_vals = df["daily_tl_rate"].values
perm_t_stats = []
for _ in range(1000):
    shuffled = rng.permutation(fx_vals)
    perm_ret = tl_vals - shuffled
    t_perm, _ = stats.ttest_1samp(perm_ret, 0)
    perm_t_stats.append(t_perm)
perm_t_stats = np.array(perm_t_stats)
perm_p = np.mean(np.abs(perm_t_stats) >= np.abs(t_stat))

print(f"{'Metrik':<35} {'Gerçek':>12} {'Placebo':>12}")
print("-" * 61)
print(f"{'Örnek büyüklüğü (gün)':<35} {n:>12} {n:>12}")
print(f"{'Yıllık ortalama carry getiri (%)':<35} {ann_carry:>12.2f} {placebo_ann:>12.2f}")
print(f"{'t-istatistiği':<35} {t_stat:>12.4f} {'—':>12}")
print(f"{'p-değeri':<35} {p_val:>12.4f} {placebo_p:>12.4f}")

print()
print("Permütasyon Testi (1000 permütasyon)")
print("-" * 45)
print(f"{'Gerçek t-istatistiği':<35} {t_stat:>8.4f}")
print(f"{'Permütasyon t dağılımı ort.':<35} {perm_t_stats.mean():>8.4f}")
print(f"{'Permütasyon t dağılımı std.':<35} {perm_t_stats.std():>8.4f}")
print(f"{'Permütasyon p-değeri':<35} {perm_p:>8.4f}")
