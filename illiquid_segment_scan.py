"""
illiquid_segment_scan.py

En düşük TL-hacimli %20'lik alt-evrende 4 hipotezi test eder ve
tüm-evren (comprehensive_scan_results.csv) ile karşılaştırır.

Hipotezler: extreme_down_reversal, extreme_up_reversal,
            volume_spike_2x, xsect_momentum_top20_lb10
Benchmark: XU100 Open fiyat serisi (data_market/XU100.csv)
Maliyet:   Her sembolün kendi HL_baseline spread tahmini (sembol bazlı)
           Senaryo A: Midas 1bps + düşük spread (%10)
           Senaryo B: AtaYatırım 38bps + yüksek spread (%20)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

from comprehensive_scan import read_stock_data, align_data

warnings.filterwarnings("ignore")

DATA_DIR      = "data"
XU100_PATH    = "data_market/XU100.csv"
SCAN_CSV      = "comprehensive_scan_results.csv"
HORIZONS      = [5, 10, 20]
ILLIQ_PCT     = 0.20          # Alt %20
LOOKBACK_DAYS = 504           # ~2 yıl işlem günü
HL_WINDOW     = 60
SCENARIOS = {
    "Midas+DUSUK":       {"commission_bps": 1.0,  "spread_frac": 0.10},
    "AtaYatirim+YUKSEK": {"commission_bps": 38.0, "spread_frac": 0.20},
}

TARGET_HYPS = {
    "extreme_down_reversal",
    "extreme_up_reversal",
    "volume_spike_2x",
    "xsect_momentum_top20_lb10",
}

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def load_xu100_open():
    df = pd.read_csv(XU100_PATH, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    return df["Open"]


def compute_hl_baseline(stock_data, symbols):
    """Her sembol için HL/Close rolling median (shift 1)."""
    hl = {}
    for sym in symbols:
        df = stock_data.get(sym)
        if df is None:
            continue
        s = df if isinstance(df, pd.DataFrame) else None
        if s is None:
            continue
        if {"High", "Low", "Close"}.issubset(s.columns):
            ratio = (s["High"] - s["Low"]) / s["Close"]
            hl[sym] = ratio.rolling(HL_WINDOW, min_periods=20).median().shift(1)
    return pd.DataFrame(hl)


def liquidity_filter(close_df, volume_df, illiq_pct=0.20, lookback=504):
    """TL hacim = Close * Volume, son `lookback` günün medyanı."""
    tl_vol = close_df * volume_df
    last_n = tl_vol.iloc[-lookback:]
    median_tl = last_n.median(axis=0).dropna()
    median_tl.sort_values(inplace=True)
    cutoff_idx = int(np.ceil(len(median_tl) * illiq_pct))
    illiquid_syms = median_tl.iloc[:cutoff_idx].index.tolist()
    return illiquid_syms, median_tl


# ---------------------------------------------------------------------------
# Sinyal event-testi (non-overlapping, XU100 excess)
# ---------------------------------------------------------------------------

def event_test_xu100(signal_bool_df, open_df, xu100_open, horizon, label, results,
                     hl_baseline_df=None):
    """
    XU100 Open fiyatını benchmark olarak kullanarak excess return hesaplar.
    Cross-sectional aggregation: her tarihte sinyalli sembollerin ortalaması.
    Non-overlapping thinning uygulanır.
    """
    trading_days = open_df.index
    fwd_stock  = np.log(open_df.shift(-(horizon + 1)) / open_df.shift(-1))
    xu100_fwd  = np.log(xu100_open.shift(-(horizon + 1)) / xu100_open.shift(-1))
    xu100_fwd  = xu100_fwd.reindex(trading_days)

    event_dates = []
    event_vals  = []
    event_costs = {s: [] for s in SCENARIOS}   # sembol-bazlı maliyet toplama

    for date in signal_bool_df.index:
        if date not in fwd_stock.index:
            continue
        xu100_ret = xu100_fwd.get(date, np.nan)
        if pd.isna(xu100_ret):
            continue
        row_signal = signal_bool_df.loc[date]
        syms_on = row_signal[row_signal].index.tolist()
        if not syms_on:
            continue
        stock_rets = fwd_stock.loc[date, syms_on].dropna()
        if stock_rets.empty:
            continue
        excess = (stock_rets - xu100_ret).mean()
        event_dates.append(date)
        event_vals.append(excess)

        # Maliyet: sembol bazlı HL_baseline
        if hl_baseline_df is not None:
            for scen_name, scen in SCENARIOS.items():
                hl_vals = []
                for sym in stock_rets.index:
                    if sym in hl_baseline_df.columns and date in hl_baseline_df.index:
                        v = hl_baseline_df.at[date, sym]
                        hl_vals.append(v if not pd.isna(v) else np.nan)
                    else:
                        hl_vals.append(np.nan)
                hl_arr = np.array(hl_vals, dtype=float)
                hl_med = np.nanmedian(hl_arr)
                hl_arr = np.where(np.isnan(hl_arr), hl_med, hl_arr)
                spread_bps = hl_arr.mean() * scen["spread_frac"] * 10000 * 2
                total_cost = spread_bps + scen["commission_bps"] * 2
                event_costs[scen_name].append(total_cost)

    if len(event_vals) < 5:
        return

    event_vals_bps = np.array(event_vals) * 10000

    # Non-overlapping thinning
    no_vals = []
    no_costs = {s: [] for s in SCENARIOS}
    last_date = None
    for i, d in enumerate(event_dates):
        if last_date is None or (d - last_date).days >= horizon:
            no_vals.append(event_vals_bps[i])
            if hl_baseline_df is not None:
                for s in SCENARIOS:
                    no_costs[s].append(event_costs[s][i])
            last_date = d

    n_no = len(no_vals)
    if n_no < 5:
        return

    no_arr = np.array(no_vals)
    t_stat, p_value = ttest_1samp(no_arr, 0)

    row = {
        "hypothesis": label,
        "horizon": horizon,
        "n_events": len(event_vals),
        "n_nonoverlap": n_no,
        "brut_excess_bps": no_arr.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
    }

    for scen_name in SCENARIOS:
        if hl_baseline_df is not None and no_costs[scen_name]:
            cost_arr = np.array(no_costs[scen_name])
            net_arr  = no_arr - cost_arr
            row[f"net_{scen_name}_bps"] = net_arr.mean()
        else:
            row[f"net_{scen_name}_bps"] = np.nan

    results.append(row)


# ---------------------------------------------------------------------------
# 4 Hipotez
# ---------------------------------------------------------------------------

def run_extreme_reversal(returns_df, open_df, xu100_open, hl_baseline_df, horizons, results):
    rank = returns_df.rank(axis=1, pct=True)
    extreme_up   = rank >= 0.95
    extreme_down = rank <= 0.05
    for horizon in horizons:
        event_test_xu100(extreme_up,   open_df, xu100_open, horizon, "extreme_up_reversal",   results, hl_baseline_df)
        event_test_xu100(extreme_down, open_df, xu100_open, horizon, "extreme_down_reversal", results, hl_baseline_df)


def run_volume_spike(volume_df, returns_df, open_df, xu100_open, hl_baseline_df, horizons, results):
    if volume_df is None:
        return
    vol_ma    = volume_df.rolling(20).mean()
    vol_ratio = volume_df / vol_ma
    signal    = vol_ratio >= 2.0
    for horizon in horizons:
        event_test_xu100(signal, open_df, xu100_open, horizon, "volume_spike_2x", results, hl_baseline_df)


def run_xsect_momentum(returns_df, open_df, xu100_open, hl_baseline_df, horizons, lookback, results):
    past_return = returns_df.rolling(lookback).sum()
    rank = past_return.rank(axis=1, pct=True)
    top_signal = rank >= 0.8
    for horizon in horizons:
        event_test_xu100(top_signal, open_df, xu100_open, horizon,
                         f"xsect_momentum_top20_lb{lookback}", results, hl_baseline_df)


# ---------------------------------------------------------------------------
# Tüm-evren referans (comprehensive_scan_results.csv)
# ---------------------------------------------------------------------------

def load_fulluni_results():
    df = pd.read_csv(SCAN_CSV)
    df = df[df["period"] == "full"]
    df = df[df["hypothesis"].isin(TARGET_HYPS)]
    return df


# ---------------------------------------------------------------------------
# Karşılaştırma tablosu
# ---------------------------------------------------------------------------

def print_comparison(illiq_df, full_df):
    SEP = "=" * 130
    print()
    print(SEP)
    print("İLLİKİT ALT-EVREN vs TÜM EVREN — 4 Hipotez Karşılaştırması")
    print(SEP)
    hdr = (f"{'Hipotez':<38} {'H':>3} {'Evren':<9} {'N_no':>6} "
           f"{'Brüt(bps)':>10} {'t-stat':>8} {'p-value':>10} {'q-value':>10} "
           f"{'Net Midas':>10} {'Net Ata':>9}")
    print(hdr)
    print("-" * 130)

    for hyp in sorted(TARGET_HYPS):
        for horizon in HORIZONS:
            # İllikit
            ill = illiq_df[(illiq_df["hypothesis"] == hyp) & (illiq_df["horizon"] == horizon)]
            # Tüm evren
            full_row = full_df[(full_df["hypothesis"] == hyp) & (full_df["horizon"] == horizon)]

            if not ill.empty:
                r = ill.iloc[0]
                net_midas = r.get("net_Midas+DUSUK_bps", np.nan)
                net_ata   = r.get("net_AtaYatirim+YUKSEK_bps", np.nan)
                q = r.get("q_value", np.nan)
                print(f"{hyp:<38} {int(horizon):>3} {'ILLİKİT':<9} {int(r['n_nonoverlap']):>6} "
                      f"{r['brut_excess_bps']:>10.1f} {r['t_stat']:>8.2f} {r['p_value']:>10.4f} "
                      f"{q:>10.4f} {net_midas:>10.1f} {net_ata:>9.1f}")

            if not full_row.empty:
                fr = full_row.iloc[0]
                bps_no  = fr.get("excess_return_bps_nonoverlap", fr.get("excess_return_bps", np.nan))
                n_no    = fr.get("event_count_nonoverlap", fr.get("event_count", np.nan))
                t_no    = fr.get("t_stat_nonoverlap", fr.get("t_stat", np.nan))
                p_no    = fr.get("p_value_nonoverlap", fr.get("p_value", np.nan))
                q_no    = fr.get("q_value_bh", np.nan)
                print(f"{'':38} {int(horizon):>3} {'TÜM':<9} {int(n_no) if not pd.isna(n_no) else '?':>6} "
                      f"{bps_no:>10.1f} {t_no:>8.2f} {p_no:>10.4f} {q_no:>10.4f} "
                      f"{'(CS-mean)':>10} {'(CS-mean)':>9}")

            if not ill.empty or not full_row.empty:
                print()

    print(SEP)
    print("NOT: Tüm-evren satırları comprehensive_scan_results.csv'den, cross-sectional mean benchmark.")
    print("     İllikit satırları XU100 Open benchmark + sembol bazlı HL_baseline maliyet.")
    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Veri yükleniyor...")
    stock_data = read_stock_data(DATA_DIR)
    close_df, volume_df, open_df = align_data(stock_data)
    returns_df = np.log(close_df / close_df.shift(1)).dropna()

    print(f"Tam evren: {close_df.shape[1]} sembol, {close_df.shape[0]} islem gunu")
    print(f"Tarih: {close_df.index.min().date()} - {close_df.index.max().date()}")

    # --- Likidite filtresi ---
    print("\nLikidite filtresi uygulanıyor...")
    if volume_df is None:
        raise RuntimeError("Hacim verisi bulunamadı.")
    illiquid_syms, median_tl = liquidity_filter(close_df, volume_df, ILLIQ_PCT, LOOKBACK_DAYS)
    print(f"İllikit alt-evren (alt {int(ILLIQ_PCT*100)}%): {len(illiquid_syms)} sembol")
    print(f"Median TL-hacim kesim noktası: {median_tl.iloc[len(illiquid_syms)-1]:,.0f} TL/gün")
    print(f"Örnek illikit semboller: {illiquid_syms[:10]}")

    # Alt-evren panelleri
    ill_close  = close_df[illiquid_syms]
    ill_open   = open_df[illiquid_syms]
    ill_vol    = volume_df[illiquid_syms] if volume_df is not None else None
    ill_ret    = returns_df[[s for s in illiquid_syms if s in returns_df.columns]]

    # HL_baseline (sembol bazlı, illikit evren)
    print("HL_baseline (per-sembol) hesaplanıyor...")
    hl_baseline_df = compute_hl_baseline(stock_data, illiquid_syms)
    hl_baseline_df = hl_baseline_df.reindex(index=open_df.index)

    # XU100
    print("XU100 yükleniyor...")
    xu100_open = load_xu100_open()

    # --- Sinyaller ---
    print("İllikit alt-evrende sinyaller çalıştırılıyor...\n")
    results = []

    run_extreme_reversal(ill_ret, ill_open, xu100_open, hl_baseline_df, HORIZONS, results)
    run_volume_spike(ill_vol, ill_ret, ill_open, xu100_open, hl_baseline_df, HORIZONS, results)
    run_xsect_momentum(ill_ret, ill_open, xu100_open, hl_baseline_df, HORIZONS, 10, results)

    if not results:
        print("Sonuç yok.")
        return

    illiq_df = pd.DataFrame(results)

    # BH-FDR (4 hipotez x 3 horizon = 12 test)
    valid = illiq_df["p_value"].notna()
    p_vals = illiq_df.loc[valid, "p_value"].values
    if len(p_vals) > 0:
        _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
        illiq_df.loc[valid, "q_value"] = q_vals
    else:
        illiq_df["q_value"] = np.nan

    # Tüm-evren referans
    print("Tüm-evren referans yükleniyor (comprehensive_scan_results.csv)...")
    full_df = load_fulluni_results()

    # --- Özet: illikit ham tablo ---
    net_midas_col = "net_Midas+DUSUK_bps"
    net_ata_col   = "net_AtaYatirim+YUKSEK_bps"

    print("\n" + "=" * 110)
    print("İLLİKİT ALT-EVREN BACKTEST — XU100 Excess Return")
    print("=" * 110)
    print(f"{'Hipotez':<38} {'H':>3} {'N_no':>6} {'Brüt(bps)':>10} {'t-stat':>8} "
          f"{'p-value':>10} {'q-value':>10} {'NetMidas':>9} {'NetAta':>8}")
    print("-" * 110)
    for _, r in illiq_df.sort_values(["hypothesis", "horizon"]).iterrows():
        nm = r.get(net_midas_col, np.nan)
        na = r.get(net_ata_col,   np.nan)
        q  = r.get("q_value", np.nan)
        print(f"{r['hypothesis']:<38} {int(r['horizon']):>3} {int(r['n_nonoverlap']):>6} "
              f"{r['brut_excess_bps']:>10.1f} {r['t_stat']:>8.2f} {r['p_value']:>10.4f} "
              f"{q:>10.4f} {nm:>9.1f} {na:>8.1f}")
    print("=" * 110)

    # --- Karşılaştırma tablosu ---
    print_comparison(illiq_df, full_df)

    # CSV kaydet
    illiq_df.to_csv("illiquid_scan_results.csv", index=False)
    print("\nilliquid_scan_results.csv kaydedildi.")


if __name__ == "__main__":
    main()
