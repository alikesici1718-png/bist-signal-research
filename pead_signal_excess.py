"""
pead_signal_excess.py

PEAD backtest — excess return (XU100'e gore).
Tum istatistikler stock_forward_return - xu100_forward_return uzerinden hesaplanir.
Maliyet yalnizca stock'un kendi maliyetidir.
"""

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

KAP_CSV = Path("kap_data/kap_financial_report_dates.csv")
DATA_DIR = Path("data")
XU100_CSV = Path("data_market/XU100.csv")
HORIZONS = [5, 10, 20]

SCENARIOS = {
    "Midas+DUSUK":       {"commission_bps": 1.0,  "spread_frac": 0.10},
    "AtaYatirim+YUKSEK": {"commission_bps": 38.0, "spread_frac": 0.20},
}


def load_price_panels():
    files = glob.glob(str(DATA_DIR / "*.csv"))
    opens, closes, highs, lows = {}, {}, {}, {}
    for f in files:
        ticker = Path(f).stem.upper()
        try:
            df = pd.read_csv(f, parse_dates=["Date"], index_col="Date")
        except Exception:
            try:
                df = pd.read_csv(f, parse_dates=[0], index_col=0)
                df.index.name = "Date"
            except Exception:
                continue
        df.sort_index(inplace=True)
        if "Open"  in df.columns: opens[ticker]  = df["Open"]
        if "Close" in df.columns: closes[ticker] = df["Close"]
        if "High"  in df.columns: highs[ticker]  = df["High"]
        if "Low"   in df.columns: lows[ticker]   = df["Low"]
    open_panel  = pd.DataFrame(opens)
    close_panel = pd.DataFrame(closes)
    high_panel  = pd.DataFrame(highs)
    low_panel   = pd.DataFrame(lows)
    idx = open_panel.index
    return open_panel.reindex(idx), close_panel.reindex(idx), high_panel.reindex(idx), low_panel.reindex(idx)


def load_xu100():
    df = pd.read_csv(XU100_CSV, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    return df["Open"]


def compute_hl_baseline(high_panel, low_panel):
    hl_ratio = (high_panel - low_panel) / ((high_panel + low_panel) / 2)
    return hl_ratio.rolling(60, min_periods=20).median().shift(1)


def load_kap_events():
    df = pd.read_csv(KAP_CSV)
    df["publish_date"] = pd.to_datetime(df["publish_date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["publish_date"])
    df["event_date"] = df["publish_date"].dt.normalize()
    return df


def build_event_returns(events_df, open_panel, hl_baseline, xu100_open, horizon):
    trading_days = open_panel.index
    rows = []
    for _, ev in events_df.iterrows():
        ticker = ev["ticker"]
        if ticker not in open_panel.columns:
            continue
        event_date = ev["event_date"]
        future_days = trading_days[trading_days > event_date]
        if len(future_days) == 0:
            continue
        entry_date = future_days[0]
        entry_pos = trading_days.get_loc(entry_date)
        exit_pos = entry_pos + horizon
        if exit_pos >= len(trading_days):
            continue
        exit_date = trading_days[exit_pos]

        entry_price = open_panel.at[entry_date, ticker]
        exit_price  = open_panel.at[exit_date, ticker]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        xu100_entry = xu100_open.get(entry_date, np.nan)
        xu100_exit  = xu100_open.get(exit_date, np.nan)
        if pd.isna(xu100_entry) or pd.isna(xu100_exit) or xu100_entry <= 0:
            continue

        stock_ret  = exit_price / entry_price - 1.0
        xu100_ret  = xu100_exit / xu100_entry - 1.0
        excess_ret = stock_ret - xu100_ret

        hl_val = hl_baseline.at[entry_date, ticker] if ticker in hl_baseline.columns else np.nan

        rows.append({
            "ticker":      ticker,
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "excess_return": excess_ret,
            "hl_baseline": hl_val,
        })
    return pd.DataFrame(rows)


def run_stats(events_df, open_panel, hl_baseline, xu100_open):
    all_results = []
    for horizon in HORIZONS:
        df = build_event_returns(events_df, open_panel, hl_baseline, xu100_open, horizon)
        if df.empty:
            continue
        excess_bps = df["excess_return"].values * 10000
        t_stat, p_value = ttest_1samp(excess_bps, 0)
        n = len(excess_bps)
        row = {
            "horizon": horizon,
            "n": n,
            "brut_excess_bps": excess_bps.mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "win_rate": (excess_bps > 0).mean(),
        }
        for scen_name, scen in SCENARIOS.items():
            spread_bps = df["hl_baseline"].fillna(df["hl_baseline"].median()) * scen["spread_frac"] * 10000 * 2
            total_cost_bps = spread_bps + scen["commission_bps"] * 2
            net_excess_bps = excess_bps - total_cost_bps.values
            row[f"net_{scen_name}_bps"] = net_excess_bps.mean()
        all_results.append(row)

    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        return result_df
    valid = result_df["p_value"].notna()
    p_vals = result_df.loc[valid, "p_value"].values
    if len(p_vals) > 0:
        _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
        result_df.loc[valid, "q_value"] = q_vals
    else:
        result_df["q_value"] = np.nan
    return result_df


def main():
    print("Fiyat panelleri yukleniyor...")
    open_panel, close_panel, high_panel, low_panel = load_price_panels()
    print(f"  {open_panel.shape[1]} sembol, {len(open_panel)} islem gunu")

    print("XU100 yukleniyor...")
    xu100_open = load_xu100()

    print("HL_baseline hesaplaniyor...")
    hl_baseline = compute_hl_baseline(high_panel, low_panel)

    print("KAP bildirimleri yukleniyor...")
    events_df = load_kap_events()
    print(f"  {len(events_df)} bildirim, {events_df['ticker'].nunique()} sembol")

    print("PEAD excess backtest calistiriliyor...\n")
    result_df = run_stats(events_df, open_panel, hl_baseline, xu100_open)

    if result_df.empty:
        print("Sonuc yok.")
        return

    scen_cols = [f"net_{s}_bps" for s in SCENARIOS]
    col_order = ["horizon", "n", "brut_excess_bps", "t_stat", "p_value", "q_value"] + scen_cols + ["win_rate"]
    result_df = result_df[col_order]

    print("=" * 96)
    print("PEAD BACKTEST SONUCLARI — EXCESS RETURN (XU100'e gore)")
    print("=" * 96)
    print(f"{'Horizon':>8} {'N':>6} {'BrutExcess':>11} {'t-stat':>8} {'p-value':>10} {'q-value':>10}", end="")
    for s in SCENARIOS:
        print(f" {'NetExcess '+s:>26}", end="")
    print(f" {'WinRate':>8}")
    print("-" * 96)
    for _, r in result_df.iterrows():
        print(f"{int(r['horizon']):>8} {int(r['n']):>6} {r['brut_excess_bps']:>11.1f} {r['t_stat']:>8.2f} {r['p_value']:>10.4f} {r['q_value']:>10.4f}", end="")
        for s in SCENARIOS:
            print(f" {r[f'net_{s}_bps']:>26.1f}", end="")
        print(f" {r['win_rate']:>8.1%}")
    print("=" * 96)
    print()
    print("NOT: config/symbols.txt statik bir listedir, survivorship bias iceriyor olabilir.")


if __name__ == "__main__":
    main()
