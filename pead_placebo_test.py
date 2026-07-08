"""
pead_placebo_test.py

PEAD plasebo testi: KAP event tarihleri yerine her sembol icin
ayni sayida RASTGELE tarih kullanarak ayni backtest mantigi uygulanir.
Amac: gozlemlenen getirinin genel piyasa trendiyle aciklanip aciklanamayacagini test etmek.
"""

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

KAP_CSV = Path("kap_data/kap_financial_report_dates.csv")
DATA_DIR = Path("data")
HORIZONS = [5, 10, 20]
SEED = 42

SCENARIOS = {
    "Midas+DUSUK":       {"commission_bps": 1.0,  "spread_frac": 0.10},
    "AtaYatirim+YUKSEK": {"commission_bps": 38.0, "spread_frac": 0.20},
}


def load_price_panels():
    files = glob.glob(str(DATA_DIR / "*.csv"))
    opens, highs, lows = {}, {}, {}
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
        if "Open" in df.columns:
            opens[ticker] = df["Open"]
        if "High" in df.columns:
            highs[ticker] = df["High"]
        if "Low" in df.columns:
            lows[ticker] = df["Low"]
    open_panel = pd.DataFrame(opens)
    high_panel = pd.DataFrame(highs)
    low_panel  = pd.DataFrame(lows)
    return open_panel, high_panel, low_panel


def compute_hl_baseline(high_panel, low_panel):
    hl_ratio = (high_panel - low_panel) / ((high_panel + low_panel) / 2)
    return hl_ratio.rolling(60, min_periods=20).median().shift(1)


def build_placebo_events(kap_df, open_panel, rng):
    """Her sembol icin KAP'taki event sayisi kadar rastgele 'sahte event tarihi' uret."""
    trading_days = open_panel.index
    counts = kap_df.groupby("ticker").size()
    rows = []
    for ticker, n_events in counts.items():
        if ticker not in open_panel.columns:
            continue
        # Cikisin mumkun olabilmesi icin son horizon kadar gunu disla (en buyuk horizon=20)
        max_horizon = max(HORIZONS)
        valid_days = trading_days[: len(trading_days) - max_horizon - 1]
        if len(valid_days) < n_events:
            chosen = valid_days
        else:
            chosen = rng.choice(valid_days, size=n_events, replace=False)
        for d in chosen:
            rows.append({"ticker": ticker, "event_date": pd.Timestamp(d)})
    return pd.DataFrame(rows)


def build_event_returns(events_df, open_panel, hl_baseline, horizon):
    trading_days = open_panel.index
    rows = []
    for _, ev in events_df.iterrows():
        ticker = ev["ticker"]
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
        hl_val = hl_baseline.at[entry_date, ticker] if ticker in hl_baseline.columns else np.nan
        rows.append({
            "ticker": ticker,
            "gross_return": exit_price / entry_price - 1.0,
            "hl_baseline": hl_val,
        })
    return pd.DataFrame(rows)


def run_stats(events_df, open_panel, hl_baseline):
    all_results = []
    for horizon in HORIZONS:
        df = build_event_returns(events_df, open_panel, hl_baseline, horizon)
        if df.empty:
            continue
        gross_bps = df["gross_return"].values * 10000
        t_stat, p_value = ttest_1samp(gross_bps, 0)
        row = {
            "horizon": horizon,
            "n": len(gross_bps),
            "brut_bps": gross_bps.mean(),
            "t_stat": t_stat,
            "p_value": p_value,
        }
        for scen_name, scen in SCENARIOS.items():
            spread_bps = df["hl_baseline"].fillna(df["hl_baseline"].median()) * scen["spread_frac"] * 10000 * 2
            net_bps = gross_bps - (spread_bps.values + scen["commission_bps"] * 2)
            row[f"net_{scen_name}_bps"] = net_bps.mean()
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
    open_panel, high_panel, low_panel = load_price_panels()
    hl_baseline = compute_hl_baseline(high_panel, low_panel)

    print("KAP event sayilari yukleniyor...")
    kap_df = pd.read_csv(KAP_CSV)

    print("Rastgele plasebo eventler uretiliyor (seed=42)...")
    rng = np.random.default_rng(SEED)
    placebo_df = build_placebo_events(kap_df, open_panel, rng)
    print(f"  {len(placebo_df)} plasebo event, {placebo_df['ticker'].nunique()} sembol")

    print("Plasebo backtest calistiriliyor...\n")
    result_df = run_stats(placebo_df, open_panel, hl_baseline)

    if result_df.empty:
        print("Sonuc yok.")
        return

    scen_cols = [f"net_{s}_bps" for s in SCENARIOS]
    print("=" * 90)
    print("PLASEBO BACKTEST SONUCLARI (rastgele tarihler, seed=42)")
    print("=" * 90)
    print(f"{'Horizon':>8} {'N':>6} {'Brut(bps)':>10} {'t-stat':>8} {'p-value':>10} {'q-value':>10}", end="")
    for s in SCENARIOS:
        print(f" {'Net '+s+' bps':>22}", end="")
    print()
    print("-" * 90)
    for _, r in result_df.iterrows():
        print(f"{int(r['horizon']):>8} {int(r['n']):>6} {r['brut_bps']:>10.1f} {r['t_stat']:>8.2f} {r['p_value']:>10.4f} {r['q_value']:>10.4f}", end="")
        for s in SCENARIOS:
            print(f" {r[f'net_{s}_bps']:>22.1f}", end="")
        print()
    print("=" * 90)


if __name__ == "__main__":
    main()
