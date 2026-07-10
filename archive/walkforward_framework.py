# NOT: Bu dosya walk-forward mantigi icermiyor, gercek walk-forward icin walkforward_multi_signal.py kullanin. Output cakismasi onlenmistir.
"""
diagnose_signals.py

Amac: volume_spike_2x ve extreme_down_reversal sinyallerinin
  1) ayni olayi mi yakaladigini (overlap)
  2) buyuklugun illiquid-hisse artefakti mi oldugunu (likidite stratifikasyonu)
  3) en ekstrem olaylarin (sembol+tarih) hangileri oldugunu (manuel inceleme icin)
kontrol eder.

Beklenen veri yapisi: data/<SYMBOL>.csv, kolonlar: Date, Open, High, Low, Close, Volume
(get_bist_data.py'nin ciktisiyla ayni format varsayiliyor)

Kullanim:
    python diagnose_signals.py
Cikti:
    diagnostic_report.txt
    outlier_events.csv
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_ROWS = 250          # comprehensive_scan.py ile tutarli esik
VOLUME_SPIKE_MULT = 2.0
EXTREME_DOWN_PCTL = 0.05  # gunun en kotu %5'i
FWD_HORIZONS = [1, 3, 5]  # forward return gunleri (excess hesaplamak icin)

EXCLUDE_NAMES = {"USDTRY", "USDTRY=X"}


def load_symbol(path):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None
        needed = {"Date", "Close", "Volume"}
        if not needed.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbols = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if name in EXCLUDE_NAMES:
            continue
        df = load_symbol(f)
        if df is not None:
            symbols[name] = df

    print(f"Loaded {len(symbols)} symbols with >= {MIN_ROWS} rows")

    # --- Build a common date index (market average) for excess return calc
    all_dates = sorted(set().union(*[set(df["Date"]) for df in symbols.values()]))
    price_cols = {}
    vol_cols = {}
    for name, df in symbols.items():
        s = df.set_index("Date")
        price_cols[name] = s["Close"]
        vol_cols[name] = s["Volume"]
    price_panel = pd.concat(price_cols, axis=1).reindex(all_dates)
    vol_panel = pd.concat(vol_cols, axis=1).reindex(all_dates)

    ret_panel = price_panel.pct_change()
    market_ret = ret_panel.mean(axis=1, skipna=True)  # simple equal-weight proxy

    # median dollar volume per symbol as liquidity proxy
    dollar_vol = (price_panel * vol_panel)
    liquidity = dollar_vol.median(axis=0, skipna=True).dropna()
    tertiles = pd.qcut(liquidity, 3, labels=["illiquid", "mid", "liquid"])

    events_volspike = []  # (symbol, date)
    events_extreme = []

    for name in symbols:
        r = ret_panel[name]
        v = vol_panel[name]
        vol_ma20 = v.rolling(20).mean()
        spike_mask = (v > VOLUME_SPIKE_MULT * vol_ma20) & vol_ma20.notna()
        for d in ret_panel.index[spike_mask.reindex(ret_panel.index, fill_value=False)]:
            events_volspike.append((name, d))

    # extreme_down: bottom 5% of cross-sectional daily returns, per day
    for d in ret_panel.index:
        day_rets = ret_panel.loc[d].dropna()
        if len(day_rets) < 20:
            continue
        cutoff = day_rets.quantile(EXTREME_DOWN_PCTL)
        losers = day_rets[day_rets <= cutoff]
        for name in losers.index:
            events_extreme.append((name, d))

    set_spike = set(events_volspike)
    set_extreme = set(events_extreme)
    overlap = set_spike & set_extreme

    overlap_pct_of_spike = len(overlap) / max(len(set_spike), 1) * 100
    overlap_pct_of_extreme = len(overlap) / max(len(set_extreme), 1) * 100

    # --- forward excess return by liquidity tertile, per signal
    def forward_excess(events, horizon):
        rows = []
        for name, d in events:
            if name not in ret_panel.columns:
                continue
            idx = ret_panel.index.get_indexer([d])[0]
            if idx == -1 or idx + horizon >= len(ret_panel.index):
                continue
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]
            stock_fwd = (1 + ret_panel.loc[fwd_dates, name]).prod() - 1
            mkt_fwd = (1 + market_ret.loc[fwd_dates]).prod() - 1
            excess = stock_fwd - mkt_fwd
            liq_bucket = tertiles.get(name, np.nan)
            rows.append((name, d, excess, liq_bucket))
        return pd.DataFrame(rows, columns=["symbol", "date", "excess", "liquidity"])

    report_lines = []
    report_lines.append(f"Symbols loaded: {len(symbols)}")
    report_lines.append(f"volume_spike_2x events: {len(set_spike)}")
    report_lines.append(f"extreme_down_reversal events: {len(set_extreme)}")
    report_lines.append(f"Overlap (same symbol+date): {len(overlap)}")
    report_lines.append(f"  -> {overlap_pct_of_spike:.1f}% of volume_spike events are also extreme_down events")
    report_lines.append(f"  -> {overlap_pct_of_extreme:.1f}% of extreme_down events are also volume_spike events")
    report_lines.append("")

    for horizon in FWD_HORIZONS:
        report_lines.append(f"--- Horizon {horizon}d ---")
        for label, events in [("volume_spike_2x", set_spike), ("extreme_down_reversal", set_extreme)]:
            df_fwd = forward_excess(events, horizon)
            if df_fwd.empty:
                report_lines.append(f"  {label}: no events with sufficient forward data")
                continue
            overall = df_fwd["excess"].mean() * 10000
            report_lines.append(f"  {label}: overall excess = {overall:.1f} bps (n={len(df_fwd)})")
            for bucket in ["illiquid", "mid", "liquid"]:
                sub = df_fwd[df_fwd["liquidity"] == bucket]
                if len(sub) == 0:
                    continue
                bps = sub["excess"].mean() * 10000
                report_lines.append(f"      {bucket:8s}: {bps:.1f} bps (n={len(sub)})")
        report_lines.append("")

    # --- outlier events (most extreme individual excess returns at horizon=5)
    df5 = forward_excess(set_spike | set_extreme, 5)
    if not df5.empty:
        df5_sorted = df5.sort_values("excess").head(30)
        df5_sorted.to_csv("outlier_events.csv", index=False)
        report_lines.append("Top 30 most extreme negative 5d-excess events saved to outlier_events.csv")
        report_lines.append("Inceleme onerisi: bu sembol+tarihleri elle kontrol et (split, temettu, limit-down var mi)")

    with open("walkforward_framework_diagnostic.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()