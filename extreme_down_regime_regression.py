"""Pre-registered regression: does macro volatility regime modulate the
extreme-down reversal signal? Interaction OLS (signal_strength x regime
dummy) on the full panel. See the Turkish pre-commitment block below for
the pre-registered expectations.
Result: no interaction edge — coeff +1.98 bps, t=0.35, p=0.72, N=1,437,255.
"""
# ON-TAAHHUT (calistirmadan once yazildi):
# Beklenen yon: sinyal_siddeti x rejim_dummy etkilesim katsayisi POZITIF
# Beklenen buyukluk: mutevazi (belki 20-50 bps rejimler arasi fark)
# Beklenen sonuc: yon dogru cikabilir ama maliyet dahil edilince muhtemelen yine negatif kalir
# Bu tek seferlik bir testtir, sonuc ne olursa olsun script bu haliyle NIHAI kabul edilir.

import os
import glob
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_DIR = "data"
MARKET_INDEX_PATH = "data_market/XU100.csv"
MIN_ROWS = 250
HORIZON = 5
EXCLUDE_NAMES = {"USDTRY", "USDTRY=X", "ISKUR"}
MAX_PLAUSIBLE_DAILY_RETURN = 1.0
IPO_SEASONING_DAYS = 60
BASELINE_WINDOW = 60
VOL_WINDOW = 20
VOL_HIGH_REGIME_PCTL = 0.80

COMMISSION_SCENARIOS_BPS = {
    "Midas_kampanyali": 1.0,
    "YapiKredi_kademeli": 22.0,
    "AtaYatirim": 38.0,
    "HalkYatirim_standart": 64.0,
}
SPREAD_SCENARIOS = {
    "DUSUK_spread": 0.10,
    "YUKSEK_spread": 0.20,
}


def load_symbol(path):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None
        needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def main():
    # --- Hisse verileri ---
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbols = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if name in EXCLUDE_NAMES:
            continue
        df = load_symbol(f)
        if df is not None:
            symbols[name] = df

    print(f"Loaded {len(symbols)} symbols")

    all_dates = sorted(set().union(*[set(df["Date"]) for df in symbols.values()]))
    price_cols, open_cols, vol_cols, hl_cols = {}, {}, {}, {}
    for name, df in symbols.items():
        s = df.set_index("Date")
        price_cols[name] = s["Close"]
        open_cols[name] = s["Open"]
        vol_cols[name] = s["Volume"]
        hl_cols[name] = (s["High"] - s["Low"]) / s["Close"]

    price_panel = pd.concat(price_cols, axis=1, sort=False).reindex(all_dates)
    open_panel = pd.concat(open_cols, axis=1, sort=False).reindex(all_dates)
    vol_panel = pd.concat(vol_cols, axis=1, sort=False).reindex(all_dates)
    hl_panel = pd.concat(hl_cols, axis=1, sort=False).reindex(all_dates)
    hl_baseline_panel = hl_panel.rolling(BASELINE_WINDOW, min_periods=20).median().shift(1)

    ret_panel = price_panel.pct_change()

    max_abs_ret = ret_panel.abs().max(axis=0)
    suspect_symbols = max_abs_ret[max_abs_ret > MAX_PLAUSIBLE_DAILY_RETURN].index.tolist()
    if suspect_symbols:
        print(f"UYARI: {len(suspect_symbols)} sembol veri hatasi supheli, cikariliyor")
        for panel in [price_panel, open_panel, vol_panel, hl_panel, ret_panel]:
            for s in suspect_symbols:
                if s in panel.columns:
                    panel.drop(columns=[s], inplace=True)
        for s in suspect_symbols:
            symbols.pop(s, None)

    # --- XU100 gercek endeks ---
    if not os.path.exists(MARKET_INDEX_PATH):
        raise FileNotFoundError(f"XU100 verisi bulunamadi: {MARKET_INDEX_PATH}. "
                                "Once get_xu100_borsapy.py calistirin.")
    xu100_df = pd.read_csv(MARKET_INDEX_PATH, parse_dates=["Date"])
    xu100_df = xu100_df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    xu100_close = xu100_df.set_index("Date")["Close"]
    market_ret = xu100_close.pct_change().reindex(all_dates)

    # --- ADIM 2: Rejim dummy (XU100 20-gun rolling vol, tum donem %80 pctl) ---
    xu100_rolling_vol = market_ret.rolling(VOL_WINDOW, min_periods=10).std()
    vol_threshold = xu100_rolling_vol.quantile(VOL_HIGH_REGIME_PCTL)
    regime_series = (xu100_rolling_vol > vol_threshold).astype(int)
    regime_series.index = pd.to_datetime(regime_series.index)
    print(f"Rejim esigi (rolling vol %80 pctl): {vol_threshold:.6f}")
    print(f"Yuksek-rejim gun sayisi: {regime_series.sum()} / {regime_series.notna().sum()}")

    # --- IPO seasoning ---
    seasoning_cutoff = {}
    for name in ret_panel.columns:
        first_valid = ret_panel[name].first_valid_index()
        if first_valid is None:
            continue
        pos = ret_panel.index.get_indexer([pd.Timestamp(first_valid)])[0]
        cutoff_pos = min(pos + IPO_SEASONING_DAYS, len(ret_panel.index) - 1)
        seasoning_cutoff[name] = ret_panel.index[cutoff_pos]

    def in_seasoning_period(name, date):
        cutoff = seasoning_cutoff.get(name)
        return cutoff is not None and date <= cutoff

    # --- ADIM 1: Sinyal siddeti (surekli) + ADIM 3: Forward return ---
    dollar_vol = (price_panel * vol_panel)
    liquidity = dollar_vol.median(axis=0, skipna=True).dropna()
    tertiles = pd.qcut(liquidity, 3, labels=["illiquid", "mid", "liquid"])

    rows = []
    for d in ret_panel.index:
        day_rets = ret_panel.loc[d].dropna()
        if len(day_rets) < 20:
            continue
        rank = day_rets.rank(pct=True)
        signal_strength = 0.5 - rank  # surekli; rank kucukse (asagi iflas) buyuk pozitif deger

        d_ts = pd.Timestamp(d)
        regime_val = regime_series.get(d_ts, np.nan)
        if np.isnan(regime_val):
            continue

        idx = ret_panel.index.get_indexer([d])[0]
        if idx == -1 or idx + HORIZON + 1 >= len(ret_panel.index):
            continue
        entry_date = ret_panel.index[idx + 1]
        exit_date = ret_panel.index[idx + HORIZON + 1]
        fwd_dates = ret_panel.index[idx + 1: idx + 1 + HORIZON]
        mkt_fwd = (1 + market_ret.loc[fwd_dates]).prod() - 1

        for name in day_rets.index:
            if in_seasoning_period(name, d):
                continue
            if name not in open_panel.columns:
                continue
            try:
                stock_fwd = open_panel.loc[exit_date, name] / open_panel.loc[entry_date, name] - 1
            except KeyError:
                continue
            if pd.isna(stock_fwd) or pd.isna(mkt_fwd):
                continue
            excess = stock_fwd - mkt_fwd

            hl_baseline = hl_baseline_panel.loc[d, name] if name in hl_baseline_panel.columns else np.nan

            rows.append({
                "ticker": name,
                "date": d,
                "signal_strength": signal_strength[name],
                "regime_dummy": int(regime_val),
                "forward_return_bps": excess * 10000,
                "hl_baseline": hl_baseline,
                "liq_bucket": tertiles.get(name, np.nan),
            })

    panel = pd.DataFrame(rows).dropna(subset=["signal_strength", "regime_dummy", "forward_return_bps"])
    print(f"Panel satir sayisi (N): {len(panel)}")

    # --- Maliyet hesabi ---
    def add_net_returns(df):
        df = df.copy()
        for comm_name, comm_bps in COMMISSION_SCENARIOS_BPS.items():
            for sp_name, sp_frac in SPREAD_SCENARIOS.items():
                spread_bps = df["hl_baseline"].fillna(df["hl_baseline"].median()) * sp_frac * 2 * 10000
                total_cost = spread_bps + comm_bps
                col = f"net_{comm_name}_{sp_name}_bps"
                df[col] = df["forward_return_bps"] - total_cost
        return df

    panel = add_net_returns(panel)

    # --- ADIM 4: Regresyonlar ---
    reg_targets = {
        "BRUT": "forward_return_bps",
        "NET_Midas_DUSUK": f"net_Midas_kampanyali_DUSUK_spread_bps",
        "NET_AtaYatirim_YUKSEK": f"net_AtaYatirim_YUKSEK_spread_bps",
    }

    print("\n" + "=" * 70)
    print("REGRESYON SONUCLARI  (horizon=5g, forward excess return, bps)")
    print("Model: forward_return ~ signal_strength + regime_dummy + signal_strength:regime_dummy")
    print("=" * 70)

    for label, dep_var in reg_targets.items():
        if dep_var not in panel.columns:
            print(f"\n[{label}] Kolon bulunamadi: {dep_var}")
            continue
        sub = panel[["signal_strength", "regime_dummy", dep_var]].dropna()
        formula = f"{dep_var} ~ signal_strength + regime_dummy + signal_strength:regime_dummy"
        result = smf.ols(formula, data=sub).fit()

        print(f"\n[{label}]  N={len(sub):,}")
        print(f"{'Degisken':<35} {'Katsayi':>10} {'Std Hata':>10} {'t-stat':>8} {'p-value':>10}")
        print("-" * 75)
        for var in ["Intercept", "signal_strength", "regime_dummy", "signal_strength:regime_dummy"]:
            if var in result.params:
                coef = result.params[var]
                se = result.bse[var]
                t = result.tvalues[var]
                p = result.pvalues[var]
                print(f"{var:<35} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>10.4f}")
        print(f"R-squared: {result.rsquared:.6f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
