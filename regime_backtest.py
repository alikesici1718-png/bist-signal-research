"""
regime_backtest.py

extreme_down_reversal (SHORT) sinyalinin, makro (USDTRY) volatilite
rejimine gore FARKLI calisip calismadigini test eder. Onceki testler
(capacity_backtest.py) tum donemi birlikte degerlendiriyordu - bu script
donemi YUKSEK ve DUSUK USDTRY volatilite rejimlerine bolup, ayni
kapasite-kisitli metodolojiyi HER IKI rejimde ayri ayri calistirir.

HIPOTEZ: Sinyalin ortalamada (tum donem) kayip uretmesi, iki farkli
rejimin ortalamasi olabilir - belki YUKSEK volatilite doneminde (panik,
asiri tepki, likidite cekilmesi) sinyal calisiyor ama DUSUK volatilite
doneminde calismiyor (ya da tersi), ve ortalama bu ikisini birbirine
karistirip "edge yok" sonucunu veriyor.

Rejim tanimi: fetch_macro_regime.py tarafindan uretilen
data_macro/usdtry_regime.csv dosyasindan okunur (USDTRY gunluk
getirisinin 20-gunluk rolling std'sinin, tum donem medyanina gore
YUKSEK/DUSUK ikiye ayrilmasi).

Ayni altyapi (IPO seasoning, ISKUR filtresi, gercek XU100 benchmark,
capacity_backtest.py ile ayni kapasite/pozisyon kisiti, ayni maliyet
senaryolari) kullanilir - sadece sonuclar rejime gore ikiye bolunur.

Kullanim:
    python regime_backtest.py
Onkosul:
    fetch_macro_regime.py once calistirilmis olmali (data_macro/usdtry_regime.csv)
Cikti:
    regime_backtest_report.txt
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
MARKET_INDEX_PATH = "data_market/XU100.csv"
# Hangi makro rejim dosyasini kullanacagimiz - fetch_macro_regime.py'de
# ACTIVE_SERIES neyse, o dosyayi burada belirtmek gerekir.
# "usdtry" -> data_macro/usdtry_regime.csv (doviz/panik proxy'si)
# "tlref"  -> data_macro/tlref_regime.csv (faiz rejimi/para politikasi proxy'si)
ACTIVE_MACRO_SOURCE = "tlref"
MACRO_REGIME_PATH = f"data_macro/{ACTIVE_MACRO_SOURCE}_regime.csv"
MIN_ROWS = 250
EXTREME_DOWN_PCTL = 0.05
FWD_HORIZONS = [1, 3, 5]

EXCLUDE_NAMES = {"USDTRY", "USDTRY=X", "ISKUR"}
MAX_PLAUSIBLE_DAILY_RETURN = 1.0
IPO_SEASONING_DAYS = 60
BASELINE_WINDOW = 60

DAILY_POSITION_CAP = 15
POSITION_PCT_OF_DAILY_VOLUME = 0.02
FIXED_POSITION_SIZE = 50_000.0

COMMISSION_SCENARIOS_BPS = {
    "Midas_kampanyali": 1.0,
    "AtaYatirim": 38.0,
    "HalkYatirim_standart": 64.0,
}
SPREAD_SCENARIOS = {
    "DUSUK_spread": 0.10,
    "YUKSEK_spread": 0.20,
}
PRIMARY_COMMISSION = "AtaYatirim"
PRIMARY_SPREAD = "YUKSEK_spread"


def load_market_index(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    return df.set_index("Date")["Close"].pct_change()


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


def load_regime_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} bulunamadi. Once 'python fetch_macro_regime.py' calistirilmali."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")["regime"]


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

    regime_map = load_regime_map(MACRO_REGIME_PATH)
    print(f"Makro rejim verisi yuklendi: {len(regime_map)} gun "
          f"(YUKSEK={sum(regime_map=='YUKSEK')}, DUSUK={sum(regime_map=='DUSUK')})")

    all_dates = sorted(set().union(*[set(df["Date"]) for df in symbols.values()]))
    price_cols, vol_cols, hl_cols = {}, {}, {}
    for name, df in symbols.items():
        s = df.set_index("Date")
        price_cols[name] = s["Close"]
        vol_cols[name] = s["Volume"]
        hl_cols[name] = (s["High"] - s["Low"]) / s["Close"]

    price_panel = pd.concat(price_cols, axis=1, sort=False).reindex(all_dates)
    vol_panel = pd.concat(vol_cols, axis=1, sort=False).reindex(all_dates)
    hl_panel = pd.concat(hl_cols, axis=1, sort=False).reindex(all_dates)

    hl_baseline_panel = hl_panel.rolling(BASELINE_WINDOW, min_periods=20).median().shift(1)
    dollar_vol_panel = (price_panel * vol_panel).shift(1)

    ret_panel = price_panel.pct_change()

    max_abs_ret = ret_panel.abs().max(axis=0)
    suspect_symbols = max_abs_ret[max_abs_ret > MAX_PLAUSIBLE_DAILY_RETURN].index.tolist()
    if suspect_symbols:
        print(f"UYARI: {len(suspect_symbols)} sembol veri hatasi supheli, cikariliyor: {suspect_symbols}")
        price_panel = price_panel.drop(columns=suspect_symbols)
        vol_panel = vol_panel.drop(columns=suspect_symbols)
        hl_panel = hl_panel.drop(columns=suspect_symbols)
        ret_panel = ret_panel.drop(columns=suspect_symbols)
        dollar_vol_panel = dollar_vol_panel.drop(columns=suspect_symbols)
        for s in suspect_symbols:
            symbols.pop(s, None)

    xu100_ret = load_market_index(MARKET_INDEX_PATH)
    if xu100_ret is not None:
        market_ret = xu100_ret.reindex(all_dates)
        market_source = f"XU100 gercek endeks ({MARKET_INDEX_PATH})"
    else:
        market_ret = ret_panel.mean(axis=1, skipna=True)
        market_source = "ESIT-AGIRLIKLI PROXY (XU100 bulunamadi!)"

    seasoning_cutoff = {}
    for name in ret_panel.columns:
        first_valid = ret_panel[name].first_valid_index()
        if first_valid is None:
            continue
        pos = ret_panel.index.get_loc(first_valid)
        cutoff_pos = min(pos + IPO_SEASONING_DAYS, len(ret_panel.index) - 1)
        seasoning_cutoff[name] = ret_panel.index[cutoff_pos]

    def in_seasoning_period(name, date):
        cutoff = seasoning_cutoff.get(name)
        return cutoff is not None and date <= cutoff

    events_by_day = {}
    for d in ret_panel.index:
        day_rets = ret_panel.loc[d].dropna()
        if len(day_rets) < 20:
            continue
        cutoff = day_rets.quantile(EXTREME_DOWN_PCTL)
        losers = day_rets[day_rets <= cutoff]
        names = [n for n in losers.index if not in_seasoning_period(n, d)]
        if names:
            events_by_day[d] = names

    total_events = sum(len(v) for v in events_by_day.values())
    print(f"extreme_down_reversal events (IPO/veri-hatasi filtreli): {total_events}")

    # Sinyal gunlerini rejime gore etiketle. Rejim, sinyal GUNUNDEKI USDTRY
    # volatilitesine gore belirleniyor (look-ahead yok, sinyal gunu ayni gun
    # zaten piyasa acilisindan itibaren bilinen bir bilgi).
    days_by_regime = {"YUKSEK": [], "DUSUK": []}
    unmatched = 0
    for d in events_by_day:
        r = regime_map.get(d, None)
        if r in ("YUKSEK", "DUSUK"):
            days_by_regime[r].append(d)
        else:
            unmatched += 1
    if unmatched > 0:
        print(f"UYARI: {unmatched} sinyal gunu makro rejim verisinde bulunamadi, atlandi")
    print(f"Rejime ayrilan gun sayisi: YUKSEK={len(days_by_regime['YUKSEK'])}, "
          f"DUSUK={len(days_by_regime['DUSUK'])}")

    def compute_capacity_constrained_daily_returns(horizon, commission_bps, spread_frac, day_list):
        daily_rows = []
        for d in day_list:
            names = events_by_day[d]
            idx = ret_panel.index.get_indexer([d])[0]
            if idx == -1 or idx + horizon >= len(ret_panel.index):
                continue
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]
            mkt_fwd = (1 + market_ret.loc[fwd_dates]).prod() - 1

            candidates = []
            for name in names:
                if name not in ret_panel.columns:
                    continue
                dvol = dollar_vol_panel.loc[d, name]
                if pd.isna(dvol) or dvol <= 0:
                    continue
                hl_baseline = hl_baseline_panel.loc[d, name]
                if pd.isna(hl_baseline):
                    continue
                stock_fwd = (1 + ret_panel.loc[fwd_dates, name]).prod() - 1
                excess = stock_fwd - mkt_fwd
                strategy_ret_bps = -excess * 10000
                spread_bps = hl_baseline * spread_frac * 10000 * 2
                net_bps = strategy_ret_bps - spread_bps - commission_bps
                cap_from_liquidity = dvol * POSITION_PCT_OF_DAILY_VOLUME
                position_size = min(cap_from_liquidity, FIXED_POSITION_SIZE)
                candidates.append((name, dvol, net_bps, position_size))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = candidates[:DAILY_POSITION_CAP]

            total_size = sum(c[3] for c in selected)
            if total_size <= 0:
                continue
            weighted_net_bps = sum(c[2] * c[3] for c in selected) / total_size

            daily_rows.append({"date": d, "portfolio_net_bps": weighted_net_bps})

        return pd.DataFrame(daily_rows)

    report = []
    report.append(f"Symbols: {len(symbols)} | Market benchmark: {market_source}")
    report.append(f"Makro rejim kaynagi: {ACTIVE_MACRO_SOURCE.upper()} ({MACRO_REGIME_PATH})")
    report.append(f"Toplam sinyal: {total_events}")
    report.append(f"Rejime ayrilan gun sayisi: YUKSEK={len(days_by_regime['YUKSEK'])}, "
                   f"DUSUK={len(days_by_regime['DUSUK'])}")
    report.append("")
    report.append("HIPOTEZ: extreme_down_reversal sinyali, USDTRY volatilite rejimine")
    report.append("gore farkli calisiyor mu? (YUKSEK vol = panik/asiri tepki donemleri)")
    report.append("")
    report.append(f"KAPASITE KISITI: gunde en fazla {DAILY_POSITION_CAP} pozisyon (capacity_backtest.py ile ayni)")
    report.append(f"ANA SENARYO: {PRIMARY_COMMISSION} komisyonu + {PRIMARY_SPREAD}")
    report.append("")

    for horizon in FWD_HORIZONS:
        report.append(f"{'='*70}")
        report.append(f"HORIZON {horizon}d")
        report.append(f"{'='*70}")

        for spread_name, spread_frac in SPREAD_SCENARIOS.items():
            for comm_name, comm_bps in COMMISSION_SCENARIOS_BPS.items():
                is_primary = (comm_name == PRIMARY_COMMISSION and spread_name == PRIMARY_SPREAD)
                tag = " *** ANA SENARYO ***" if is_primary else ""
                report.append(f"[{comm_name:20s} + {spread_name:14s}]{tag}")

                for regime_name in ["YUKSEK", "DUSUK"]:
                    day_list = days_by_regime[regime_name]
                    df_daily = compute_capacity_constrained_daily_returns(
                        horizon, comm_bps, spread_frac, day_list
                    )
                    if df_daily.empty:
                        report.append(f"    {regime_name:8s}: veri yok")
                        continue
                    mean_bps = df_daily["portfolio_net_bps"].mean()
                    win_rate = (df_daily["portfolio_net_bps"] > 0).mean() * 100
                    n_days = len(df_daily)
                    std_bps = df_daily["portfolio_net_bps"].std()
                    sharpe_like = (mean_bps / std_bps) if std_bps > 0 else np.nan
                    report.append(f"    {regime_name:8s}: NET_ort={mean_bps:.1f} bps | "
                                   f"kazanma={win_rate:.1f}% | n_gun={n_days} | "
                                   f"std={std_bps:.1f} | ort/std={sharpe_like:.3f}")
        report.append("")

    report.append("ONEMLI NOTLAR:")
    report.append("- Rejim, sinyal gunundeki USDTRY 20-gunluk rolling volatilitesinin")
    report.append("  tum donem medyanina gore YUKSEK/DUSUK ikiye ayrilmasiyla belirlenir")
    report.append("  (bkz. fetch_macro_regime.py). Kaba/basit bir ayrim, ML tabanli degil.")
    report.append("- Eger iki rejim arasinda buyuk fark VARSA: sinyalin ortalamasi iki")
    report.append("  farkli davranisi maskelemis olabilir - rejime kosullu bir strateji")
    report.append("  dusunulebilir (ama bu, yeni bir overfitting riski de tasir - rejim")
    report.append("  sinirinin kendisi de bir serbestlik derecesidir).")
    report.append("- Eger iki rejim arasinda BUYUK FARK YOKSA: bu, sinyalin zayifliginin")
    report.append("  makro volatiliteden bagimsiz oldugunu, yapisal bir sorun oldugunu")
    report.append("  gosterir (rejim ayrimi sonucu degistirmiyor).")

    with open("regime_backtest_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    main()
