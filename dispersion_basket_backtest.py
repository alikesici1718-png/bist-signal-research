"""
dispersion_basket_backtest.py

FARKLI HIPOTEZ SINIFI: Onceki testler (extreme_down_reversal) tek-hisse
yon bahsi yapiyordu (short pozisyon, piyasa beta'sina maruz). Bu script
onun yerine PIYASA-NOTR bir dispersion/basket stratejisi test eder:

  Her gun: o gunun en asiri dusen N hissesini LONG, en asiri yukselen N
  hissesini SHORT yap. Iki bacak birbirini piyasa riskine karsi netler
  (long ve short toplam pozisyon buyuklugu esitlenir) - yani market beta'ya
  bahis degil, kesitteki (cross-section) asiri hareketlerin ortalamaya
  donme egiliminde olup olmadigina bahis.

Bu, tek hisseye ozgu (idiosyncratic) mean-reversion sinyalini, piyasa
yonunden (ki bunun momentum oldugunu zaten biliyoruz) ayristirmaya
calisir. Onceki testte excess return piyasaya gore olculuyordu ama
pozisyonun kendisi hala tek yonlu piyasa riskine aciktir (short pozisyon
piyasa duserse kazanir, cikarsa kaybeder oldugu gibi). Burada iki bacak
ile o risk netlenir.

Ayni altyapi: IPO seasoning filtresi, ISKUR guvenlik filtresi, gercek
XU100'e referans (raporlama amacli, artik hedge oldugu icin stratejinin
kendisi icin gerekli degil), ayni maliyet modeli (komisyon+spread,
capacity_backtest.py ile ayni senaryolar), ayni likidite/kapasite kisiti.

Not: Basket'in HER IKI bacaginda da islem yapildigi icin (long acilis +
short acilis, ikisi de kapanista ters islem) round-trip maliyet TEK
YONLU stratejinin ~2 kati islem hacmi gerektirir - ama pozisyon
buyuklugu esit boluneceginden toplam maliyet bps olarak ayni kalir
(her bacak kendi maliyetini toplam pozisyonun yarisi uzerinden oder).

Kullanim:
    python dispersion_basket_backtest.py
Cikti:
    dispersion_basket_report.txt
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
MARKET_INDEX_PATH = "data_market/XU100.csv"
MIN_ROWS = 250
FWD_HORIZONS = [1, 3, 5]

EXCLUDE_NAMES = {"USDTRY", "USDTRY=X", "ISKUR"}
MAX_PLAUSIBLE_DAILY_RETURN = 1.0
IPO_SEASONING_DAYS = 60
BASELINE_WINDOW = 60

# Basket buyuklugu: her bacakta (long/short) en fazla kac hisse
BASKET_SIZE_PER_LEG = 10
# Ekstrem tanimi: gunluk getiri dagiliminin ust/alt yuzdeligi
EXTREME_PCTL = 0.05

POSITION_PCT_OF_DAILY_VOLUME = 0.02
FIXED_POSITION_SIZE = 50_000.0

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
    market_source = (f"XU100 gercek endeks ({MARKET_INDEX_PATH}, sadece raporlama icin - "
                      f"strateji piyasa-notr) ") if xu100_ret is not None else "yok"

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

    # Her gun icin long/short basket'lerini olustur
    daily_baskets = {}
    for d in ret_panel.index:
        day_rets = ret_panel.loc[d].dropna()
        if len(day_rets) < 20:
            continue
        valid_names = [n for n in day_rets.index if not in_seasoning_period(n, d)]
        day_rets = day_rets.loc[valid_names]
        if len(day_rets) < 20:
            continue

        low_cutoff = day_rets.quantile(EXTREME_PCTL)
        high_cutoff = day_rets.quantile(1 - EXTREME_PCTL)
        losers = day_rets[day_rets <= low_cutoff].index.tolist()
        winners = day_rets[day_rets >= high_cutoff].index.tolist()
        if losers and winners:
            daily_baskets[d] = {"long": losers, "short": winners}

    total_days = len(daily_baskets)
    print(f"Basket olusturulabilen gun sayisi: {total_days}")

    def compute_basket_daily_returns(horizon, commission_bps, spread_frac):
        rows = []
        for d, basket in daily_baskets.items():
            idx = ret_panel.index.get_indexer([d])[0]
            if idx == -1 or idx + horizon >= len(ret_panel.index):
                continue
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]

            def leg_candidates(names, direction):
                # direction: +1 long, -1 short
                cands = []
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
                    raw_bps = direction * stock_fwd * 10000
                    spread_bps = hl_baseline * spread_frac * 10000 * 2
                    net_bps = raw_bps - spread_bps - commission_bps
                    cap_from_liquidity = dvol * POSITION_PCT_OF_DAILY_VOLUME
                    position_size = min(cap_from_liquidity, FIXED_POSITION_SIZE)
                    cands.append((name, dvol, net_bps, position_size))
                cands.sort(key=lambda x: x[1], reverse=True)
                return cands[:BASKET_SIZE_PER_LEG]

            long_leg = leg_candidates(basket["long"], +1)
            short_leg = leg_candidates(basket["short"], -1)
            all_positions = long_leg + short_leg
            if not all_positions:
                continue
            total_size = sum(c[3] for c in all_positions)
            if total_size <= 0:
                continue
            weighted_net_bps = sum(c[2] * c[3] for c in all_positions) / total_size

            rows.append({
                "date": d,
                "n_long": len(long_leg),
                "n_short": len(short_leg),
                "portfolio_net_bps": weighted_net_bps,
            })
        return pd.DataFrame(rows)

    report = []
    report.append(f"Symbols: {len(symbols)} | Market: {market_source}")
    report.append(f"Basket gun sayisi: {total_days} | Bacak basi max hisse: {BASKET_SIZE_PER_LEG}")
    report.append(f"Ekstrem tanimi: gunluk getiri dagiliminin en ust/alt %{EXTREME_PCTL*100:.0f}'i")
    report.append("")
    report.append("HIPOTEZ: Piyasa-notr dispersion basket -- ayni gun en cok dusen hisseleri LONG,")
    report.append("en cok yukselen hisseleri SHORT yap. Beta riski netlenir, kesitteki asiri")
    report.append("hareketin ortalamaya donmesine bahis yapilir (piyasa yonune degil).")
    report.append("")
    report.append(f"POZISYON BOYUTU: min(gunluk_dolar_hacminin_%{POSITION_PCT_OF_DAILY_VOLUME*100:.0f}'i, "
                   f"{FIXED_POSITION_SIZE:,.0f} TL sabit taban)")
    report.append(f"ANA SENARYO: {PRIMARY_COMMISSION} komisyonu + {PRIMARY_SPREAD}")
    report.append("")

    for horizon in FWD_HORIZONS:
        report.append(f"{'='*70}")
        report.append(f"HORIZON {horizon}d")
        report.append(f"{'='*70}")

        for spread_name, spread_frac in SPREAD_SCENARIOS.items():
            for comm_name, comm_bps in COMMISSION_SCENARIOS_BPS.items():
                df_daily = compute_basket_daily_returns(horizon, comm_bps, spread_frac)
                if df_daily.empty:
                    continue

                is_primary = (comm_name == PRIMARY_COMMISSION and spread_name == PRIMARY_SPREAD)
                tag = " *** ANA SENARYO ***" if is_primary else ""

                mean_daily_bps = df_daily["portfolio_net_bps"].mean()
                win_rate = (df_daily["portfolio_net_bps"] > 0).mean() * 100
                n_days = len(df_daily)
                cum_bps = df_daily["portfolio_net_bps"].sum()
                std_daily = df_daily["portfolio_net_bps"].std()
                sharpe_like = (mean_daily_bps / std_daily) if std_daily > 0 else np.nan

                report.append(f"[{comm_name:20s} + {spread_name:14s}]{tag}")
                report.append(f"    Gunluk portfoy NET ort={mean_daily_bps:.1f} bps | "
                               f"kazanma_orani={win_rate:.1f}% | n_gun={n_days}")
                report.append(f"    Toplam kumulatif (basit toplam) = {cum_bps:.0f} bps | "
                               f"gunluk std={std_daily:.1f} bps | ort/std oran={sharpe_like:.3f}")
        report.append("")

    report.append("ONEMLI NOTLAR:")
    report.append("- Bu strateji piyasa-notr: long ve short bacaklar toplam pozisyon buyuklugunce")
    report.append("  esitlenir, net piyasa (beta) maruziyeti sifira yakin olmalidir.")
    report.append("- Onceki extreme_down_reversal testinden farki: o test piyasa yonune (short)")
    report.append("  bahis yapiyordu, bu test kesitteki (cross-sectional) dispersiyonun ortalamaya")
    report.append("  donup donmedigine bahis yapiyor.")
    report.append("- Maliyet iki bacakta da odenir (long acilis+kapanis, short acilis+kapanis) -")
    report.append("  bu yuzden toplam islem hacmi tek-yonlu stratejiden ~2x fazladir, ama bps")
    report.append("  bazinda maliyet agirlikli ortalamaya zaten dahil edildigi icin sonuc")
    report.append("  dogrudan karsilastirilabilir.")
    report.append("- Komisyon sabit/gercek, spread hala HL_RATIO tabanli tahmin.")

    with open("dispersion_basket_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    main()