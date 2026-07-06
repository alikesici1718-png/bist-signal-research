# ❌ ARŞİVLENDİ: extreme_down_reversal karsiz cikti (3 bagimsiz maliyet yontemiyle dogrulandi, tum senaryolarda net negatif). Detay: check_net_returns.py, capacity_backtest.py, check_net_returns_cs.py sonuclarina bakiniz.
"""
capacity_backtest.py

extreme_down_reversal (SHORT) sinyalinin, GERCEKCI kapasite kisitlari
altinda ne kadar getiri uretecegini test eder. Onceki testler ("check_net_returns.py")
her sinyale sinirsiz sermaye ile girilebiliyormus gibi hesapliyordu - bu
gercekci degil. Bu script iki ek kisit ekliyor:

  1) GUNLUK POZISYON TAVANI (N=15): Bir gunde N'den fazla sinyal tetiklenirse
     hepsine giremezsin. En likit N sinyal secilir (gercek hayatta illiquid
     isimlere zaten buyuk pozisyon giremezsin, oncelik dogal olarak likide
     kayar).

  2) LIKIDITEYE GORE POZISYON BOYUTU: Her pozisyon, gunluk dolar hacminin
     %2'sini VEYA sabit taban tutari (50,000 TL) - hangisi kucukse - asamaz.
     Bu, market impact'i (fiyatin kendine karsi hareket etmesi) orutuk
     olarak modele katar; hacminin buyuk bir kismini tek islemde almaya
     calisan bir pozisyon gercekte o fiyattan dolmaz.

Maliyet senaryosu (ana/varsayilan, "gercekci-kotumser" secim):
  AtaYatirim komisyonu (38bps round-trip, gercek tarife) +
  YUKSEK_spread (HL_baseline'in %20'si, round-trip icin x2)
Diger 3 komisyon x 2 spread kombinasyonu da yan bilgi olarak raporlanir.

Portfoy getirisi: her gun, o gun secilen pozisyonlarin (agirlikli) ortalama
NET getirisi olarak hesaplanir - yani "gunluk portfoy getirisi" serisi
uretilir, tek tek islem ortalamasi degil. Bu, kapasite kisitinin gercek
etkisini (bazi gunler cok sinyal var ama hepsine giremiyorsun) yansitir.

Kullanim:
    python capacity_backtest.py
Cikti:
    capacity_backtest_report.txt
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
MARKET_INDEX_PATH = "data_market/XU100.csv"
MIN_ROWS = 250
EXTREME_DOWN_PCTL = 0.05
FWD_HORIZONS = [1, 3, 5]

EXCLUDE_NAMES = {"USDTRY", "USDTRY=X", "ISKUR"}
MAX_PLAUSIBLE_DAILY_RETURN = 1.0
IPO_SEASONING_DAYS = 60
BASELINE_WINDOW = 60

# --- Kapasite kisiti parametreleri ---
DAILY_POSITION_CAP = 15          # bir gunde en fazla N pozisyon
POSITION_PCT_OF_DAILY_VOLUME = 0.02   # pozisyon buyuklugu, gunluk dolar hacminin en fazla %2'si
FIXED_POSITION_SIZE = 50_000.0   # TL, sabit taban tutar (goreceli birim)

# --- Maliyet senaryolari (check_net_returns.py ile ayni mantik) ---
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
# Ana/varsayilan senaryo (raporun basinda vurgulanacak, "gercekci-kotumser")
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
    # Gunluk dolar hacmi (bugunku degil, dunku - look-ahead onlemek icin shift)
    dollar_vol_panel = (price_panel * vol_panel).shift(1)

    ret_panel = price_panel.pct_change()

    max_abs_ret = ret_panel.abs().max(axis=0)
    suspect_symbols = max_abs_ret[max_abs_ret > MAX_PLAUSIBLE_DAILY_RETURN].index.tolist()
    if suspect_symbols:
        print(f"UYARI: {len(suspect_symbols)} sembol veri hatasi supheli, cikariliyor: {suspect_symbols}")
        price_panel = price_panel.drop(columns=suspect_symbols)
        open_panel = open_panel.drop(columns=suspect_symbols)
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

    dollar_vol_median = (price_panel * vol_panel).median(axis=0, skipna=True).dropna()
    tertiles = pd.qcut(dollar_vol_median, 3, labels=["illiquid", "mid", "liquid"])

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

    # Gun bazinda event listesi (kapasite kisiti icin gun bazinda gruplamamiz gerekiyor)
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
    print(f"Toplam gun sayisi (en az 1 sinyal olan): {len(events_by_day)}")

    def compute_capacity_constrained_daily_returns(horizon, commission_bps, spread_frac):
        """
        Her gun icin: o gunun sinyallerini likiditeye gore sirala, en likit
        DAILY_POSITION_CAP tanesini sec, her birine likidite/sabit tavanina
        gore agirlik ver, agirlikli ortalama NET getiriyi (bps) portfoy
        getirisi olarak don.
        """
        daily_rows = []
        for d, names in events_by_day.items():
            idx = ret_panel.index.get_indexer([d])[0]
            if idx == -1 or idx + horizon + 1 >= len(ret_panel.index):
                continue
            entry_date = ret_panel.index[idx + 1]
            exit_date = ret_panel.index[idx + horizon + 1]
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
                stock_fwd = open_panel.loc[exit_date, name] / open_panel.loc[entry_date, name] - 1
                excess = stock_fwd - mkt_fwd
                strategy_ret_bps = -excess * 10000  # short yonlu brut, bps
                spread_bps = hl_baseline * spread_frac * 10000 * 2
                net_bps = strategy_ret_bps - spread_bps - commission_bps
                # pozisyon buyuklugu: dolar_hacmi*%2 ile sabit tavan tutarindan kucuk olan
                cap_from_liquidity = dvol * POSITION_PCT_OF_DAILY_VOLUME
                position_size = min(cap_from_liquidity, FIXED_POSITION_SIZE)
                candidates.append((name, dvol, net_bps, position_size))

            if not candidates:
                continue

            # kapasite tavani: en likit DAILY_POSITION_CAP tanesini sec
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = candidates[:DAILY_POSITION_CAP]

            total_size = sum(c[3] for c in selected)
            if total_size <= 0:
                continue
            weighted_net_bps = sum(c[2] * c[3] for c in selected) / total_size

            daily_rows.append({
                "date": d,
                "n_signals": len(names),
                "n_selected": len(selected),
                "capped": len(names) > DAILY_POSITION_CAP,
                "portfolio_net_bps": weighted_net_bps,
            })

        return pd.DataFrame(daily_rows)

    report = []
    report.append(f"Symbols: {len(symbols)} | Market benchmark: {market_source}")
    report.append(f"Toplam sinyal: {total_events} | Sinyalli gun sayisi: {len(events_by_day)}")
    report.append("")
    report.append(f"KAPASITE KISITI: gunde en fazla {DAILY_POSITION_CAP} pozisyon "
                   f"(en likit olanlar oncelikli secilir)")
    report.append(f"POZISYON BOYUTU: min(gunluk_dolar_hacminin_%{POSITION_PCT_OF_DAILY_VOLUME*100:.0f}'i, "
                   f"{FIXED_POSITION_SIZE:,.0f} TL sabit taban)")
    report.append(f"ANA SENARYO (gercekci-kotumser): {PRIMARY_COMMISSION} komisyonu + {PRIMARY_SPREAD}")
    report.append("")

    for horizon in FWD_HORIZONS:
        report.append(f"{'='*70}")
        report.append(f"HORIZON {horizon}d")
        report.append(f"{'='*70}")

        for spread_name, spread_frac in SPREAD_SCENARIOS.items():
            for comm_name, comm_bps in COMMISSION_SCENARIOS_BPS.items():
                df_daily = compute_capacity_constrained_daily_returns(horizon, comm_bps, spread_frac)
                if df_daily.empty:
                    continue

                is_primary = (comm_name == PRIMARY_COMMISSION and spread_name == PRIMARY_SPREAD)
                tag = " *** ANA SENARYO ***" if is_primary else ""

                mean_daily_bps = df_daily["portfolio_net_bps"].mean()
                win_rate = (df_daily["portfolio_net_bps"] > 0).mean() * 100
                n_days = len(df_daily)
                n_capped = df_daily["capped"].sum()
                # kumulatif getiri (basit toplama, gunluk portfoy getirileri birbirinden bagimsiz degil
                # ama buyukluk mertebesi icin toplam bps yeterli)
                cum_bps = df_daily["portfolio_net_bps"].sum()
                std_daily = df_daily["portfolio_net_bps"].std()
                sharpe_like = (mean_daily_bps / std_daily) if std_daily > 0 else np.nan

                report.append(f"[{comm_name:20s} + {spread_name:14s}]{tag}")
                report.append(f"    Gunluk portfoy NET ort={mean_daily_bps:.1f} bps | "
                               f"kazanma_orani={win_rate:.1f}% | n_gun={n_days} | "
                               f"kapasite_asilan_gun={n_capped} ({100*n_capped/n_days:.0f}%)")
                report.append(f"    Toplam kumulatif (basit toplam) = {cum_bps:.0f} bps | "
                               f"gunluk std={std_daily:.1f} bps | ort/std oran={sharpe_like:.3f}")
        report.append("")

    report.append("ONEMLI NOTLAR:")
    report.append("- Bu backtest, gunluk portfoy getirisini kapasite kisitli olarak hesaplar.")
    report.append("  Onceki (check_net_returns.py) test, her sinyale sinirsiz sermaye ile")
    report.append("  girilebiliyormus gibi hesapliyordu - bu daha gerceklikten uzakti.")
    report.append("- Pozisyon boyutu kisiti (dolar hacminin %2'si) market impact'i ortuk olarak")
    report.append("  modele katar: hacmin cok kucuk bir kismini alan pozisyonlar o fiyattan")
    report.append("  gerceklesir varsayimi, hacmin buyuk kismini almaya calisanlar gerceklesmez.")
    report.append("- 'kapasite_asilan_gun' orani yuksekse (signal sayisi cap'i asiyorsa), bu")
    report.append("  stratejinin olcek/capacity limiti oldugunu gosterir - portfoyu buyutmek")
    report.append("  getiriyi orantili artirmaz.")
    report.append("- Komisyon sabit/gercek, spread hala HL_RATIO tabanli tahmin - bkz.")
    report.append("  check_net_returns.py notlari.")

    with open("capacity_backtest_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    main()