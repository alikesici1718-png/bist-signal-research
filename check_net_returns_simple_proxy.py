# ❌ ARŞİVLENDİ: extreme_down_reversal karsiz cikti (3 bagimsiz maliyet yontemiyle dogrulandi, tum senaryolarda net negatif). Detay: check_net_returns.py, capacity_backtest.py, check_net_returns_cs.py sonuclarina bakiniz.
"""
check_net_returns_simple_proxy.py

Amac: Corwin-Schultz estimatoru formul hatasi (surekli negatif alpha, spread
her zaman 0'a kirpiliyor) verdigi icin GUVENILMEZ cikti. Bunun yerine daha
basit, matematiksel olarak her zaman pozitif ve dogrulanabilir iki proxy
kullaniyoruz:

  1. HL_RATIO = (High - Low) / Close  -- gunluk fiyat araligi, Close'a oranla.
     Bu SAF spread degil (volatilite de icerir), ama likidite ile ters
     iliskili olmasi beklenir: dar/derin piyasada (liquid) High-Low araligi
     kucuk olur, illiquid'de buyuk olur -- CONSISTENT bir SIRALAMA (ranking)
     sinyali olarak guvenilir, mutlak seviye olarak degil.

  2. AMIHUD_ILLIQ = |return| / DollarVolume -- literaturde (Amihud 2002)
     standart illiquidity olcusu. Fiyatin, birim islem hacminde ne kadar
     hareket ettigini gosterir -- yuksekse dusuk likidite/yuksek market impact.

Bu iki proxy'yi KOMBINE ETMIYORUZ (birbirinin farkli seyleri olcmesi
nedeniyle basit bir agirlikli ortalama yaniltici olur). Bunun yerine:
  - HL_RATIO'yu, sembol bazinda MEDYAN gunluk aralik olarak alip, bunu
    "tek yon islem maliyeti" ust siniri gibi kullaniyoruz (spread'in
    HL_RATIO'nun bir kesri oldugu bilinen literatur bulgusuna dayanarak,
    kaba bir katsayiyla: tek yon maliyet ~= HL_RATIO / 4).
  - Bu, "spread HL araliginin belli bir kesridir" varsayimina dayanir --
    KESIN DEGIL, ama negatif/sifir cikma riski yok, yonlendirici bir
    BUYUKLUK MERTEBESI verir.

Bu script, oncekilerle (sabit varsayim: 60/25bps, ve bozuk CS: 0bps)
KARSILASTIRMALI olarak sonucu gosterir.

Kullanim:
    python check_net_returns_simple_proxy.py
Cikti:
    net_returns_simple_proxy_report.txt
    hl_ratio_by_symbol.csv
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_ROWS = 250
EXTREME_DOWN_PCTL = 0.05
FWD_HORIZONS = [1, 3, 5]
EXCLUDE_NAMES = {"USDTRY", "USDTRY=X"}

# HL_RATIO'dan tek-yon maliyete gecis katsayisi -- ACIMASIZ STRES TESTI:
# normalde literatur ~0.15-0.25 arasi kullanir, biz kasitli olarak YUKSEK
# bir katsayi (0.5) kullanip USTUNE sabit bir taban maliyet ekliyoruz --
# amac "bu edge en kotu, en pesimist senaryoda da ayakta kalir mi" sorusu.
SPREAD_FRACTION_OF_RANGE = 0.5
FLAT_EXTRA_COST_BPS = 20.0  # ek sabit taban maliyet (komisyon+slippage guvenlik payi)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_TXT = os.path.join(OUTPUT_DIR, "net_returns_simple_proxy_report.txt")
HL_CSV = os.path.join(OUTPUT_DIR, "hl_ratio_by_symbol.csv")


def load_symbol(path):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None
        if not {"Date", "Open", "High", "Low", "Close"}.issubset(df.columns):
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

    print(f"Loaded {len(symbols)} symbols")

    all_dates = pd.to_datetime(
        sorted(set().union(*[df["Date"].unique() for df in symbols.values()]))
    )

    close_series, open_series, vol_series, hl_series = {}, {}, {}, {}
    for name, df in symbols.items():
        s = df.set_index("Date")
        close_series[name] = s["Close"].reindex(all_dates)
        open_series[name] = s["Open"].reindex(all_dates)
        vol_series[name] = s["Volume"].reindex(all_dates) if "Volume" in df.columns else pd.Series(np.nan, index=all_dates)
        hl_ratio = (s["High"] - s["Low"]) / s["Close"].replace(0, np.nan)
        hl_series[name] = hl_ratio.reindex(all_dates)

    close_panel = pd.concat(close_series, axis=1)
    open_panel = pd.concat(open_series, axis=1)
    vol_panel = pd.concat(vol_series, axis=1)
    hl_panel = pd.concat(hl_series, axis=1)
    for p in (close_panel, open_panel, vol_panel, hl_panel):
        p.index.name = None

    ret_panel = close_panel.pct_change()
    market_ret = ret_panel.mean(axis=1, skipna=True).fillna(0)

    # saglik kontrolu: HL_RATIO her zaman pozitif olmali (yapisal olarak High>=Low)
    assert (hl_panel.dropna() >= 0).all().all(), "HL_RATIO negatif cikti -- veri hatasi var"
    print("Saglik kontrolu OK: HL_RATIO her zaman >= 0")

    median_hl_by_symbol = hl_panel.median(axis=0, skipna=True).dropna().sort_values(ascending=False)
    median_hl_by_symbol.to_frame("median_hl_ratio").to_csv(HL_CSV)
    print(f"\nEn yuksek HL_RATIO (en illiquid gorunen) 5 sembol:\n{median_hl_by_symbol.head(5)}")
    print(f"\nEn dusuk HL_RATIO (en liquid gorunen) 5 sembol:\n{median_hl_by_symbol.tail(5)}")

    dollar_vol = close_panel * vol_panel
    liquidity = dollar_vol.median(axis=0, skipna=True).dropna()
    tertiles = pd.qcut(liquidity, 3, labels=["illiquid", "mid", "liquid"]) if len(liquidity) >= 3 else pd.Series(dtype="object")

    def get_bucket(symbol):
        if symbol in tertiles.index and pd.notna(tertiles.get(symbol)):
            return tertiles[symbol]
        return "mid"

    events = []
    for d in ret_panel.index:
        day_rets = ret_panel.loc[d].dropna()
        if len(day_rets) < 20:
            continue
        cutoff = day_rets.quantile(EXTREME_DOWN_PCTL)
        losers = day_rets[day_rets <= cutoff]
        for name in losers.index:
            events.append((name, d))
    df_events = pd.DataFrame(events, columns=["symbol", "date"])
    print(f"\nTotal extreme_down events: {len(df_events)}")

    report = []
    report.append("*** ACIMASIZ STRES TESTI: yuksek maliyet varsayimi (0.5x range + 20bps sabit) ***")
    report.append(f"Total extreme_down_reversal events: {len(df_events)}")
    report.append("")
    report.append("=== HL_RATIO (High-Low)/Close -- likidite proxy sagligi ===")
    report.append(f"Universe medyan HL_RATIO: {median_hl_by_symbol.median()*10000:.1f} bps/gun")
    for bucket in ["illiquid", "mid", "liquid"]:
        syms = [s for s in median_hl_by_symbol.index if get_bucket(s) == bucket]
        if not syms:
            continue
        med = median_hl_by_symbol.loc[syms].median()
        report.append(f"  {bucket:8s}: medyan HL_RATIO={med*10000:.1f} bps/gun, n_sembol={len(syms)}")
    report.append("(Beklenti: illiquid > mid > liquid siralamasi olmali -- bu, proxy'nin")
    report.append(" likidite ile mantikli iliski kurdugunu dogrular.)")
    report.append("")

    tahmini_tek_yon_maliyet = median_hl_by_symbol * SPREAD_FRACTION_OF_RANGE
    report.append(f"Tahmini tek-yon maliyet (HL_RATIO x {SPREAD_FRACTION_OF_RANGE}):")
    for bucket in ["illiquid", "mid", "liquid"]:
        syms = [s for s in tahmini_tek_yon_maliyet.index if get_bucket(s) == bucket]
        if not syms:
            continue
        med = tahmini_tek_yon_maliyet.loc[syms].median()
        report.append(f"  {bucket:8s}: tek_yon={med*10000:.1f} bps, round_trip={med*20000:.1f} bps")
    report.append("")
    report.append(f"(KARSILASTIRMA -- onceki sabit varsayimlar: illiquid=150, mid=60, liquid=25bps round-trip.")
    report.append(f" Bu proxy'nin ciktisi bunlara ne kadar yakinsa, o kadar guveniyoruz.)")
    report.append("")

    def forward_net_profit(events_df, horizon):
        rows = []
        for name, d in zip(events_df["symbol"], events_df["date"]):
            if name not in ret_panel.columns or name not in open_panel.columns:
                continue
            idx = ret_panel.index.get_indexer([d])[0]
            if idx < 0 or idx + horizon + 1 >= len(ret_panel.index):
                continue
            entry_date = ret_panel.index[idx + 1]
            exit_date = ret_panel.index[idx + horizon + 1]
            entry_price = open_panel.loc[entry_date, name]
            exit_price = open_panel.loc[exit_date, name]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                continue
            stock_fwd = exit_price / entry_price - 1
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]
            mkt_fwd = (1 + market_ret.loc[fwd_dates]).prod() - 1
            gross_excess = stock_fwd - mkt_fwd

            bucket = get_bucket(name)
            if bucket == "illiquid":
                continue

            if name not in tahmini_tek_yon_maliyet.index:
                continue
            round_trip_cost = tahmini_tek_yon_maliyet[name] * 2 + FLAT_EXTRA_COST_BPS / 10000.0

            net_profit = -gross_excess - round_trip_cost
            rows.append({"symbol": name, "date": d, "gross": gross_excess,
                         "net": net_profit, "liquidity": bucket, "cost": round_trip_cost})
        return pd.DataFrame(rows)

    report.append("=== Net kar (HL_RATIO proxy maliyeti ile, mid+liquid) ===")
    for horizon in FWD_HORIZONS:
        df_res = forward_net_profit(df_events, horizon)
        if df_res.empty:
            report.append(f"--- Horizon {horizon}d: veri yok ---")
            continue
        report.append(f"--- Horizon {horizon}d (n={len(df_res)}) ---")
        report.append(f"  Ortalama tahmini maliyet (round-trip): {df_res['cost'].mean()*10000:.1f} bps")
        report.append(f"  BRUT excess (long yon)   : {df_res['gross'].mean()*10000:.1f} bps")
        report.append(f"  NET KAR (kisa, proxy maliyet): {df_res['net'].mean()*10000:.1f} bps")
        report.append(f"  NET KAR pozitif oran      : {(df_res['net']>0).mean()*100:.1f}%")
        for bucket in ["mid", "liquid"]:
            sub = df_res[df_res["liquidity"] == bucket]
            if len(sub) == 0:
                continue
            report.append(f"    {bucket:8s}: net_kar={sub['net'].mean()*10000:.1f} bps, "
                           f"maliyet={sub['cost'].mean()*10000:.1f} bps, n={len(sub)}, "
                           f"kazanma={100*(sub['net']>0).mean():.1f}%")
        report.append("")

    report.append("YORUM: SPREAD_FRACTION_OF_RANGE=0.25 KESIN DEGIL -- kaba bir varsayim.")
    report.append("Bu script'in asil degeri, HL_RATIO siralamasinin (illiquid>mid>liquid)")
    report.append("mantikli olup olmadigini dogrulamak ve maliyetin buyukluk mertebesini")
    report.append("gormek. Kesin bir 'edge var/yok' hukmu icin GERCEK bid-ask verisi ya da")
    report.append("gercek bir aracidan emir denemesi hala gerekli.")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))
    print(f"\nRapor kaydedildi: {REPORT_TXT}")


if __name__ == "__main__":
    main()