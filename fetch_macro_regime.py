"""
fetch_macro_regime.py

TCMB EVDS'den USDTRY (dolar/TL) alis kurunu ceker ve gunluk volatilite
rejimini (YUKSEK / DUSUK) hesaplar. Bu, extreme_down_reversal sinyalinin
makro volatilite rejimine gore farkli calisip calismadigini test etmek
icin bir on-adimdir (bkz. regime_backtest.py).

Rejim tanimi: USDTRY gunluk getirisinin son 20 gunluk rolling std'si,
tum donemin medyanina gore YUKSEK/DUSUK olarak ikiye ayrilir. Bu kaba
ama seffaf bir rejim ayrimi - ML tabanli bir rejim tespiti degil,
kasitli olarak basit tutuluyor cunku amac makro volatilitenin KABACA
bir etkisi var mi diye bakmak, hassas bir rejim modeli kurmak degil.

Kullanim:
    python fetch_macro_regime.py
Cikti:
    data_macro/usdtry_regime.csv  (Date, usdtry_close, usdtry_ret, usdtry_vol20, regime)
"""

import pandas as pd
import numpy as np
from evds import evdsAPI

API_KEY = "Mub6hUbWuk"
START_DATE = "01-01-2015"   # veri setinizin kapsadigi araligi genis tutmak icin
END_DATE = "05-07-2026"     # bugune kadar (gelecek tarih vermek hataya sebep olabilir)

# Hangi seriyi cekecegimiz - USDTRY veya TLREF secilebilir.
# TLREF (Turk Lirasi Gecelik Referans Faiz Orani), faiz rejimi/likidite
# kosullarinin bir proxy'si. USDTRY volatilitesinden FARKLI bir sey olcer:
# USDTRY = doviz riski/panik, TLREF = para politikasi/faiz rejimi.
SERIES_CONFIG = {
    "usdtry": {
        "code": "TP.DK.USD.A.YTL",
        "output_prefix": "usdtry",
    },
    "tlref": {
        "code": "TP.BISTTLREF.KAPANIS",
        "output_prefix": "tlref",
    },
}
ACTIVE_SERIES = "tlref"  # "usdtry" veya "tlref" olarak degistirilebilir

OUTPUT_DIR = "data_macro"

VOL_WINDOW = 20


def fetch_evds_series(series_code, start_date, end_date, api_key):
    """
    evds paketini kullanir - bu paket EVDS3 endpoint degisikligine uyumlu
    hale getirilmis (bkz. GitHub fatihmete/evds). Ham requests ile eski
    evds2.tcmb.gov.tr/service/evds/ URL'sini cagirmak artik sadece HTML
    web sayfasi donduruyor (endpoint tasinmis).

    EVDS API'si tek istekte donen satir sayisini sinirliyor (~1000 gunluk
    veri). Bu yuzden tarih araligini yillik parcalara bolup her yili ayri
    cekip birlestiriyoruz.
    """
    client = evdsAPI(api_key)
    start_year = int(start_date.split("-")[-1])
    end_year = int(end_date.split("-")[-1])

    all_chunks = []
    for year in range(start_year, end_year + 1):
        chunk_start = f"01-01-{year}" if year != start_year else start_date
        chunk_end = f"31-12-{year}" if year != end_year else end_date
        print(f"  -> {chunk_start} - {chunk_end} cekiliyor...")
        try:
            chunk_df = client.get_data([series_code], startdate=chunk_start, enddate=chunk_end)
            all_chunks.append(chunk_df)
        except Exception as e:
            print(f"     UYARI: {year} yili icin cekim basarisiz: {e}")

    df = pd.concat(all_chunks, ignore_index=True)
    return df


def main():
    config = SERIES_CONFIG[ACTIVE_SERIES]
    series_code = config["code"]
    prefix = config["output_prefix"]
    output_file = f"{prefix}_regime.csv"

    print(f"EVDS'den {series_code} ({ACTIVE_SERIES}) cekiliyor ({START_DATE} - {END_DATE})...")
    df = fetch_evds_series(series_code, START_DATE, END_DATE, API_KEY)
    print(f"Cekilen satir sayisi: {len(df)}")
    print(df.head())
    print(df.tail())

    # evds paketi Tarih kolonunu ve seri kodunu nokta->altcizgi cevirerek
    # dondurur. Dinamik bul.
    date_col = "Tarih" if "Tarih" in df.columns else df.columns[0]
    value_col = [c for c in df.columns if c != date_col][0]
    print(f"Kullanilan tarih kolonu: {date_col}, deger kolonu: {value_col}")

    # ONEMLI: format acikca belirtiliyor (gun-ay-yil), otomatik algilama
    # bazi satirlari yanlis parse edip NaT'a dusurebiliyordu.
    df["Tarih"] = pd.to_datetime(df[date_col], format="%d-%m-%Y", errors="coerce")
    n_bad_dates = df["Tarih"].isna().sum()
    if n_bad_dates > 0:
        print(f"UYARI: {n_bad_dates} satirin tarihi parse edilemedi, atiliyor")

    close_col = f"{prefix}_close"
    ret_col = f"{prefix}_ret"
    vol_col = f"{prefix}_vol20"

    df = df[["Tarih", value_col]].rename(columns={"Tarih": "Date", value_col: close_col})
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["Date", close_col])
    df = df.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)

    df[ret_col] = df[close_col].pct_change()
    df[vol_col] = df[ret_col].rolling(VOL_WINDOW, min_periods=10).std()

    median_vol = df[vol_col].median()
    df["regime"] = np.where(df[vol_col] >= median_vol, "YUKSEK", "DUSUK")

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_file)
    df.to_csv(out_path, index=False)

    print(f"\nMedyan 20-gunluk volatilite ({prefix}): {median_vol:.5f}")
    print(f"YUKSEK rejim gun sayisi: {(df['regime']=='YUKSEK').sum()}")
    print(f"DUSUK rejim gun sayisi: {(df['regime']=='DUSUK').sum()}")
    print(f"\nCikti kaydedildi: {out_path}")
    print(df[["Date", close_col, ret_col, vol_col, "regime"]].tail(10))


if __name__ == "__main__":
    main()