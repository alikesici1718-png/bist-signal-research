import os
import pandas as pd
from evds import evdsAPI
from dotenv import load_dotenv

load_dotenv(".env")
api = evdsAPI(os.environ["EVDS_API_KEY"])

data = api.get_data(["TP.APIFON4"], startdate="10-07-2025", enddate="10-07-2026")

df = data[["Tarih", "TP_APIFON4"]].rename(columns={"Tarih": "date", "TP_APIFON4": "policy_rate"})
df = df.dropna(subset=["policy_rate"])
df["policy_rate"] = df["policy_rate"].astype(str).str.replace(",", ".").astype(float)

out = "data/carry_trade/tcmb_policy_rate.csv"
df.to_csv(out, index=False)

print(f"Satır sayısı: {len(df)}")
print(f"Tarih aralığı: {df['date'].min()} — {df['date'].max()}")
print(f"Kaydedildi: {out}")
