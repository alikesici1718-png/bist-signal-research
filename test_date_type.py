"""Debug script: test whether pykap's disclosure API accepts string vs
datetime.date arguments. One-off tooling; no analysis output.
"""
import pykap
from datetime import date, datetime
import traceback

comp = pykap.BISTCompany('THYAO')

print("=== TEST A: string tarih ===")
try:
    r = comp.get_historical_disclosure_list(
        fromdate="2023-01-01",
        todate="2024-01-01",
        disclosure_type="FR",
        subject="4028328c594bfdca01594c0af9aa0057"
    )
    print(f"Basarili: {len(r)} sonuc")
except Exception as e:
    traceback.print_exc()

print()
print("=== TEST B: date objesi ===")
try:
    fd = datetime.strptime("2023-01-01", "%Y-%m-%d").date()
    td = datetime.strptime("2024-01-01", "%Y-%m-%d").date()
    print(f"fd tipi: {type(fd)}, degeri: {fd}")
    r = comp.get_historical_disclosure_list(
        fromdate=fd,
        todate=td,
        disclosure_type="FR",
        subject="4028328c594bfdca01594c0af9aa0057"
    )
    print(f"Basarili: {len(r)} sonuc")
except Exception as e:
    traceback.print_exc()
