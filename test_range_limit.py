import pykap
import traceback

ranges = [
    ("2023-01-01", "2024-01-01"),   # 1 yil (calisti)
    ("2022-01-01", "2024-01-01"),   # 2 yil
    ("2021-01-01", "2024-01-01"),   # 3 yil
    ("2020-01-01", "2024-01-01"),   # 4 yil
    ("2019-01-01", "2024-01-01"),   # 5 yil
    ("2020-01-01", "2026-07-06"),   # ~6.5 yil (patladi)
]

comp = pykap.BISTCompany('THYAO')

for fd, td in ranges:
    print(f"=== {fd} -> {td} ===")
    try:
        r = comp.get_historical_disclosure_list(
            fromdate=fd,
            todate=td,
            disclosure_type="FR",
            subject="4028328c594bfdca01594c0af9aa0057"
        )
        print(f"Basarili: {len(r)} sonuc")
    except Exception as e:
        print(f"HATA: {e}")
    print()
