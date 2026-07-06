import requests
import json

url = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"

payload = {
    "fromDate": "2023-01-01",
    "toDate": "2024-01-01",
    "disclosureClass": "FR",
    "subjectList": ["4028328c594bfdca01594c0af9aa0057"],
    "mkkMemberOidList": ["4028e4a1416e696301416f37201c5f2e"],
    "inactiveMkkMemberOidList": [],
    "bdkMemberOidList": [],
    "fromSrc": False,
    "disclosureIndexList": []
}

headers_variants = [
    {},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
     "Content-Type": "application/json",
     "Origin": "https://www.kap.org.tr",
     "Referer": "https://www.kap.org.tr/tr/"}
]

for i, headers in enumerate(headers_variants):
    print(f"=== Deneme {i+1}: headers={list(headers.keys())} ===")
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        print(f"Status: {r.status_code}")
        print(f"Response (ilk 500 karakter): {r.text[:500]}")
    except Exception as e:
        print(f"Hata: {e}")
    print()
