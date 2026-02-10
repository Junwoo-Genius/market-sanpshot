import requests
import csv
import os
from datetime import datetime

BASE_URL = "https://stooq.pl/q/d/l/"
DATA_DIR = "public/csv"

os.makedirs(DATA_DIR, exist_ok=True)

def fetch_daily_csv(symbol: str):
    """
    stooq에서 일봉 CSV 전체(최대 제공 범위)를 가져와서
    종목별 CSV 파일로 저장
    """
    sym = symbol.lower()
    url = f"{BASE_URL}?s={sym}&i=d"

    r = requests.get(url, timeout=20)
    if r.status_code != 200 or len(r.text) < 50:
        print(f"[FAIL] {symbol}: no data")
        return

    lines = r.text.strip().splitlines()
    reader = csv.reader(lines)
    header = next(reader)

    rows = []
    for row in reader:
        # 날짜, 시가, 고가, 저가, 종가, 거래량
        rows.append(row)

    if not rows:
        print(f"[EMPTY] {symbol}")
        return

    out_path = f"{DATA_DIR}/{symbol.upper()}_daily.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[OK] {symbol} -> {out_path} ({len(rows)} rows)")


def main():
    with open("tickers.txt") as f:
        symbols = [line.strip() for line in f if line.strip()]

    for sym in symbols:
        fetch_daily_csv(sym)


if __name__ == "__main__":
    main()
