import os
import json
import time
import requests
from datetime import datetime, timezone

API_KEY = os.environ["ALPHAVANTAGE_KEY"]

RSI_PERIOD = 14
EMA_PERIOD = 20
SMA_PERIOD = 60

MAX_RETRIES = 8
INITIAL_BACKOFF = 20
BACKOFF_CAP = 120
OUTPUTSIZE = "compact"


def read_tickers():
    path = os.environ.get("TICKERS_FILE", "tickers.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip().upper() for l in f if l.strip() and not l.strip().startswith("#")]


def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(len(values) - period, len(values)):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += -diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100 - (100 / (1 + rs))


def fetch_daily_adjusted(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": OUTPUTSIZE,
        "apikey": API_KEY,
    }

    backoff = INITIAL_BACKOFF
    last_keys = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[{symbol}] attempt {attempt}/{MAX_RETRIES}", flush=True)

        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        last_keys = list(data.keys())
        print(f"[{symbol}] keys={list(data.keys())}", flush=True)

        ts = data.get("Time Series (Daily)")
        if ts:
            dates = sorted(ts.keys())
            close = [float(ts[d]["4. close"]) for d in dates]
            volume = [int(float(ts[d]["6. volume"])) for d in dates]
            return dates, close, volume

        if "Information" in data or "Note" in data:
            time.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_CAP)
            continue

        if "Error Message" in data:
            raise RuntimeError(f"{symbol} invalid")

        time.sleep(backoff)
        backoff = min(backoff * 2, BACKOFF_CAP)

    raise RuntimeError(f"{symbol}: No daily data. keys={last_keys}")


def main():
    tickers = read_tickers()

    report = {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": {}
    }

    for sym in tickers:
        dates, close, volume = fetch_daily_adjusted(sym)

        report["tickers"][sym] = {
            "last_date": dates[-1],
            "close": close[-1],
            "volume": volume[-1],
            "rsi14": rsi(close, RSI_PERIOD),
            "ema20": ema(close, EMA_PERIOD),
            "sma60": sma(close, SMA_PERIOD),
        }

        time.sleep(3)

    os.makedirs("public", exist_ok=True)
    with open("public/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
