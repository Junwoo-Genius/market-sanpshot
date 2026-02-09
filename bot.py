import os
import json
import csv
import requests
from io import StringIO
from datetime import datetime, timezone

# === 고정 파라미터 (일/주/월 동일) ===
RSI_PERIOD = 14
EMA_PERIOD = 20
SMA_PERIOD = 60

# Stooq (미국 주식: aapl.us 형태)
STOOQ_BASE = "https://stooq.com/q/d/l/"

TICKERS_FILE = os.environ.get("TICKERS_FILE", "tickers.txt")
OUT_PATH = os.environ.get("OUT_PATH", "public/report.json")


def read_tickers(path=TICKERS_FILE):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.upper())
    return out


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
    """
    Wilder's RSI (TradingView/대부분 지표에 더 근접)
    """
    if len(values) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_indicators(close_series):
    return {
        "rsi14": rsi(close_series, RSI_PERIOD),
        "ema20": ema(close_series, EMA_PERIOD),
        "sma60": sma(close_series, SMA_PERIOD),
    }


def to_stooq_symbol(us_symbol: str) -> str:
    return f"{us_symbol.lower()}.us"


def fetch_stooq_daily(us_symbol: str, keep_last: int = 3000):
    """
    Stooq CSV:
      https://stooq.com/q/d/l/?s=aapl.us&i=d
    전체를 읽은 뒤 Date 정렬 후 '최신 keep_last개'만 사용해서
    last_date가 과거로 고정되는 문제를 방지한다.
    """
    s = to_stooq_symbol(us_symbol)
    params = {"s": s, "i": "d"}
    r = requests.get(STOOQ_BASE, params=params, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    if not text or "No data" in text:
        raise RuntimeError(f"{us_symbol}: no data from stooq")

    reader = csv.DictReader(StringIO(text))
    rows = []
    for row in reader:
        if not row.get("Date"):
            continue
        rows.append(row)

    if len(rows) < 10:
        raise RuntimeError(f"{us_symbol}: insufficient rows ({len(rows)})")

    rows.sort(key=lambda x: x["Date"])
    rows = rows[-keep_last:]  # ✅ 최신 데이터 유지

    dates, o, h, l, c, v = [], [], [], [], [], []
    for row in rows:
        dates.append(row["Date"])
        o.append(float(row["Open"]))
        h.append(float(row["High"]))
        l.append(float(row["Low"]))
        c.append(float(row["Close"]))
        v.append(int(float(row["Volume"])))

    return dates, o, h, l, c, v


def _iso_week_key(date_str: str):
    dt = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]))
    iso = dt.isocalendar()
    return int(iso.year), int(iso.week)


def resample_weekly(dates, o, h, l, c, v):
    buckets = {}
    for i, d in enumerate(dates):
        key = _iso_week_key(d)
        if key not in buckets:
            buckets[key] = {"fi": i, "li": i, "high": h[i], "low": l[i], "vol": v[i]}
        else:
            b = buckets[key]
            b["li"] = i
            b["high"] = max(b["high"], h[i])
            b["low"] = min(b["low"], l[i])
            b["vol"] += v[i]

    out_dates, out_o, out_h, out_l, out_c, out_v = [], [], [], [], [], []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        fi, li = b["fi"], b["li"]
        out_dates.append(dates[li])
        out_o.append(o[fi])
        out_h.append(b["high"])
        out_l.append(b["low"])
        out_c.append(c[li])
        out_v.append(b["vol"])

    return {"dates": out_dates, "open": out_o, "high": out_h, "low": out_l, "close": out_c, "volume": out_v}


def resample_monthly(dates, o, h, l, c, v):
    buckets = {}
    for i, d in enumerate(dates):
        key = (int(d[0:4]), int(d[5:7]))
        if key not in buckets:
            buckets[key] = {"fi": i, "li": i, "high": h[i], "low": l[i], "vol": v[i]}
        else:
            b = buckets[key]
            b["li"] = i
            b["high"] = max(b["high"], h[i])
            b["low"] = min(b["low"], l[i])
            b["vol"] += v[i]

    out_dates, out_o, out_h, out_l, out_c, out_v = [], [], [], [], [], []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        fi, li = b["fi"], b["li"]
        out_dates.append(dates[li])
        out_o.append(o[fi])
        out_h.append(b["high"])
        out_l.append(b["low"])
        out_c.append(c[li])
        out_v.append(b["vol"])

    return {"dates": out_dates, "open": out_o, "high": out_h, "low": out_l, "close": out_c, "volume": out_v}


def main():
    tickers = read_tickers()

    report = {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "stooq",
        "params": {"rsi": RSI_PERIOD, "ema": EMA_PERIOD, "sma": SMA_PERIOD},
        "tickers": {},
    }

    for sym in tickers:
        try:
            dates, o, h, l, c, v = fetch_stooq_daily(sym, keep_last=3000)
        except Exception as e:
            report["tickers"][sym] = {"error": str(e)}
            continue

        daily_ind = compute_indicators(c)

        w = resample_weekly(dates, o, h, l, c, v)
        m = resample_monthly(dates, o, h, l, c, v)

        weekly_ind = compute_indicators(w["close"])
        monthly_ind = compute_indicators(m["close"])

        report["tickers"][sym] = {
            "daily": {
                "last_date": dates[-1],
                "last_close": c[-1],
                "last_volume": v[-1],
                **daily_ind,
            },
            "weekly": {
                "last_date": w["dates"][-1] if w["dates"] else None,
                "last_close": w["close"][-1] if w["close"] else None,
                **weekly_ind,
            },
            "monthly": {
                "last_date": m["dates"][-1] if m["dates"] else None,
                "last_close": m["close"][-1] if m["close"] else None,
                **monthly_ind,
            },
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
```0
