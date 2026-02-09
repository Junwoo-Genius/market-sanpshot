import os
import json
import requests
from datetime import datetime, timezone

API_KEY = os.environ["ALPHAVANTAGE_KEY"]

RSI_PERIOD = 14
EMA_PERIOD = 20
SMA_PERIOD = 60

def read_tickers(path="tickers.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]

def fetch_daily_adjusted(symbol: str):
    """
    Alpha Vantage: TIME_SERIES_DAILY_ADJUSTED
    returns daily OHLC + adjusted close + volume
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    ts = data.get("Time Series (Daily)")
    if not ts:
        # Rate limit / invalid symbol / temporary issue
        raise RuntimeError(f"{symbol}: No daily time series. payload keys={list(data.keys())}")

    # sort by date ascending (old -> new)
    dates = sorted(ts.keys())
    closes = [float(ts[d]["4. close"]) for d in dates]
    volumes = [int(float(ts[d]["6. volume"])) for d in dates]
    return dates, closes, volumes

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
    # 위 계산은 전체 구간 EMA라서 초기값 영향이 있음.
    # 실전에서는 충분히 긴 히스토리를 쓰거나, period 이후부터 계산해도 됨.
    # 여기서는 compact(약 100일)로도 안정적으로 동작.
    return e

def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def pct_change(today, prev):
    if prev == 0:
        return 0.0
    return (today - prev) / prev * 100.0

def classify_today(p):
    # 너의 템플릿용 (기준은 필요하면 바꿔서 고정 가능)
    if p >= 10:
        return f"상승(+{p:.0f}%)"
    if p <= -10:
        return f"하락({p:.0f}%)"
    return f"보합(+{p:.0f}% 내외)"

def classify_flow(vols_last5_latest_to_old):
    seq = vols_last5_latest_to_old
    up = all(seq[i] >= seq[i+1] for i in range(len(seq)-1))
    down = all(seq[i] <= seq[i+1] for i in range(len(seq)-1))
    if up:
        return "📈 증가 지속"
    if down:
        return "📉 감소 지속"
    mx, mn = max(seq), min(seq)
    if mn and (mx - mn) / mn < 0.15:
        return "⚖️ 보합 유지"
    return "📉 증가 실패"

def top_label(today_state, flow_state):
    # 거래량만으로 보수적 라벨링
    if today_state.startswith("상승") and flow_state == "📈 증가 지속":
        return "🟢 신규 자금 유입 우위"
    if today_state.startswith("하락") and flow_state == "📉 감소 지속":
        return "🟡 기존 포지션 정리 우위"
    return "⚪ 관망"

def main():
    tickers = read_tickers()

    out = {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "alphavantage",
        "interval": "1d",
        "params": {
            "rsi": RSI_PERIOD,
            "ema": EMA_PERIOD,
            "sma": SMA_PERIOD
        },
        "tickers": {}
    }

    for sym in tickers:
        dates, closes, volumes = fetch_daily_adjusted(sym)

        if len(volumes) < 6:
            raise RuntimeError(f"{sym}: Not enough data points for last5 + prev day")

        # latest values
        last_close = closes[-1]
        last_vol = volumes[-1]
        prev_vol = volumes[-2]
        p = pct_change(last_vol, prev_vol)

        vols_last5 = volumes[-5:]               # old -> new
        vols_last5_latest = list(reversed(vols_last5))  # new -> old

        today_state = classify_today(p)
        flow_state = classify_flow(vols_last5_latest)
        label = top_label(today_state, flow_state)

        out["tickers"][sym] = {
            "last_date": dates[-1],
            "last_close": last_close,
            "last_volume": last_vol,
            "prev_volume": prev_vol,
            "vol_pct": p,
            "vol_state": today_state,
            "flow_state": flow_state,
            "top_label": label,
            "rsi14": rsi(closes, RSI_PERIOD),
            "ema20": ema(closes[-100:], EMA_PERIOD),  # compact 범위 안전
            "sma60": sma(closes, SMA_PERIOD),
            "volumes_last5_latest_to_old": vols_last5_latest
        }

    os.makedirs("public", exist_ok=True)
    with open("public/report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
