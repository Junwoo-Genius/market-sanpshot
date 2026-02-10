import json
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) JSON 자동 로드
# -----------------------------
def load_daily_json(symbol: str, base_url: str) -> pd.DataFrame:
    """
    base_url 예시:
      - "https://junwoo-genius.github.io/market-snapshot/prices"
      -> 실제 요청 URL: {base_url}/{symbol}_daily.json
    """
    url = f"{base_url}/{symbol}_daily.json"

    with urllib.request.urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    # payload["data"] 안에 date/open/high/low/close/volume이 있다고 가정
    df = pd.DataFrame(payload["data"]).copy()

    # 표준 컬럼명으로 통일
    rename_map = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"JSON data is missing columns: {missing}")

    return df


# -----------------------------
# 2) 최근 N개월 필터 + 지표
# -----------------------------
def prepare_recent(df: pd.DataFrame, months: int = 3) -> pd.DataFrame:
    end_date = df["Date"].max()
    start_date = end_date - pd.DateOffset(months=months)
    out = df[df["Date"] >= start_date].copy()

    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["SMA60"] = out["Close"].rolling(60).mean()

    return out


# -----------------------------
# 3) 매물대(Volume Profile) 기반 지지·저항 레벨 추출
# -----------------------------
def volume_profile_levels(df: pd.DataFrame, bins: int = 20, top_n: int = 3) -> list[float]:
    # Close 가격대에 거래량을 가중치로 히스토그램
    hist, edges = np.histogram(df["Close"], bins=bins, weights=df["Volume"])
    centers = (edges[:-1] + edges[1:]) / 2

    # 거래량 큰 구간 top_n 선택
    idx = np.argsort(hist)[-top_n:]
    levels = sorted(float(centers[i]) for i in idx)
    return levels


# -----------------------------
# 4) 차트 생성 (거래량 아래 패널, 지지/저항 수평선 표기)
# -----------------------------
def plot_price_volume_sr(
    df: pd.DataFrame,
    symbol: str,
    levels: list[float],
    out_path: str = "chart.png",
):
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # 가격 패널
    ax_price.plot(df["Date"], df["Close"], label="Close")
    ax_price.plot(df["Date"], df["EMA20"], label="EMA20")
    ax_price.plot(df["Date"], df["SMA60"], label="SMA60")

    # 지지/저항(매물대 상위 구간)
    x_text = df["Date"].iloc[-1]
    for lvl in levels:
        ax_price.axhline(lvl, linestyle="--", linewidth=1.4)
        ax_price.text(x_text, lvl, f" {lvl:.2f}", va="center")

    ax_price.set_title(f"{symbol} (Recent) | Close + EMA20 + SMA60 | S/R from Volume Profile")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")

    # 거래량 패널 (겹치지 않게 아래)
    ax_vol.bar(df["Date"], df["Volume"])
    ax_vol.set_ylabel("Volume")
    ax_vol.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# 5) 실행 예시
# -----------------------------
if __name__ == "__main__":
    SYMBOL = "IREN"
    BASE_URL = "https://junwoo-genius.github.io/market-snapshot/prices"  # <- 너가 배포하는 경로로 변경
    MONTHS = 3
    BINS = 20
    TOP_N = 3
    OUT = f"{SYMBOL}_3M_SR_separate_volume.png"

    df_all = load_daily_json(SYMBOL, BASE_URL)
    df_recent = prepare_recent(df_all, months=MONTHS)
    sr_levels = volume_profile_levels(df_recent, bins=BINS, top_n=TOP_N)

    plot_price_volume_sr(df_recent, SYMBOL, sr_levels, out_path=OUT)
    print("Saved:", OUT)
