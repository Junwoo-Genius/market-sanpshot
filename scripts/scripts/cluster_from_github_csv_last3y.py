import os
import json
import time
import hashlib
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ====== 학습 대상 32종목 ======
TICKERS = [
    "NVDA", "TSLA", "PLTR", "IREN", "BE", "ASTS", "CRCL", "HOOD", "OKLO", "NBIS", "ABAT", "AMD", "RGTI",
    "AAPL", "LLY", "SMR", "FLNC", "IONQ", "RIVN", "QBTS", "MU", "TSM", "INTC", "NVO", "RKLB", "ADBE",
    "NFLX", "GOOGL", "MSFT", "META", "UNH", "AVGO"
]

OUT_DIR = "public/cluster"
os.makedirs(OUT_DIR, exist_ok=True)

# ✅ 단일 소스 고정: GitHub RAW (네 저장소)
RAW_BASE = "https://raw.githubusercontent.com/Junwoo-Genius/market-snapshot/main/public/csv"


def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch_daily_from_github_raw(ticker: str):
    """
    단일 소스: https://raw.githubusercontent.com/.../public/csv/{TICKER}_daily.csv
    Columns expected: Date,Open,High,Low,Close,Volume
    """
    url = f"{RAW_BASE}/{ticker}_daily.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; market-snapshot-bot/1.0)",
        "Accept": "text/csv,*/*",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    r = requests.get(url, headers=headers, timeout=30)
    status = r.status_code
    r.raise_for_status()

    b = r.content
    sha = hashlib.sha256(b).hexdigest()

    df = pd.read_csv(io.BytesIO(b))
    need = {"Date", "Close"}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"{ticker}: missing columns {need - set(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    meta = {
        "ticker": ticker,
        "source_url": url,
        "download_time_utc": now_utc_iso(),
        "http_status": status,
        "bytes": len(b),
        "sha256": sha,
        "row_count": int(len(df)),
        "date_range": [
            df["Date"].min().strftime("%Y-%m-%d"),
            df["Date"].max().strftime("%Y-%m-%d"),
        ] if len(df) else None,
    }
    return df, meta


def last_3y(df: pd.DataFrame):
    if df.empty:
        return df
    end = df["Date"].max()
    start = end - relativedelta(years=3)
    return df[df["Date"] >= start].copy().reset_index(drop=True)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def slope_pct(series: pd.Series, n: int) -> float:
    if len(series) < n + 1:
        return np.nan
    a = series.iloc[-n - 1]
    b = series.iloc[-1]
    if pd.isna(a) or pd.isna(b) or a == 0:
        return np.nan
    return (b / a - 1.0) * 100.0


def disparity_pct(x: float, ma: float) -> float:
    if pd.isna(x) or pd.isna(ma) or ma == 0:
        return np.nan
    return (x / ma - 1.0) * 100.0


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.set_index("Date").resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()
    return w


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    m = df.set_index("Date").resample("M").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()
    return m


def build_features(df_d: pd.DataFrame):
    d = df_d.copy()
    d["EMA20"] = ema(d["Close"], 20)
    d["EMA60"] = ema(d["Close"], 60)
    d["RSI14"] = rsi_wilder(d["Close"], 14)

    end_idx = d["Close"].idxmax()
    end = d.loc[end_idx]
    end_date = end["Date"]
    end_close = float(end["Close"])

    w = to_weekly(d)
    w["EMA20"] = ema(w["Close"], 20)
    w["EMA60"] = ema(w["Close"], 60)
    w["RSI14"] = rsi_wilder(w["Close"], 14)

    m = to_monthly(d)
    m["EMA20"] = ema(m["Close"], 20)
    m["RSI14"] = rsi_wilder(m["Close"], 14)

    w_end = w[w["Date"] >= end_date].head(1)
    if w_end.empty:
        w_end = w.tail(1)
    w_end_row = w_end.iloc[0]

    m_end = m[m["Date"] >= end_date].head(1)
    if m_end.empty:
        m_end = m.tail(1)
    m_end_row = m_end.iloc[0]

    feat = {}
    # Slope
    feat["D20_slope"] = slope_pct(d.loc[:end_idx, "EMA20"].tail(260), 5)
    feat["D60_slope"] = slope_pct(d.loc[:end_idx, "EMA60"].tail(400), 5)
    feat["D_diff"] = feat["D20_slope"] - feat["D60_slope"]

    feat["W20_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA20"].tail(120), 4)
    feat["W60_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA60"].tail(160), 4)

    feat["M3_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(60), 3)
    feat["M6_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(80), 6)

    # Disparity 7
    feat["d_DISP20"] = disparity_pct(end_close, float(end["EMA20"]))
    feat["d_DISP60"] = disparity_pct(end_close, float(end["EMA60"]))
    feat["D_SPREAD"] = disparity_pct(float(end["EMA20"]), float(end["EMA60"]))

    feat["w_DISP20"] = disparity_pct(float(w_end_row["Close"]), float(w_end_row["EMA20"]))
    feat["w_DISP60"] = disparity_pct(float(w_end_row["Close"]), float(w_end_row["EMA60"]))
    feat["w_SPREAD"] = disparity_pct(float(w_end_row["EMA20"]), float(w_end_row["EMA60"]))

    feat["M_DISP20"] = disparity_pct(float(m_end_row["Close"]), float(m_end_row["EMA20"]))

    # RSI
    feat["D_RSI_END"] = float(end["RSI14"])
    feat["W_RSI_END"] = float(w_end_row["RSI14"])
    feat["M_RSI_END"] = float(m_end_row["RSI14"])

    feat["_END_DATE"] = end_date.strftime("%Y-%m-%d")
    feat["_END_CLOSE"] = float(end_close)
    return feat


def percentile_dict(arr: np.ndarray):
    return {
        "P10": float(np.nanpercentile(arr, 10)),
        "P25": float(np.nanpercentile(arr, 25)),
        "Med": float(np.nanpercentile(arr, 50)),
        "P75": float(np.nanpercentile(arr, 75)),
        "P90": float(np.nanpercentile(arr, 90)),
    }


def main():
    run_meta = {
        "run_time_utc": now_utc_iso(),
        "source": "github_raw_daily_csv",
        "source_pattern": f"{RAW_BASE}/{{TICKER}}_daily.csv",
        "universe": TICKERS,
        "window": "last_3y",
        "k": 5,
        "features": [
            "D20_slope", "D60_slope", "D_diff",
            "W20_slope", "W60_slope",
            "M3_slope", "M6_slope",
            "d_DISP20", "d_DISP60", "D_SPREAD",
            "w_DISP20", "w_DISP60", "w_SPREAD",
            "M_DISP20",
            "D_RSI_END", "W_RSI_END", "M_RSI_END"
        ]
    }

    metas, rows, failed = [], [], []

    for t in TICKERS:
        try:
            df, meta = fetch_daily_from_github_raw(t)
            df3 = last_3y(df)
            meta["last3y_row_count"] = int(len(df3))
            meta["last3y_date_range"] = [
                df3["Date"].min().strftime("%Y-%m-%d"),
                df3["Date"].max().strftime("%Y-%m-%d"),
            ] if len(df3) else None

            f = build_features(df3)
            f["Ticker"] = t
            rows.append(f)
            metas.append(meta)

        except Exception as e:
            failed.append({"ticker": t, "error": str(e)})

        time.sleep(0.1)

    # 실패 증거 저장
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "source_fingerprints.json"), "w", encoding="utf-8") as f:
        json.dump({"tickers": metas, "failed": failed}, f, ensure_ascii=False, indent=2)

    if len(rows) == 0:
        raise SystemExit("All tickers failed to download from GitHub RAW daily.csv. Check source_fingerprints.json")

    feat_df = pd.DataFrame(rows).set_index("Ticker")

    feature_cols = [c for c in run_meta["features"] if not c.startswith("_")]
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    used = list(X.index)
    dropped = [t for t in feat_df.index if t not in used]

    run_meta["used_count"] = len(used)
    run_meta["dropped_count"] = len(dropped)
    run_meta["failed_fetch_count"] = len(failed)
    run_meta["dropped_tickers"] = dropped
    run_meta["failed_tickers"] = [x["ticker"] for x in failed]

    # overwrite meta with counts
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=50)
    labels = kmeans.fit_predict(Xs)

    out = X.copy()
    out["Cluster"] = labels

    # (1) numeric bands
    bands = {}
    for cl in sorted(out["Cluster"].unique()):
        sub = out[out["Cluster"] == cl]
        b = {}
        for col in feature_cols:
            b[col] = percentile_dict(sub[col].to_numpy())
        bands[str(int(cl))] = {"count": int(len(sub)), "bands": b}

    # separation rank
    sep = {}
    for col in feature_cols:
        between = out.groupby("Cluster")[col].mean().var()
        within = out.groupby("Cluster")[col].var().mean()
        sep[col] = float(between / (within + 1e-9))
    sep_rank = sorted(sep.items(), key=lambda x: x[1], reverse=True)

    essential = [k for k, _ in sep_rank[:3]]
    optional = [k for k, _ in sep_rank[3:6]]
    rule2 = {
        "separation_rank_top10": sep_rank[:10],
        "essential_top3": essential,
        "optional_next3": optional
    }

    # score-based
    centers = kmeans.cluster_centers_

    def score_one(xrow: np.ndarray):
        xs = scaler.transform([xrow])[0]
        dists = np.linalg.norm(centers - xs, axis=1)
        mx, mn = float(dists.max()), float(dists.min())
        scores = (mx - dists) / (mx - mn + 1e-9) * 100.0
        order = np.argsort(scores)[::-1]
        c1, c2 = int(order[0]), int(order[1])
        return c1, float(scores[c1]), c2, float(scores[c2]), float(scores[c1] - scores[c2])

    scores_rows = []
    for tkr in used:
        c1, s1, c2, s2, delta = score_one(X.loc[tkr].to_numpy())
        scores_rows.append({
            "Ticker": tkr,
            "Top1_Cluster": c1,
            "Top1_Score": s1,
            "Top2_Cluster": c2,
            "Top2_Score": s2,
            "Delta": delta
        })
    scores_df = pd.DataFrame(scores_rows).sort_values(["Top1_Score", "Delta"], ascending=[False, False])

    assign = out[["Cluster"]].copy()
    assign = assign.join(feat_df[["_END_DATE", "_END_CLOSE"]], how="left")

    assign.to_csv(os.path.join(OUT_DIR, "cluster_assign.csv"), encoding="utf-8")
    scores_df.to_csv(os.path.join(OUT_DIR, "cluster_scores.csv"), index=False, encoding="utf-8")

    with open(os.path.join(OUT_DIR, "cluster_bands.json"), "w", encoding="utf-8") as f:
        json.dump(bands, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "cluster_rule2_essential_optional.json"), "w", encoding="utf-8") as f:
        json.dump(rule2, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
