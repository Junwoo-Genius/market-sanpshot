import os
import json
import time
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# UNIVERSE (FIXED)
# =========================
TICKERS = [
    "NVDA", "TSLA", "PLTR", "IREN", "BE", "ASTS", "CRCL", "HOOD", "OKLO", "NBIS", "ABAT", "AMD", "RGTI",
    "AAPL", "LLY", "SMR", "FLNC", "IONQ", "RIVN", "QBTS", "MU", "TSM", "INTC", "NVO", "RKLB", "ADBE",
    "NFLX", "GOOGL", "MSFT", "META", "UNH", "AVGO"
]

# =========================
# PATH RESOLUTION (HARD FIX)
# =========================
def find_repo_root():
    """
    Actions에서도, 어떤 working-directory에서도 'repo root'를 확정.
    - 1순위: GITHUB_WORKSPACE (Actions에서 가장 신뢰)
    - 2순위: 현재 경로에서 상위로 올라가며 'public/csv'가 있는 지점 탐색
    """
    gw = os.environ.get("GITHUB_WORKSPACE")
    if gw and os.path.isdir(gw):
        return os.path.abspath(gw)

    cur = os.path.abspath(os.getcwd())
    for _ in range(12):
        if os.path.isdir(os.path.join(cur, "public", "csv")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.getcwd())

ROOT = find_repo_root()
OUT_DIR = os.path.join(ROOT, "public", "cluster")
CSV_DIR = os.path.join(ROOT, "public", "csv")

def csv_path(ticker: str) -> str:
    # ✅ 파일명 형식 고정: public/csv/BE_daily.csv
    return os.path.join(CSV_DIR, f"{ticker}_daily.csv")

# =========================
# UTILS
# =========================
def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def hard_print_env_and_fs():
    """
    ✅ 워크플로우 수정 없이도, 로그만 보면 원인 확정되도록 stdout에 강제 출력.
    """
    print("=== CLUSTER DEBUG START ===")
    print("UTC:", now_utc_iso())
    print("CWD:", os.path.abspath(os.getcwd()))
    print("REPO_ROOT:", ROOT)
    print("CSV_DIR:", CSV_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("GITHUB_WORKSPACE:", os.environ.get("GITHUB_WORKSPACE"))
    print("GITHUB_REF:", os.environ.get("GITHUB_REF"))
    print("GITHUB_SHA:", os.environ.get("GITHUB_SHA"))

    print("\n--- DIR EXISTS CHECK ---")
    print("exists(public):", os.path.isdir(os.path.join(ROOT, "public")))
    print("exists(public/csv):", os.path.isdir(CSV_DIR))
    print("exists(public/cluster):", os.path.isdir(os.path.join(ROOT, "public", "cluster")))

    print("\n--- CSV LIST (top 30) ---")
    if os.path.isdir(CSV_DIR):
        files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith("_daily.csv")])
        print("csv_count:", len(files))
        print("sample:", files[:30])
    else:
        print("csv_count: 0 (CSV_DIR missing)")

    print("\n--- SANITY OPEN TEST (AAPL, BE) ---")
    for t in ["AAPL", "BE"]:
        p = csv_path(t)
        print(f"try_open {t} -> {p} exists={os.path.exists(p)}")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, nrows=5)
                print(f"{t} head columns:", list(df.columns))
            except Exception as e:
                print(f"{t} read_csv ERROR:", repr(e))
    print("=== CLUSTER DEBUG END ===\n")

# =========================
# DATA LOADER (LOCAL ONLY)
# =========================
def read_local_daily(ticker: str):
    path = csv_path(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{ticker}: missing local csv -> {path}")

    df = pd.read_csv(path)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{ticker}: local csv missing columns {missing} -> {path}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    if len(df) < 60:
        raise RuntimeError(f"{ticker}: too few rows after parse ({len(df)}) -> {path}")

    meta = {
        "ticker": ticker,
        "source_path": path,
        "sha256": sha256_file(path),
        "row_count": int(len(df)),
        "date_range": [
            df["Date"].min().strftime("%Y-%m-%d"),
            df["Date"].max().strftime("%Y-%m-%d"),
        ],
    }
    return df, meta

# =========================
# INDICATORS / FEATURES
# =========================
def last_3y(df: pd.DataFrame):
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
    feat["D20_slope"] = slope_pct(d.loc[:end_idx, "EMA20"].tail(260), 5)
    feat["D60_slope"] = slope_pct(d.loc[:end_idx, "EMA60"].tail(400), 5)
    feat["D_diff"] = feat["D20_slope"] - feat["D60_slope"]

    feat["W20_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA20"].tail(120), 4)
    feat["W60_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA60"].tail(160), 4)

    feat["M3_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(60), 3)
    feat["M6_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(80), 6)

    feat["d_DISP20"] = disparity_pct(end_close, float(end["EMA20"]))
    feat["d_DISP60"] = disparity_pct(end_close, float(end["EMA60"]))
    feat["D_SPREAD"] = disparity_pct(float(end["EMA20"]), float(end["EMA60"]))

    feat["w_DISP20"] = disparity_pct(float(w_end_row["Close"]), float(w_end_row["EMA20"]))
    feat["w_DISP60"] = disparity_pct(float(w_end_row["Close"]), float(w_end_row["EMA60"]))
    feat["w_SPREAD"] = disparity_pct(float(w_end_row["EMA20"]), float(w_end_row["EMA60"]))

    feat["M_DISP20"] = disparity_pct(float(m_end_row["Close"]), float(m_end_row["EMA20"]))

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

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ✅ 1) 여기서 이미 “CSV가 보이는지”가 로그로 확정됨
    hard_print_env_and_fs()

    metas, rows, failed = [], [], []

    for t in TICKERS:
        try:
            df, meta = read_local_daily(t)
            df3 = last_3y(df)
            meta["last3y_rows"] = int(len(df3))
            f = build_features(df3)
            f["Ticker"] = t
            rows.append(f)
            metas.append(meta)
        except Exception as e:
            failed.append({"ticker": t, "path": csv_path(t), "error": str(e)})

    # evidence files (always)
    with open(os.path.join(OUT_DIR, "source_fingerprints.json"), "w", encoding="utf-8") as f:
        json.dump({"tickers": metas, "failed": failed}, f, ensure_ascii=False, indent=2)

    if len(rows) == 0:
        raise SystemExit(
            "All tickers failed. (This is now fully explainable from the DEBUG output above.)"
        )

    feat_df = pd.DataFrame(rows).set_index("Ticker")
    feature_cols = [c for c in feat_df.columns if not c.startswith("_")]
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(X) < 5:
        raise SystemExit(f"Not enough valid rows for K=5 clustering. valid={len(X)}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=50)
    labels = kmeans.fit_predict(Xs)

    out = X.copy()
    out["Cluster"] = labels

    bands = {}
    for cl in sorted(out["Cluster"].unique()):
        sub = out[out["Cluster"] == cl]
        b = {col: percentile_dict(sub[col].to_numpy()) for col in feature_cols}
        bands[str(int(cl))] = {"count": int(len(sub)), "bands": b}

    assign = out[["Cluster"]].copy()
    assign = assign.join(feat_df[["_END_DATE", "_END_CLOSE"]], how="left")

    assign.to_csv(os.path.join(OUT_DIR, "cluster_assign.csv"), encoding="utf-8")
    with open(os.path.join(OUT_DIR, "cluster_bands.json"), "w", encoding="utf-8") as f:
        json.dump(bands, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()# UTILS
# =========================
def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_probably_html(text_head: str) -> bool:
    t = (text_head or "").lstrip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html") or "<html" in t[:2000]

def validate_csv_header(text: str) -> bool:
    """
    반드시 Date,Open,High,Low,Close,Volume 헤더가 있어야 CSV로 인정
    """
    if not text:
        return False
    first = text.splitlines()[0].strip().replace(" ", "")
    return first.lower() == "date,open,high,low,close,volume"

def read_csv_bytes_to_df(b: bytes, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(b))
    required = {"Date", "Close"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"{ticker}: missing columns {required - set(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    if len(df) < 30:
        raise RuntimeError(f"{ticker}: too few rows after parse ({len(df)})")
    return df

# =========================
# DATA LOADING (FINAL)
# =========================
def load_daily_df(ticker: str):
    """
    ✅ Final rule:
    1) local public/csv/{TICKER}_daily.csv exists -> use it
    2) else download from RAW_BASE/{TICKER}_daily.csv with retry/backoff/validation and save locally
    """
    os.makedirs(CSV_DIR, exist_ok=True)

    local_path = os.path.join(CSV_DIR, f"{ticker}_daily.csv")
    if os.path.exists(local_path):
        # Local load
        df = pd.read_csv(local_path)
        required = {"Date", "Close"}
        if not required.issubset(set(df.columns)):
            raise RuntimeError(f"{ticker}: local csv missing columns {required - set(df.columns)}: {local_path}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

        meta = {
            "ticker": ticker,
            "mode": "local",
            "source_path": local_path,
            "read_time_utc": now_utc_iso(),
            "sha256": sha256_file(local_path),
            "row_count": int(len(df)),
            "date_range": [
                df["Date"].min().strftime("%Y-%m-%d"),
                df["Date"].max().strftime("%Y-%m-%d"),
            ],
        }
        return df, meta

    # Remote download
    url = f"{RAW_BASE}/{ticker}_daily.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; market-snapshot-bot/2.0)",
        "Accept": "text/csv,*/*",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            status = r.status_code
            text_head = (r.text[:400] if r.text else "")

            if status != 200:
                raise RuntimeError(f"HTTP {status} (head={text_head[:120]!r})")

            if is_probably_html(text_head):
                raise RuntimeError(f"HTML response detected (head={text_head[:120]!r})")

            text = r.text
            if not validate_csv_header(text):
                raise RuntimeError(f"Invalid CSV header (head={text_head[:120]!r})")

            b = r.content
            df = read_csv_bytes_to_df(b, ticker)

            # Save to local cache (so next runs are network-free)
            with open(local_path, "wb") as f:
                f.write(b)

            meta = {
                "ticker": ticker,
                "mode": "remote_then_saved",
                "source_url": url,
                "download_time_utc": now_utc_iso(),
                "http_status": status,
                "sha256": sha256_bytes(b),
                "saved_path": local_path,
                "row_count": int(len(df)),
                "date_range": [
                    df["Date"].min().strftime("%Y-%m-%d"),
                    df["Date"].max().strftime("%Y-%m-%d"),
                ],
            }
            return df, meta

        except Exception as e:
            last_err = str(e)
            if attempt < RETRIES:
                time.sleep(BACKOFF_BASE_SEC * (2 ** (attempt - 1)))
            else:
                break

    raise RuntimeError(f"{ticker}: remote download failed after {RETRIES} retries: {last_err}")

# =========================
# INDICATORS / FEATURES
# =========================
def last_3y(df: pd.DataFrame):
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
    """
    End = last3y 구간의 max Close(종가 최고점)
    Slopes: D N=5, W N=4, M N=3/6
    """
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

    # align end row
    w_end = w[w["Date"] >= end_date].head(1)
    if w_end.empty:
        w_end = w.tail(1)
    w_end_row = w_end.iloc[0]

    m_end = m[m["Date"] >= end_date].head(1)
    if m_end.empty:
        m_end = m.tail(1)
    m_end_row = m_end.iloc[0]

    feat = {}

    feat["D20_slope"] = slope_pct(d.loc[:end_idx, "EMA20"].tail(260), 5)
    feat["D60_slope"] = slope_pct(d.loc[:end_idx, "EMA60"].tail(400), 5)
    feat["D_diff"] = feat["D20_slope"] - feat["D60_slope"]

    feat["W20_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA20"].tail(120), 4)
    feat["W60_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA60"].tail(160), 4)

    feat["M3_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(60), 3)
    feat["M6_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(80), 6)

    feat["d_DISP20"] = disparity_pct(end_close, float(end["EMA20"]))
    feat["d_DISP60"] = disparity_pct(end_close, float(end["EMA60"]))
    feat["D_SPREAD"] = disparity_pct(float(end["EMA20"]), float(end["EMA60"]))

    feat["w_DISP20"] = disparity_pct(float(w_end_row["Close"]), float(w_end_row["EMA20"]))
    feat["w_DISP60"] = disparity_pct(float(w_end_row["Close"]), float(w_end_row["EMA60"]))
    feat["w_SPREAD"] = disparity_pct(float(w_end_row["EMA20"]), float(w_end_row["EMA60"]))

    feat["M_DISP20"] = disparity_pct(float(m_end_row["Close"]), float(m_end_row["EMA20"]))

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

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    run_meta = {
        "run_time_utc": now_utc_iso(),
        "source_policy": "local_first_then_raw_download_save",
        "raw_base": RAW_BASE,
        "csv_dir": CSV_DIR,
        "universe": TICKERS,
        "window": "last_3y",
        "k": 5,
        "network": {"timeout": HTTP_TIMEOUT, "retries": RETRIES, "backoff_base_sec": BACKOFF_BASE_SEC},
        "features": [
            "D20_slope", "D60_slope", "D_diff",
            "W20_slope", "W60_slope",
            "M3_slope", "M6_slope",
            "d_DISP20", "d_DISP60", "D_SPREAD",
            "w_DISP20", "w_DISP60", "w_SPREAD",
            "M_DISP20",
            "D_RSI_END", "W_RSI_END", "M_RSI_END"
        ],
    }

    metas, rows, failed = [], [], []

    # evidence: what local files exist right now
    local_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith("_daily.csv")]) if os.path.isdir(CSV_DIR) else []
    run_meta["local_csv_files_count"] = len(local_files)
    run_meta["local_csv_files_sample"] = local_files[:25]

    for t in TICKERS:
        try:
            df, meta = load_daily_df(t)
            df3 = last_3y(df)

            meta["last3y_row_count"] = int(len(df3))
            meta["last3y_date_range"] = [
                df3["Date"].min().strftime("%Y-%m-%d"),
                df3["Date"].max().strftime("%Y-%m-%d"),
            ]

            feat = build_features(df3)
            feat["Ticker"] = t

            rows.append(feat)
            metas.append(meta)

        except Exception as e:
            failed.append({"ticker": t, "error": str(e)})

        time.sleep(SLEEP_BETWEEN_TICKERS)

    # always write evidence
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "source_fingerprints.json"), "w", encoding="utf-8") as f:
        json.dump({"tickers": metas, "failed": failed}, f, ensure_ascii=False, indent=2)

    if len(rows) == 0:
        # hard stop with evidence already written
        raise SystemExit(
            "All tickers failed. Check public/cluster/source_fingerprints.json for per-ticker error evidence."
        )

    feat_df = pd.DataFrame(rows).set_index("Ticker")
    feature_cols = [c for c in run_meta["features"] if not c.startswith("_")]
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    used = list(X.index)
    dropped = [t for t in feat_df.index if t not in used]

    run_meta["used_count"] = len(used)
    run_meta["dropped_count"] = len(dropped)
    run_meta["failed_count"] = len(failed)
    run_meta["failed_tickers"] = [x["ticker"] for x in failed]
    run_meta["dropped_tickers"] = dropped

    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=50)
    labels = kmeans.fit_predict(Xs)

    out = X.copy()
    out["Cluster"] = labels

    # numeric bands per cluster
    bands = {}
    for cl in sorted(out["Cluster"].unique()):
        sub = out[out["Cluster"] == cl]
        b = {col: percentile_dict(sub[col].to_numpy()) for col in feature_cols}
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

    # save outputs
    assign = out[["Cluster"]].copy()
    assign = assign.join(feat_df[["_END_DATE", "_END_CLOSE"]], how="left")

    assign.to_csv(os.path.join(OUT_DIR, "cluster_assign.csv"), encoding="utf-8")
    with open(os.path.join(OUT_DIR, "cluster_bands.json"), "w", encoding="utf-8") as f:
        json.dump(bands, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "cluster_rule2_essential_optional.json"), "w", encoding="utf-8") as f:
        json.dump(rule2, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
