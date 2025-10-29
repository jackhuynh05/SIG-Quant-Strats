import alpaca_trade_api as tradeapi
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time, json, os, math
from xgboost import XGBClassifier

# Alpaca API credentials
API_KEY    = 'PKH4XPYGKRN6B4TAE2V26F2CWP'
API_SECRET = '8Uv6mMdZssZEE47d9jQGmp5jJQspBLYezm99ihJUUAJd'
BASE_URL   = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)


# Config
UNIVERSE = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMGN","AMT","AMZN","AVGO",
    "AXP","BA","BAC","BK","BKNG","BLK","BMY","BRK.B","C","CAT",
    "CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX",
    "DHR","DIS","DOW","DUK","EMR","EXC","F","FDX","LRCX","GE",
    "GILD","GM","PGR","APH","GS","HD","HON","IBM","INTC","INTU",
    "ISRG","JNJ","PLTR","KO","LIN","LLY","LMT","LOW","MAR","MCD",
    "MDLZ","MDT","MET","META","MMM","MO","MRK","MS","MSFT","NEE",
    "NFLX","NKE","NVDA","ORCL","PEP","PFE","PG","PM","PYPL","QCOM",
    "RTX","SBUX","SO","SPG","T","TGT","TMO","TMUS","TXN","UNH",
    "UNP","UPS","USB","V","VZ","WBA","WFC","WMT","XOM"
]

BENCHMARK = "SPY"

START_DAYS = 365 * 7          # ~7 years of daily data
FREQ = "W-FRI"                 # weekly rebal, features on Friday closes
MOM_LOOKBACK = 12              # 12 weeks momentum
TOP_N = 10
RISK_ON = True
RISK_MA_WEEKS = 40             # 40-week SMA (≈200 trading days)
SHORT_MA = 10
LONG_MA  = 30
RSI_LEN  = 14

DOLLARS_PER_POSITION = 1000.0  # per selected symbol
EPS = 1e-6

STATE_FILE  = "trading_state_group2.json"
TRADES_FILE = "trade_log_group2.json"

# State & Trade Log
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "last_run": None,
        "summary": {
            "total_rebalances": 0,
            "total_trades": 0,
            "total_profit": 0.0
        },
        "positions": {}  # symbol -> {qty, avg_entry}
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    return {"trades": [], "cumulative_profit": 0.0}

def save_trades(trades):
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=4)


# Data fetch (Alpaca daily)
def fetch_daily(symbols, days_back=START_DAYS):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt
    )
    bars = stock_data_client.get_stock_bars(req)
    df = bars.df.reset_index()
    if df.empty:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={"symbol":"ticker"})
    df = df[["timestamp","ticker","open","high","low","close","volume"]]
    wide = df.pivot(index="timestamp", columns="ticker", values=["open","high","low","close","volume"]).sort_index()
    # Make columns MultiIndex consistent: (ticker, Field)
    wide.columns = pd.MultiIndex.from_tuples([(c[1], c[0].title()) for c in wide.columns], names=["ticker","field"])
    return wide

def last_close(df, symbol):
    try:
        return float(df[(symbol, "Close")].dropna().iloc[-1])
    except Exception:
        return None


# Indicators / Features
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def compute_features_from_daily(df_px: pd.DataFrame) -> pd.DataFrame:
    # Resample to weekly (Friday close)
    tickers = sorted({c[0] for c in df_px.columns})
    frames = []
    for tk in tickers:
        if (tk, "Close") not in df_px.columns: 
            continue
        s = df_px[(tk, "Close")].dropna()
        w = s.resample(FREQ).last().dropna()
        if len(w) < max(LONG_MA, MOM_LOOKBACK) + 5:
            continue
        ret_1p = w.pct_change()
        ma_s = w.rolling(SHORT_MA).mean()
        ma_l = w.rolling(LONG_MA).mean()
        cross = (ma_s > ma_l).astype(int)
        r = rsi(w, RSI_LEN)
        mom = w.pct_change(MOM_LOOKBACK)
        feat = pd.DataFrame({
            "ret_fwd": ret_1p.shift(-1),
            "ret_1p": ret_1p,
            "ma_s": ma_s,
            "ma_l": ma_l,
            "ma_cross": cross,
            "rsi": r,
            "mom": mom
        })
        feat.index.name = "date"
        feat["ticker"] = tk
        frames.append(feat)
    feats = pd.concat(frames).dropna()
    feats = feats.reset_index().set_index(["date","ticker"]).sort_index()
    return feats

def market_breadth_sentiment_from_daily(df_px: pd.DataFrame) -> pd.Series:
    tickers = sorted({c[0] for c in df_px.columns})
    weekly = {}
    for tk in tickers:
        if (tk, "Close") not in df_px.columns: 
            continue
        s = df_px[(tk, "Close")].dropna().resample(FREQ).last()
        if s.empty: 
            continue
        weekly[tk] = s
    wdf = pd.DataFrame(weekly).dropna(how="all")
    ret = wdf.pct_change()
    breadth = (ret > 0).sum(axis=1) / max(ret.shape[1], 1)
    out = breadth.shift(1).rename("sentiment")
    out.index.name = "date"
    return out.dropna()

def prepare_training_table(df_px: pd.DataFrame) -> pd.DataFrame:
    feats = compute_features_from_daily(df_px)
    sent = market_breadth_sentiment_from_daily(df_px)
    f = feats.reset_index().merge(sent.reset_index(), on="date", how="left").dropna()
    f["label_up"] = (f["ret_fwd"] > 0).astype(int)
    return f.set_index(["date","ticker"]).sort_index()

def compute_risk_on(df_px: pd.DataFrame) -> int:
    """Return 1 if risk-on (SPY weekly close > 40W SMA), else 0."""
    s = df_px[(BENCHMARK, "Close")].dropna()
    wf = s.resample(FREQ).last().dropna()
    sma = wf.rolling(RISK_MA_WEEKS).mean()
    if len(wf) == 0 or len(sma) == 0:
        return 1
    return int(wf.iloc[-1] > sma.iloc[-1])


# Model
def build_model():
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )

FEATURES = ["ret_1p","ma_s","ma_l","ma_cross","rsi","mom","sentiment"]

def pick_top_n(df_px: pd.DataFrame, universe: list, top_n: int) -> list:
    # Build training table from daily bars (weekly features)
    feats = prepare_training_table(df_px)

    # Split into train (all but last week) and "current" week for scoring
    f = feats.reset_index()
    weeks = sorted(f["date"].unique())
    if len(weeks) < 80:
        # Not enough weeks to train robustly
        return []

    train_weeks = weeks[:-1]
    score_week  = weeks[-1]

    train_df = f[f["date"].isin(train_weeks) & f["ticker"].isin(universe)]
    cur_df   = f[(f["date"] == score_week) & f["ticker"].isin(universe)]

    # Train
    X_train = train_df[FEATURES]
    y_train = train_df["label_up"]
    if len(y_train.unique()) < 2:
        return []

    model = build_model()
    model.fit(X_train, y_train)

    # Score current week
    X_cur = cur_df[FEATURES]
    prob_up = model.predict_proba(X_cur)[:, 1]
    cur_df = cur_df.assign(prob_up=prob_up)

    # Momentum-boosted ranking
    mom_rank = cur_df["mom"].rank(pct=True, method="average").fillna(0.5)
    score = cur_df["prob_up"] * (0.5 + 0.5 * mom_rank)
    cur_df = cur_df.assign(score=score)

    picks = list(cur_df.nlargest(top_n, "score")["ticker"])
    return picks


# Orders / Portfolio
def get_position(symbol):
    try:
        pos = api.get_position(symbol)
        return float(pos.qty), float(pos.avg_entry_price)
    except Exception:
        return 0.0, 0.0

def set_target_dollars(symbol, target_dollars, last_price):
    """Fractional target using qty delta."""
    if last_price is None or last_price <= 0:
        return
    desired_qty = float(target_dollars) / float(last_price)
    current_qty, _ = get_position(symbol)
    delta = desired_qty - current_qty
    if delta > EPS:
        api.submit_order(symbol=symbol, qty=float(delta), side='buy', type='market', time_in_force='gtc')
        print(f"BUY {symbol} ~{delta:.6f} to target ${target_dollars:.0f}.")
    elif delta < -EPS:
        api.submit_order(symbol=symbol, qty=float(abs(delta)), side='sell', type='market', time_in_force='gtc')
        print(f"SELL {symbol} ~{abs(delta):.6f} to target ${target_dollars:.0f}.")
    else:
        print(f"{symbol}: already near target.")

def liquidate_symbol(symbol):
    url = f"{BASE_URL}/v2/positions/{symbol}?percentage=100"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
    r = requests.delete(url, headers=headers)
    print(f"Liquidate {symbol}: {r.text}")

def sync_portfolio_to_selection(df_px, selection):
    # Sell any universe positions not in selection
    open_positions = {p.symbol: p for p in api.list_positions()}
    for sym in list(open_positions.keys()):
        if sym in UNIVERSE and sym not in selection:
            liquidate_symbol(sym)

    # Buy/adjust targets for selected names
    for sym in selection:
        px = last_close(df_px, sym)
        if px is None:
            print(f"{sym}: no price, skip.")
            continue
        set_target_dollars(sym, DOLLARS_PER_POSITION, px)


# Weekly run
def run_weekly_rebalance():
    print("Fetching daily data (Alpaca)...")
    symbols = list(dict.fromkeys(UNIVERSE + [BENCHMARK]))
    df_px = fetch_daily(symbols)

    if df_px.empty:
        print("No price data downloaded. Skipping.")
        return

    # Risk-on filter from SPY weekly trend
    risk = compute_risk_on(df_px) if RISK_ON else 1
    print(f"Risk-on = {risk}")

    state = load_state()
    trades = load_trades()

    if risk == 0:
        # Risk-off: move to cash (universe only)
        print("Risk-off: liquidating universe to cash.")
        for sym in UNIVERSE:
            liquidate_symbol(sym)
        state["last_run"] = datetime.now().isoformat()
        state["summary"]["total_rebalances"] += 1
        save_state(state); save_trades(trades)
        return

    # Pick top N
    selection = pick_top_n(df_px, UNIVERSE, TOP_N)
    print(f"Selected (Top {TOP_N}): {selection}")

    # Sync portfolio to selection
    sync_portfolio_to_selection(df_px, selection)

    # Light logging (positions snapshot)
    pos_after = {p.symbol: {"qty": float(p.qty), "avg_entry": float(p.avg_entry_price)} for p in api.list_positions()}
    state["positions"] = pos_after
    state["last_run"] = datetime.now().isoformat()
    state["summary"]["total_rebalances"] += 1
    save_state(state); save_trades(trades)

# Scheduler
def wait_until_next_monday_1005_local():
    """
    Sleep until next Monday ~10:05 local time (simple cadence).
    """
    now = datetime.now()
    # find next Monday
    days_ahead = (0 - now.weekday()) % 7  # Monday=0
    target = now + timedelta(days=days_ahead)
    target = target.replace(hour=10, minute=5, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=7)
    delta = (target - now).total_seconds()
    mins = delta / 60.0
    print(f"Sleeping ~{mins:.1f} minutes until next Monday 10:05.")
    time.sleep(max(60, delta))  # at least 60s

def trading_bot_sig_ml_weekly():
    account = api.get_account()
    print(f"Buying Power: {account.buying_power}")

    while True:
        print("\n=== Weekly Rebalance (SIG ML Momentum) ===")
        try:
            run_weekly_rebalance()
        except Exception as e:
            print("Rebalance error:", e)
        wait_until_next_monday_1005_local()

# Entrypoint
if __name__ == "__main__":
    trading_bot_sig_ml_weekly()
