# group 1, QQQ Arbitrage
import alpaca_trade_api as tradeapi
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time, json, os, math, requests


# Alpaca API credentials
API_KEY    = ''
API_SECRET = ''
BASE_URL   = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)


# Universe / Weights
BASKET = ["NVDA","AAPL","MSFT","AMZN","META","AVGO","GOOG","TSLA","NFLX","PLTR","COST","AMD","ASML","CSCO"]
QQQ    = "QQQ"
UNIVERSE = sorted(list(set(BASKET + [QQQ])))

# If you have true QQQ component weights for these 14, put them here (raw, will be normalized).
# Defaults to equal weights.
WEIGHTS = {sym: 1.0 for sym in BASKET}  # <-- replace with actual weights if desired


# Parameters
START_DAYS = 365 * 5
Z_WIN      = 60                 # rolling window for spread Z
RSI_LEN    = 14
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9
VOL_LOOK   = 20                 # realized vol lookback (on spread)
VOL_THRESH = 0.02               # 2% daily realized vol threshold (strict by your spec)
Z_LONG     = -1.0
Z_SHORT    = +1.0
Z_EXIT_ABS = 0.05               # close when z approximates 0
Z_STOP_AWAY= 1.5                # adverse move from entry z to stop

MAX_GROSS_DOLLARS = 10000.0
LEG_DOLLARS       = MAX_GROSS_DOLLARS / 2.0  # $5k long + $5k short when active
EPS = 1e-6

STATE_FILE  = "trading_state_group1.json"
TRADES_FILE = "trade_log_group1.json"


# State / Trades
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "last_run": None,
        "regime": "neutral",  # 'neutral', 'long_basket', 'short_basket'
        "open_trade": None,   # {"side": "long_basket"/"short_basket", "z_entry": float, "opened_at": ts}
        "summary": {
            "total_rebalances": 0,
            "total_trades": 0,
            "profitable_trades": 0,
            "unprofitable_trades": 0,
            "total_profit": 0.0
        }
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


# Data
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"symbol": "ticker"})
    df = df[["timestamp","ticker","open","high","low","close","volume"]]
    wide = df.pivot(index="timestamp", columns="ticker", values=["open","high","low","close","volume"]).sort_index()
    wide.columns = pd.MultiIndex.from_tuples([(c[1], c[0].title()) for c in wide.columns], names=["ticker","field"])
    return wide

def last_close(df, symbol):
    try:
        return float(df[(symbol,"Close")].dropna().iloc[-1])
    except Exception:
        return None


# Indicators on SPREAD
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def macd(series: pd.Series, fast=MACD_FAST, slow=MACD_SLOW, sig=MACD_SIG):
    macd_line = ema(series, fast) - ema(series, slow)
    signal = ema(macd_line, sig)
    hist = macd_line - signal
    return macd_line, signal, hist

def rsi(series: pd.Series, length=RSI_LEN):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def realized_vol(series: pd.Series, lookback=VOL_LOOK):
    r = series.pct_change().dropna()
    if len(r) < lookback:
        return np.inf
    return float(r.iloc[-lookback:].std())


# Spread construction
def normalize_weights(weights_dict):
    total = sum(max(0.0, float(v)) for v in weights_dict.values())
    if total <= 0:
        return {k: 1.0/len(weights_dict) for k in weights_dict}
    return {k: float(v)/total for k,v in weights_dict.items()}

def basket_price_series(df_px: pd.DataFrame, weights: dict) -> pd.Series:
    # equal/weighted average of closes
    w = normalize_weights(weights)
    cols = [(tk,"Close") for tk in w.keys() if (tk,"Close") in df_px.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = df_px[cols].copy()
    # weighted sum each day
    ws = pd.Series(0.0, index=sub.index)
    for tk, wt in w.items():
        if (tk,"Close") in sub.columns:
            ws = ws.add(sub[(tk,"Close")] * wt, fill_value=0.0)
    return ws

def rolling_rebased_index(series: pd.Series, window: int) -> pd.Series:
    """
    Rebase each rolling window to 1 at the window start:
    idx_t = price_t / price_{t-window+1}
    """
    s = series.dropna().copy()
    base = s.rolling(window).apply(lambda x: x[0], raw=True)
    return s / base

def compute_spread_signals(df_px: pd.DataFrame):
    # Build basket and QQQ closes
    basket = basket_price_series(df_px, WEIGHTS)
    qqq    = df_px[(QQQ,"Close")].dropna() if (QQQ,"Close") in df_px.columns else pd.Series(dtype=float)
    common = basket.index.intersection(qqq.index)
    basket, qqq = basket.loc[common], qqq.loc[common]

    if len(common) < max(Z_WIN, VOL_LOOK, RSI_LEN, MACD_SLOW+MACD_SIG) + 5:
        raise RuntimeError("Not enough data to compute signals.")

    # Rolling rebased indices
    b_idx = rolling_rebased_index(basket, Z_WIN)
    q_idx = rolling_rebased_index(qqq,    Z_WIN)

    spread = (b_idx / q_idx) - 1.0
    spread = spread.dropna()

    # Z-score over rolling window
    mean = spread.rolling(Z_WIN).mean()
    std  = spread.rolling(Z_WIN).std(ddof=1)
    z    = (spread - mean) / std

    # RSI on spread
    rsi_spread = rsi(spread, RSI_LEN)

    # MACD on spread
    macd_line, macd_sig, _ = macd(spread)
    macd_bull = (macd_line.iloc[-1] > macd_sig.iloc[-1]) and (macd_line.iloc[-2] <= macd_sig.iloc[-2])
    macd_bear = (macd_line.iloc[-1] < macd_sig.iloc[-1]) and (macd_line.iloc[-2] >= macd_sig.iloc[-2])

    # 20D realized volatility (daily) of spread
    vol_spread = realized_vol(spread, VOL_LOOK)

    latest = spread.index[-1]
    out = {
        "latest_date": str(latest.date()),
        "z": float(z.iloc[-1]),
        "rsi": float(rsi_spread.iloc[-1]),
        "vol20": float(vol_spread),
        "macd_bull": bool(macd_bull),
        "macd_bear": bool(macd_bear),
        # debug values
        "spread": float(spread.iloc[-1]),
        "mean": float(mean.iloc[-1]),
        "std": float(std.iloc[-1]) if std.iloc[-1] == std.iloc[-1] else None
    }
    return out


# Orders / Positions
def get_position(symbol):
    try:
        p = api.get_position(symbol)
        return float(p.qty), float(p.avg_entry_price)
    except Exception:
        return 0.0, 0.0

def set_target_dollars_long(symbol, target_dollars, last_price):
    if last_price is None or last_price <= 0:
        print(f"{symbol}: no price, skip.")
        return
    desired_qty = float(target_dollars) / float(last_price)
    cur_qty, _ = get_position(symbol)
    delta = desired_qty - cur_qty
    if delta > EPS:
        api.submit_order(symbol=symbol, qty=float(delta), side="buy", type="market", time_in_force="day")
        print(f"BUY {symbol} ~{delta:.6f} to ${target_dollars:.0f}.")
    elif delta < -EPS:
        api.submit_order(symbol=symbol, qty=float(abs(delta)), side="sell", type="market", time_in_force="day")
        print(f"SELL {symbol} ~{abs(delta):.6f} (reduce) toward ${target_dollars:.0f}.")
    else:
        print(f"{symbol}: near target long.")

def set_target_dollars_short(symbol, target_dollars, last_price):
    if last_price is None or last_price <= 0:
        print(f"{symbol}: no price, skip.")
        return
    desired_qty = -float(target_dollars) / float(last_price)  # negative qty target
    cur_qty, _ = get_position(symbol)
    delta = desired_qty - cur_qty
    if delta < -EPS:
        # need to sell more (increase short)
        api.submit_order(symbol=symbol, qty=float(abs(delta)), side="sell", type="market", time_in_force="day")
        print(f"SHORT {symbol} ~{abs(delta):.6f} to ${target_dollars:.0f}.")
    elif delta > EPS:
        # need to buy to reduce short
        api.submit_order(symbol=symbol, qty=float(abs(delta)), side="buy", type="market", time_in_force="day")
        print(f"COVER {symbol} ~{abs(delta):.6f} (reduce) toward ${target_dollars:.0f}.")
    else:
        print(f"{symbol}: near target short.")

def liquidate_symbol(symbol):
    url = f"{BASE_URL}/v2/positions/{symbol}?percentage=100"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
    r = requests.delete(url, headers=headers)
    print(f"Liquidate {symbol}: {r.text}")

def estimate_current_gross(df_px):
    gross = 0.0
    for sym in UNIVERSE:
        try:
            qty, _ = get_position(sym)
            px = last_close(df_px, sym)
            if px is not None:
                gross += abs(qty) * px
        except Exception:
            pass
    return gross

def enforce_gross_cap(plan_long_dollars, plan_short_dollars, df_px):
    current = estimate_current_gross(df_px)
    planned = plan_long_dollars + plan_short_dollars
    limit = MAX_GROSS_DOLLARS
    if current + planned <= limit + 1e-6:
        return plan_long_dollars, plan_short_dollars
    scale = max(0.0, (limit - current) / max(planned, 1e-9))
    return plan_long_dollars * scale, plan_short_dollars * scale


# Execution helpers
def open_long_basket_short_qqq(df_px, state):
    long_dol, short_dol = enforce_gross_cap(LEG_DOLLARS, LEG_DOLLARS, df_px)
    # allocate long dollars across basket by WEIGHTS
    w = normalize_weights(WEIGHTS)
    for sym, wt in w.items():
        px = last_close(df_px, sym)
        set_target_dollars_long(sym, long_dol * wt, px)
    # short QQQ
    qpx = last_close(df_px, QQQ)
    set_target_dollars_short(QQQ, short_dol, qpx)
    state["regime"] = "long_basket"

def open_short_basket_long_qqq(df_px, state):
    long_dol, short_dol = enforce_gross_cap(LEG_DOLLARS, LEG_DOLLARS, df_px)
    # short basket, long QQQ
    w = normalize_weights(WEIGHTS)
    for sym, wt in w.items():
        px = last_close(df_px, sym)
        set_target_dollars_short(sym, short_dol * wt, px)
    qpx = last_close(df_px, QQQ)
    set_target_dollars_long(QQQ, long_dol, qpx)
    state["regime"] = "short_basket"

def liquidate_all():
    for sym in UNIVERSE:
        liquidate_symbol(sym)

def compute_pair_pnl_snapshot(df_px, side, entry_snapshot=None):
    """
    Optional: could track per-leg entries; for simplicity, we snapshot
    aggregate using live positions vs last close for a rough realized pnl on close.
    """
    pnl = 0.0
    # basket legs
    for sym in BASKET + [QQQ]:
        qty, avg = get_position(sym)
        px = last_close(df_px, sym)
        if px is None or abs(qty) <= EPS: 
            continue
        if qty > 0:
            pnl += (px - avg) * qty
        else:
            pnl += (avg - px) * abs(qty)
    return float(pnl)

def should_open_long(sig):
    return (
        (sig["z"] <= Z_LONG) and
        (sig["rsi"] < 30.0) and
        sig["macd_bull"] and
        (sig["vol20"] < VOL_THRESH)
    )

def should_open_short(sig):
    return (
        (sig["z"] >= Z_SHORT) and
        (sig["rsi"] > 70.0) and
        sig["macd_bear"] and
        (sig["vol20"] < VOL_THRESH)
    )

def stop_hit(sig, open_trade):
    if not open_trade:
        return False
    side = open_trade["side"]
    z_ent = float(open_trade["z_entry"])
    z_now = float(sig["z"])
    if side == "long_basket":
        # adverse is z moving further negative by 1.5
        return (z_now <= z_ent - Z_STOP_AWAY)
    else:
        # adverse is z moving further positive by 1.5
        return (z_now >= z_ent + Z_STOP_AWAY)

def exit_hit(sig, open_trade):
    if not open_trade:
        return False
    # exit when z back near 0
    return abs(float(sig["z"])) <= Z_EXIT_ABS

# =========================
# Daily decision loop
# =========================
def run_daily():
    print("Fetching daily data (Alpaca)...")
    df_px = fetch_daily(UNIVERSE)
    if df_px.empty:
        print("No data; skip.")
        return

    sig = compute_spread_signals(df_px)
    print(f"Signals @ {sig['latest_date']}: z={sig['z']:.2f} | rsi={sig['rsi']:.1f} | "
          f"vol20={sig['vol20']:.4f} | macd_bull={sig['macd_bull']} | macd_bear={sig['macd_bear']}")

    state = load_state()
    open_trade = state.get("open_trade")

    # 1) Manage exits first (stop/target)
    if state["regime"] in ("long_basket","short_basket") and open_trade:
        if stop_hit(sig, open_trade):
            print("STOP hit: liquidating pair.")
            pnl = compute_pair_pnl_snapshot(df_px, open_trade["side"])
            liquidate_all()
            trades = load_trades()
            trades["trades"].append({
                "closed_at": datetime.now().isoformat(),
                "reason": "z_stop",
                "side": open_trade["side"],
                "z_entry": float(open_trade["z_entry"]),
                "z_exit": float(sig["z"]),
                "pnl": float(pnl)
            })
            trades["cumulative_profit"] += float(pnl)
            state["summary"]["total_profit"] += float(pnl)
            state["summary"]["total_trades"] += 1
            if pnl >= 0: state["summary"]["profitable_trades"] += 1
            else:        state["summary"]["unprofitable_trades"] += 1
            state["regime"] = "neutral"
            state["open_trade"] = None
            save_trades(trades); save_state(state)
            return

        if exit_hit(sig, open_trade):
            print("EXIT: z back near 0; liquidating pair.")
            pnl = compute_pair_pnl_snapshot(df_px, open_trade["side"])
            liquidate_all()
            trades = load_trades()
            trades["trades"].append({
                "closed_at": datetime.now().isoformat(),
                "reason": "z_exit_zero",
                "side": open_trade["side"],
                "z_entry": float(open_trade["z_entry"]),
                "z_exit": float(sig["z"]),
                "pnl": float(pnl)
            })
            trades["cumulative_profit"] += float(pnl)
            state["summary"]["total_profit"] += float(pnl)
            state["summary"]["total_trades"] += 1
            if pnl >= 0: state["summary"]["profitable_trades"] += 1
            else:        state["summary"]["unprofitable_trades"] += 1
            state["regime"] = "neutral"
            state["open_trade"] = None
            save_trades(trades); save_state(state)
            return

    # 2) If neutral, check entries
    if state["regime"] == "neutral":
        if should_open_long(sig):
            print("OPEN: Long Basket / Short QQQ.")
            open_long_basket_short_qqq(df_px, state)
            state["open_trade"] = {
                "side": "long_basket",
                "z_entry": float(sig["z"]),
                "opened_at": datetime.now().isoformat()
            }
            save_state(state)
            return

        if should_open_short(sig):
            print("OPEN: Short Basket / Long QQQ.")
            open_short_basket_long_qqq(df_px, state)
            state["open_trade"] = {
                "side": "short_basket",
                "z_entry": float(sig["z"]),
                "opened_at": datetime.now().isoformat()
            }
            save_state(state)
            return

    # 3) If in a trade but neither stop nor exit hit, hold
    print("No change; holding regime:", state["regime"])
    state["last_run"] = datetime.now().isoformat()
    state["summary"]["total_rebalances"] += 1
    save_state(state)

# Scheduler (Daily ~10:05)
def wait_until_next_day_1005_local():
    now = datetime.now()
    target = now.replace(hour=10, minute=5, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    while target.weekday() >= 5:  # skip weekend
        target = target + timedelta(days=1)
    delta = (target - now).total_seconds()
    mins = delta / 60.0
    print(f"Sleeping ~{mins:.1f} minutes until next trading day 10:05.")
    time.sleep(max(60, delta))

def trading_bot_nas14_qqq_daily():
    account = api.get_account()
    print(f"Buying Power: {account.buying_power}")
    while True:
        print("\n=== Daily Rebalance (NAS14 vs QQQ MR) ===")
        try:
            run_daily()
        except Exception as e:
            print("Rebalance error:", e)
        wait_until_next_day_1005_local()


# Entrypoint
if __name__ == "__main__":
    trading_bot_nas14_qqq_daily()
