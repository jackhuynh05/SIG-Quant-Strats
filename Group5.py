# Group 5: Hourly MA(log-returns) Sign Strategy with 2% Trailing Stop
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import requests
import time
import json
import os
import math


# Alpaca API credentials
API_KEY = ''
API_SECRET = ''
BASE_URL = 'https://paper-api.alpaca.markets'

# Files
STATE_FILE = 'trading_state_group5.json'
TRADES_FILE = 'trade_log_group5.json'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Universe / Parameters
UNIVERSE = ['AXON', 'TSLA', 'COIN', 'INTC', 'SHOP', 'PYPL', 'ASAN', 'NVDA', 'MSTR', 'ARM']

TIMEFRAME = TimeFrame.Hour     # 1-hour bars
LOOKBACK_DAYS = 60             # enough to cover hourly windows comfortably
MA_WINDOW = 5                  # 5 one-hour bars
TRAIL_PCT = 0.02               # 2% trailing stop
DOLLARS_PER_TRADE = 1000.0     # flat sizing per symbol (long or short)
EPS = 1e-6

# Helpers: data, state, logs
def get_historical_data(symbol, time_frame, days=LOOKBACK_DAYS):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=time_frame,
        start=start_date,
        end=end_date,
    )
    bars = stock_data_client.get_stock_bars(req)
    df = bars.df.reset_index()
    if df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def get_data(symbols, time_frame):
    data_dict = {}
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        data_dict[symbol] = get_historical_data(symbol, time_frame, days=LOOKBACK_DAYS)
    return data_dict

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
        # Per-symbol persistent state
        state = {
            "summary": {
                "total_trades": 0,
                "profitable_trades": 0,
                "unprofitable_trades": 0,
                "total_profit": 0.0
            },
            "symbols": {
                sym: {
                    "regime": "flat",           # 'flat', 'long', 'short', 'stopped'
                    "open_trade": None,         # dict when active
                    "stop_anchor": None,        # float: trail reference (high for long / low for short)
                    "cooldown_sign": None       # None or +1/-1 (sign when stopped; must flip to re-arm)
                } for sym in UNIVERSE
            }
        }
    return state

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f:
            trades = json.load(f)
    else:
        trades = {
            "trades": [],
            "cumulative_profit": 0.0
        }
    return trades

def save_trades(trades):
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=4)

# Indicators / Signals
def ma_log_return_sign(close_series: pd.Series, window: int = MA_WINDOW):
    """
    Returns sign of moving average of log returns over 'window' bars.
    +1 if >0, -1 if <0, 0 if exactly 0 or insufficient data.
    """
    s = close_series.dropna()
    if len(s) < window + 1:
        return 0, 0.0
    rets = np.log(s).diff()
    ma = rets.rolling(window=window).mean().iloc[-1]
    if ma > 0:  return +1, float(ma)
    if ma < 0:  return -1, float(ma)
    return 0, float(ma)

# Order / Position Helpers
def get_last_close(df):
    if df is None or df.empty:
        return None
    return float(df['close'].iloc[-1])

def get_position(symbol):
    try:
        pos = api.get_position(symbol)
        qty = float(pos.qty)
        side = 'long' if qty > 0 else 'short'
        return {'symbol': symbol, 'qty': qty, 'side': side, 'avg_entry': float(pos.avg_entry_price)}
    except Exception:
        return None

# --- REPLACED: separate helpers for fractional longs vs whole-share shorts ---

def submit_long_fractional_order(symbol, qty, side):
    """
    Fractional LONG entries/exits allowed. side in {'buy','sell'}.
    Always use DAY for fractional orders.
    """
    if qty <= EPS:
        return
    api.submit_order(symbol=symbol, qty=float(qty), side=side,
                     type='market', time_in_force='day')

def submit_short_whole_order(symbol, qty, side):
    """
    Whole-share SHORT entries/exits only. side in {'sell','buy'}.
    'sell' opens/increases a short, 'buy' covers/reduces a short.
    """
    q = int(math.floor(float(qty)))
    if q < 1:
        print(f"{symbol}: computed short qty < 1 share; skipping.")
        return
    api.submit_order(symbol=symbol, qty=q, side=side,
                     type='market', time_in_force='day')

def liquidate_symbol(symbol):
    url = f"{BASE_URL}/v2/positions/{symbol}?percentage=100"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
    response = requests.delete(url, headers=headers)
    print(f"Liquidating {symbol}: {response.text}")

# Trade Open/Close per Symbol
# --- REPLACED: longs stay fractional; shorts floor to whole shares ---
def open_symbol_trade(symbol, direction, price, state, trades):
    """
    direction: 'long' or 'short'
    """
    try:
        if direction == 'long':
            qty = DOLLARS_PER_TRADE / price                    # fractional ok
            submit_long_fractional_order(symbol, qty, 'buy')
            stop_anchor = price  # for long, track highest since entry
        else:
            qty_float = DOLLARS_PER_TRADE / price              # must be whole for shorts
            qty = int(math.floor(qty_float))
            if qty < 1:
                print(f"{symbol}: short size < 1 share; skipping open.")
                return state, trades
            submit_short_whole_order(symbol, qty, 'sell')
            stop_anchor = price  # for short, track lowest since entry

        print(f"Opened {direction.upper()} {symbol}: qty={float(qty):.6f} @ ~{price:.2f}")

        # Record open in state/log
        state["symbols"][symbol]["regime"] = direction
        state["symbols"][symbol]["stop_anchor"] = float(stop_anchor)
        state["symbols"][symbol]["open_trade"] = {
            "symbol": symbol,
            "opened_at": datetime.now().isoformat(),
            "side": direction,
            "entry_px": float(price),
            "qty": float(qty)
        }
        state["summary"]["total_trades"] += 1

        trades["trades"].append({
            "symbol": symbol,
            "opened_at": state["symbols"][symbol]["open_trade"]["opened_at"],
            "side": direction,
            "entry_px": float(price),
            "qty": float(qty)
        })

    except tradeapi.rest.APIError as e:
        print(f"API Error opening {symbol}: {e}")

    return state, trades

# --- REPLACED: close longs fractionally; cover shorts with whole shares ---
def close_symbol_trade(symbol, price, state, trades, reason="exit"):
    ot = state["symbols"][symbol]["open_trade"]
    if not ot:
        return state, trades

    try:
        qty_open = float(ot["qty"])

        if ot["side"] == "long":
            # fractional sell to close long
            submit_long_fractional_order(symbol, qty_open, 'sell')
            pnl = (price - ot["entry_px"]) * qty_open
        else:
            # whole-share buy to cover short
            qty_cover = int(round(qty_open))
            if qty_cover < 1:
                qty_cover = 1
            submit_short_whole_order(symbol, qty_cover, 'buy')
            pnl = (ot["entry_px"] - price) * qty_cover

        # Update summary
        state["summary"]["total_profit"] += float(pnl)
        if pnl >= 0:
            state["summary"]["profitable_trades"] += 1
        else:
            state["summary"]["unprofitable_trades"] += 1

        # Update last trade log for symbol
        for t in reversed(trades["trades"]):
            if t.get("symbol") == symbol and "closed_at" not in t:
                t["closed_at"] = datetime.now().isoformat()
                t["exit_px"] = float(price)
                t["pnl"] = float(pnl)
                t["reason"] = reason
                break

        trades["cumulative_profit"] += float(pnl)
        print(f"Closed {symbol} ({reason}). P&L=${pnl:.2f}")

        # Clear state for symbol
        state["symbols"][symbol]["open_trade"] = None
        state["symbols"][symbol]["stop_anchor"] = None

    except tradeapi.rest.APIError as e:
        print(f"API Error closing {symbol}: {e}")

    return state, trades

# Trailing Stop Check/Update
def update_trailing_stop(symbol, price, state):
    sym_state = state["symbols"][symbol]
    regime = sym_state["regime"]
    anchor = sym_state["stop_anchor"]

    if sym_state["open_trade"] is None or anchor is None:
        return 'ok', anchor

    if regime == 'long':
        # update highest
        anchor = max(anchor, price)
        drawdown = (anchor - price) / anchor if anchor > 0 else 0.0
        sym_state["stop_anchor"] = float(anchor)
        if drawdown >= TRAIL_PCT:
            return 'hit', anchor
        return 'ok', anchor

    if regime == 'short':
        # update lowest
        anchor = min(anchor, price)
        adverse = (price - anchor) / anchor if anchor > 0 else 0.0
        sym_state["stop_anchor"] = float(anchor)
        if adverse >= TRAIL_PCT:
            return 'hit', anchor
        return 'ok', anchor

    return 'ok', anchor

# Main Trading Loop (Hourly)
def trading_bot_group5():
    account = api.get_account()
    print(f"Buying Power: {account.buying_power}")

    state = load_state()
    trades = load_trades()

    while True:
        print("\n=== Hourly check (Group 5) ===")

        # Fetch latest hourly data for all symbols
        global data_dict
        data_dict = get_data(UNIVERSE, time_frame=TIMEFRAME)

        for symbol in UNIVERSE:
            df = data_dict[symbol]
            price = get_last_close(df)
            if price is None:
                print(f"{symbol}: price missing, skip.")
                continue

            # Signal: sign of MA of log-returns (last 5 hourly bars)
            sig, ma_val = ma_log_return_sign(df['close'], window=MA_WINDOW)
            # sig in {+1, -1, 0}
            print(f"{symbol}: MA5(logret)={ma_val:.6f} -> sig={sig}")

            sym_state = state["symbols"][symbol]
            regime = sym_state["regime"]
            cooldown_sign = sym_state["cooldown_sign"]

            # Update trailing stop if in a position
            if regime in ('long', 'short'):
                status, anchor = update_trailing_stop(symbol, price, state)
                if status == 'hit':
                    # trailing stop triggered -> close and mark 'stopped'
                    state, trades = close_symbol_trade(symbol, price, state, trades, reason="trailing_stop")
                    sym_state["regime"] = "stopped"
                    sym_state["cooldown_sign"] = sig  # wait until this sign flips before re-arming
                    save_state(state); save_trades(trades)
                    continue  # no new action this bar after stop

                # If signal flips against current side -> reverse immediately
                if regime == 'long' and sig < 0:
                    state, trades = close_symbol_trade(symbol, price, state, trades, reason="signal_flip")
                    sym_state["regime"] = "flat"
                    save_state(state); save_trades(trades)
                    # open short now
                    state, trades = open_symbol_trade(symbol, "short", price, state, trades)
                    save_state(state); save_trades(trades)
                    continue

                if regime == 'short' and sig > 0:
                    state, trades = close_symbol_trade(symbol, price, state, trades, reason="signal_flip")
                    sym_state["regime"] = "flat"
                    save_state(state); save_trades(trades)
                    # open long now
                    state, trades = open_symbol_trade(symbol, "long", price, state, trades)
                    save_state(state); save_trades(trades)
                    continue

                # If sig == 0, do nothing while in position
                continue

            # If we're flat or stopped, consider entries
            if regime == 'stopped':
                # After a stop, hold cash until the MA sign CHANGES from when stopped
                if cooldown_sign is None:
                    sym_state["cooldown_sign"] = sig  # initialize
                if sig != cooldown_sign and sig != 0:
                    # re-armed: open in current sign direction
                    side = "long" if sig > 0 else "short"
                    sym_state["cooldown_sign"] = None
                    state, trades = open_symbol_trade(symbol, side, price, state, trades)
                    save_state(state); save_trades(trades)
                else:
                    print(f"{symbol}: stopped, waiting for sign change (cooldown_sign={cooldown_sign}).")
                continue

            if regime == 'flat':
                if sig > 0:
                    state, trades = open_symbol_trade(symbol, "long", price, state, trades)
                    save_state(state); save_trades(trades)
                elif sig < 0:
                    state, trades = open_symbol_trade(symbol, "short", price, state, trades)
                    save_state(state); save_trades(trades)
                else:
                    # sig == 0: stay flat
                    pass
                continue

        # Sleep until next hour + small buffer
        wait_until_next_hour_plus_buffer()

def wait_until_next_hour_plus_buffer():
    now = datetime.now()
    target = now.replace(minute=5, second=0, microsecond=0)
    if now.minute >= 5:
        target = target + timedelta(hours=1)
    # Weekend skip not strictly necessary for hourly bars, but harmless:
    while target.weekday() >= 5:
        target = target + timedelta(days=1)
        target = target.replace(hour=10)  # mid-morning on Monday
    delta = (target - now).total_seconds()
    mins = delta / 60.0
    print(f"Waiting ~{mins:.1f} minutes for next hourly check.")
    time.sleep(max(30, delta))  # at least 30s

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    trading_bot_group5()
