import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykalman import KalmanFilter
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
STATE_FILE = 'trading_state.json'
TRADES_FILE = 'trade_log.json'


# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)


# Symbols / Parameters
# Use NVDA & AMD for pair signal; hold SPY in neutral
PAIR_LONGABLE = 'NVDA'
PAIR_SHORTABLE = 'AMD'
BENCHMARK = 'SPY'

# Universe to fetch
symbols_to_fetch = [PAIR_LONGABLE, PAIR_SHORTABLE, BENCHMARK]

# Strategy parameters
TIMEFRAME = TimeFrame.Day
LOOKBACK_DAYS = 1462
Z_TRADE = 2.0          # enter/flip threshold
Z_RISK_OFF = 3.0       # hard stop -> liquidate pair, hold SPY
TARGET_VOL = 0.10      # 10% annualized target for NVDA/AMD legs
BASE_DOLLARS_PER_LEG = 1000.0
NEUTRAL_DOLLARS = 10000.0
PAIR_ACTIVE_SPY_DOLLARS = 8000.0
MAX_GROSS_DOLLARS = 10000.0
EPS = 1e-6             # tiny threshold to avoid spam orders with ~0 qty deltas


# data, state, logs
def get_historical_data(symbol, time_frame, days=LOOKBACK_DAYS):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=time_frame,
        start=start_date,
        end=end_date,
    )
    bars = stock_data_client.get_stock_bars(request_params)
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
        state = {
            "regime": "neutral",  # 'neutral', 'long_pair', 'short_pair'
            "total_trades": 0,
            "profitable_trades": 0,
            "unprofitable_trades": 0,
            "total_profit": 0.0,
            "open_trade": None
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


# Signal / Sizing
def kalman_spread(nvda_close, amd_close):
    # Align by index and compute NVDA - AMD spread
    merged = pd.merge(
        nvda_close.to_frame('close_nvda'),
        amd_close.to_frame('close_amd'),
        left_index=True, right_index=True, how='inner'
    )
    spread = merged['close_nvda'] - merged['close_amd']

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )
    state_means, _ = kf.filter(spread.values)
    spread_mean = state_means.flatten()

    # z-score of current point
    resid = spread.values - spread_mean
    var = np.var(resid)
    if var <= 0:
        z = 0.0
    else:
        z = (spread.values[-1] - spread_mean[-1]) / math.sqrt(var)
    return z, spread.values[-1], spread_mean[-1]

def annualized_vol(close_series, lookback=60):
    # daily returns, annualized by sqrt(252)
    if len(close_series) < lookback + 1:
        return 0.30  # fallback assumption
    px = close_series[-(lookback+1):]
    rets = px.pct_change().dropna()
    vol = rets.std() * math.sqrt(252)
    return float(vol) if vol and vol > 1e-8 else 0.30

def target_dollar_allocation(price, realized_vol, base_dollars=BASE_DOLLARS_PER_LEG, target_vol=TARGET_VOL):
    # Vol targeting around a $1,000 "base". Scale ~ target_vol / realized_vol
    if realized_vol <= 1e-8:
        realized_vol = 0.30
    dollars = base_dollars * (target_vol / realized_vol)
    # keep it sane: between 50% and 200% of base
    dollars = max(0.5 * base_dollars, min(2.0 * base_dollars, dollars))
    qty = dollars / price  # FRACTIONAL shares allowed
    return float(qty), float(dollars)

# Order / Positioning 
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

def liquidate_symbol(symbol):
    url = f"{BASE_URL}/v2/positions/{symbol}?percentage=100"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
    response = requests.delete(url, headers=headers)
    print(f"Liquidating {symbol} position: {response.text}")

def set_spy_target_dollars(target_dollars, spy_price):
    # Adjust SPY position to approximate target dollars 
    if spy_price is None or spy_price <= 0:
        print("SPY price unavailable, skipping SPY rebalance.")
        return
    desired_qty = target_dollars / spy_price

    current = get_position(BENCHMARK)
    current_qty = float(current['qty']) if current else 0.0

    delta = desired_qty - current_qty
    if delta > EPS:
        try:
            api.submit_order(symbol=BENCHMARK, qty=float(delta), side='buy', type='market', time_in_force='gtc')
            print(f"Bought ~{delta:.6f} {BENCHMARK} to reach target ${target_dollars:.0f}.")
        except tradeapi.rest.APIError as e:
            print(f"API Error (buy SPY): {e}")
    elif delta < -EPS:
        try:
            api.submit_order(symbol=BENCHMARK, qty=float(abs(delta)), side='sell', type='market', time_in_force='gtc')
            print(f"Sold ~{abs(delta):.6f} {BENCHMARK} to reach target ${target_dollars:.0f}.")
        except tradeapi.rest.APIError as e:
            print(f"API Error (sell SPY): {e}")
    else:
        print("SPY already near target allocation.")

def gross_exposure_estimate(spy_price, spy_target, nvda_price, nvda_qty, amd_price, amd_qty):
    gross = 0.0
    if spy_price and spy_target:
        gross += min(spy_target, MAX_GROSS_DOLLARS)
    if nvda_price and nvda_qty:
        gross += abs(nvda_qty) * nvda_price
    if amd_price and amd_qty:
        gross += abs(amd_qty) * amd_price
    return gross

# Pair Trade Execution
def open_pair_trade(z, nvda_px, amd_px, nvda_vol, amd_vol, state, trades):
    # z < -2 => NVDA undervalued vs AMD => long NVDA, short AMD
    # z >  2 => NVDA overvalued vs AMD => short NVDA, long AMD
    long_nvda = z < -Z_TRADE
    short_nvda = z > Z_TRADE

    if not long_nvda and not short_nvda:
        return state, trades

    spy_target = PAIR_ACTIVE_SPY_DOLLARS

    # Vol-targeted sizes (fractional)
    nvda_qty, nvda_dollars = target_dollar_allocation(nvda_px, nvda_vol)
    amd_qty,  amd_dollars  = target_dollar_allocation(amd_px,  amd_vol)

    # Cap gross exposure
    gross_est = gross_exposure_estimate(nvda_px, spy_target, nvda_px, nvda_qty, amd_px, amd_qty)
    if gross_est > MAX_GROSS_DOLLARS:
        # scale both legs down proportionally
        legs_dollars = nvda_qty * nvda_px + amd_qty * amd_px
        scale = (MAX_GROSS_DOLLARS - spy_target) / max(1.0, legs_dollars)
        scale = max(0.1, min(1.0, scale))
        nvda_qty *= scale
        amd_qty  *= scale

    # Rebalance SPY first
    set_spy_target_dollars(spy_target, get_last_close(data_dict[BENCHMARK]))

    # Submit orders for the pair (fractional qty)
    try:
        if long_nvda:
            # BUY NVDA, SELL AMD
            if nvda_qty > EPS:
                api.submit_order(symbol=PAIR_LONGABLE, qty=float(nvda_qty), side='buy', type='market', time_in_force='gtc')
            if amd_qty > EPS:
                api.submit_order(symbol=PAIR_SHORTABLE, qty=float(amd_qty), side='sell', type='market', time_in_force='gtc')
            regime = 'long_pair'
            side_desc = 'Long NVDA / Short AMD'
            nvda_qty_signed = +nvda_qty
            amd_qty_signed  = -amd_qty
        else:
            # SELL NVDA, BUY AMD
            if nvda_qty > EPS:
                api.submit_order(symbol=PAIR_LONGABLE, qty=float(nvda_qty), side='sell', type='market', time_in_force='gtc')
            if amd_qty > EPS:
                api.submit_order(symbol=PAIR_SHORTABLE, qty=float(amd_qty), side='buy', type='market', time_in_force='gtc')
            regime = 'short_pair'
            side_desc = 'Short NVDA / Long AMD'
            nvda_qty_signed = -nvda_qty
            amd_qty_signed  = +amd_qty

        print(f"Opened pair: {side_desc} | NVDA qty={nvda_qty:.6f}, AMD qty={amd_qty:.6f}")

        # Record open in state + trade log
        state["regime"] = regime
        state["open_trade"] = {
            "opened_at": datetime.now().isoformat(),
            "z_enter": float(z),
            "side": regime,
            "nvda_entry_px": float(nvda_px),
            "amd_entry_px": float(amd_px),
            "nvda_qty": float(nvda_qty_signed),  # +long / -short
            "amd_qty": float(amd_qty_signed)     # +long / -short
        }
        state["total_trades"] += 1

        trade_rec = {
            "opened_at": state["open_trade"]["opened_at"],
            "side": side_desc,
            "z_enter": float(z),
            "nvda_entry_px": float(nvda_px),
            "amd_entry_px": float(amd_px),
            "nvda_qty": float(nvda_qty_signed),
            "amd_qty": float(amd_qty_signed)
        }
        trades["trades"].append(trade_rec)

    except tradeapi.rest.APIError as e:
        print(f"API Error opening pair: {e}")

    return state, trades

def close_pair_trade(z, nvda_px, amd_px, state, trades, reason="exit"):
    ot = state.get("open_trade")
    if not ot:
        return state, trades

    try:
        # For NVDA
        nvda_qty_open = float(ot["nvda_qty"])
        if abs(nvda_qty_open) > EPS:
            if nvda_qty_open > 0:
                api.submit_order(symbol=PAIR_LONGABLE, qty=float(abs(nvda_qty_open)), side='sell', type='market', time_in_force='gtc')
            else:
                api.submit_order(symbol=PAIR_LONGABLE, qty=float(abs(nvda_qty_open)), side='buy', type='market', time_in_force='gtc')

        # For AMD
        amd_qty_open = float(ot["amd_qty"])
        if abs(amd_qty_open) > EPS:
            if amd_qty_open > 0:
                api.submit_order(symbol=PAIR_SHORTABLE, qty=float(abs(amd_qty_open)), side='sell', type='market', time_in_force='gtc')
            else:
                api.submit_order(symbol=PAIR_SHORTABLE, qty=float(abs(amd_qty_open)), side='buy', type='market', time_in_force='gtc')

        # Compute P&L
        nvda_pnl = (nvda_px - ot["nvda_entry_px"]) * ot["nvda_qty"]
        amd_pnl  = (amd_px  - ot["amd_entry_px"])  * ot["amd_qty"]
        trade_pnl = float(nvda_pnl + amd_pnl)

        # Update state/trades
        state["total_profit"] += trade_pnl
        if trade_pnl >= 0:
            state["profitable_trades"] += 1
        else:
            state["unprofitable_trades"] += 1

        for t in reversed(trades["trades"]):
            if "closed_at" not in t:
                t["closed_at"]   = datetime.now().isoformat()
                t["z_exit"]      = float(z)
                t["nvda_exit_px"]= float(nvda_px)
                t["amd_exit_px"] = float(amd_px)
                t["pnl"]         = trade_pnl
                t["reason"]      = reason
                break

        trades["cumulative_profit"] += trade_pnl

        print(f"Closed pair trade ({reason}). P&L = ${trade_pnl:.2f}")

        # Clear open trade and set neutral
        state["open_trade"] = None
        state["regime"] = "neutral"

    except tradeapi.rest.APIError as e:
        print(f"API Error closing pair: {e}")

    return state, trades


# Main Trading Loop (Daily)
def trading_bot():
    account = api.get_account()
    print(f"Current Buying Power: {account.buying_power}")

    state = load_state()
    trades = load_trades()

    while True:
        print("\n=== Daily check for trading signals ===")

        global data_dict
        data_dict = get_data(symbols_to_fetch, time_frame=TIMEFRAME)

        nvda_df = data_dict[PAIR_LONGABLE]
        amd_df  = data_dict[PAIR_SHORTABLE]
        spy_df  = data_dict[BENCHMARK]

        nvda_px = get_last_close(nvda_df)
        amd_px  = get_last_close(amd_df)
        spy_px  = get_last_close(spy_df)

        if nvda_px is None or amd_px is None or spy_px is None:
            print("Price data missing. Skipping this cycle.")
            time.sleep(60 * 60 * 24)
            continue

        # Signal
        z, spread_now, spread_mean_now = kalman_spread(nvda_df['close'], amd_df['close'])
        print(f"Spread NVDA-AMD: last={spread_now:.2f}, kf_mean={spread_mean_now:.2f}, z={z:.2f}")

        # Vols for sizing (annualized)
        nvda_vol = annualized_vol(nvda_df['close'])
        amd_vol  = annualized_vol(amd_df['close'])
        print(f"Annualized vol: NVDA={nvda_vol:.2%}, AMD={amd_vol:.2%}")

        # Risk-off
        if state["regime"] in ("long_pair", "short_pair") and abs(z) >= Z_RISK_OFF:
            print("Risk-off trigger hit (|z| >= 3). Liquidating pair and moving to SPY only.")
            state, trades = close_pair_trade(z, nvda_px, amd_px, state, trades, reason="risk_off")
            set_spy_target_dollars(NEUTRAL_DOLLARS, spy_px)
            save_state(state); save_trades(trades)
            wait_until_next_close_plus_buffer()
            continue

        # Neutral band -> SPY only
        if abs(z) < Z_TRADE:
            if state["regime"] in ("long_pair", "short_pair"):
                print("Neutral zone (|z| < 2) while pair active. Closing pair and holding SPY.")
                state, trades = close_pair_trade(z, nvda_px, amd_px, state, trades, reason="neutral_zone")
            set_spy_target_dollars(NEUTRAL_DOLLARS, spy_px)
            save_state(state); save_trades(trades)
            wait_until_next_close_plus_buffer()
            continue

        # Pair active zone (|z| >= 2)
        if z <= -Z_TRADE:
            if state["regime"] == "long_pair":
                print("Already in long_pair. Maintain position and SPY at $8k.")
                set_spy_target_dollars(PAIR_ACTIVE_SPY_DOLLARS, spy_px)
            elif state["regime"] == "short_pair":
                print("Signal flipped to long_pair. Closing short_pair and opening long_pair.")
                state, trades = close_pair_trade(z, nvda_px, amd_px, state, trades, reason="flip")
                state, trades = open_pair_trade(z, nvda_px, amd_px, nvda_vol, amd_vol, state, trades)
            else:
                print("Opening long_pair from neutral.")
                state, trades = open_pair_trade(z, nvda_px, amd_px, nvda_vol, amd_vol, state, trades)

        elif z >= Z_TRADE:
            if state["regime"] == "short_pair":
                print("Already in short_pair. Maintain position and SPY at $8k.")
                set_spy_target_dollars(PAIR_ACTIVE_SPY_DOLLARS, spy_px)
            elif state["regime"] == "long_pair":
                print("Signal flipped to short_pair. Closing long_pair and opening short_pair.")
                state, trades = close_pair_trade(z, nvda_px, amd_px, state, trades, reason="flip")
                state, trades = open_pair_trade(z, nvda_px, amd_px, nvda_vol, amd_vol, state, trades)
            else:
                print("Opening short_pair from neutral.")
                state, trades = open_pair_trade(z, nvda_px, amd_px, nvda_vol, amd_vol, state, trades)

        save_state(state)
        save_trades(trades)
        wait_until_next_close_plus_buffer()

def wait_until_next_close_plus_buffer():
    now = datetime.now()
    target = now.replace(hour=16, minute=10, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    while target.weekday() >= 5:
        target = target + timedelta(days=1)
    delta = (target - now).total_seconds()
    mins = delta / 60.0
    print(f"Waiting ~{mins:.1f} minutes until next daily check.")
    time.sleep(max(60, delta))  # at least 60 seconds safety

# run bit
if __name__ == "__main__":
    trading_bot()
