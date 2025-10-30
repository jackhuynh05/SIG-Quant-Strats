# group 4 Energy, Transport arbitrage
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

# Universe / Config
ENERGY = ["EOG", "EIX", "NEE", "BP", "GEVO"]
TRANSP = ["UPS", "FDX", "DAL", "UAL"]
UNIVERSE = sorted(list(set(ENERGY + TRANSP)))

START_DAYS = 365 * 5     # ~5 years daily history
FREQ = "W-FRI"           # compute indicators from daily -> weekly Fri close
MA_FAST = 20
MA_SLOW = 60
RSI_LEN = 14
TRAIL_PCT = 0.05         # 5% trailing stop per leg
TARGET_VOL = 0.125       # 12.5% target (within 10–15% band)
VOL_LOOKBACK = 60        # days for realized vol estimate
MAX_GROSS_DOLLARS = 10000.0   # total live capital cap (gross)
PAIR_SECTOR_DOLLARS = MAX_GROSS_DOLLARS / 2.0  # per sector when pair is active
EPS = 1e-6

STATE_FILE  = "trading_state_group4.json"
TRADES_FILE = "trade_log_group4.json"


# State & Trade Log
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "last_run": None,
        "regime": "neutral",  # 'neutral', 'long_energy', 'long_transport'
        "summary": {
            "total_rebalances": 0,
            "total_trades": 0,
            "profitable_trades": 0,
            "unprofitable_trades": 0,
            "total_profit": 0.0
        },
        # per-symbol trailing stop anchors (+ highest for long, lowest for short)
        "stop_anchor": {sym: None for sym in UNIVERSE},
        # open legs snapshot for P&L on close
        "open_legs": {}  # sym -> {"qty": float, "entry_px": float, "side": "long"/"short"}
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"symbol": "ticker"})
    df = df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]
    wide = df.pivot(index="timestamp", columns="ticker", values=["open", "high", "low", "close", "volume"]).sort_index()
    # MultiIndex columns: (ticker, Field)
    wide.columns = pd.MultiIndex.from_tuples([(c[1], c[0].title()) for c in wide.columns], names=["ticker", "field"])
    return wide

def last_close(df, symbol):
    try:
        return float(df[(symbol, "Close")].dropna().iloc[-1])
    except Exception:
        return None

# Indicators
def rsi(series: pd.Series, length: int = RSI_LEN) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def realized_ann_vol(close_series: pd.Series, lookback: int = VOL_LOOKBACK) -> float:
    s = close_series.dropna()
    if len(s) < lookback + 1:
        return 0.25  # fallback 25%
    rets = s.pct_change().dropna().iloc[-lookback:]
    v = float(rets.std()) * math.sqrt(252)
    return v if v > 1e-6 else 0.25

def basket_close(df_px: pd.DataFrame, tickers: list) -> pd.Series:
    # Equal-weight average close across tickers (only those with data on a day)
    cols = [(tk, "Close") for tk in tickers if (tk, "Close") in df_px.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = df_px[cols].copy()
    return sub.mean(axis=1)

def compute_signals(df_px: pd.DataFrame):
    # Build weekly closes for baskets
    e_close = basket_close(df_px, ENERGY).resample(FREQ).last().dropna()
    t_close = basket_close(df_px, TRANSP).resample(FREQ).last().dropna()
    common_idx = e_close.index.intersection(t_close.index)
    e_close, t_close = e_close.loc[common_idx], t_close.loc[common_idx]

    ratio = (e_close / t_close).dropna()
    r_ma20 = ratio.rolling(MA_FAST).mean()
    r_ma60 = ratio.rolling(MA_SLOW).mean()

    # Momentum (basket MA cross)
    e_ma20, e_ma60 = e_close.rolling(MA_FAST).mean(), e_close.rolling(MA_SLOW).mean()
    t_ma20, t_ma60 = t_close.rolling(MA_FAST).mean(), t_close.rolling(MA_SLOW).mean()

    # RSI on baskets for overextension filter
    e_rsi = rsi(e_close)
    t_rsi = rsi(t_close)

    latest = ratio.index[-1]
    sig = {
        "latest_date": str(latest.date()),
        "ratio": float(ratio.iloc[-1]),
        "ratio_ma20": float(r_ma20.iloc[-1]) if not np.isnan(r_ma20.iloc[-1]) else None,
        "ratio_ma60": float(r_ma60.iloc[-1]) if not np.isnan(r_ma60.iloc[-1]) else None,
        "e_mom_up": bool(e_ma20.iloc[-1] > e_ma60.iloc[-1]),
        "t_mom_up": bool(t_ma20.iloc[-1] > t_ma60.iloc[-1]),
        "e_rsi": float(e_rsi.iloc[-1]) if not np.isnan(e_rsi.iloc[-1]) else 50.0,
        "t_rsi": float(t_rsi.iloc[-1]) if not np.isnan(t_rsi.iloc[-1]) else 50.0
    }
    return sig


# Sizing / Orders
def get_position(symbol):
    try:
        pos = api.get_position(symbol)
        return float(pos.qty), float(pos.avg_entry_price)
    except Exception:
        return 0.0, 0.0

def set_target_dollars(symbol, target_dollars, last_price):
    if last_price is None or last_price <= 0:
        print(f"{symbol}: no price, skip target.")
        return
    desired_qty = float(target_dollars) / float(last_price)
    current_qty, _ = get_position(symbol)
    delta = desired_qty - current_qty
    if delta > EPS:
        api.submit_order(symbol=symbol, qty=float(delta), side="buy", type="market", time_in_force="day")
        print(f"BUY {symbol} ~{delta:.6f} to ${target_dollars:.0f}.")
    elif delta < -EPS:
        api.submit_order(symbol=symbol, qty=float(abs(delta)), side="sell", type="market", time_in_force="day")
        print(f"SELL {symbol} ~{abs(delta):.6f} to ${target_dollars:.0f}.")
    else:
        print(f"{symbol}: near target.")

def liquidate_symbol(symbol):
    url = f"{BASE_URL}/v2/positions/{symbol}?percentage=100"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
    r = requests.delete(url, headers=headers)
    print(f"Liquidate {symbol}: {r.text}")

def sector_weights_by_inverse_vol(df_px, tickers: list, sector_dollars: float) -> dict:
    # Vol targeting: weight ∝ 1 / realized_vol, then scale to sector dollars
    vols = {}
    for tk in tickers:
        s = df_px[(tk, "Close")].dropna() if (tk, "Close") in df_px.columns else pd.Series(dtype=float)
        v = realized_ann_vol(s)
        vols[tk] = max(v, 1e-6)
    inv = {tk: 1.0 / v for tk, v in vols.items()}
    total_inv = sum(inv.values()) if inv else 1.0
    dollars = {tk: sector_dollars * (inv[tk] / total_inv) for tk in tickers}
    return dollars  # per symbol target dollars (long or short)

def open_pair_long_energy(df_px, state):
    # Allocate long ENERGY, short TRANSP with vol-target sizing
    e_targets = sector_weights_by_inverse_vol(df_px, ENERGY, PAIR_SECTOR_DOLLARS)
    t_targets = sector_weights_by_inverse_vol(df_px, TRANSP, PAIR_SECTOR_DOLLARS)

    # Buy energy
    for sym, dol in e_targets.items():
        px = last_close(df_px, sym)
        set_target_dollars(sym, dol, px)
        state["open_legs"][sym] = {"qty": get_position(sym)[0], "entry_px": px, "side": "long"}

    # Short transport
    for sym, dol in t_targets.items():
        px = last_close(df_px, sym)
        # short by selling to target negative qty
        desired_qty = -(dol / px) if (px and px > 0) else 0.0
        current_qty, _ = get_position(sym)
        delta = desired_qty - current_qty
        if delta < -EPS:  # need to sell more (make qty more negative)
            api.submit_order(symbol=sym, qty=float(abs(delta)), side="sell", type="market", time_in_force="day")
            print(f"SHORT {sym} ~{abs(delta):.6f} to ${dol:.0f}.")
        elif delta > EPS: # reduce short (buy back)
            api.submit_order(symbol=sym, qty=float(abs(delta)), side="buy", type="market", time_in_force="day")
            print(f"COVER {sym} ~{abs(delta):.6f} (reduce) toward ${dol:.0f}.")
        else:
            print(f"{sym}: short near target.")
        state["open_legs"][sym] = {"qty": get_position(sym)[0], "entry_px": px, "side": "short"}

def open_pair_long_transport(df_px, state):
    # Allocate long TRANSP, short ENERGY with vol-target sizing
    t_targets = sector_weights_by_inverse_vol(df_px, TRANSP, PAIR_SECTOR_DOLLARS)
    e_targets = sector_weights_by_inverse_vol(df_px, ENERGY, PAIR_SECTOR_DOLLARS)

    # Buy transport
    for sym, dol in t_targets.items():
        px = last_close(df_px, sym)
        set_target_dollars(sym, dol, px)
        state["open_legs"][sym] = {"qty": get_position(sym)[0], "entry_px": px, "side": "long"}

    # Short energy
    for sym, dol in e_targets.items():
        px = last_close(df_px, sym)
        desired_qty = -(dol / px) if (px and px > 0) else 0.0
        current_qty, _ = get_position(sym)
        delta = desired_qty - current_qty
        if delta < -EPS:
            api.submit_order(symbol=sym, qty=float(abs(delta)), side="sell", type="market", time_in_force="day")
            print(f"SHORT {sym} ~{abs(delta):.6f} to ${dol:.0f}.")
        elif delta > EPS:
            api.submit_order(symbol=sym, qty=float(abs(delta)), side="buy", type="market", time_in_force="day")
            print(f"COVER {sym} ~{abs(delta):.6f} (reduce) toward ${dol:.0f}.")
        else:
            print(f"{sym}: short near target.")
        state["open_legs"][sym] = {"qty": get_position(sym)[0], "entry_px": px, "side": "short"}

def liquidate_universe():
    for sym in UNIVERSE:
        liquidate_symbol(sym)


# Trailing Stop Management
def update_trailing_and_check_stop(df_px, sym, state) -> bool:
    """
    Update trailing anchor; return True if stop hit and position closed.
    """
    px = last_close(df_px, sym)
    if px is None:
        return False
    qty, _ = get_position(sym)
    if abs(qty) <= EPS:
        state["stop_anchor"][sym] = None
        return False

    anchor = state["stop_anchor"].get(sym)
    if qty > 0:  # long
        anchor = px if anchor is None else max(anchor, px)
        drawdown = (anchor - px) / anchor if anchor > 0 else 0.0
        state["stop_anchor"][sym] = float(anchor)
        if drawdown >= TRAIL_PCT:
            # sell to close long
            api.submit_order(symbol=sym, qty=float(abs(qty)), side="sell", type="market", time_in_force="day")
            print(f"TRAIL STOP HIT (LONG) {sym}: sold ~{abs(qty):.6f}")
            return True
    else:        # short
        anchor = px if anchor is None else min(anchor, px)
        adverse = (px - anchor) / anchor if anchor > 0 else 0.0
        state["stop_anchor"][sym] = float(anchor)
        if adverse >= TRAIL_PCT:
            # buy to close short
            api.submit_order(symbol=sym, qty=float(abs(qty)), side="buy", type="market", time_in_force="day")
            print(f"TRAIL STOP HIT (SHORT) {sym}: bought ~{abs(qty):.6f}")
            return True
    return False

def pairwide_trailing_stop_check(df_px, state) -> bool:
    """
    If any leg hits a trailing stop, close the whole pair (symmetry).
    Returns True if pair was closed.
    """
    hit = False
    for sym in UNIVERSE:
        if update_trailing_and_check_stop(df_px, sym, state):
            hit = True
    if hit:
        print("Pair stop triggered: liquidating all legs for symmetry.")
        liquidate_universe()
        # compute P&L snapshot
        pnl = 0.0
        for sym, leg in state["open_legs"].items():
            exit_px = last_close(df_px, sym)
            if exit_px is None:
                continue
            if leg["side"] == "long":
                pnl += (exit_px - leg["entry_px"]) * leg["qty"]
            else:
                pnl += (leg["entry_px"] - exit_px) * abs(leg["qty"])
        trades = load_trades()
        trades["trades"].append({
            "closed_at": datetime.now().isoformat(),
            "reason": "pair_trailing_stop",
            "regime": state["regime"],
            "pnl": float(pnl)
        })
        trades["cumulative_profit"] += float(pnl)
        state["summary"]["total_profit"] += float(pnl)
        if pnl >= 0: state["summary"]["profitable_trades"] += 1
        else:        state["summary"]["unprofitable_trades"] += 1
        state["open_legs"] = {}
        state["regime"] = "neutral"
        save_trades(trades)
        save_state(state)
        return True
    return False


# Decision & Execution
def run_weekly_rebalance():
    print("Fetching daily data (Alpaca)...")
    df_px = fetch_daily(UNIVERSE)
    if df_px.empty:
        print("No data. Skip.")
        return

    sig = compute_signals(df_px)
    print(f"Signals @ {sig['latest_date']}: ratio={sig['ratio']:.4f} | "
          f"ma20={sig['ratio_ma20']:.4f} | ma60={sig['ratio_ma60']:.4f} | "
          f"e_mom_up={sig['e_mom_up']} | t_mom_up={sig['t_mom_up']} | "
          f"e_rsi={sig['e_rsi']:.1f} | t_rsi={sig['t_rsi']:.1f}")

    state = load_state()

    # Pair-level trailing stop check (first, fast exit if hit)
    if state["regime"] in ("long_energy", "long_transport"):
        if pairwide_trailing_stop_check(df_px, state):
            return  # already handled

    # Determine desired regime from rules
    want_regime = "neutral"
    if (sig["ratio"] is not None) and (sig["ratio_ma60"] is not None):
        if (sig["ratio"] > sig["ratio_ma60"]) and sig["e_mom_up"] and (sig["e_rsi"] < 70.0):
            want_regime = "long_energy"
        elif (sig["ratio"] < sig["ratio_ma60"]) and sig["t_mom_up"] and (sig["t_rsi"] < 70.0):
            want_regime = "long_transport"
        else:
            want_regime = "neutral"

    # Mean-reversion exit rule: ratio crossing back across MA20
    if state["regime"] == "long_energy" and (sig["ratio_ma20"] is not None) and (sig["ratio"] <= sig["ratio_ma20"]):
        print("Exit long_energy: ratio reverted under MA20.")
        liquidate_universe()
        # log pnl
        pnl = 0.0
        for sym, leg in state["open_legs"].items():
            exit_px = last_close(df_px, sym)
            if exit_px is None: continue
            if leg["side"] == "long":
                pnl += (exit_px - leg["entry_px"]) * leg["qty"]
            else:
                pnl += (leg["entry_px"] - exit_px) * abs(leg["qty"])
        trades = load_trades()
        trades["trades"].append({
            "closed_at": datetime.now().isoformat(),
            "reason": "mean_revert_ma20",
            "regime": "long_energy",
            "pnl": float(pnl)
        })
        trades["cumulative_profit"] += float(pnl)
        state["summary"]["total_profit"] += float(pnl)
        if pnl >= 0: state["summary"]["profitable_trades"] += 1
        else:        state["summary"]["unprofitable_trades"] += 1
        state["open_legs"] = {}
        state["regime"] = "neutral"
        save_trades(trades)
        save_state(state)

    if state["regime"] == "long_transport" and (sig["ratio_ma20"] is not None) and (sig["ratio"] >= sig["ratio_ma20"]):
        print("Exit long_transport: ratio reverted above MA20.")
        liquidate_universe()
        # log pnl
        pnl = 0.0
        for sym, leg in state["open_legs"].items():
            exit_px = last_close(df_px, sym)
            if exit_px is None: continue
            if leg["side"] == "long":
                pnl += (exit_px - leg["entry_px"]) * leg["qty"]
            else:
                pnl += (leg["entry_px"] - exit_px) * abs(leg["qty"])
        trades = load_trades()
        trades["trades"].append({
            "closed_at": datetime.now().isoformat(),
            "reason": "mean_revert_ma20",
            "regime": "long_transport",
            "pnl": float(pnl)
        })
        trades["cumulative_profit"] += float(pnl)
        state["summary"]["total_profit"] += float(pnl)
        if pnl >= 0: state["summary"]["profitable_trades"] += 1
        else:        state["summary"]["unprofitable_trades"] += 1
        state["open_legs"] = {}
        state["regime"] = "neutral"
        save_trades(trades)
        save_state(state)

    # Recompute desired regime after possible exits
    if (sig["ratio"] is not None) and (sig["ratio_ma60"] is not None):
        if (sig["ratio"] > sig["ratio_ma60"]) and sig["e_mom_up"] and (sig["e_rsi"] < 70.0):
            want_regime = "long_energy"
        elif (sig["ratio"] < sig["ratio_ma60"]) and sig["t_mom_up"] and (sig["t_rsi"] < 70.0):
            want_regime = "long_transport"
        else:
            want_regime = "neutral"

    # Adjust portfolio to desired regime
    if want_regime == "neutral":
        if state["regime"] != "neutral":
            print("Going NEUTRAL: liquidate all legs.")
            liquidate_universe()
            state["open_legs"] = {}
            state["regime"] = "neutral"
    elif want_regime == "long_energy":
        if state["regime"] != "long_energy":
            print("Switching to LONG ENERGY / SHORT TRANSPORT.")
            liquidate_universe()
            state["open_legs"] = {}
            open_pair_long_energy(df_px, state)
            state["regime"] = "long_energy"
    elif want_regime == "long_transport":
        if state["regime"] != "long_transport":
            print("Switching to LONG TRANSPORT / SHORT ENERGY.")
            liquidate_universe()
            state["open_legs"] = {}
            open_pair_long_transport(df_px, state)
            state["regime"] = "long_transport"

    # Update trailing anchors for any open legs
    for sym in UNIVERSE:
        px = last_close(df_px, sym)
        if px is None:
            continue
        qty, _ = get_position(sym)
        if abs(qty) <= EPS:
            state["stop_anchor"][sym] = None
        else:
            anchor = state["stop_anchor"].get(sym)
            if qty > 0:
                state["stop_anchor"][sym] = float(px if anchor is None else max(anchor, px))
            else:
                state["stop_anchor"][sym] = float(px if anchor is None else min(anchor, px))

    state["last_run"] = datetime.now().isoformat()
    state["summary"]["total_rebalances"] += 1
    save_state(state)


# Scheduler (Weekly)
def wait_until_next_monday_1005_local():
    """
    Sleep until next Monday ~10:05 local time.
    """
    now = datetime.now()
    days_ahead = (0 - now.weekday()) % 7  # Monday=0
    target = now + timedelta(days=days_ahead)
    target = target.replace(hour=10, minute=5, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=7)
    # Skip weekends logic not required here (we target Monday)
    delta = (target - now).total_seconds()
    mins = delta / 60.0
    print(f"Sleeping ~{mins:.1f} minutes until next Monday 10:05.")
    time.sleep(max(60, delta))  # at least 60s

def trading_bot_energy_transport():
    account = api.get_account()
    print(f"Buying Power: {account.buying_power}")

    while True:
        print("\n=== Weekly Rebalance (Energy vs Transport Ratio) ===")
        try:
            run_weekly_rebalance()
        except Exception as e:
            print("Rebalance error:", e)
        wait_until_next_monday_1005_local()


# Entrypoint
if __name__ == "__main__":
    trading_bot_energy_transport()
