from flask import Flask, request, jsonify
import json
import os
import requests
import pandas as pd
# The Python-Binance Client for API interactions
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException # New import for error handling
from binance.enums import * # New import for order types (SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, etc.)
from datetime import datetime
import time
# Imports for Technical Analysis (TA-Lib)
from talib import SMA, EMA, RSI, ATR, BBANDS
import hmac, hashlib # For raw balance fetch
from ml_strategy import get_ml_signal
import logging
import os

# Create logs folder if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging to CSV file
logging.basicConfig(
    filename='logs/trades.csv',
    level=logging.INFO,
    format='%(asctime)s,%(levelname)s,%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define the Futures Testnet URL (Global constant for the functions)
FUTURES_TESTNET_URL = 'https://testnet.binancefuture.com'

# ==============================================================================
# 1. FLASK INITIALIZATION (MUST RUN FIRST)
# ==============================================================================

# Initialize Flask application
app = Flask(__name__)

# Load configuration
config = {}
config_path = 'config.json'

if os.path.exists(config_path):
    try:
        with open(config_path) as f:
            config = json.load(f)
        print("✅ Configuration loaded.")
    except Exception as e:
        print(f"❌ Error loading config.json: {e}")
else:
    print("⚠️ WARNING: config.json not found. Using empty configuration.")

# ==============================================================================
# 2. TRADING BOT FUNCTIONS (FROM MAIN.IPYNB)
# ==============================================================================

def get_testnet_client():
    """
    Initializes and returns a Binance Client connected to the Futures Testnet.

    CRITICAL FIX: Removed 'base_url' from Client constructor as it causes an error
    in extremely old library versions. We rely on the client's internal routing
    when calling futures-prefixed methods (futures_ping, futures_klines, etc.).
    """
    api_key = config.get('binance_key')
    api_secret = config.get('binance_secret')

    if not api_key or not api_secret:
        print("❌ Error: API key or secret missing from config.json.")
        return None

    try:
        # CRITICAL FIX: Initialize standard Client *without* base_url, as it is not supported
        client = Client(
            api_key=api_key,
            api_secret=api_secret,
            tld='com' # tld is generally supported, we keep it.
        )

        # Verify connectivity using a standard futures method
        client.futures_ping()
        print("✅ Binance Testnet client initialized and connected.")
        return client
    except Exception as e:
        # Check for API Key error
        if "APIError(code=-2015)" in str(e) or "Authentication failed" in str(e):
            print("❌ Could not initialize Testnet client. Error: API Key / Permission Issue. Please verify your 'binance_key' and 'binance_secret'.")
        else:
            print(f"❌ Could not initialize Testnet client. Error: {e}")
            print("\n!!! URGENT: Your python-binance library is too old and lacks critical features like 'base_url'. Please update it using: pip install --upgrade python-binance !!!")
        return None


# CELL 2: Revised 'fetch_data' Function (Uses Binance API History Loop)
def fetch_data(symbol='BTCUSDT', interval='1d', start_date_str="2017-08-17"):
    """
    Fetches long-term historical data (1d interval) from the public Binance API,
    and recent 1h klines and current price from the authenticated Testnet client.

    Args:
        symbol (str): The trading pair (e.g., 'BTCUSDT').
        interval (str): The historical kline interval (e.g., '1d').
        start_date_str (str): The date to start fetching history from.

    Returns:
        dict: A dictionary containing historical_klines, recent_1h, resampled data,
              and current_price. Returns None on failure.
    """
    # The 'config' dictionary must be defined in Cell 1 and accessible globally.
    try:
        # Initialize Client for PUBLIC (non-futures) historical data.
        # We use a client without special URL for public API endpoints
        client = Client(
            api_key=config.get('binance_key'), # Use .get for safety
            api_secret=config.get('binance_secret') # Use .get for safety
        )
        print("Binance client initialized for public data fetch.")

        # Initialize Testnet client for recent data and current price
        testnet_client = get_testnet_client()
        if testnet_client is None:
            return None

        # --- 1. Historical Data (Looping Fetch from PUBLIC API) ---
        print(f"Starting historical {interval} kline fetch for {symbol} from {start_date_str}...")

        start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)
        now_ts = int(time.time() * 1000)
        all_data = []

        # Use public API for deep history
        public_url = "https://api.binance.com/api/v3/klines"

        while start_ts < now_ts:
            # Fetch 1000 candles at a time
            params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "limit": 1000}

            r = requests.get(public_url, params=params)
            r.raise_for_status() # Raise error for bad response codes (4xx, 5xx)
            chunk = r.json()

            if not chunk or len(chunk) < 2:
                break

            all_data.extend(chunk)
            # Set next start time to last candle's open time + 1ms to avoid overlap
            start_ts = chunk[-1][0] + 1

            print(f"Fetched {len(chunk)} candles, continuing from: {datetime.fromtimestamp(start_ts/1000).strftime('%Y-%m-%d')}")
            time.sleep(0.5) # Politeness delay to avoid rate limits

        print(f"Total historical candles fetched: {len(all_data)}")

        # Convert historical data to DataFrame
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
        hist_df = pd.DataFrame(all_data, columns=cols)

        # Clean historical data
        if not hist_df.empty:
            numeric_cols = ["open", "high", "low", "close", "volume"]
            hist_df[numeric_cols] = hist_df[numeric_cols].apply(pd.to_numeric)
            hist_df['timestamp'] = pd.to_datetime(hist_df['open_time'], unit='ms')
            # Drop unnecessary columns after conversion
            hist_df.drop(columns=['open_time', 'close_time', 'ignore', 'taker_buy_base', 'taker_buy_quote'], inplace=True)
            hist_df.set_index('timestamp', inplace=True)

        # --- 2. Recent 24hr Kliness (1h interval) from Testnet Client ---
        print("Fetching recent 24h klines (1h) from Testnet...")
        # Use the testnet_client created above (endpoint already changed)
        klines = testnet_client.futures_klines(
            symbol=symbol,
            interval='1h',
            limit=500
        )

        recent_df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

        # Clean recent data
        numeric_cols_recent = ['open', 'high', 'low', 'close', 'volume']
        recent_df[numeric_cols_recent] = recent_df[numeric_cols_recent].apply(pd.to_numeric)
        recent_df['timestamp'] = pd.to_datetime(recent_df['open_time'], unit='ms')
        recent_df.set_index('timestamp', inplace=True)
        recent_df.drop(columns=['open_time', 'close_time', 'ignore', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'], inplace=True)


        # Resample recent data (used for quick, aggregated view)
        resampled_4h = recent_df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        resampled_1d = recent_df.resample('1d').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

        # --- 3. Current Price from Testnet Client ---
        # Use the testnet_client created above (endpoint already changed)
        current_price = float(testnet_client.futures_symbol_ticker(
            symbol=symbol
        )['price'])
        print(f"Current Testnet Price: {current_price}")

        return {
            'historical_klines': hist_df,        # Full history (1d by default)
            'recent_1h': recent_df,             # Last 24 hours (1h candles)
            'resampled_4h': resampled_4h,         # 4h candles from recent data
            'resampled_1d': resampled_1d,         # 1d candle from recent data
            'current_price': current_price
        }

    except requests.HTTPError as he:
        print(f"❌ API HTTP Error: {he}. Check symbol, interval, or rate limits.")
        return None
    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        return None


# CELL 4: Indicator Helper Functions (NEW)

def fibonacci_levels(high, low):
    """
    Calculates key Fibonacci retracement levels based on a given high and low range.
    """
    diff = high - low
    return {
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
    }

def compute_indicators(df):
    """
    Computes a set of standard technical indicators using TA-Lib on the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'close', 'high', and 'low' columns.

    Returns:
        dict: A dictionary of indicator values, or None on failure.
    """
    try:
        # We assume the input DataFrame (e.g., recent_1h) is already cleaned and numeric.
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        indicators = {
            # Simple Moving Averages
            'sma50': SMA(closes, timeperiod=50)[-1] if len(closes) >= 50 else None,
            'sma200': SMA(closes, timeperiod=200)[-1] if len(closes) >= 200 else None,

            # Exponential Moving Averages
            'ema12': EMA(closes, timeperiod=12)[-1] if len(closes) >= 12 else None,
            'ema26': EMA(closes, timeperiod=26)[-1] if len(closes) >= 26 else None,

            # Momentum and Volatility Indicators
            'rsi14': RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else None,
            'atr14': ATR(highs, lows, closes, timeperiod=14)[-1] if len(closes) >= 14 else None,
            'bollinger_upper': BBANDS(closes, timeperiod=20)[0][-1] if len(closes) >= 20 else None,
            'bollinger_middle': BBANDS(closes, timeperiod=20)[1][-1] if len(closes) >= 20 else None,
            'bollinger_lower': BBANDS(closes, timeperiod=20)[2][-1] if len(closes) >= 20 else None,

            # Fibonacci Retracements
            'fib_levels': fibonacci_levels(max(highs), min(lows))
        }

        # Store to CSV
        # We need to flatten the nested 'fib_levels' dict before saving to CSV.

        # Create a flattened dictionary for CSV
        csv_indicators = {}
        for key, value in indicators.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    csv_indicators[f"fib_{sub_key.replace('%', '').replace('.', '_')}"] = sub_value
            else:
                csv_indicators[key] = value

        # Ensure the 'data' directory exists before writing
        if not os.path.exists('data'):
            os.makedirs('data')
            
        pd.DataFrame([csv_indicators]).to_csv('data/indicators.csv', index=False)

        print("✅ Indicators computed and saved to data/indicators.csv")
        return indicators

    except Exception as e:
        print(f"❌ Indicator error: {e}")
        return None

# CELL 7: Strategy Definition (PHASE 3 START - Corrected NameError)

def generate_signal(indicators, current_price, atr_multiplier=2.0):
    """
    Generates a trading signal (BUY/SELL/HOLD) based on multiple indicator confirmations
    and defines Stop Loss (SL) and Take Profit (TP) levels using ATR.
    Strategy:
    1. Base signal on SMA crossover (Long-term trend).
    2. Confirmation by RSI (Oversold/Overbought).
    3. Final confirmation by Bollinger Bands (Reversion to mean setup).
    4. Confidence boost if near Fibonacci 61.8% level.
    Args:
        indicators (dict): The dictionary of calculated indicator values.
        current_price (float): The current market price.
        atr_multiplier (float): Multiplier for ATR to calculate SL/TP levels (Risk Management).
    Returns:
        dict: The complete signal dictionary including risk management levels.
    """
    signal = 'HOLD'
    confidence = 0.5
    rationale = "No clear signal."
    stop_loss = None
    take_profit = None
    # Extract necessary indicator values, using .get for safety
    rsi = indicators.get('rsi14')
    atr = indicators.get('atr14')
    bb_upper = indicators.get('bollinger_upper')
    bb_middle = indicators.get('bollinger_middle') # Ensure bb_middle is extracted
    bb_lower = indicators.get('bollinger_lower')
    sma50 = indicators.get('sma50')
    sma200 = indicators.get('sma200')
    fib_61_8 = indicators.get('fib_levels', {}).get('61.8%')
    # Data validation for essential short-term components for SL/TP
    if any(val is None for val in [rsi, atr, bb_upper, bb_middle, bb_lower, fib_61_8]):
        rationale = "Insufficient short-term data (RSI, ATR, BBands, Fib levels) to generate a signal."
        return {
            "signal": signal,
            "confidence": 0.0,
            "rationale": rationale,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            'atr14': atr
        }
    # Check for proximity to Fibonacci 61.8% level (within 1.0 ATR distance)
    is_near_fib_61_8 = abs(current_price - fib_61_8) < atr * 1.0
    # --- Check for SMA Crossover availability (required for signal) ---
    is_sma_ready = sma50 is not None and sma200 is not None
    # --- BUY Logic ---
    if rsi < 60 and current_price <= bb_middle:
        signal = 'BUY'
        confidence = 0.75
        rationale = (
            f"Strong BUY signal: RSI is low at {rsi:.2f}, "
            f"and price ({current_price:,.2f}) is at or below the Bollinger middle ({bb_middle:,.2f})."
        )
        # Boost confidence if SMA bullish
        if is_sma_ready and sma50 > sma200:
            confidence = min(1.0, confidence + 0.1)
            rationale += f" Bullish SMA crossover (SMA50>{sma50:,.2f} > SMA200>{sma200:,.2f})."
        # Boost confidence if near Fibonacci support
        if is_near_fib_61_8:
            confidence = min(1.0, confidence + 0.1)
            rationale += f" Near Fibonacci 61.8% support ({fib_61_8:,.2f})."
        # Risk Management for LONG position (SL = 2 ATR below, TP = 3 ATR above)
        stop_loss = current_price - (atr * atr_multiplier)
        take_profit = current_price + (atr * atr_multiplier * 1.5) # 1.5R reward
    # --- SELL Logic ---
    elif rsi > 40 and current_price >= bb_upper:
        signal = 'SELL'
        confidence = 0.65
        rationale = (
            f"Strong SELL signal: RSI is high at {rsi:.2f}, "
            f"and price ({current_price:,.2f}) is at or above the Bollinger middle ({bb_middle:,.2f})."
        )
        # Boost confidence if SMA bearish
        if is_sma_ready and sma50 < sma200:
            confidence = min(1.0, confidence + 0.1)
            rationale += f" Bearish SMA crossover (SMA50<{sma50:,.2f} < SMA200<{sma200:,.2f})."
        # Boost confidence if near Fibonacci resistance
        if is_near_fib_61_8:
            confidence = min(1.0, confidence + 0.1)
            rationale += f" Near Fibonacci 61.8% resistance ({fib_61_8:,.2f})."
        # Risk Management for SHORT position (SL = 2 ATR above, TP = 3 ATR below)
        stop_loss = current_price + (atr * atr_multiplier)
        take_profit = current_price - (atr * atr_multiplier * 1.5) # 1.5R reward
    # --- HOLD Logic (Default or if SMA Crossover is not ready) ---
    else:
        if not is_sma_ready:
            rationale = "HOLD: Strategy requires SMA 50/200 crossover which is not available in the current window."
            confidence = 0.0 # Lowest confidence if core strategy can't run
        else:
            # More specific HOLD rationale using BB middle
            # This logic now correctly uses the extracted bb_middle variable.
            if current_price > bb_middle:
                rationale = f"HOLD: Conditions not met for BUY/SELL. Price ({current_price:,.2f}) is in the upper half of the Bollinger Bands (BB Mid: {bb_middle:,.2f}, RSI: {rsi:.2f})."
            else:
                rationale = f"HOLD: Conditions not met for BUY/SELL. Price ({current_price:,.2f}) is in the lower half of the Bollinger Bands (BB Mid: {bb_middle:,.2f}, RSI: {rsi:.2f})."
    return {
        "signal": signal,
        "confidence": confidence,
        "rationale": rationale,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        # Pass ATR value for use in position sizing calculation
        'atr14': atr
    }
    
# cell 11:
def get_testnet_balance():
    import time, hmac, hashlib, requests, json

    try:
        # We assume config is globally loaded, but we load it again just to be safe
        with open('config.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Error: config.json not found for balance check.")
        return None
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON in config.json.")
        return None

    api_key = config.get('binance_key')
    api_secret = config.get('binance_secret')

    if not api_key or not api_secret:
        print("❌ Error: API key or secret missing from config.json.")
        return None

    api_secret_encoded = api_secret.encode()

    base_url = 'https://testnet.binancefuture.com'
    endpoint = '/fapi/v2/balance'
    timestamp = int(time.time() * 1000)
    query_string = f'timestamp={timestamp}'
    signature = hmac.new(api_secret_encoded, query_string.encode(), hashlib.sha256).hexdigest()
    headers = {'X-MBX-APIKEY': api_key}
    url = f'{base_url}{endpoint}?{query_string}&signature={signature}'

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP/Request error fetching balance: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error fetching balance: {e}")
        return None


def calculate_position(signal_data, current_price, max_risk_pct=0.01, leverage=10, min_qty=0.0001):
    """
    Calculates the position size based on max risk percentage and the defined Stop Loss (SL) level.
    """
    if signal_data['signal'] == 'HOLD' or signal_data.get('stop_loss') is None:
        print("Position calculation skipped: Signal is HOLD or Stop Loss is missing.")
        return None

    try:
        # ✅ Use raw balance fetch
        balance_info = get_testnet_balance()
        if balance_info is None:
            raise ValueError("Failed to retrieve balance information.")

        usdt_item = next((item for item in balance_info if item['asset'] == 'USDT'), None)
        if usdt_item is None:
            raise ValueError("USDT balance not found in futures account.")

        balance = float(usdt_item['availableBalance'])

        # 🔢 Risk calculations
        risk_capital = balance * max_risk_pct
        stop_loss = signal_data['stop_loss']
        take_profit = signal_data['take_profit']
        risk_per_unit = abs(current_price - stop_loss)

        if risk_per_unit == 0 or risk_per_unit < 0.01:
            raise ValueError(f"Risk per unit is too small ({risk_per_unit:,.2f}), cannot calculate quantity safely.")

        quantity = max(min_qty, risk_capital / risk_per_unit)
        margin = (quantity * current_price) / leverage

        print(f"Account Balance: {balance:,.2f} USDT")
        print(f"Max Risk Capital: {risk_capital:,.2f} USDT")
        print(f"Risk Per Unit (Entry to SL): {risk_per_unit:,.2f}")

        return {
            'quantity': round(quantity, 4),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'margin_required': round(margin, 2),
            'estimated_risk_pct': max_risk_pct
        }

    except Exception as e:
        print(f"❌ Risk calc error: {e}")
        return None

# NEW FUNCTION: Order Execution
def execute_trade(symbol, trade_plan, leverage=10):
    """
    Executes a trade on the Binance Futures Testnet based on the final trade plan.

    1. Sets leverage.
    2. Places a MARKET order for entry.
    3. Places a corresponding STOP_MARKET/TAKE_PROFIT_MARKET order pair (O.C.O. strategy).
    """
    client = get_testnet_client()
    if not client:
        return {'status': 'error', 'message': 'Client initialization failed.'}

    signal = trade_plan.get('signal')
    quantity = trade_plan.get('quantity')
    stop_loss = trade_plan.get('stop_loss')
    take_profit = trade_plan.get('take_profit')
    current_price = trade_plan.get('price') # Assuming 'price' is passed in the plan

    if signal not in ['BUY', 'SELL'] or quantity <= 0:
        return {'status': 'error', 'message': f'Invalid signal or quantity: {signal}, {quantity}'}

    # Determine order parameters
    if signal == 'BUY':
        side = SIDE_BUY
        closing_side = SIDE_SELL
        stop_loss_type = 'STOP_MARKET'
        take_profit_type = 'TAKE_PROFIT_MARKET'
        # For a long position, SL is below entry, TP is above entry
        stop_price = round(stop_loss, 2)
        profit_price = round(take_profit, 2)
    else: # SELL (Short)
        side = SIDE_SELL
        closing_side = SIDE_BUY
        stop_loss_type = 'STOP_MARKET'
        take_profit_type = 'TAKE_PROFIT_MARKET'
        # For a short position, SL is above entry, TP is below entry
        stop_price = round(stop_loss, 2)
        profit_price = round(take_profit, 2)


    # 1. Set Leverage (Must be done before placing orders)
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"✅ Set {symbol} leverage to {leverage}x.")
    except Exception as e:
        print(f"❌ Error setting leverage: {e}")
        # Note: If leverage fails, the trade will still attempt to execute, but might fail due to insufficient margin.
        # We proceed to the order placement anyway.

    # 2. Place MARKET Entry Order
    entry_order_result = None
    try:
        # Note: Binance Futures uses the same side for market entry and closing orders
        # The quantity must be positive for both.
        entry_order_result = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"✅ MARKET Entry Order Placed: {signal} {quantity} {symbol} @ {current_price:,.2f}")
    except BinanceAPIException as e:
        print(f"❌ Binance API Error placing entry order: {e}")
        return {'status': 'error', 'message': f'Entry Order Failed: {e.code} - {e.message}'}
    except BinanceRequestException as e:
        print(f"❌ Binance Request Error placing entry order: {e}")
        return {'status': 'error', 'message': f'Entry Order Request Failed: {e}'}
    except Exception as e:
        print(f"❌ General Error placing entry order: {e}")
        return {'status': 'error', 'message': f'Entry Order Failed: {e}'}


    # 3. Place OCO (Stop Loss/Take Profit) Orders
    oco_result = None
    try:
        # Futures OCO is a bit different; we must place a Stop Market and Take Profit Market order separately.
        # Stop-Loss Order (Stop Market)
        stop_loss_order = client.futures_create_order(
            symbol=symbol,
            side=closing_side, # Opposite of the entry side
            type=stop_loss_type,
            quantity=quantity,
            # We use a stop price slightly better than the SL price to ensure execution
            stopPrice=stop_price,
            closePosition='true' # Ensure it closes the entire position upon trigger
        )

        # Take-Profit Order (Take Profit Market)
        take_profit_order = client.futures_create_order(
            symbol=symbol,
            side=closing_side, # Opposite of the entry side
            type=take_profit_type,
            quantity=quantity,
            # We use a stop price slightly better than the TP price to ensure execution
            stopPrice=profit_price,
            closePosition='true' # Ensure it closes the entire position upon trigger
        )
        print(f"✅ SL/TP Orders Placed: SL @ {stop_price:,.2f}, TP @ {profit_price:,.2f}")
        oco_result = {
            'stop_loss_order': stop_loss_order,
            'take_profit_order': take_profit_order
        }

    except BinanceAPIException as e:
        # If OCO fails, we MUST cancel the entry order to prevent an open, unprotected position
        print(f"❌ OCO Order Placement Failed: {e}. Attempting to cancel entry order...")
        try:
            # We attempt to cancel ALL open orders for the symbol for safety
            client.futures_cancel_all_open_orders(symbol=symbol)
            print("⚠️ All open orders cancelled to protect from unprotected position.")
        except Exception as cancel_e:
            print(f"❌ Failed to cancel orders: {cancel_e}")
        return {'status': 'error', 'message': f'SL/TP Orders Failed. Entry Order was cancelled. Error: {e.message}'}
    except Exception as e:
        print(f"❌ General Error placing OCO orders: {e}")
        return {'status': 'error', 'message': f'SL/TP Orders Failed: {e}'}


    return {
        'status': 'success',
        'entry_order': entry_order_result,
        'oco_orders': oco_result,
        'final_plan': trade_plan
    }


# CELL 9: Simple Backtest Function (NEW)

def simple_backtest(hist_df, indicators_func, min_data_points=200):
    """
    Upgraded backtest with PnL, equity curve, slippage/fees, and metrics.
    Skips HOLD signals and handles None SL/TP.
    """
    if not isinstance(hist_df, pd.DataFrame):
        raise TypeError("hist_df must be a pandas DataFrame")

    equity = 10000.0
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [equity]

    for i in range(min_data_points, len(hist_df)):
        window = hist_df.iloc[i-min_data_points:i]
        indicators = indicators_func(window)
        price = hist_df['close'].iloc[i]
        signal = generate_signal(indicators, price)

        # Skip if HOLD or no SL/TP
        if signal['signal'] == 'HOLD' or signal['stop_loss'] is None or signal['take_profit'] is None:
            equity_curve.append(equity)
            trades.append({"date": hist_df.index[i], "equity": equity, "signal": signal['signal'], "price": price})
            continue

        # Slippage (0.05%)
        if signal['signal'] == 'BUY':
            price *= 1.0005
        elif signal['signal'] == 'SELL':
            price *= 0.9995

        # Execution logic
        if position == 0 and signal['signal'] == 'BUY':
            position = 1
            entry_price = price
        elif position == 0 and signal['signal'] == 'SELL':
            position = -1
            entry_price = price
        elif position == 1 and (price <= signal['stop_loss'] or price >= signal['take_profit']):
            pnl = (price - entry_price) / entry_price * 10000  # 1 BTC position sim
            equity += pnl
            position = 0
        elif position == -1 and (price >= signal['stop_loss'] or price <= signal['take_profit']):
            pnl = (entry_price - price) / entry_price * 10000
            equity += pnl
            position = 0

        equity_curve.append(equity)
        trades.append({"date": hist_df.index[i], "equity": equity, "signal": signal['signal'], "price": price})

    df = pd.DataFrame(trades)
    df['equity_curve'] = equity_curve[1:]

    # Metrics
    final_equity = equity_curve[-1]
    cagr = (final_equity / 10000) ** (365 / len(df)) - 1 if final_equity > 0 else -1
    max_dd = ((pd.Series(equity_curve).cummax() - equity_curve) / pd.Series(equity_curve).cummax()).max()
    print(f"CAGR: {cagr:.1%}, Max DD: {max_dd:.1%}, Final Equity: ${final_equity:,.0f}")

    return df


# --- MAIN EXECUTION LOGIC ---

def main():
    print("--- 🚀 Starting Crypto Trading Strategy Application 🚀 ---")

    # 1. Setup
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory.")

    # 2. Data Fetch
    data = fetch_data()
    if data is None:
        print("🔴 Application terminated due to data fetch failure.")
        return

    # Export historical data for later use
    data['historical_klines'].to_csv('data/BTCUSDT_historical_1d.csv')
    print("✅ Historical klines saved to data/BTCUSDT_historical_1d.csv")

    # 3. Indicator Calculation (using recent 1H data)
    print("\n--- CALCULATING INDICATORS (Recent 1H) ---")
    indicators = compute_indicators(data['recent_1h'])
    if indicators is None:
        print("🔴 Application terminated due to indicator calculation failure.")
        return

    # Flatten indicators for CSV saving
    csv_indicators = {}
    for key, value in indicators.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                csv_indicators[f"fib_{sub_key.replace('%', '').replace('.', '_')}"] = sub_value
        else:
            csv_indicators[key] = value
    pd.DataFrame([csv_indicators]).to_csv('data/indicators.csv', index=False)
    print("✅ Indicators results saved to data/indicators.csv")

    # 4. Live Strategy
    current_price = data['current_price']
    trade_signal = generate_signal(indicators, current_price)

    print("\n--- LIVE TRADING SIGNAL ---")
    print(f"💰 Symbol: BTCUSDT")
    print(f"Current Price: {current_price:,.2f}")
    print(f"➡️ Signal: {trade_signal['signal']}")
    print(f"Confidence: {trade_signal['confidence']:.2f}")
    print(f"Stop Loss: {trade_signal['stop_loss']:,.2f}" if trade_signal['stop_loss'] is not None else "Stop Loss: N/A")
    print(f"Take Profit: {trade_signal['take_profit']:,.2f}" if trade_signal['take_profit'] is not None else "Take Profit: N/A")
    print(f"Rationale: {trade_signal['rationale']}")

    # 5. Position Calculation
    position_data = None
    if trade_signal['signal'] != 'HOLD' and trade_signal['stop_loss'] is not None:
        print("\n--- POSITION SIZING (Risk Management) ---")
        position_data = calculate_position(trade_signal, current_price)

        if position_data:
            print("✅ Position Calculated:")
            print(f"  Trade Direction: {trade_signal['signal']}")
            print(f"  Quantity: {position_data['quantity']:,.4f} BTC")
            print(f"  Entry Price: {current_price:,.2f}")
            print(f"  Margin Required (10x Lev): {position_data['margin_required']:,.2f} USDT")
            print(f"  Max Risk % per Trade: {position_data['estimated_risk_pct'] * 100:.2f}%")

            # 6. Trade Execution (Manual Test Run Only)
            print("\n--- EXECUTING TRADE (Test Run) ---")
            final_plan = {**trade_signal, **position_data, 'price': current_price}
            execution_result = execute_trade('BTCUSDT', final_plan)

            if execution_result['status'] == 'success':
                 print(f"✅ Trade Execution Successful. Order ID: {execution_result['entry_order']['orderId']}")
            else:
                 print(f"❌ Trade Execution Failed: {execution_result['message']}")

        else:
            print("⚠️ Position data is None (Calculation failed, see error above).")
    else:
        print("✅ Signal is HOLD or SL is missing. No position calculated or executed.")


    # 7. Backtest (using historical 1D data)
    print("\n--- RUNNING SIMPLE BACKTEST (1D Historical Data) ---")
    backtest_results = simple_backtest(data['historical_klines'], compute_indicators)
    backtest_results.to_csv('data/backtest.csv', index=False)
    print("✅ Backtest complete. Results saved to data/backtest.csv")
    print("\n--- ✅ Application Finished Successfully ✅ ---")



def ensemble_vote(rule_signal, llm_setup, ml_signal):
    """
    Combines signals from rules, LLM, and ML into a final ensemble decision.
    Requires at least 2/3 agreement and confidence >0.7, otherwise HOLD.
    Also checks ML liquidation risk.
    """
    signals = [rule_signal['signal'], llm_setup['action'], ml_signal['signal']]
    confidences = [rule_signal['confidence'], llm_setup['confidence'], ml_signal['confidence']]
    # Most common signal (majority vote)
    vote = max(set(signals), key=signals.count)
    # Require 2/3 agreement and min confidence
    if signals.count(vote) < 2 or min(confidences) < 0.7:
        vote = 'HOLD'
    # Liquidation safety override
    if ml_signal.get('liq_risk', False):
        vote = 'HOLD'
    return {
        'ensemble_signal': vote,
        'ensemble_confidence': round(sum(confidences)/3, 2),
        'details': {
            'rules': rule_signal,
            'llm': llm_setup,
            'ml': ml_signal
        }
    }

if __name__ == "__main__":
    main()
# IMPORTANT: This file assumes you have defined the necessary imports (e.g., from flask import...) 
# and helper functions (fetch_data, compute_indicators, generate_signal, calculate_position, execute_trade) 
# at the beginning of your script.

# ==============================================================================
# 1. WEBHOOK ROUTE: /fetch_data (Raw Signal Data for External Analysis/AI)
# ==============================================================================
@app.route('/fetch_data', methods=['POST'])
def fetch_data_webhook():
    """
    Receives a request, fetches market data and indicators, and returns
    them along with the initial signal generated by the strategy.
    (Used by Make.com Low Confidence Path to feed the LLM.)
    """
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    try:
        data_req = request.json
        # NOTE: Using .get() is crucial here to prevent KeyError if the sender fails to provide 'symbol'
        symbol = data_req.get('symbol', 'BTCUSDT').upper() 

        print(f"--- Received /fetch_data request for {symbol} (Raw Data) ---")

        # 1. Fetch Data
        market_data = fetch_data(symbol)
        if market_data is None or market_data.get('recent_1h') is None:
             return jsonify({"status": "error", "message": "Failed to fetch market data."}), 500

        recent_df = market_data['recent_1h']
        current_price = market_data['current_price']

        # 2. Compute Indicators
        indicators = compute_indicators(recent_df)

        # 3. Generate Signal (Initial)
        signal = generate_signal(indicators, current_price)

        # 4. Construct Payload
        payload = {
            'status': 'data_fetched',
            'symbol': symbol,
            'current_price': current_price,
            'indicators': indicators,
            'signal_data': signal
        }

        print(f"Initial Signal for AI refinement: {signal['signal']}")
        return jsonify(payload), 200

    except Exception as e:
        print(f"Error in /fetch_data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ==============================================================================
# 2. WEBHOOK ROUTE: /signal (Trade Planning Pipeline)
#    (The Audit Entry Point - Calculates Plan, DOES NOT Execute)
# ==============================================================================
@app.route('/signal', methods=['POST'])
def receive_signal():
    """
    Receives a signal request, runs the full strategy and position sizing
    pipeline, and returns the final, executable trade plan to Make.com for audit.
    """
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    webhook_data = request.json
    symbol = webhook_data.get('symbol', 'BTCUSDT').upper()
    signal_source = webhook_data.get('signal_source', 'TradingView/Direct')

    print(f"\n--- Processing /signal request for {symbol} from {signal_source} ---")

    try:
        # 1-2. Fetch Data & Compute Indicators
        market_data = fetch_data(symbol=symbol)
        if not market_data: return jsonify({'status': 'error', 'message': 'Failed to fetch market data.'}), 500

        indicators = compute_indicators(market_data['recent_1h'])
        if not indicators: return jsonify({'status': 'error', 'message': 'Failed to compute indicators.'}), 500
        current_price = market_data['current_price']

        # 3. Generate Signal
        trade_signal = generate_signal(indicators, current_price)

        # 4. Calculate Position Size (ONLY if a trade is signaled)
        position_data = None
        if trade_signal['signal'] not in ['HOLD', 'NONE']:
            position_data = calculate_position(trade_signal, current_price)

            if not position_data:
                # Execution blocked due to position sizing failure (e.g., zero risk)
                trade_signal['signal'] = 'HOLD'
                trade_signal['rationale'] = f"HOLD (Execution Blocked): {trade_signal['rationale']} but position calculation failed."
                print(f"⚠️ Calculation blocked execution. Final signal: HOLD.")
                final_response = trade_signal
            else:
                # Merge the signal and position data into the final plan
                final_plan = {**trade_signal, **position_data, 'price': current_price}
                final_response = final_plan
                print(f"✅ Trade Plan Ready: {final_plan['signal']} {final_plan['quantity']:,} @ {current_price:,.2f}")

        else:
            final_response = trade_signal
            print(f"⚠️ Final signal: HOLD (No trade generated).")

        # Return the full plan to Make.com for audit/execution decision
        return jsonify({
            'status': 'plan_generated',
            'symbol': symbol,
            'price': current_price,
            'trade_plan': final_response # This is the key payload for Make.com
        }), 200

    except Exception as e:
        print(f"Error in /signal: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ==============================================================================
# 3. WEBHOOK ROUTE: /execute_trade (Final Trade Execution)
#    (NEW ROUTE FOR MAKE.COM HTTP Module 11/7)
# ==============================================================================
@app.route('/execute_trade', methods=['POST'])
def execute_trade_webhook():
    """
    Receives the final audited trade plan from Make.com and executes the trade
    on Binance Testnet.
    """
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    try:
        final_plan = request.json
        # NOTE: Make.com might wrap the payload, so we must be defensive.
        # We assume the incoming JSON body IS the final_plan object.
        symbol = final_plan.get('symbol', 'BTCUSDT').upper() 

        print(f"\n--- EXECUTING AUDITED TRADE for {symbol} ---")

        # Check if the plan is valid for execution
        if final_plan.get('signal') in ['BUY', 'SELL']:
            # Call the helper function that handles the Binance API call
            execution_result = execute_trade(symbol, final_plan)

            if execution_result['status'] == 'success':
                print(f"✅ Execution Complete. Entry Order ID: {execution_result['entry_order']['orderId']}")
            else:
                print(f"❌ Execution Failed: {execution_result['message']}")

            # Return the plan plus the execution result back to Make.com
            return jsonify({**final_plan, 'execution_result': execution_result}), 200
        else:
            # Handle cases where the final plan is a HOLD/NONE signal
            return jsonify({'status': 'hold', 'message': 'Trade plan signal is HOLD or NONE. No execution required.'}), 200

    except Exception as e:
        print(f"Error in /execute_trade: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
        
# ==============================================================================
# NEW: MAIN PIPELINE ENDPOINT
# ==============================================================================
 

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    """
    Main pipeline endpoint: Chains data fetch, indicators, rules, LLM, ML, ensemble, position sizing, and execution.
    Triggered by Make.com scheduler.
    """
    print("\n" + "="*50)
    print("RUN_PIPELINE ENDPOINT CALLED!")
    print("Request headers:", request.headers)
    print("Request data:", request.data)
    print("Request JSON:", request.get_json())
    print("="*50 + "\n")
    
    if not request.is_json:
        print("ERROR: Not JSON")
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    data_req = request.json
    print(f"Received payload: {data_req}")
    symbol = data_req.get('symbol', 'BTCUSDT').upper()
    print(f"--- Running full pipeline for {symbol} ---")

    try:
        # 1. Fetch data
        market_data = fetch_data(symbol=symbol)
        if not market_data:
            return jsonify({'status': 'error', 'message': 'Data fetch failed'}), 500

        indicators = compute_indicators(market_data['recent_1h'])
        if not indicators:
            return jsonify({'status': 'error', 'message': 'Indicators failed'}), 500

        current_price = market_data['current_price']

        # 2. Rules layer
        rule_signal = generate_signal(indicators, current_price)

        # 3. LLM layer (use your existing run_ai_pipeline)
        compact_payload = {
            'symbol': symbol,
            'current': {'price': current_price},
            'indicators': indicators,
            'history': market_data['recent_1h'].to_dict(orient='records')
        }
        llm_result = run_ai_pipeline(compact_payload, get_testnet_client())
        llm_setup = llm_result.get('chosen_setup', {'action': 'HOLD', 'confidence': 0.0})

        # 4. ML layer (requires ml_strategy.py)
        ml_signal = get_ml_signal(market_data['recent_1h'], indicators)
        
        # Log ML signal to trades.csv
        logging.info(f"ML Signal: {ml_signal}")

        # 5. Ensemble
        ensemble = ensemble_vote(rule_signal, llm_setup, ml_signal)

        # 6. Position & Execution (if trade)
        final_plan = None
        if ensemble['ensemble_signal'] != 'HOLD':
            position_data = calculate_position(ensemble, current_price)
            if position_data:
                final_plan = {**ensemble, **position_data, 'price': current_price}
                execution_result = execute_trade(symbol, final_plan)
                final_plan['execution'] = execution_result
            else:
                final_plan = ensemble
                final_plan['ensemble_signal'] = 'HOLD'
                final_plan['reason'] = 'Position sizing failed'
        else:
            final_plan = ensemble

        # 7. Log and send to Make.com
        send_to_make_com(final_plan, config.get('make_webhook_url'))

        return jsonify({'status': 'success', 'final_plan': final_plan}), 200

    except Exception as e:
        print(f"Pipeline error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ==============================================================================
# 4. APPLICATION RUNNER
# ==============================================================================
if __name__ == '__main__':
    print("--- STARTING TRADING WEBHOOK SERVER ---")
    # Setting host='0.0.0.0' allows access from outside the container/machine
    # Ensure this is always the last block of code executed
    app.run(host='0.0.0.0', port=5000, debug=True)