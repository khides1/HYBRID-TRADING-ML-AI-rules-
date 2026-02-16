import json
import os
import time
from google import genai
from typing import List, Dict, Union, Any, Optional
import requests

# --- CONFIGURATION & INITIALIZATION ---
def load_config(file_path="config.json") -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Config file not found at {file_path}.")
        return {}
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON in {file_path}")
        return {}

config = load_config()

GEMINI_API_KEY = config.get("gemini_key") or os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: 'gemini_key' missing. Using mock mode.")

MAKE_COM_WEBHOOK_URL = config.get("make_webhook_url", "https://hook.eu2.make.com/4xw9fjdt6kzafyyotwar8h6kw3negib8")

# --- DATA STRUCTURES ---
TradeSetup = Dict[str, Union[str, float, int, Dict[str, Any]]]

# --- VALIDATION HELPER ---
def validate_price_range(setup: TradeSetup, current_price: float, tolerance_percent: float = 1.0) -> bool:
    try:
        entry = float(setup.get('entry', 0))
        stop_loss = float(setup.get('stop_loss', 0))
        take_profit = float(setup.get('take_profit', 0))
        action = setup.get('action')
        if not all(isinstance(p, (int, float)) for p in [entry, stop_loss, take_profit]):
            return False
        min_price = current_price * (1 - tolerance_percent / 100)
        max_price = current_price * (1 + tolerance_percent / 100)
        is_in_range = (min_price <= entry <= max_price and
                       min_price <= stop_loss <= max_price and
                       min_price <= take_profit <= max_price)
        if not is_in_range:
            return False
        if action == 'BUY':
            is_logic_valid = (entry > stop_loss) and (take_profit > entry)
        elif action == 'SELL':
            is_logic_valid = (entry < stop_loss) and (take_profit < entry)
        elif action == 'HOLD':
            is_logic_valid = True
        else:
            is_logic_valid = False
        return is_logic_valid
    except Exception as e:
        print(f"Price validation error: {e}")
        return False

# --- MOCK FUNCTIONS ---
def mock_analyze_market(compact_payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = compact_payload['symbol']
    price = compact_payload['current']['price']
    summary = "Consolidation above $95k" if price > 95000 else "Bearish pressure, bounce off support"
    return {
        "market_summary": summary,
        "key_levels": {"support": 95000.0, "resistance": 96500.0, "pivot": 95750.0},
        "indicators_summary": {"RSI": "Oversold", "MACD": "Bearish crossover"},
        "symbols_analyzed": [symbol],
    }

def mock_generate_trade_setups(analysis: Dict[str, Any], current_price: float) -> List[TradeSetup]:
    symbol = analysis['symbols_analyzed'][0]
    return [
        {
            "symbol": symbol, "action": "BUY", "order_type": "LIMIT",
            "entry": current_price * 0.9995, "stop_loss": current_price * 0.992,
            "take_profit": current_price * 1.015, "risk_level": "med",
            "confidence": 0.90, "rationale": "Bullish divergence"
        },
        {
            "symbol": symbol, "action": "SELL", "order_type": "MARKET",
            "entry": current_price * 1.0005, "stop_loss": current_price * 1.008,
            "take_profit": current_price * 0.990, "risk_level": "low",
            "confidence": 0.65, "rationale": "Rejection at resistance"
        }
    ]

# --- MODULE A: ANALYZE MARKET (Live Gemini) ---
def analyze_market(compact_payload: Dict[str, Any], model: str = "gemini-1.5-flash") -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        print("Gemini key missing. Using mock analysis.")
        return mock_analyze_market(compact_payload)

    try:
        print("--- Running live Gemini analysis ---")
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [
                    {"text": "You are a sophisticated cryptocurrency market analyst. "
                     "Analyze the provided market data (symbol, current price, indicators, history) "
                     "and return ONLY a JSON object with: "
                     "market_summary (string), key_levels (object: {support: float, resistance: float, pivot: float}), "
                     "indicators_summary (object: {RSI: string, MACD: string}), symbols_analyzed (array of strings). "
                     "Do not include any other text."},
                    {"text": "Market Data:\n" + json.dumps(compact_payload, indent=2)}
                ]}
            ]
        )
        text = response.text.strip()
        analysis = json.loads(text)
        return analysis
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return {"error": "gemini_analysis_failed", "message": str(e)}

# --- MODULE B: GENERATE TRADE SETUPS (Live Gemini) ---
def generate_trade_setups(analysis: Dict[str, Any], current_price: float, model: str = "gemini-1.5-flash", max_retries: int = 3) -> Union[List[TradeSetup], Dict[str, str]]:
    if not GEMINI_API_KEY:
        print("Gemini key missing. Using mock setups.")
        return mock_generate_trade_setups(analysis, current_price)

    try:
        print(f"Attempting live Gemini setups...")
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [
                    {"text": "You are a disciplined crypto trading strategist. "
                     "Based on the analysis JSON, propose 0-5 trade setups. "
                     "Entry, stop_loss, and take_profit prices MUST be close to current_price. "
                     "Return ONLY a JSON object with key 'setups': array of objects with: "
                     "symbol, action (BUY|SELL|HOLD), order_type (MARKET|LIMIT), "
                     "entry (float), stop_loss (float), take_profit (float), "
                     "risk_level (low|med|high), confidence (0-1), rationale (string). "
                     "Do not include any other text."},
                    {"text": "Analysis:\n" + json.dumps(analysis) + f"\nCurrent_price: {current_price}"}
                ]}
            ]
        )
        text = response.text.strip()
        raw_data = json.loads(text)
        setups = raw_data.get('setups', [])
        if not isinstance(setups, list):
            raise ValueError("No 'setups' list in response.")
        validated_setups = [s for s in setups if validate_price_range(s, current_price, 1.0)]
        return validated_setups
    except Exception as e:
        print(f"Gemini setups error: {e}")
        return mock_generate_trade_setups(analysis, current_price)

# --- MODULE C: VALIDATE RISK AND QUANTITY ---
def validate_risk_and_quantity(entry: float, stop_loss: float, balance: float, max_risk_pct: float = 0.01, leverage: int = 10, min_qty: float = 0.0001) -> Dict[str, Union[str, float, int]]:
    risk_capital = balance * max_risk_pct
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit < 1e-8:
        return {"status": "ERROR_ZERO_RISK", "quantity": 0.0, "margin_required": 0.0, "estimated_risk_pct": 0.0}
    quantity = risk_capital / risk_per_unit
    if quantity < min_qty:
        return {"status": "SKIP_TOO_SMALL", "quantity": round(quantity, 8), "margin_required": 0.0, "estimated_risk_pct": max_risk_pct}
    margin_required = (quantity * entry) / leverage
    return {
        "status": "VALID",
        "quantity": round(quantity, 8),
        "margin_required": round(margin_required, 3),
        "estimated_risk_pct": max_risk_pct
    }

# --- MAIN ORCHESTRATION FUNCTION ---
def run_ai_pipeline(compact_payload: Dict[str, Any], binance_client: Any) -> Dict[str, Any]:
    analysis = analyze_market(compact_payload)
    if "error" in analysis:
        return {"status": "analysis_error", "detail": analysis}

    current_price = compact_payload['current']['price']

    setups = generate_trade_setups(analysis, current_price)
    if isinstance(setups, dict) and "error" in setups:
        return {"status": "setups_error", "detail": setups}

    if not setups:
        return {"status": "no_setups"}

    chosen = max(setups, key=lambda s: s.get('confidence', 0.0))
    print(f"Chosen setup (Conf: {chosen.get('confidence', 'N/A')}): {chosen.get('action')} @ {chosen.get('entry')}")

    try:
        bal_resp = binance_client.futures_account_balance()
        usdt_item = next((i for i in bal_resp if i['asset'] == 'USDT'), None)
        balance = float(usdt_item['availableBalance']) if usdt_item and 'availableBalance' in usdt_item else 1000.0
    except Exception as e:
        print(f"Balance fetch error: {e}. Using default $1000.")
        balance = 1000.0

    size = validate_risk_and_quantity(
        chosen.get('entry', current_price),
        chosen.get('stop_loss', current_price * 0.99),
        balance
    )
    if size.get('status') != 'VALID':
        return {"status": "position_sizing_failed", "detail": size}

    trade_plan = {
        "status": "TRADE_PLAN_READY",
        "analysis": analysis,
        "chosen_setup": chosen,
        "position": size
    }
    return trade_plan

# --- WEBHOOK SENDER ---
def send_to_make_com(payload: Dict[str, Any], url: str):
    if not url:
        print("CRITICAL: MAKE_COM_WEBHOOK_URL not configured.")
        return
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries}: Sending to Make.com")
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code in [200, 202, 302]:
                print(f"✅ Make.com received (Status: {response.status_code})")
                return True
            else:
                print(f"❌ HTTP Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    print("--- Send failed after retries ---")
    return False

# --- MOCK CLIENT ---
class MockBinanceClient:
    def __init__(self, mock_balance: float = 10000.00):
        self._balance = mock_balance
    def futures_account_balance(self):
        return [{"asset": "USDT", "availableBalance": str(self._balance)}]
    def futures_create_order(self, **params):
        return {"orderId": int(time.time() * 1000), "status": "NEW"}

if __name__ == "__main__":
    MOCK_SYMBOL = "BTC/USDT"
    MOCK_PRICE = 96500.00
    MOCK_PAYLOAD = {
        "symbol": MOCK_SYMBOL,
        "current": {"price": MOCK_PRICE, "timestamp": 1678886400000},
        "indicators": {"RSI": 35.0, "MACD": -50.0, "BB_W": 2.5},
        "history": [{"t": 1678886400000, "o": 96500.0, "h": 96510.5, "l": 96490.1, "c": 96495.0, "v": 150.0}]
    }
    mock_binance = MockBinanceClient()
    results = run_ai_pipeline(MOCK_PAYLOAD, mock_binance)
    print("\n--- COMPLETE TRADE PLAN ---")
    print(json.dumps(results, indent=4))
    if results.get('status') == 'TRADE_PLAN_READY':
        send_to_make_com(results, MAKE_COM_WEBHOOK_URL)
    else:
        print(f"Skipping send. Status: {results.get('status')}")