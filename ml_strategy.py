import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any

# --- CONFIGURATION ---
MODEL_PATH = 'models/xgb_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Load pre-trained model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ ML model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ML model/scaler: {e}")
    model = None
    scaler = None

def extract_ml_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts ML features from a kline DataFrame (same as in notebook 02).
    Returns a dict of features for the last row.
    """
    if len(df) < 20:
        return {"error": "Insufficient data for feature extraction"}

    df = df.copy()
    # Returns and log returns
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    # Volatility
    df['volatility'] = df['return'].rolling(20).std()
    # Volume change
    df['volume_change'] = df['volume'].pct_change()
    # Cumulative Volume Delta (CVD) approximation
    df['delta'] = df['close'] - df['open']
    df['cvd'] = (df['volume'] * df['delta']).cumsum()
    # Order imbalance (proxy)
    df['imbalance'] = df['volume'].rolling(10).mean() / df['volume'].rolling(20).mean()
    # Liquidation risk proxy
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    df['liq_risk'] = (df['close'] - df['low'].rolling(50).min()) / df['atr']
    # Lags
    for lag in [1, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

    # Drop NaN rows
    df.dropna(inplace=True)

    if df.empty:
        return {"error": "No valid features after cleaning"}

    # Return features for the last row
    feature_cols = [
        'return', 'log_return', 'volatility', 'volume_change', 'cvd', 'imbalance', 'liq_risk',
        'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10',
        'volume_lag_1', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10'
    ]

    return df[feature_cols].iloc[-1].to_dict()

def get_ml_signal(df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts trading signal using the trained XGBoost model.
    Returns dict with signal, confidence, liq_risk.
    """
    if model is None or scaler is None:
        return {"signal": "HOLD", "confidence": 0.0, "liq_risk": False, "error": "Model not loaded"}

    try:
        # Extract features
        features_dict = extract_ml_features(df)
        if "error" in features_dict:
            return {"signal": "HOLD", "confidence": 0.0, "liq_risk": False, "error": features_dict["error"]}

        # Convert to DataFrame for scaling
        features_df = pd.DataFrame([features_dict])

        # Scale features
        scaled_features = scaler.transform(features_df)

        # Predict probabilities (model outputs [prob_SELL, prob_BUY, prob_HOLD])
        prob = model.predict_proba(scaled_features)[0]

        # Map classes: 0 = SELL, 1 = BUY, 2 = HOLD (from your training)
        sell_prob = prob[0]
        buy_prob = prob[1]
        hold_prob = prob[2]

        # Determine signal
        if buy_prob > 0.65:
            signal = "BUY"
            confidence = buy_prob
        elif sell_prob > 0.65:
            signal = "SELL"
            confidence = sell_prob
        else:
            signal = "HOLD"
            confidence = max(sell_prob, buy_prob, hold_prob)

        # Liquidation risk from feature
        liq_risk_prob = features_dict.get('liq_risk', 0.0)
        liq_risk = liq_risk_prob > 0.7  # High risk threshold

        return {
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "liq_risk": liq_risk,
            "liq_risk_prob": round(float(liq_risk_prob), 2)
        }

    except Exception as e:
        print(f"ML prediction error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "liq_risk": False, "error": str(e)}

if __name__ == "__main__":
    df = pd.read_csv('data/btcusdt_1h.csv', index_col='open_time', parse_dates=True)
    indicators = {"rsi14": 50, "atr14": 100}  # Dummy
    result = get_ml_signal(df, indicators)
    print("ML Signal:", result)