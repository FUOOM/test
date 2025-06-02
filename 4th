import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import yfinance as yf

# === SETUP LOGGING ===
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# === CONSTANTS ===
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = mt5.TIMEFRAME_M5
BAR_COUNT = 1000
LOT_SIZE = 0.01
SL_POINTS = 100
TP_POINTS = 200
MAX_LOSS_PER_TRADE = 0.01   # 1% of balance
MAX_DRAWDOWN = 0.10         # 10% account drawdown

# === GLOBAL STATE ===
model_dict = {}         # Stores (model, scaler, features, last_train_time, last_bar_time) for each symbol
open_trades = {}        # Stores open trades per symbol

def get_mt5_data(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def get_yahoo_data(symbol, interval='5m', lookback_days=5):
    """
    Download historical OHLCV data from Yahoo Finance.
    symbol: e.g., 'DX-Y.NYB' (DXY Index), 'GC=F' (Gold Futures), 'EURUSD=X', etc.
    interval: '1m', '5m', '15m', '1h', '1d', etc.
    lookback_days: How many days of data to pull
    """
    end = datetime.now()
    start = end - pd.Timedelta(days=lookback_days)
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        return None
    df.index = df.index.tz_localize(None)  # Remove timezone for merging
    df.rename(columns=lambda x: f'yf_{x.lower()}', inplace=True)
    return df

def merge_external_with_mt5(mt5_df, yahoo_df):
    """
    Merge Yahoo Finance data into MT5 DataFrame by timestamp (forward-fill missing).
    Only keeps MT5 timestamps as index.
    """
    combined = mt5_df.join(yahoo_df, how='left')
    combined.fillna(method='ffill', inplace=True)
    return combined

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr = pd.concat([
        (high - low),
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def make_hft_features(df):
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['ma_fast'] = df['close'].rolling(window=20).mean()
    df['ma_slow'] = df['close'].rolling(window=50).mean()
    df['ma_diff'] = df['ma_fast'] - df['ma_slow']
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['atr'] = calculate_atr(df, 14)
    df.dropna(inplace=True)
    return df

def label_maker(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    return df

def train_xgb(df):
    features = ['rsi', 'ma_diff', 'atr', 'minute_sin', 'minute_cos', 'hour', 'dxy_close']
    X = df[features].values
    y = df['target'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, features

def get_account_info():
    info = mt5.account_info()
    if info is None:
        raise Exception("MT5: account info fetch failed!")
    return info

def check_risk_limits():
    info = get_account_info()
    balance = info.balance
    equity = info.equity
    drawdown = (balance - equity) / balance
    if drawdown > MAX_DRAWDOWN:
        logging.warning(f"Drawdown exceeded: {drawdown:.2%}. Trading halted.")
        return False
    return True

def execute_trade(symbol, prediction):
    info = mt5.symbol_info(symbol)
    if not info or not info.visible:
        mt5.symbol_select(symbol, True)
    point = info.point
    price = mt5.symbol_info_tick(symbol).ask if prediction == 1 else mt5.symbol_info_tick(symbol).bid
    sl = price - SL_POINTS * point if prediction == 1 else price + SL_POINTS * point
    tp = price + TP_POINTS * point if prediction == 1 else price - TP_POINTS * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if prediction == 1 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "python script order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    logging.info(f"Trade for {symbol} ({'BUY' if prediction==1 else 'SELL'}): {result}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade failed: {result.retcode} ({result.comment})")
        return False
    open_trades[symbol] = {"type": prediction, "ticket": result.order}
    return True

def close_trade(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for pos in positions:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
            "deviation": 10,
            "magic": 234000,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        logging.info(f"Closed trade for {symbol}: {result}")

def main_loop():
    if not mt5.initialize():
        print("MT5 init failed")
        logging.critical("MT5 failed to initialize")
        return

    try:
        prev_bar_times = {s: None for s in SYMBOLS}
        while True:
            if not check_risk_limits():
                break

            for symbol in SYMBOLS:
                print(f"\nProcessing {symbol}...")
                df = get_mt5_data(symbol, TIMEFRAME, BAR_COUNT)
                if df is None or df.empty:
                    logging.warning(f"No data for {symbol}")
                    continue

                # Fetch DXY index as example external data (change 'DX-Y.NYB' if needed)
                dxy_yahoo = get_yahoo_data('DX-Y.NYB', interval='5m', lookback_days=5)
                if dxy_yahoo is not None:
                    df = merge_external_with_mt5(df, dxy_yahoo[['yf_close']])
                    df.rename(columns={'yf_close': 'dxy_close'}, inplace=True)
                else:
                    df['dxy_close'] = np.nan

                df = make_hft_features(df)
                df = label_maker(df)
                last_bar_time = df.index[-1]

                # Only retrain model when a new bar arrives
                if symbol not in model_dict or prev_bar_times[symbol] != last_bar_time:
                    print(f"Retraining model for {symbol} (new bar detected)")
                    model, scaler, features = train_xgb(df)
                    model_dict[symbol] = (model, scaler, features, datetime.now(), last_bar_time)
                    prev_bar_times[symbol] = last_bar_time

                model, scaler, features, _, _ = model_dict[symbol]
                last = df.iloc[-1]
                X_pred = scaler.transform(last[features].values.reshape(1, -1))
                y_pred = int(model.predict(X_pred)[0])
                logging.info(f"{symbol} prediction: {y_pred} ({'BUY' if y_pred==1 else 'SELL'})")

                # Get open positions and manage them
                positions = mt5.positions_get(symbol=symbol)
                pos_type = None
                if positions and len(positions) > 0:
                    pos_type = positions[0].type  # Only first position considered here
                    # If signal reverses, close old and open new
                    if (y_pred == 1 and pos_type == mt5.ORDER_TYPE_SELL) or (y_pred == 0 and pos_type == mt5.ORDER_TYPE_BUY):
                        logging.info(f"Signal reversed on {symbol}, closing old trade")
                        close_trade(symbol)
                        execute_trade(symbol, y_pred)
                    else:
                        logging.info(f"Trade exists for {symbol}. No new trade.")
                else:
                    execute_trade(symbol, y_pred)

            time.sleep(60)  # Wait 1 minute for new bar
    except KeyboardInterrupt:
        print("Manual interrupt. Shutting down...")
    except Exception as e:
        logging.critical(f"Unhandled error: {e}")
    finally:
        mt5.shutdown()
        print("Bot stopped. See trading_bot.log for details.")

def backtest(symbol, train_size=0.7):
    print(f"Backtesting {symbol}...")
    df = get_mt5_data(symbol, TIMEFRAME, BAR_COUNT)
    if df is None or df.empty:
        print("No data")
        return
    dxy_yahoo = get_yahoo_data('DX-Y.NYB', interval='5m', lookback_days=5)
    if dxy_yahoo is not None:
        df = merge_external_with_mt5(df, dxy_yahoo[['yf_close']])
        df.rename(columns={'yf_close': 'dxy_close'}, inplace=True)
    else:
        df['dxy_close'] = np.nan
    df = make_hft_features(df)
    df = label_maker(df)
    features = ['rsi', 'ma_diff', 'atr', 'minute_sin', 'minute_cos', 'hour', 'dxy_close']
    X = df[features].values
    y = df['target'].values
    split = int(len(df) * train_size)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    print(f"Backtest accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    # Uncomment to backtest:
    # for s in SYMBOLS: backtest(s)

    # Live trading:
    main_loop()
