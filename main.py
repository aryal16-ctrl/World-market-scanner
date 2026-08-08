import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()
BOT_PROFILE = "https://t.me/MastermindSTK_bot"

ASSET_PORTFOLIO = [
    # INDIA 🇮🇳
    {
        "symbol": "^NSEI",
        "name": "Nifty 50",
        "exchange": "NSE (Index)",
        "country": "India 🇮🇳",
    },
    {
        "symbol": "^NSEBANK",
        "name": "Bank Nifty",
        "exchange": "NSE (Index)",
        "country": "India 🇮🇳",
    },
    {
        "symbol": "RELIANCE.NS",
        "name": "Reliance Industries",
        "exchange": "NSE (Equities)",
        "country": "India 🇮🇳",
    },
    {
        "symbol": "HDFCBANK.NS",
        "name": "HDFC Bank",
        "exchange": "NSE (Equities)",
        "country": "India 🇮🇳",
    },
    {
        "symbol": "TCS.NS",
        "name": "TCS",
        "exchange": "NSE (Equities)",
        "country": "India 🇮🇳",
    },
    {
        "symbol": "INFY.NS",
        "name": "Infosys",
        "exchange": "NSE (Equities)",
        "country": "India 🇮🇳",
    },
    {
        "symbol": "ICICIBANK.NS",
        "name": "ICICI Bank",
        "exchange": "NSE (Equities)",
        "country": "India 🇮🇳",
    },
    # USA 🇺🇸
    {
        "symbol": "AAPL",
        "name": "Apple",
        "exchange": "NASDAQ",
        "country": "USA 🇺🇸",
    },
    {
        "symbol": "NVDA",
        "name": "NVIDIA",
        "exchange": "NASDAQ",
        "country": "USA 🇺🇸",
    },
    {
        "symbol": "TSLA",
        "name": "Tesla",
        "exchange": "NASDAQ",
        "country": "USA 🇺🇸",
    },
    {
        "symbol": "MSFT",
        "name": "Microsoft",
        "exchange": "NASDAQ",
        "country": "USA 🇺🇸",
    },
    {
        "symbol": "AMZN",
        "name": "Amazon",
        "exchange": "NASDAQ",
        "country": "USA 🇺🇸",
    },
    {
        "symbol": "JPM",
        "name": "JPMorgan Chase",
        "exchange": "NYSE",
        "country": "USA 🇺🇸",
    },
    # UNITED KINGDOM 🇬🇧
    {
        "symbol": "^FTSE",
        "name": "FTSE 100",
        "exchange": "LSE",
        "country": "United Kingdom 🇬🇧",
    },
    # JAPAN 🇯🇵
    {
        "symbol": "^N225",
        "name": "Nikkei 225",
        "exchange": "TSE",
        "country": "Japan 🇯🇵",
    },
]


def build_features(df):
  df = df.copy()
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  close = df["Close"].squeeze()
  df["Returns"] = close.pct_change()
  gain = df["Returns"].clip(lower=0).rolling(14).mean()
  loss = df["Returns"].clip(upper=0).abs().rolling(14).mean()
  df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
  df["EMA_20"] = close.ewm(span=20).mean()
  df["EMA_50"] = close.ewm(span=50).mean()
  df["Target"] = np.where(close.shift(-1) > close, 1, 0)
  return df.dropna()


def run_global_scanner():
  grouped_results = defaultdict(lambda: defaultdict(list))
  for asset in ASSET_PORTFOLIO:
    symbol = asset["symbol"]
    try:
      data = yf.download(
          symbol,
          period="60d",
          interval="1h",
          progress=False,
          auto_adjust=False,
      )
      if data.empty:
        continue

      df = build_features(data)
      features = ["Returns", "RSI", "EMA_20", "EMA_50"]

      if len(df) < 50:
        continue

      train_size = int(len(df) * 0.8)
      train, test = df.iloc[:train_size], df.iloc[train_size:]

      model = XGBClassifier(
          n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
      )
      model.fit(train[features], train["Target"])

      preds = model.predict(test[features])
      win_rate = accuracy_score(test["Target"], preds) * 100
      latest_prob = float(model.predict_proba(df[features].iloc[[-1]])[0][1])

      signal = (
          "BUY 🟢"
          if latest_prob > 0.60
          else ("SELL 🔴" if latest_prob < 0.40 else "NEUTRAL 🟡")
      )

      grouped_results[asset["country"]][asset["exchange"]].append({
          "symbol": symbol,
          "name": asset["name"],
          "win_rate": round(win_rate, 1),
          "signal": signal,
          "confidence": round(latest_prob * 100, 1),
      })
    except Exception as e:
      print(f"Error processing {symbol}: {e}")
  return grouped_results


def format_report(grouped_results):
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
  report = (
      f"🤖 Bot: {BOT_PROFILE}\n🌍 GLOBAL HOURLY MARKET SCANNER 📈\n🕒 Time:"
      f" {timestamp}\n═══════════════════════════\n\n"
  )
  for country, exchanges in grouped_results.items():
    report += f"REGION: {country.upper()}\n"
    for exchange, assets in exchanges.items():
      report += f"🏛 Exchange: {exchange}\n"
      for item in assets:
        report += (
            f"• {item['name']} ({item['symbol']})\n  ├ Win Rate:"
            f" {item['win_rate']:.1f}%\n  ├ Signal: {item['signal']}\n  └"
            f" Confidence: {item['confidence']:.1f}%\n"
        )
      report += "\n"
  return report


if __name__ == "__main__":
  data = run_global_scanner()
  if data:
    msg = format_report(data)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    res = requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=15)
    print("Delivered:" if res.status_code == 200 else f"Failed: {res.text}")
