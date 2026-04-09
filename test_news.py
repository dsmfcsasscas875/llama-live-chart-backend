import yfinance as yf
import json

def check_yfinance_news(symbol):
    ticker = yf.Ticker(symbol)
    news = ticker.news
    print(f"News for {symbol}:")
    for item in news[:5]:
        print(f"Title: {item.get('title')}")
        print(f"Publisher: {item.get('publisher')}")
        print(f"Link: {item.get('link')}")
        print("-" * 20)

if __name__ == "__main__":
    check_yfinance_news("AAPL")
    print("\n" + "="*40 + "\n")
    check_yfinance_news("BTC-USD")
