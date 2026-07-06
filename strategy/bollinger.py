import yfinance as yf
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import sys

# Custom rolling mean
def rolling_mean(x, window):
    result = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(window - 1, len(x)):
        result[i] = np.mean(x[i - window + 1:i + 1])
    return result

# Custom rolling std
def rolling_std(x, window):
    result = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(window - 1, len(x)):
        result[i] = np.std(x[i - window + 1:i + 1])
    return result

# ✅ Strategy class
class BollingerBandsStrategy(Strategy):
    def init(self):
        close = self.data.Close
        window = 20
        self.ma = self.I(lambda x: rolling_mean(x, window), close)
        self.std = self.I(lambda x: rolling_std(x, window), close)
        self.upper = self.I(lambda ma, std: ma + 2 * std, self.ma, self.std)
        self.lower = self.I(lambda ma, std: ma - 2 * std, self.ma, self.std)

    def next(self):
        price = self.data.Close[-1]
        ma, upper, lower = self.ma[-1], self.upper[-1], self.lower[-1]

        if np.isnan(ma) or np.isnan(upper) or np.isnan(lower):
            return

        if price < lower:
            self.buy()
        elif price > upper:
            self.sell()
        elif self.position:
            self.position.close()

# ✅ Backtest runner
def run_strategy(ticker="AAPL"):
    print(f"\n📈 Running strategy on {ticker}...\n")
    df = yf.download(ticker, start="2024-01-01", end="2025-05-06", auto_adjust=True)

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)  # Get the first level of the MultiIndex

    # Ensure proper structure
    try:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except KeyError as e:
        print(f"❌ KeyError: {e}")
        return

    df.index.name = 'Date'

    if df.empty:
        print(f"❌ No data for {ticker}.")
        return  

    bt = Backtest(df, BollingerBandsStrategy, cash=10000, commission=0.002)
    stats = bt.run()

    # Print key stats
    print(stats[['Start', 'End', 'Duration', 'Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]', '# Trades']])

    # Export results
    stats['_trades'].to_csv(f"{ticker}_trades.csv", index=False)
    stats.to_frame().to_csv(f"{ticker}_backtest_summary.csv")
    print(f"\n✅ Exported: {ticker}_trades.csv and {ticker}_backtest_summary.csv")

    bt.plot()

# ✅ Main entry
if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_strategy(ticker)
