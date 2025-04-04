from rich.console import Console
import threading
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from datetime import datetime, timedelta
from src.utils import to_yi_round2

warnings.filterwarnings("ignore")

console = Console()
print = console.print

# Major US indices - 主要关注这三个指数
US_INDICES = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
}

def get_index_data(index_symbol, period="3y"):
    """
    Get historical data for a US market index using yfinance
    """
    try:
        # Download data from Yahoo Finance
        ticker = yf.Ticker(index_symbol)
        df = ticker.history(period=period)
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        })
        
        return df
    except Exception as e:
        print(f"Error fetching data for {index_symbol}: {str(e)}")
        return None

def KAMA(prices, er_window=10, fast_period=5, slow_period=30):
    """
    Kaufman's Adaptive Moving Average
    """
    # Calculate price changes and volatility
    change = prices.diff()
    volatility = prices.rolling(window=er_window).apply(
        lambda x: abs(x.max() - x.min()), raw=False
    )

    # Calculate Efficiency Ratio (ER)
    er = change / volatility

    # Calculate smoothing constants
    sc_fast = 2 / (fast_period + 1)
    sc_slow = 2 / (slow_period + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow).fillna(0)

    # Initialize KAMA array
    kama = pd.Series(index=prices.index)
    kama.iloc[0] = prices.iloc[0]

    # Calculate KAMA values
    for i in range(1, len(prices)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            prices.iloc[i] - kama.iloc[i - 1]
        )

    return kama.to_numpy()

class USIndexAnalyzer:
    def __init__(self):
        self.results = []
        self.results_lock = threading.Lock()
        self.backtrack_year = 3
        self.day_offset = 60
        self.buy_sum = 4
        self.sell_sum = 4
        self.enable_backtest = False

        # Investment parameters
        self.yearly_investment = 0
        self.safe_margin_range = -0.15
        self.max_margin = -0.3
        self.min_margin = -0.05
        self.initial_investment_list = [200000]
        self.offset = 0
        self.period = "10y"
        
        # 存储VIX数据
        self.vix_data = None

    @staticmethod
    def get_status(price_change, volume_change):
        """
        Determine market status based on price and volume changes
        """
        if pd.isna(price_change) or pd.isna(volume_change):
            return np.nan
            
        if volume_change < -0.005:
            volume_status = "缩量"
        elif volume_change >= 0.005:
            volume_status = "放量"
        else:
            volume_status = "平量"

        if price_change < -0.005:
            price_status = "下跌"
        elif price_change >= 0.005:
            price_status = "上涨"
        else:
            price_status = "平盘"
            
        return f"{volume_status}{price_status}"

    def analyze_index(self, index_symbol):
        """
        Analyze a specific US market index
        """
        print(f"\nAnalyzing {US_INDICES.get(index_symbol, index_symbol)}...")
        
        # 获取指数数据
        history_data = get_index_data(index_symbol, self.period)
        if history_data is None or history_data.empty:
            print(f"Failed to get data for {index_symbol}")
            return
            
        # 确保有VIX数据
        if self.vix_data is None:
            print("Fetching VIX data first...")
            vix_ticker = yf.Ticker("^VIX")
            self.vix_data = vix_ticker.history(period=self.period)
            self.vix_data = self.vix_data.rename(columns={'Close': 'close'})

        # 计算基本指标
        current_pos = history_data["close"]
        current_change = current_pos.pct_change()
        volume = history_data["volume"]
        volume_change = history_data["volume"].pct_change()
        
        # 计算移动平均线
        mean_20day = current_pos.rolling(window=20).mean()
        mean_30day = current_pos.rolling(window=30).mean()
        mean_60day = current_pos.rolling(window=60).mean()
        mean_120day = current_pos.rolling(window=120).mean()

        # 计算偏差指标
        bias_20 = ((current_pos - mean_20day) / mean_20day).fillna(0)
        bias_30 = ((current_pos - mean_30day) / mean_30day).fillna(0)
        bias_60 = ((current_pos - mean_60day) / mean_60day).fillna(0)
        bias_120 = ((current_pos - mean_120day) / mean_120day).fillna(0)

        # 计算KAMA
        kama = KAMA(current_pos)
        bias_kama = ((current_pos - pd.Series(kama, index=current_pos.index)) / pd.Series(kama, index=current_pos.index)).fillna(0)

        # 获取市场状态
        status_list = [self.get_status(pc, vc) for pc, vc in zip(current_change, volume_change)]
        
        # 获取VIX数据并合并
        vix_close = self.vix_data["close"]
        # 确保日期索引匹配
        vix_close = vix_close[:len(current_pos)]
        
        # 准备最终报告
        final_report = {
            "Date": current_pos.index.strftime('%Y-%m-%d').tolist(),
            "Close": current_pos.tolist(),
            "Daily Change": [f"{x:.2%}" for x in current_change.tolist()],
            "Volume": list((map(to_yi_round2, volume.tolist()))),
            "Volume Change": [f"{x:.2%}" for x in volume_change.tolist()],
            "20-day MA": mean_20day.tolist(),
            "20-day Bias": [f"{x:.2%}" for x in bias_20.tolist()],
            "30-day MA": mean_30day.tolist(),
            "30-day Bias": [f"{x:.2%}" for x in bias_30.tolist()],
            "60-day MA": mean_60day.tolist(),
            "60-day Bias": [f"{x:.2%}" for x in bias_60.tolist()],
            "120-day MA": mean_120day.tolist(),
            "120-day Bias": [f"{x:.2%}" for x in bias_120.tolist()],
            "KAMA": kama.tolist(),
            "KAMA Bias": [f"{x:.2%}" for x in bias_kama.tolist()],
            "VIX": vix_close.tolist(),  # 添加VIX作为单独的列
            "Market Status": status_list
        }

        # 保存到CSV
        df = pd.DataFrame(final_report)
        # 反转数据框，使最新的数据在第一行
        df = df.iloc[::-1].reset_index(drop=True)
        df.to_csv(f"./index_analysis/{index_symbol}_analysis.csv", index=False)
        print(f"Analysis saved for {US_INDICES.get(index_symbol, index_symbol)}")

    def run_analysis(self):
        """
        Run analysis for all major US indices
        """
        # 首先获取VIX数据
        print("Fetching VIX data first...")
        self.analyze_index("^VIX")
        
        # 然后分析其他指数
        for index_symbol in ["^GSPC", "^IXIC"]:
            self.analyze_index(index_symbol)

def main():
    """
    Main function to run the analysis
    """
    try:
        # Create output directory
        import os
        os.makedirs("./index_analysis", exist_ok=True)

        # Initialize and run analyzer
        analyzer = USIndexAnalyzer()
        analyzer.run_analysis()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 