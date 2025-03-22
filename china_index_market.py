from rich.console import Console
import threading
import akshare as ak
import pandas as pd
import json
from retrying import retry
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.utils import *

console = Console()
print = console.print

# stock_zh_a_spot_em(https://akshare.akfamily.xyz/data/stock/stock.html#id10)
# stock_individual_info_em（https://akshare.akfamily.xyz/data/stock/stock.html#id8）
# 关键指标：stock_financial_abstract（https://akshare.akfamily.xyz/data/stock/stock.html#id191）
# 个数指标api：stock_a_indicator_lg（https://akshare.akfamily.xyz/data/stock/stock.html#id257）


def get_index_data(index_code):
    """
    获取指数数据
    """
    history_data = ak.stock_zh_index_daily_em(symbol=index_code)
    pass


def KAMA(prices, er_window=10, fast_period=5, slow_period=30):
    # 计算价格变化和波动性
    change = prices.diff()
    volatility = prices.rolling(window=er_window).apply(
        lambda x: abs(x.max() - x.min()), raw=False
    )

    # 计算效率比率 (Efficiency Ratio)
    er = change / volatility

    # 计算平滑因子
    sc_fast = 2 / (fast_period + 1)
    sc_slow = 2 / (slow_period + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow).fillna(0)

    # 初始化KAMA数组
    kama = pd.Series(index=prices.index)
    kama.iloc[0] = prices.iloc[0]  # 第一个值初始化为第一个价格

    # 迭代计算KAMA值
    for i in range(1, len(prices)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            prices.iloc[i] - kama.iloc[i - 1]
        )

    return kama.to_numpy()


class MultiprocessChina:
    def __init__(self):
        # 获取上证指数
        self.shangzheng = ak.stock_zh_index_daily_em(symbol="sh000001")
        self.shenzheng = ak.stock_zh_index_daily_em(symbol="sz399001")
        chuangye = ak.stock_zh_index_daily_em(symbol="sz399006")
        self.results = []  # 全局结果列表
        self.results_lock = threading.Lock()  # 用于保护结果的锁
        # self.rt_info = ak.stock_zh_index_spot_sina()  # 获取指数信息
        self.index_list = [
            {"sh000001": self.shangzheng},  # 上证指数
            # "sz399995",  # 基建工程
            # "sz399975",  # 证券公司
            # "sz399300",  # 沪深300
            # "sz399006",  # 创业板指
            # "sh000905",  # 中证500
            # "sz399296",  # 创成长
            # "HSI",  # 恒生指数
            "HSTECH",  # 恒生科技指数
        ]
        self.backtrack_year = 3  # 回溯年数
        self.day_offset = 60  # 回溯天数
        self.buy_sum = 4
        self.sell_sum = 4
        self.enable_backtest = False

        # 初始化资金记录
        self.yearly_investment = 0
        self.safe_marin_range = -0.15
        self.max_margin = -0.3
        self.min_margin = -0.05
        self.initial_investment_list = [200000]
        self.offset = 0  # 购买日期延后天数



    def fetch_data_for_code(self, code):
        data = get_index_data(str(code))
        if data:
            with self.results_lock:
                self.results.append(data)
        return data

    def backtrack(self):
        for index_code in self.index_list:
            if isinstance(index_code, str) and index_code.startswith("HS"):
                self.backtrack_bias(index_code, func=ak.stock_hk_index_daily_sina)
            else:
                self.backtrack_bias(index_code)
            print("\n")

    @staticmethod
    def get_status(c1, c2):
        # c1: 当前涨跌幅, c2: 当前成交量涨跌幅，量价关系
        if pd.isna(c1) or pd.isna(c2):
            return np.nan
        if c2 < -0.005:
            amount_p = "缩量"
        elif c2 >= 0.005:
            amount_p = "放量"
        else:
            amount_p = "平量"

        if c1 < -0.005:
            price_p = "下跌"
        elif c1 >= 0.005:
            price_p = "上涨"
        else:
            price_p = "平盘"
        return amount_p + price_p

    def get_buy_sell_status(
        self,
        current_pos,
        current_change,
        amount_change,
        bias,
        status,
        cumulative_return,
        buy_percentile,
        sell_percentile,
        kama,
        bias_kama,
        top_bottom_margin,
    ):
        bias_buy, buy_pos, change_buy, amount_buy = buy_percentile
        bias_sell, sell_pos, change_sell, amount_sell = sell_percentile
        top_margin, bottom_margin = top_bottom_margin
        buy_option = np.array(
            [
                status == "放量上涨",
                current_change < change_buy,
                amount_change < amount_buy,
                bias < bias_buy,
                bias_kama > 0.01,
                current_pos < bottom_margin,
                (bias < 0 and bias_kama > 0),
                current_pos < buy_pos,
            ],
            dtype=bool,
        )

        sell_option = np.array(
            [
                status != "放量上涨",
                current_change > change_sell,
                amount_change > amount_sell,
                bias > bias_sell,
                bias_kama < -0.01,
                current_pos > top_margin,
                (bias > 0 and bias_kama < 0),
                current_pos > sell_pos,
            ],
            dtype=bool,
        )
        buy_sum = sum(buy_option)
        sell_sum = sum(sell_option)

        if pd.isna(bias) or pd.isna(status):
            return "", buy_option, sell_option
        elif buy_sum > self.buy_sum:
            return "买入", buy_option, sell_option
        elif sell_sum > self.sell_sum:
            return "卖出", buy_option, sell_option
        else:
            return "", buy_option, sell_option

    def value_backtest(
        self,
        current_pos,
        current_change_list,
        amount_change_list,
        bias_list,
        status_list,
        kama_list,
        bias_kama_list,
    ):

        def operate(cash, stocks, index_price, percent, is_buy=True):
            if is_buy:
                # 买入时购买股票
                stocks += percent * cash / index_price
                cash -= percent * cash
            else:
                # 卖出时将股票转换为现金
                cash += stocks * percent * index_price
                stocks -= stocks * percent
            return cash, stocks

        def revserse_percentile(values_list, specific_value):
            # 计算特定值在列表中的百分位
            # 首先对列表进行排序
            sorted_values = np.sort(np.abs(values_list))
            specific_value = abs(specific_value)
            # 计算特定值在排序后列表中的位置
            position = np.sum(sorted_values < specific_value) / len(sorted_values)
            return position

        def buy_sell_policy(cash, stocks, safe_margin, index_price, operate_percent):
            if cumulative_return < safe_margin and stocks > 0 and stock_value[-1] < 0:
                # 卖出时将股票转换为现金，最高优先级，止损
                cash, stocks = operate(cash, stocks, index_price, 1.0, False)
                safe_margin = cumulative_return + self.safe_marin_range
                buy_sell_status.append("卖出")
                operate_percent = 1.0
                extra.update({"policy": "止损卖出"})
            elif action == "卖出" and stocks > 0:
                # 卖出时将股票转换为现金
                cash, stocks = operate(
                    cash,
                    stocks,
                    index_price,
                    operate_percent,
                    False,
                )
                safe_margin = cumulative_return + self.safe_marin_range
                buy_sell_status.append("卖出")
                operate_percent = operate_percent
                extra.update({"policy": "策略卖出"})
            elif action == "买入":
                cash, stocks = operate(
                    cash,
                    stocks,
                    index_price,
                    operate_percent,
                )
                buy_sell_status.append("买入")
                operate_percent = operate_percent
                extra.update({"policy": "策略买入"})
            else:
                buy_sell_status.append("")
                operate_percent = 0
                extra.update({"policy": "持有"})
            return cash, stocks, safe_margin

        # 初始化params
        stocks = 0
        stock_value = []
        stock_list = []
        safe_margin = self.safe_marin_range
        portfolio_value_list = self.initial_investment_list.copy()
        cash_list = self.initial_investment_list.copy()
        cumulative_returns = []
        buy_sell_status = []
        safe_margin_list = []
        cumulative_return = 0
        cumulative_diff = 0
        extra_list = []

        # 遍历上证指数和买卖列表来计算收益
        for i in range(len(current_pos)):
            extra = {}
            buy_percentile = [
                np.percentile(bias_list[: i + 1], 20),
                np.percentile(current_pos[: i + 1], 10),
                np.nanpercentile(current_change_list[: i + 1], 10),
                np.nanpercentile(amount_change_list[: i + 1], 10),
            ]
            sell_percentile = [
                np.percentile(bias_list[: i + 1], 95),
                np.percentile(current_pos[: i + 1], 80),
                np.nanpercentile(current_change_list[: i + 1], 90),
                np.nanpercentile(amount_change_list[: i + 1], 90),
            ]
            if i > 1:
                top_margin, bottom_margin = np.percentile(
                    current_pos[:i], 95
                ), np.percentile(current_pos[:i], 5)
            else:
                top_margin, bottom_margin = 0, 0

            cash = cash_list[-1]
            initial_investment = self.initial_investment_list[-1]
            index_price = current_pos[i - self.offset]

            if index_price < bottom_margin:
                self.safe_marin_range = max(self.max_margin, self.safe_marin_range * 2)
            elif index_price > top_margin:
                self.safe_marin_range = max(self.min_margin, self.safe_marin_range / 2)

            action, buy_option, sell_option = self.get_buy_sell_status(
                index_price,
                current_change_list[i - self.offset],
                amount_change_list[i - self.offset],
                bias_list[i - self.offset],
                status_list[i - self.offset],
                cumulative_return,
                buy_percentile,
                sell_percentile,
                kama_list[i - self.offset],
                bias_kama_list[i - self.offset],
                (top_margin, bottom_margin),
            )

            # operate_percent = revserse_percentile(bias_list, bias_list[i - self.offset])
            operate_percent = 1.0
            if i >= self.day_offset:
                if (i + 1) % 20 == 0:
                    cash += self.yearly_investment
                    initial_investment += self.yearly_investment
                cash, stocks, safe_margin = buy_sell_policy(
                    cash, stocks, safe_margin, index_price, operate_percent
                )
            # 计算投资组合的当前市值
            current_value = cash + stocks * index_price
            cumulative_return = (
                current_value - initial_investment
            ) / initial_investment
            # 根据cumulate_return进行调整safe margin
            if i > 1:
                cumulative_diff = cumulative_return - cumulative_returns[i - 1]
            if cumulative_diff > 0 and cumulative_return > safe_margin or stocks == 0:
                safe_margin += cumulative_diff

            extra.update(
                {
                    "buy_sum": sum(buy_option).tolist(),
                    "sell_sum": sum(sell_option).tolist(),
                    "buy_option": buy_option.tolist(),
                    "sell_option": sell_option.tolist(),
                    "safe_margin": to_percentage(safe_margin),
                    "operate_percent": to_percentage(operate_percent),
                    "investment": initial_investment,
                }
            )
            extra_list.append(extra)
            stock_list.append(stocks)
            cash_list.append(cash)
            portfolio_value_list.append(current_value)
            stock_value.append(stocks * index_price)
            self.initial_investment_list.append(initial_investment)
            cumulative_returns.append(cumulative_return)

        # 计算累计收益率
        cumulative_returns = [
            (value / initial_investment - 1)
            for value, initial_investment in zip(
                portfolio_value_list, self.initial_investment_list
            )
        ]
        # 计算最终收益率
        final_value = portfolio_value_list[-1]
        return_rate = cumulative_returns[-1] * 100

        # 输出结果
        print("最终市值:", final_value)
        print("最终投入:", self.initial_investment_list[-1])
        print("收益率: {:.2f}%".format(return_rate))
        with open("temp.txt", "a") as f:
            f.write(f"{return_rate}\n")
        print(
            "年平均收益率: {:.2f}%".format(
                return_rate / ((len(current_pos) - self.day_offset) / 250)
            )
        )
        print(
            "年化波动率: {:.2f}%".format(
                np.std(cumulative_returns) * np.sqrt(250) * 100
            )
        )
        print("最大回撤: {:.2f}%".format(np.min(cumulative_returns) * 100))
        print(
            "夏普比率: {:.2f}".format(
                return_rate / (np.std(cumulative_returns) * np.sqrt(250))
            )
        )
        print("交易次数:", sum([1 for x in buy_sell_status if x]))
        return (
            cumulative_returns[1:],
            portfolio_value_list[1:],
            stock_value,
            stock_list,
            buy_sell_status,
            extra_list,
        )

    def backtrack_bias(self, code: str, func=ak.stock_zh_index_daily_em):
        """
        获取指数的回溯指数10年60日均值偏离度
        """
        if isinstance(code, str):
            history_data = func(symbol=code)
        else:
            history_data = list(code.values())[0]
            code = list(code.keys())[0]

        print(f"history data days: {len(history_data)}")
        date = history_data["date"].iloc[-self.backtrack_year * 250 - self.day_offset :]
        try:
            amount = history_data["amount"].iloc[
                -self.backtrack_year * 250 - self.day_offset :
            ]
        except:
            amount = history_data["volume"].iloc[
                -self.backtrack_year * 250 - self.day_offset :
            ]
        amount_change = amount.pct_change()
        current_pos = history_data["close"].iloc[
            -self.backtrack_year * 250 - self.day_offset :
        ]
        current_change = current_pos.pct_change()
        mean_60day = (
            history_data["close"]
            .rolling(window=60)
            .mean()[-self.backtrack_year * 250 - self.day_offset :]
        )
        mean_30day = (
            history_data["close"]
            .rolling(window=30)
            .mean()[-self.backtrack_year * 250 - self.day_offset :]
        )
        mean_20day = (
            history_data["close"]
            .rolling(window=20)
            .mean()[-self.backtrack_year * 250 - self.day_offset :]
        )
        mean_120day = (
            history_data["close"]
            .rolling(window=120)
            .mean()[-self.backtrack_year * 250 - self.day_offset :]
        )

        # 计算偏离度百分位
        status_list = list(
            map(self.get_status, current_change.tolist(), amount_change.tolist())
        )
        bias_list = ((current_pos - mean_60day) / mean_60day).tolist()
        bias_30_list = ((current_pos - mean_30day) / mean_30day).tolist()
        bias_20_list = ((current_pos - mean_20day) / mean_20day).tolist()
        bias_120_list = ((current_pos - mean_120day) / mean_120day).tolist()

        kama_list = KAMA(current_pos)
        bias_kama_list = ((current_pos - kama_list) / kama_list).tolist()

        current_pos_list = current_pos.tolist()

        if self.enable_backtest:
            (
                daily_returns,
                portfolio_value,
                stock_value,
                stock_list,
                buy_sell_status,
                extra_list,
            ) = self.value_backtest(
                current_pos_list,
                current_change.tolist(),
                amount_change.tolist(),
                bias_list,
                status_list,
                kama_list.tolist(),
                bias_kama_list,
            )
        total_amount = (
            self.shangzheng["amount"]
            .iloc[-self.backtrack_year * 250 - self.day_offset :]
            .reset_index()
            + self.shenzheng["amount"]
            .iloc[-self.backtrack_year * 250 - self.day_offset :]
            .reset_index()
        )
        total_amount_change = total_amount.pct_change()

        final_report = {
            "日期": date.tolist(),
            "上证指数": self.shangzheng["close"]
            .iloc[-self.backtrack_year * 250 - self.day_offset :]
            .tolist(),
            "两市成交额": list(map(to_yi_round2, total_amount["amount"].tolist())),
            "两市成交额涨跌幅": list(
                map(to_percentage, total_amount_change["amount"].tolist())
            ),
            "当前点位": current_pos_list,
            "日涨跌幅": list(map(to_percentage, current_change.tolist())),
            "20日均线": mean_20day.tolist(),
            "20均线偏离度": list(map(to_percentage, bias_20_list)),
            "30日均线": mean_30day.tolist(),
            "30均线偏离度": list(map(to_percentage, bias_30_list)),
            "60日均线": mean_60day.tolist(),
            "60均线偏离度": list(map(to_percentage, bias_list)),
            "120日均线": mean_120day.tolist(),
            "120均线偏离度": list(map(to_percentage, bias_120_list)),
            "成交额": list(map(to_yi_round2, amount.tolist())),
            "成交额涨跌幅": list(map(to_percentage, amount_change.tolist())),
            "本日状态": status_list,
            # "买卖点位": buy_sell_status,
        }
        if self.enable_backtest:
            final_report.update(
                {
                    "收益率": list(map(to_percentage, daily_returns)),
                    "总资产": list(map(lambda x: round(x, 2), portfolio_value)),
                    "总持仓": list(map(lambda x: round(x, 2), stock_value)),
                    "总持仓数": list(map(lambda x: round(x, 2), stock_list)),
                    "附录": extra_list,
                }
            )
            # 提示买卖
            if buy_sell_status[-1]:
                print(f"当前指数{code}: {buy_sell_status[-1]}", style="bold red")
        # revert date
        final_report = {
            k: v[-self.backtrack_year * 250 :][::-1] for k, v in final_report.items()
        }
        # print({k: len(v) for k, v in final_report.items()})  # debugshow
        save_csv(final_report, None, f"./指数分析/{code}_index_bias.csv")


def test_api():
    data = ak.stock_hk_index_spot_em()
    data.to_csv("hk_index_data.csv")
    print(data)


if __name__ == "__main__":
    """
    ! 3. 设置指数的计算方式

    """
    # [上证 "sh000001", 中证500"sh000905", 创业板指"sz399006", 沪深300"sz399300"]
    mpc = MultiprocessChina()
    mpc.backtrack()
    # for year in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     for buy in range(0, 8):
    #         for sell in range(0, 8):
    #             print(f"buy: {buy}, sell: {sell}")
    #             mpc.buy_sum = buy
    #             mpc.sell_sum = sell
    #             mpc.backtrack()
    # test_api()
