from rich.progress import Progress
from rich import print
import os
import akshare as ak
import pandas as pd
import time
from retrying import retry
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading
import warnings

warnings.filterwarnings("ignore")

from src.utils import *
from src.metrics import *
from src.reports import *

# stock_zh_a_spot_em(https://akshare.akfamily.xyz/data/stock/stock.html#id10)
# stock_individual_info_em（https://akshare.akfamily.xyz/data/stock/stock.html#id8）
# 关键指标：stock_financial_abstract（https://akshare.akfamily.xyz/data/stock/stock.html#id191）
# 个数指标api：stock_a_indicator_lg（https://akshare.akfamily.xyz/data/stock/stock.html#id257）


@retry(wait_fixed=3000, stop_max_attempt_number=2)
def get_stock_data(code: str, rt_info: pd.DataFrame, sy_info: pd.DataFrame):
    ######################### basic information #########################
    # 获取个股最新财务指标数据
    rt_stock_info = rt_info[rt_info["代码"] == code]

    # 获取个股商誉
    sy_stock_info = sy_info[sy_info["股票代码"] == code]
    if sy_stock_info.empty:
        shangyu_report = {"商誉": 0, "商誉占比": 0}
    else:
        shangyu_report = {
            "商誉": to_yi_round2(sy_stock_info["商誉"].values[0]),
            "商誉占比": round(sy_stock_info["商誉占净资产比例"].values[0], 4),
        }

    name = rt_stock_info["名称"].values[0]
    if "退市" in name:
        print(f"{code} 已退市")
        return None
    rt_price = rt_stock_info["最新价"].values[0]
    if np.isnan(rt_price):
        print(f"{code} 无最新价: {rt_price=}")
        return None
    PB = rt_stock_info["市净率"].values[0]
    stock_info = ak.stock_individual_info_em(symbol=code)

    # 基础信息
    name = stock_info[stock_info["item"] == "股票简称"]["value"].values[0]
    sector = stock_info[stock_info["item"] == "行业"]["value"].values[0]
    total_value = stock_info[stock_info["item"] == "总市值"]["value"].values[0]
    total_volume = stock_info[stock_info["item"] == "总股本"]["value"].values[0]
    if total_volume == "-":
        print(f"{code} 总股本未揭露")
        return None
    if total_value == "-":
        total_value = total_volume * rt_price

    # 财报信息
    financial_data = ak.stock_financial_abstract_ths(symbol=code)
    fa_date = int(financial_data["报告期"].values[0].replace("-", ""))
    start_date = fa_date - 5
    try:
        fa_price = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=str(start_date),
            end_date=str(fa_date),
        )
        fa_value = total_volume * float(fa_price["收盘"].values[0])
    except:
        return None
    financial_report = load_ths_report(financial_data, total_volume)
    benefit_report = get_benefits_report(code)
    debts_report = get_debts_report(code)
    cash_report = get_cash_report(code)
    out_dict = {
        "股票名称": name,
        "代码": str(code),
        "行业": sector,
        "总市值": to_yi_round2(total_value),
        "财报季市值": to_yi_round2(fa_value),
        "当前价格": rt_price,
        "总股本": to_yi_round2(total_volume),
    }
    out_dict.update(financial_report)
    out_dict.update(benefit_report)
    out_dict.update(shangyu_report)
    out_dict.update(cash_report)
    out_dict.update(debts_report)

    ######################### advance metrices #########################
    # 估值指标
    PS = round(out_dict["总市值"] / out_dict["营业总收入-TTM"], 2)
    PE_TTM = round(
        out_dict["总市值"] / (out_dict["扣非净利润-TTM"] - out_dict["商誉"]), 2
    )
    PE_TTM_fr = round(
        out_dict["财报季市值"] / (out_dict["扣非净利润-TTM"] - out_dict["商誉"]), 2
    )
    # 计算减去商誉后的真实ROE
    ROE = round(PB / PE_TTM, 2)
    out_dict.update(
        {
            "市净率": PB,
            "市销率": PS,
            "市盈率-TTM": PE_TTM,
            "市盈率-TTM(财报)": PE_TTM_fr,
            "ROE": ROE,
        }
    )
    # compute in advance
    current_value = out_dict.pop("当前价格")
    basic_eps = out_dict.pop("basic_eps")
    basic_cps = out_dict.pop("base_cps")
    pe_ttm = out_dict.pop("市盈率-TTM")

    # 股息率 -> 每股分红
    dividend_year_ratio = get_dividend_year_ratio(code, total_value)
    dividend_per_volume = rt_price * dividend_year_ratio
    dpv = round(dividend_per_volume, 4)
    dividend_yield = round(dpv / current_value, 4)

    # ROE > 12%的季度数
    cumulate_roe_list, roe_larger_12 = get_roe_list(out_dict)

    # 60日均线
    mean60day = get_mean60day(code)

    # 自由现金流率
    fcf_rate = get_fcf_rate(out_dict)

    # 盈利质量
    profit_quality = get_profit_quality(out_dict)

    # 输出结果
    final_report = {
        "股票名称": out_dict.pop("股票名称"),
        "代码": out_dict.pop("代码"),
        "行业": out_dict.pop("行业"),
        "总市值": out_dict.pop("总市值"),
        "当前价格": current_value,
        "总股本": out_dict.pop("总股本"),
        "扣非净利润-TTM": out_dict.pop("扣非净利润-TTM"),
        "扣非净利润同比-TTM": to_percentage(out_dict.pop("扣非净利润同比-TTM")),
        "扣非利润年复合同比": to_percentage(round(out_dict.pop("net_compound_g"), 4)),
        "扣非净利润增速-TTM": to_percentage(out_dict.pop("扣非净利润增速-TTM")),
        "当季扣非净利润同比": to_percentage(out_dict.pop("当季扣非净利润同比")),
        "当季扣非净利润增速": to_percentage(out_dict.pop("当季扣非净利润增速")),
        "营业总收入-TTM": out_dict.pop("营业总收入-TTM"),
        "营业总收入同比-TTM": to_percentage(out_dict.pop("营业总收入同比-TTM")),
        "营业总收入年复合同比": to_percentage(
            round(out_dict.pop("sale_flow_compound_g"), 4)
        ),
        "营业总收入增速-TTM": to_percentage(out_dict.pop("营业总收入增速-TTM")),
        "当季营业总收入同比": to_percentage(out_dict.pop("当季营业总收入同比")),
        "当季营业总收入增速": to_percentage(out_dict.pop("当季营业总收入增速")),
        "当季毛利率": to_percentage(out_dict.pop("当季毛利率")),
        "当季毛利率同比": to_percentage(out_dict.pop("当季毛利率同比")),
        "当季毛利率同比增速": to_percentage(out_dict.pop("当季毛利率同比增速")),
        "自由现金流": to_yi_round2(out_dict.pop("fcf_ttm")),
        "自由现金流复合增长率": to_percentage(out_dict.pop("fcf_compund_g")),
        "速动资产/流动负债": out_dict.pop("速动资产/流动负债"),
        "资产负债率": to_percentage(out_dict.pop("资产负债率")),
        "少数股东损益": out_dict.pop("少数股东损益"),
        "当季三费同比": to_percentage(out_dict.pop("当季三费同比")),
        "商誉": out_dict.pop("商誉"),
        "商誉占比": to_percentage(out_dict.pop("商誉占比")),
        "市净率": out_dict.pop("市净率"),
        "市销率": out_dict.pop("市销率"),
        "市盈率-TTM": pe_ttm,
        "市盈率-TTM(财报)": out_dict.pop("市盈率-TTM(财报)"),
        "ROE": out_dict.pop("ROE"),
        "ROE_qualified-TTM": roe_larger_12,
        "每股分红": dpv,
        "股息率": to_percentage(dividend_yield),
        "分红率": to_percentage(dividend_yield * pe_ttm),
        "eps-TTM": round(basic_eps, 4),
        "eps-占比": to_percentage(round(basic_eps / current_value, 4)),
        "eps-TTM同比": to_percentage(round(out_dict.pop("eps_yoy_rate"), 4)),
        "eps-TTM同比增速": to_percentage(round(out_dict.pop("eps_yoy_speed"), 4)),
        "cps-TTM": round(basic_cps, 4),
        "cps-占比": to_percentage(round(basic_cps / current_value, 4)),
        "cps-TTM同比": to_percentage(round(out_dict.pop("cps_yoy_rate"), 4)),
        "cps-TTM同比增速": to_percentage(round(out_dict.pop("cps_yoy_speed"), 4)),
        "60日均线": round(mean60day.pop("mean_60day"), 4),
        "60日均线偏离": to_percentage(round(mean60day.pop("mean_60day_bias"), 4)),
        "盈利质量": to_percentage(round(profit_quality, 4)),
        "自由现金流率": to_percentage(round(fcf_rate, 4)),
    }
    del out_dict
    return final_report


class MultiprocessChina:
    def __init__(self):
        self.results = []  # 全局结果列表
        self.results_lock = threading.Lock()  # 用于保护结果的锁

        self.rt_info = ak.stock_zh_a_spot_em()  # 获取股票信息
        self.sy_info = ak.stock_sy_em()

    def fetch_data_for_code(self, code):
        code = "{code:0>6}".format(code=code.strip())  # 补全6位代码
        data = get_stock_data(str(code), self.rt_info, self.sy_info)
        if data:
            with self.results_lock:
                self.results.append(data)
        return data

    def multiprocess_main(self):
        stock_codes = self.rt_info["代码"].tolist()  # 获取股票代码列表

        save_path = f"./daily_report/stock_data_{current_date}.csv"
        prev_df = None
        existed_codes = []

        # 检查是否存在以前的记录
        if os.path.exists(save_path):
            prev_df = pd.read_csv(save_path)
            existed_codes = prev_df["代码"].tolist()

        with ThreadPoolExecutor() as executor:
            with tqdm(total=len(stock_codes), desc="获取股票数据") as pbar:
                futures = {
                    executor.submit(self.fetch_data_for_code, code): code
                    for code in stock_codes
                }
                for future in futures:
                    code = futures[future]
                    try:
                        future.result()  # 等待线程完成
                        if prev_df is None or int(code) not in existed_codes:
                            pbar.update(1)
                    except Exception as e:
                        print("获取数据失败: {}, 股票代码: {}".format(e, code))

        save_csv(self.results, prev_df, save_path)  # 保存结果

    def test_single(self):
        # stock_codes = self.rt_info["代码"].tolist()  # 获取股票代码列表
        stock_codes = ["000858"]
        # stock_codes = ["601328"]
        with tqdm(total=len(stock_codes), desc="获取股票数据") as pbar:
            for code in stock_codes:
                pbar.update(1)
                test_api(code)
                data = self.fetch_data_for_code(code)
                print(data)
                break


def test_api(code):
    debt = ak.stock_financial_debt_ths(code)
    benefit = ak.stock_financial_benefit_ths(code)
    cash = ak.stock_financial_cash_ths(code)
    abstract = ak.stock_financial_abstract_ths(code)

    debt.to_csv("./temp/debt.csv", index=False)
    benefit.to_csv("./temp/benefit.csv", index=False)
    cash.to_csv("./temp/cash.csv", index=False)
    abstract.to_csv("./temp/abstract.csv", index=False)


if __name__ == "__main__":
    # MultiprocessChina().multiprocess_main()
    MultiprocessChina().test_single()
