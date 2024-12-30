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

from utils import (
    to_percentage,
    to_yi_round2,
    parse_chinese_number,
    parse_percentage,
    save_csv,
    current_date,
    get_yoy_rate_and_speed,
    return_last_4q,
)

# stock_zh_a_spot_em(https://akshare.akfamily.xyz/data/stock/stock.html#id10)
# stock_individual_info_em（https://akshare.akfamily.xyz/data/stock/stock.html#id8）
# 关键指标：stock_financial_abstract（https://akshare.akfamily.xyz/data/stock/stock.html#id191）
# 个数指标api：stock_a_indicator_lg（https://akshare.akfamily.xyz/data/stock/stock.html#id257）


def load_ths_report(financial_data, total_volume):
    if "扣非净利润" in financial_data.columns:
        net_profit, profit_list = return_last_4q(
            "扣非净利润",
            financial_data,
            parse_chinese_number,
            func=np.sum,
            is_cumulate=True,
        )
    else:
        print("扣非净利润未揭露")
        net_profit, profit_list = return_last_4q(
            "净利润",
            financial_data,
            parse_chinese_number,
            func=np.sum,
            is_cumulate=True,
        )
    net_profit_yoy, net_profit_speed, net_compound_g = get_yoy_rate_and_speed(
        profit_list, get_TTM=True
    )
    cur_net_profit_yoy, cur_net_profit_speed, _ = get_yoy_rate_and_speed(profit_list)

    # 经营现金流估计
    base_cps, cps_list = return_last_4q(
        "每股经营现金流", financial_data, float, func=np.sum, is_cumulate=True
    )
    cps_yoy_rate, cps_yoy_speed, cps_compund_g = get_yoy_rate_and_speed(
        cps_list, get_TTM=True, use_min_value=False
    )

    sale_flow, sale_list = return_last_4q(
        "营业总收入",
        financial_data,
        parse_chinese_number,
        func=np.sum,
        is_cumulate=True,
    )
    sale_flow_yoy, sale_flow_speed, sale_flow_compound_g = get_yoy_rate_and_speed(
        sale_list, get_TTM=True
    )
    cur_sale_flow_yoy, cur_sale_flow_speed, _ = get_yoy_rate_and_speed(sale_list)
    profit_per_value, _ = return_last_4q(
        "基本每股收益", financial_data, float, func=np.sum, is_cumulate=True
    )
    if "速动比率" in financial_data.columns:
        cash_flow_to_debt_ratio, _ = return_last_4q("速动比率", financial_data, float)
    else:
        cash_flow_to_debt_ratio = -1
    debt_ratio, _ = return_last_4q("资产负债率", financial_data, parse_percentage)

    # 净资产收益率

    return {
        "扣非净利润-TTM": to_yi_round2(net_profit),
        "扣非净利润同比-TTM": round(net_profit_yoy, 4),
        "扣非净利润增速-TTM": round(net_profit_speed, 4),
        "当季扣非净利润同比": round(cur_net_profit_yoy, 4),
        "当季扣非净利润增速": round(cur_net_profit_speed, 4),
        "营业总收入-TTM": to_yi_round2(sale_flow),
        "营业总收入同比-TTM": round(sale_flow_yoy, 4),
        "营业总收入增速-TTM": round(sale_flow_speed, 4),
        "当季营业总收入同比": round(cur_sale_flow_yoy, 4),
        "当季营业总收入增速": round(cur_sale_flow_speed, 4),
        "速动资产/流动负债": round(cash_flow_to_debt_ratio, 4),
        "资产负债率": round(debt_ratio, 4),
        "base_cps": base_cps,
        "cps_yoy_rate": cps_yoy_rate,
        "cps_yoy_speed": cps_yoy_speed,
        "net_compound_g": net_compound_g,
        "sale_flow_compound_g": sale_flow_compound_g,
        "cps_list": cps_list,
        # "基本每股收益-TTM": round(profit_per_value, 4),
    }


def get_dividend_year_ratio(code: str, total_value: float):
    try:
        temp_list = ak.stock_fhps_detail_ths(symbol=code).iloc[-2:]["分红总额"].tolist()
    except:
        temp_list = (
            ak.stock_fhps_detail_ths(symbol=code).iloc[-2:]["AH分红总额"].tolist()
        )
    year_dividend = 0
    for temp in temp_list:
        if not "亿" in temp and not "万" in temp:
            continue
        year_dividend += parse_chinese_number(temp)
    dividend_year_ratio = year_dividend / total_value

    return dividend_year_ratio


def get_debts_report(code: str):
    debts_df = ak.stock_financial_debt_ths(code)
    # 归母股东权益

    _, equity_list = return_last_4q(5, debts_df, parse_chinese_number, func=np.sum)
    return {"equity_list": equity_list}


def get_cash_report(code: str):
    cash_df = ak.stock_financial_cash_ths(code)
    if "购建固定资产、无形资产和其他长期资产支付的现金" in cash_df.columns:
        cap_exp_ttm, capexp_list = return_last_4q(
            "购建固定资产、无形资产和其他长期资产支付的现金",
            cash_df,
            parse_chinese_number,
            func=np.sum,
            is_cumulate=True,
        )
    else:
        cap_exp_ttm = 0

    if "*经营活动产生的现金流量净额" in cash_df.columns:
        op_cash_flow_ttm, op_cash_flow_list = return_last_4q(
            "*经营活动产生的现金流量净额",
            cash_df,
            parse_chinese_number,
            func=np.sum,
            is_cumulate=True,
        )

    # 自由现金流
    fcf_ttm = op_cash_flow_ttm - cap_exp_ttm
    fcf_list = [ocf - capexp for ocf, capexp in zip(op_cash_flow_list, capexp_list)]
    fcf_yoy_rate, fcf_yoy_speed, fcf_compund_g = get_yoy_rate_and_speed(
        fcf_list, get_TTM=True, use_min_value=False
    )

    return {
        "captal_expense": cap_exp_ttm,
        "capexp_list": capexp_list,
        "fcf_ttm": fcf_ttm,
        "fcf_yoy_rate": fcf_yoy_rate,
        "fcf_yoy_speed": fcf_yoy_speed,
        "fcf_compund_g": fcf_compund_g,
    }


def get_benefits_report(code: str):
    benefits_df = ak.stock_financial_benefit_ths(code)
    three_cost = []
    if "销售费用" in benefits_df.columns:
        sell_cost = np.array(
            return_last_4q(
                "销售费用", benefits_df, parse_chinese_number, is_cumulate=True
            )[1],
            dtype=np.float64,
        )
        three_cost.append(sell_cost)
    if "管理费用" in benefits_df.columns:
        manage_cost = np.array(
            return_last_4q(
                "管理费用", benefits_df, parse_chinese_number, is_cumulate=True
            )[1]
        )
        three_cost.append(manage_cost)
    if "研发费用" in benefits_df.columns:
        dev_cost = np.array(
            return_last_4q(
                "研发费用", benefits_df, parse_chinese_number, is_cumulate=True
            )[1],
            dtype=np.float64,
        )
        three_cost.append(dev_cost)
    if "财务费用" in benefits_df.columns:
        financial_cost = np.array(
            return_last_4q(
                "财务费用", benefits_df, parse_chinese_number, is_cumulate=True
            )[1],
            dtype=np.float64,
        )
        three_cost.append(financial_cost)
    three_cost = np.sum(three_cost, axis=0)
    if isinstance(three_cost, np.ndarray):
        threecost_yoy_rate, threecost_yoy_speed, _ = get_yoy_rate_and_speed(three_cost)
    else:
        threecost_yoy_rate, threecost_yoy_speed = -1, -1
    # 少数股东损益
    if "少数股东损益" not in benefits_df.columns:
        minor_shares = 0
    else:
        minor_shares = return_last_4q(
            "少数股东损益",
            benefits_df,
            parse_chinese_number,
            func=np.sum,
            is_cumulate=True,
        )[0]
    # 基本每股收益
    if "（一）基本每股收益" in benefits_df.columns:
        basic_eps, eps_list = return_last_4q(
            "（一）基本每股收益",
            benefits_df,
            float,
            func=np.sum,
            is_cumulate=True,
        )
    else:
        basic_eps = 0
    eps_yoy_rate, eps_yoy_speed, _ = get_yoy_rate_and_speed(
        eps_list, get_TTM=True, use_min_value=False
    )
    # 毛利率
    _, income_list = return_last_4q(
        3, benefits_df, parse_chinese_number, func=np.sum, is_cumulate=True
    )
    _, expense_list = return_last_4q(
        4, benefits_df, parse_chinese_number, func=np.sum, is_cumulate=True
    )
    gross_profit_rate_list = (
        np.array(income_list) - np.array(expense_list)
    ) / np.array(income_list)
    gross_profit_yoy_rate, gross_profit_yoy_speed, _ = get_yoy_rate_and_speed(
        gross_profit_rate_list, use_min_value=False
    )
    gross_profit_rate = (income_list[0] - expense_list[0]) / income_list[0]
    # 扣非净利润
    _, net_profit_list = return_last_4q(
        5, benefits_df, parse_chinese_number, func=np.sum, is_cumulate=True
    )
    return {
        "当季毛利率": round(gross_profit_rate, 4),
        "当季毛利率同比": round(gross_profit_yoy_rate, 4),
        "当季毛利率同比增速": round(gross_profit_yoy_speed, 4),
        "少数股东损益": to_yi_round2(minor_shares, 4),
        "当季三费同比": round(threecost_yoy_rate, 4),
        "net_profit_list": net_profit_list,
        "basic_eps": basic_eps,
        "eps_yoy_rate": eps_yoy_rate,
        "eps_yoy_speed": eps_yoy_speed,
    }


def get_roe_list(net_profit_list, equity_list):
    threshold = 0.12

    # 定义一个函数来计算ROE-TTM
    def calculate_roe_ttm(quarterly_roe):
        roe_ttm = []
        for i in range(len(quarterly_roe) - 3):  # 确保有足够的季度
            ttm = sum(quarterly_roe[i : i + 4])  # 求和四个季度的ROE
            roe_ttm.append(ttm)  # 将TTM结果添加到列表中
        return roe_ttm

    net_profit_list = np.array(net_profit_list)
    net_profit_list[net_profit_list < 0] = 0
    equity_list = np.array(equity_list)
    equity_list[equity_list < 0] = -1

    if len(net_profit_list) != len(equity_list):
        net_profit_list = net_profit_list[: len(equity_list)]
    quater_roe_list = net_profit_list / equity_list
    cumulate_roe_list = calculate_roe_ttm(quater_roe_list)

    # get the number of ROE larger than 12%
    count = 0
    for roe_ttm in cumulate_roe_list:
        if roe_ttm >= threshold:
            count += 1
        else:
            break
    return cumulate_roe_list, count


def get_mean60day(code):
    """
    获取指数的回溯指数10年60日均值偏离度
    """
    history_data = ak.stock_zh_a_hist(symbol=code)
    mean_60day = history_data["收盘"].rolling(window=60).mean().iloc[-1]
    mean_60day_bias = (history_data["收盘"].iloc[-1] - mean_60day) / mean_60day
    return {"mean_60day": mean_60day, "mean_60day_bias": mean_60day_bias}


@retry(wait_fixed=3000, stop_max_attempt_number=2)
def get_stock_data(code: str, rt_info: pd.DataFrame, sy_info: pd.DataFrame):
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
    # 财报期 市值

    # ROE, net_profit, net_profit_yoy, cash_flow = load_sina_report(financial_data)
    financial_report = load_ths_report(financial_data, total_volume)
    # 股息率 -> 每股分红
    dividend_year_ratio = get_dividend_year_ratio(code, total_value)
    dividend_per_volume = rt_price * dividend_year_ratio
    # 财报信息
    benefit_report = get_benefits_report(code)
    debts_report = get_debts_report(code)
    cumulate_roe_list, roe_larger_12 = get_roe_list(
        benefit_report.pop("net_profit_list"), debts_report.pop("equity_list")
    )
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
            "ROE_qualified-TTM": roe_larger_12,
        }
    )

    # compute in advance
    dpv = round(dividend_per_volume, 4)
    current_value = out_dict.pop("当前价格")
    basic_eps = out_dict.pop("basic_eps")
    basic_cps = out_dict.pop("base_cps")
    dividend_yield = round(dpv / current_value, 4)
    pe_ttm = out_dict.pop("市盈率-TTM")

    # 60日均线
    mean60day = get_mean60day(code)

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
        "ROE_qualified-TTM": out_dict.pop("ROE_qualified-TTM"),
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
        stock_codes = ["688382"]
        # stock_codes = ["603075"]
        with tqdm(total=len(stock_codes), desc="获取股票数据") as pbar:
            for code in stock_codes:
                pbar.update(1)
                data = self.fetch_data_for_code(code)
                print(data)
                break


def test_api():
    code = "000568"
    # debts_df = ak.stock_financial_debt_ths(code)
    # print(debts_df)
    debt = ak.stock_financial_debt_ths(code)
    benefit = ak.stock_financial_benefit_ths(code)
    cash = ak.stock_financial_cash_ths(code)
    abstract = ak.stock_financial_abstract_ths(code)

    debt.to_csv("debt.csv", index=False)
    benefit.to_csv("benefit.csv", index=False)
    cash.to_csv("cash.csv", index=False)
    abstract.to_csv("abstract.csv", index=False)


if __name__ == "__main__":
    MultiprocessChina().multiprocess_main()
    # MultiprocessChina().test_single()
    # test_api()
