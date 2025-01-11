import akshare as ak
import numpy as np

from src.utils import parse_chinese_number, HUNDRED_MILLION


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


def get_roe_list(out_dict):
    net_profit_list = out_dict["net_profit_list"]
    equity_list = out_dict["net_equity_list"]
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


def get_fcf_rate(out_dict):
    fcf_ttm = out_dict["fcf_ttm"]
    total_value = out_dict["总市值"] * HUNDRED_MILLION
    monetary_funds = out_dict["monetary_funds"]
    total_liabilities = out_dict["total_liabilities"]
    fcf_rate = fcf_ttm / (total_value + total_liabilities - monetary_funds)
    return fcf_rate


def get_profit_quality(out_dict):
    """
    获取利润质量
    """
    net_equity = out_dict["net_equity_list"][0]
    sale_profit_ttm = out_dict["sale_profit_ttm"]
    op_cash_flow_ttm = out_dict["op_cash_flow_ttm"]
    total_liabilities = out_dict["total_liabilities"]
    profit_quality = (op_cash_flow_ttm - sale_profit_ttm) / (
        net_equity + total_liabilities
    )
    return profit_quality
