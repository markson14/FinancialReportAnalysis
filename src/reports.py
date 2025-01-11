import numpy as np
import akshare as ak
from rich import print
from src.utils import *

__all__ = [
    "load_ths_report",
    "get_debts_report",
    "get_cash_report",
    "get_benefits_report",
]


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


def get_debts_report(code: str):
    # 负债表中数据都是当下数据，所以不需要累计，取最新计算即可
    debts_df = ak.stock_financial_debt_ths(code)
    # 归母股东权益

    _, net_equity_list = return_last_4q(5, debts_df, parse_chinese_number)

    # 货币资金
    if "货币资金" not in debts_df.columns:
        print(f"{code}: 货币资金未揭露")
        monetary_funds_list = [0]
    else:
        _, monetary_funds_list = return_last_4q(
            "货币资金", debts_df, parse_chinese_number
        )

    # 负债总额
    _, total_liabilities_list = return_last_4q(
        "*负债合计", debts_df, parse_chinese_number
    )

    return {
        "net_equity_list": net_equity_list,
        "monetary_funds": monetary_funds_list[0],
        "total_liabilities": total_liabilities_list[0],
    }


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
        "op_cash_flow_ttm": op_cash_flow_ttm,
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
    # 营业利润
    sale_profit_ttm, _ = return_last_4q(
        "三、营业利润", benefits_df, parse_chinese_number, func=np.sum, is_cumulate=True
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
        "sale_profit_ttm": sale_profit_ttm,
    }
