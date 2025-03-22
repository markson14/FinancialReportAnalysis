from rich import print
import akshare as ak
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import os

warnings.filterwarnings("ignore")

from src.utils import *
from src.metrics import *
from src.reports import *

# stock_zh_a_spot_em(https://akshare.akfamily.xyz/data/stock/stock.html#id10)
# stock_individual_info_em（https://akshare.akfamily.xyz/data/stock/stock.html#id8）
# 关键指标：stock_financial_abstract（https://akshare.akfamily.xyz/data/stock/stock.html#id191）
# 个数指标api：stock_a_indicator_lg（https://akshare.akfamily.xyz/data/stock/stock.html#id257）


async def get_data_async(code: str):
    code = f"{code:0>6}"
    loop = asyncio.get_running_loop()

    # 创建线程池执行器
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 并行执行四个财务数据请求
        financial_data, benefits_data, debts_data, cash_data = await asyncio.gather(
            loop.run_in_executor(executor, ak.stock_financial_abstract_ths, code),
            loop.run_in_executor(executor, ak.stock_financial_benefit_ths, code),
            loop.run_in_executor(executor, ak.stock_financial_debt_ths, code),
            loop.run_in_executor(executor, ak.stock_financial_cash_ths, code),
        )
    time.sleep(0.1)

    # 获取最新财务报告日期，financial_data为正序排列
    fa_date = int(financial_data["报告期"].values[-1].replace("-", ""))

    # 并行获取历史数据和股票信息
    with ThreadPoolExecutor(max_workers=2) as executor:
        fa_price, stock_info = await asyncio.gather(
            loop.run_in_executor(
                executor,
                ak.stock_zh_a_hist,
                code,
                "daily",
                str(fa_date - 5),
                str(fa_date),
                "qfq",
            ),
            loop.run_in_executor(executor, ak.stock_individual_info_em, code),
        )
    return financial_data, benefits_data, debts_data, cash_data, fa_price, stock_info


async def get_stock_data(code: str, rt_info: pd.DataFrame, sy_info: pd.DataFrame):
    """改造为真正的异步函数"""
    try:
        result = await get_data_async(code)
        if not result:
            return None

        financial_data, benefits_data, debts_data, cash_data, fa_price, stock_info = (
            result
        )

        # 基础信息验证
        rt_stock_info = rt_info[rt_info["代码"] == code]
        if rt_stock_info.empty:
            print(f"{code} 未找到实时信息")
            return None

        name = rt_stock_info["名称"].values[0]
        if "退市" in name:
            print(f"{code} 已退市")
            return None

        rt_price = rt_stock_info["最新价"].values[0]
        if np.isnan(rt_price):
            print(f"{code} 无最新价: {rt_price=}")
            return None

        PB = rt_stock_info["市净率"].values[0]
        if np.isnan(PB):
            print(f"{code} 无市净率数据")
            return None

        # 获取个股商誉
        sy_stock_info = sy_info[sy_info["股票代码"] == code]
        shangyu_report = {
            "商誉": (
                to_yi_round2(sy_stock_info["商誉"].values[0])
                if not sy_stock_info.empty
                else 0
            ),
            "商誉占比": (
                round(sy_stock_info["商誉占净资产比例"].values[0], 4)
                if not sy_stock_info.empty
                else 0
            ),
        }

        # 获取股票基本信息
        try:
            name = stock_info[stock_info["item"] == "股票简称"]["value"].values[0]
            sector = stock_info[stock_info["item"] == "行业"]["value"].values[0]
            total_value = stock_info[stock_info["item"] == "总市值"]["value"].values[0]
            total_volume = stock_info[stock_info["item"] == "总股本"]["value"].values[0]
        except (IndexError, KeyError) as e:
            print(f"{code} 获取基本信息失败: {str(e)}")
            return None

        if total_volume == "-":
            print(f"{code} 总股本未揭露")
            return None

        if total_value == "-":
            total_value = total_volume * rt_price

        # 财报信息处理
        try:
            fa_value = total_volume * float(fa_price["收盘"].values[0])
        except (IndexError, ValueError) as e:
            print(f"{code} 计算财报市值失败: {str(e)}")
            return None

        # 获取各类报告
        financial_report = load_ths_report(financial_data, total_volume)
        benefit_report = get_benefits_report(benefits_data, code)
        debts_report = get_debts_report(debts_data, code)
        cash_report = get_cash_report(cash_data, code)

        # 合并基础信息
        out_dict = {
            "股票名称": name,
            "代码": str(code),
            "行业": sector,
            "总市值": to_yi_round2(total_value),
            "财报季市值": to_yi_round2(fa_value),
            "当前价格": rt_price,
            "总股本": to_yi_round2(total_volume),
        }

        # 合并所有报告
        out_dict.update(financial_report)
        out_dict.update(benefit_report)
        out_dict.update(shangyu_report)
        out_dict.update(cash_report)
        out_dict.update(debts_report)

        # 计算高级指标
        try:
            PS = round(out_dict["总市值"] / out_dict["营业总收入-TTM"], 2)
            PE_TTM = round(
                out_dict["总市值"] / (out_dict["扣非净利润-TTM"] - out_dict["商誉"]), 2
            )
            PE_TTM_fr = round(
                out_dict["财报季市值"]
                / (out_dict["扣非净利润-TTM"] - out_dict["商誉"]),
                2,
            )
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
        except (KeyError, ZeroDivisionError) as e:
            print(f"{code} 计算估值指标失败: {str(e)}")
            return None

        # 计算其他指标
        current_value = out_dict.pop("当前价格")
        basic_eps = out_dict.pop("basic_eps")
        basic_cps = out_dict.pop("base_cps")
        pe_ttm = out_dict.pop("市盈率-TTM")

        # 计算股息率
        dividend_year_ratio = get_dividend_year_ratio(code, total_value)
        dividend_per_volume = rt_price * dividend_year_ratio
        dpv = round(dividend_per_volume, 4)
        dividend_yield = round(dpv / current_value, 4)

        # 计算ROE指标
        cumulate_roe_list, roe_larger_12 = get_roe_list(out_dict)

        # 计算均线指标
        mean60day = get_mean60day(code)

        # 计算自由现金流率
        fcf_rate = get_fcf_rate(out_dict)

        # 计算盈利质量
        profit_quality = get_profit_quality(out_dict)

        # 计算RO指标
        ROA = get_RO_metrics(out_dict)

        # 构建最终报告
        final_report = {
            "股票名称": out_dict.pop("股票名称"),
            "代码": out_dict.pop("代码"),
            "行业": out_dict.pop("行业"),
            "总市值": out_dict.pop("总市值"),
            "当前价格": current_value,
            "总股本": out_dict.pop("总股本"),
            "扣非净利润-TTM": out_dict.pop("扣非净利润-TTM"),
            "扣非净利润同比-TTM": to_percentage(out_dict.pop("扣非净利润同比-TTM")),
            "扣非利润年复合同比": to_percentage(
                round(out_dict.pop("net_compound_g"), 4)
            ),
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
            "总负债": to_yi_round2(out_dict.pop("total_liabilities")),
            "总资产": to_yi_round2(out_dict.pop("total_equity")),
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

        return final_report

    except Exception as e:
        print(f"{code} 处理失败: {str(e)}")
        return None


class AsyncChinaAnalyzer:
    def __init__(self, max_concurrency=3):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rt_info = None
        self.sy_info = None

    async def initialize(self):
        """异步初始化基础数据"""
        self.rt_info, self.sy_info = ak.stock_zh_a_spot_em(), ak.stock_sy_em()

    async def process_code(self, code):
        """带并发限制的异步处理"""
        async with self.semaphore:
            return await get_stock_data(code, self.rt_info, self.sy_info)

    async def run_analysis(self, codes):
        """主运行逻辑"""
        await self.initialize()
        tasks = [self.process_code(code) for code in codes]
        results = []

        # 使用asyncio.as_completed实现实时进度显示
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await future
            if result:
                results.append(result)
        return results


async def main():
    try:
        # 创建输出目录
        os.makedirs("./daily_report", exist_ok=True)

        # 初始化分析器
        print("初始化分析器...")
        analyzer = AsyncChinaAnalyzer(max_concurrency=5)

        # 获取股票代码列表
        print("获取股票代码列表...")
        try:
            codes = ak.stock_zh_a_spot_em()["代码"].tolist()
            print(f"成功获取 {len(codes)} 个股票代码")
        except Exception as e:
            print(f"获取股票代码列表失败: {str(e)}")
            raise

        # 运行分析
        print("开始分析股票数据...")
        results = await analyzer.run_analysis(codes)

        # 保存结果
        if results:
            df = pd.DataFrame(results)
            save_path = f"./daily_report/stock_data_{current_date}.csv"
            df.to_csv(save_path, index=False)
            print(f"成功保存 {len(results)} 条数据到 {save_path}")

            # 输出统计信息
            print(f"分析完成: 成功 {len(results)}/{len(codes)} 个股票")
        else:
            logger.warning("没有获取到任何数据")

    except Exception as e:
        print(f"程序执行出错: {str(e)}", exc_info=True)
        raise


def test_api(code):
    """测试API调用"""
    try:
        print(f"测试API调用: {code}")

        # 获取数据
        debt = ak.stock_financial_debt_ths(code)
        benefit = ak.stock_financial_benefit_ths(code)
        cash = ak.stock_financial_cash_ths(code)
        abstract = ak.stock_financial_abstract_ths(code)

        # 保存数据
        os.makedirs("./temp", exist_ok=True)
        debt.to_csv("./temp/debt.csv", index=False)
        benefit.to_csv("./temp/benefit.csv", index=False)
        cash.to_csv("./temp/cash.csv", index=False)
        abstract.to_csv("./temp/abstract.csv", index=False)

        print("API测试完成")

    except Exception as e:
        print(f"API测试失败: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常退出: {str(e)}", exc_info=True)
