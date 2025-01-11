import re, os
import pandas as pd
from datetime import datetime
import pytz
import numpy as np


__all__ = [
    "current_date",
    "to_yi_round2",
    "to_percentage",
    "parse_chinese_number",
    "parse_percentage",
    "save_csv",
    "return_last_4q",
    "get_yoy_rate_and_speed",
]

# 获取当前日期
current_date = datetime.now(pytz.timezone("Asia/shanghai")).strftime("%Y%m%d")
print(f"{current_date=}")
HUNDRED_MILLION = 100000000
MIN_VALUE = 2000 * 10000


def to_yi_round2(value, round_num=2):
    return round(value / HUNDRED_MILLION, round_num)


def to_percentage(value, round_num=2):
    return f"{round(value * 100, round_num)}%"


def parse_chinese_number(s):
    if isinstance(s, bool):
        return 0
    units = {
        "万亿": 1000000000000,
        "亿": 100000000,
        "千万": 10000000,
        "百万": 1000000,
        "十万": 100000,
        "万": 10000,
        "千": 1000,
        "百": 100,
        "十": 10,
    }
    pattern = re.compile(r"([+-]?\d*\.?\d+)(万亿|亿|千万|百万|十万|万|千|百|十)?$")
    match = pattern.match(s)
    if not match:
        raise ValueError("Invalid input string")
    num, unit = match.groups()
    num = float(num)
    if unit and unit in units:
        num *= units[unit]
    return num


def parse_percentage(s):
    # 移除百分号并转换为浮点数
    if s:
        return float(s.strip("%")) / 100
    else:
        return 0.0


def save_csv(data: list, prev_df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df = pd.DataFrame(data)
    if prev_df is not None:
        result_df = pd.concat([prev_df, result_df], ignore_index=True)
    result_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig",
    )
    print("数据获取完成,已保存到文件")


def return_last_4q(
    x, financial_data, parse_func, func=np.mean, is_cumulate=False, lasting=13
):
    if is_cumulate:
        last_12_list = []
        date_list = financial_data.iloc[0:lasting]["报告期"].tolist()
        for idx, date in enumerate(date_list[:-1]):
            if "03-31" in date:
                cur_q = parse_func(financial_data.iloc[idx][x])
            elif "06-30" in date or "09-30" in date or "12-31" in date:
                q_cum = parse_func(financial_data.iloc[idx][x])
                q_qum_1 = parse_func(financial_data.iloc[idx + 1][x])
                cur_q = q_cum - q_qum_1
            last_12_list.append(cur_q)
    else:
        if isinstance(x, int):
            last_12_list = financial_data.iloc[0 : lasting - 1, x].tolist()
        else:
            last_12_list = financial_data.iloc[0 : lasting - 1][x].tolist()
        last_12_list = [parse_func(y) for y in last_12_list]
    return func(last_12_list[:4]), last_12_list


def get_yoy_rate_and_speed(last_8_list, get_TTM=False, use_min_value=True):
    if get_TTM:
        this_year = np.sum(last_8_list[:4])
        last_year = np.sum(last_8_list[4:8])
        if use_min_value and last_year < MIN_VALUE and last_year >= 0:
            last_year = MIN_VALUE
        yoy_rate = (this_year - last_year) / abs(last_year)
        if len(last_8_list) < 9:
            yoy_rate_last = yoy_rate
        else:
            year_before_last = np.sum(last_8_list[8:12])
            if use_min_value and year_before_last < MIN_VALUE and year_before_last >= 0:
                year_before_last = MIN_VALUE
            yoy_rate_last = (last_year - year_before_last) / abs(year_before_last)
        yoy_speed = yoy_rate - yoy_rate_last
        # print(
        #     f"{this_year=}, {last_year=}, {year_before_last=}, {yoy_rate=}, {yoy_rate_last=}, {yoy_speed=}\n"
        # )
        compound_growth_rate = np.mean([yoy_rate, yoy_rate_last])
        return yoy_rate, yoy_speed, compound_growth_rate
    else:
        # 根据每期的数值，计算增长率和增速
        this_q = last_8_list[0]
        last_q = last_8_list[4]
        if use_min_value and last_q < MIN_VALUE and last_q >= 0:
            last_q = MIN_VALUE
        yoy_rate = (this_q - last_q) / abs(last_q)
        if len(last_8_list) < 5:
            yoy_rate_last = 0
        else:
            q_before_last = last_8_list[8]
            if use_min_value and q_before_last < MIN_VALUE and q_before_last >= 0:
                q_before_last = MIN_VALUE
            yoy_rate_last = (last_q - q_before_last) / abs(q_before_last)
        yoy_speed = yoy_rate - yoy_rate_last
        return yoy_rate, yoy_speed, None


if __name__ == "__main__":
    # 测试各种情况
    test_strings = [
        "62.62亿",
        "0.72千万",
        "-1.5万",
        "500",
        "+3500",
        "0.1千",
        "2.5百万",
        "3.8十万",
        "9999",
        "-0.01亿",
        "1.23亿",
        "4567万",
        "78.9千",
        "0.5百",
        "10十",
    ]

    for ts in test_strings:
        try:
            result = parse_chinese_number(ts)
            print(f"{ts}: {result}")
        except ValueError as e:
            print(f"{ts}: Error - {str(e)}")
