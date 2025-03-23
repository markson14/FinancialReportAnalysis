# 中国股市分析工具

## 项目概述

本项目是一个用于分析中国股市数据的 Python 工具集，提供了多个模块用于获取和分析股票市场数据，包括指数分析、个股分析、市场状态判断等功能。项目采用异步编程和多线程技术，提高了数据获取和处理效率。

## 主要功能模块

### 1. 指数市场分析 (china_index_market.py)

该模块专注于中国主要股指的数据分析，主要功能包括：

- **数据获取**：使用 `akshare` 库获取上证指数、深证指数等的历史数据
- **技术指标计算**：实现 KAMA（Kaufman's Adaptive Moving Average）等指标的计算
- **多线程处理**：通过 `threading` 模块实现高效的数据处理
- **回测功能**：支持股指历史数据的策略回测
- **市场状态判断**：分析市场涨跌幅和成交量变化

### 2. 个股分析 (china_market_async.py)

该模块专注于个股的财务数据分析，采用异步编程提高效率：

- **异步数据获取**：使用 `asyncio` 和 `ThreadPoolExecutor` 实现高效的并发数据获取
- **财务数据分析**：
  - 获取个股财务摘要数据
  - 分析利润表数据
  - 分析资产负债表数据
  - 分析现金流量表数据
- **数据处理与存储**：支持数据清洗和 CSV 格式存储
- **进度显示**：使用 `rich` 和 `tqdm` 提供友好的进度显示

### 3. 美国市场分析 (american_market.py)

提供美国市场数据的获取和分析功能。

## 项目结构

```
StockProject/
├── src/                    # 源代码目录
│   ├── utils.py           # 工具函数
│   ├── metrics.py         # 指标计算
│   └── reports.py         # 报告生成
├── daily_report/          # 日报相关
├── test/                  # 测试文件
├── Documents/             # 文档
├── 指数分析/              # 指数分析相关
├── 宏观分析/              # 宏观分析相关
├── china_index_market.py  # 指数市场分析
├── china_market_async.py  # 异步个股分析
```

## 环境要求

- Python 3.8+
- 依赖包：见 requirements.txt

## 安装说明

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd StockProject
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

1. **指数分析**：
```python
python china_index_market.py
```

2. **个股分析**：
```python
python china_market_async.py
```

3. **美国市场分析**：
```python
python american_market.py
```

## 注意事项

- 使用前请确保网络连接正常
- 数据获取可能需要一定时间，请耐心等待
- 建议使用代理以提高数据获取的稳定性

## 贡献与反馈

欢迎提交 Issue 和 Pull Request 来帮助改进项目。如果您有任何问题或建议，请随时提出。

## 许可证

MIT License