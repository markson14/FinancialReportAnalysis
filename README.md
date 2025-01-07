## README for 中国市场分析脚本

该仓库包含一个用于分析中国股票财务数据的Python脚本，使用了Akshare库。该脚本能够获取股票数据、处理财务报告，并计算多种财务指标，以提供对股票表现的深入分析。

### 特性

- **数据获取**：脚本可以实时获取多只股票的信息及历史财务数据。

- **财务指标计算**：计算关键财务指标，包括：
  - 扣非净利润及其增长率
  - 经营现金流
  - 每股收益（EPS）
  - 股东权益回报率（ROE）
  - 债务比率及其他相关财务指标

- **多线程支持**：利用多线程并发获取数据，提高处理多只股票时的性能。

- **错误处理**：实现重试逻辑以处理数据获取中的临时错误。

- **CSV导出**：将处理后的股票数据保存为CSV文件，以便进一步分析。

### 环境要求

运行此脚本前，请确保安装以下Python包：

- `akshare`
- `pandas`
- `numpy`
- `rich`
- `tqdm`
- `retrying`

可以使用pip安装这些包：

```bash
pip install akshare pandas numpy rich tqdm retrying
```

### 使用方法

1. **克隆仓库**：
   将此仓库克隆到本地机器。

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **运行脚本**：
   执行脚本以开始获取和处理股票数据。

   ```bash
   python china_market.py
   ```

3. **查看结果**：
   执行后，结果将保存在`./daily_report/`目录下，文件名格式为`stock_data_<current_date>.csv`。

### 函数概述

- **load_ths_report(financial_data, total_volume)**：加载并处理特定股票的财务报告。

- **get_dividend_year_ratio(code, total_value)**：根据历史分红数据计算年度分红比率。

- **get_debts_report(code)**：获取特定股票的债务相关财务信息。

- **get_cash_report(code)**：分析现金流量表并计算自由现金流。

- **get_benefits_report(code)**：从财务报表中提取利润相关指标。

- **get_stock_data(code, rt_info, sy_info)**：主函数，用于收集所有相关股票数据并计算各种指标。

### 示例

要测试单只股票的获取，可以在脚本中修改`test_single`方法以包含所需的股票代码：

```python
def test_single(self):
    stock_codes = ["688382"]  # 替换为所需的股票代码
    ...
```

### 贡献

欢迎贡献！如果您有建议或改进，请创建一个问题或提交拉取请求。

### 许可证

该项目根据MIT许可证进行许可。有关详细信息，请参阅LICENSE文件。

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/16191020/579dc910-6c55-496d-ab3f-045fdc67b4d5/china_market.py