# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# jojo_quant (韭韭量化) - jojo指标选股工具

基于 jojo 复合动量指标的全市场扫描工具。扫描范围覆盖 NASDAQ + NYSE 全部股票和商品期货。

## 可用命令

### 每日扫描

```bash
# 扫描策略1信号（超买动量）
python3 screener.py --strategy 1

# 扫描策略2信号（超卖反转）
python3 screener.py --strategy 2

# 扫描全部策略
python3 screener.py --strategy all

# 限制返回数量
python3 screener.py --strategy 1 --top 20
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--strategy` | `all` | `1`=超买动量, `2`=超卖反转, `all`=全部 |
| `--top N` | 全部 | 只显示前 N 个结果 |
| `--days N` | 120 | 下载 N 天历史数据用于信号检测 |
| `--batch N` | 200 | 每批下载的 ticker 数量 |

### 历史回测

```bash
# 回测单只或多只股票
python3 backtest.py TSLA NVDA HOOD --years 3

# 使用 TradingView CSV 数据回测
python3 backtest.py --csv data.csv --use-tv --label TSLA
```

### 生成回测报告

```bash
# 生成13只股票的完整回测报告（自动推送GitHub和S3）
python3 generate_report.py

# 只生成不推送
python3 generate_report.py --no-push --no-s3
```

> `generate_report.py` 默认推送到 GitHub 和 S3（见下方"外部依赖"）。无凭证或调试时务必加 `--no-push --no-s3`。

### 测试与调试

```bash
# 回测逻辑测试（基于 assert，无 pytest 配置；这是项目唯一的测试入口）
python3 test_logic.py

# 校验 jojo 指标与 TradingView 导出 CSV 的差异
python3 validate.py

# 输出 jojo 各子指标分量（RSI/WR/CMO/KD/TSI/ADX）用于排查
python3 debug_indicators.py

# 比较 4 种基金选股排名策略
python3 compare_ranking.py
```

## 策略说明

### 策略1: 超买动量
- **买入**: jojo指标上穿 76
- **卖出**: jojo指标下穿 68
- **过滤**: ATR%(14) >= 2.0（过滤低波动股）
- **止损**: -20%
- **适合**: 高波动股票（TSLA, NVDA, RKLB 等）

### 策略2: 超卖反转
- **买入**: jojo指标在 28 以下拐头向上
- **卖出**: jojo指标上穿 51，或再次下穿 28
- **止损**: -20%
- **适合**: 超卖反弹机会

## 输出格式

每个信号包含以下字段：

| 字段 | 说明 |
|------|------|
| ticker | 标的代码 |
| name | 英文名称 |
| cn_name | 中文名称（常见标的） |
| industry | 行业分类 |
| mkt_cap_fmt | 市值（格式化） |
| close | 最新收盘价 |
| jojo | jojo指标当前值 |
| atr_pct | ATR%（仅策略1） |
| bt_trades | 历史回测交易次数（2009年至今） |
| bt_win_rate | 历史胜率% |
| bt_total_pnl | 历史累计收益% |
| bt_pf | 盈亏比 (Profit Factor) |
| bt_max_dd | 最大回撤% |
| {regime}_trades | 当前市场环境下的交易次数 |
| {regime}_win% | 当前市场环境下的胜率 |
| {regime}_pnl% | 当前市场环境下的累计收益 |
| {regime}_pf | 当前市场环境下的盈亏比 |
| {regime}_dd% | 当前市场环境下的最大回撤 |

> **市场环境判断**: SPX 收盘价 >= SMA(225) 为牛市，< SMA(225) 为熊市。输出中 {regime} 会根据当前 SPX 状态自动替换为"牛市"或"熊市"，只展示当前环境对应的数据。

## 过滤规则

- 股票：市值 >= 1B USD，排除 ETF
- 商品期货：不做市值过滤

## 覆盖范围

- **股票**: NASDAQ + NYSE 全部（约 6000+）
- **商品期货**: 黄金(GC=F), 白银(SI=F), 原油(CL=F), 天然气(NG=F), 铜(HG=F), 铂金(PL=F)

## 架构

模块职责与数据流：

- **`indicators.py`** — 纯 pandas/numpy，无 I/O。导出 `compute_jojo(df)`，内部由 6 个子指标合成（`_rsi` / `_willr` / `_cmo` / `_stoch` / `_tsi` / `_dmi_adx`，配合 `_rma` / `_ema` 平滑）。被其他所有模块复用。
- **`backtest.py`** — 暴露 `backtest_strategy1()` / `backtest_strategy2()`（numpy 向量化模拟）以及编排函数 `run_backtest()`（下载 → 指标 → 策略 → 按市场环境拆分指标）。被 `screener.py` 与 `generate_report.py` 调用。
- **`screener.py`** — 每日扫描入口：`yfinance` 批量下载 OHLC → `compute_jojo` → 当日信号筛选 → 用 `run_backtest()` 给每行附加历史回测指标（整段 + 当前 SPX 环境子集） → 排名输出。
- **市场环境**: 由 `^GSPC` 收盘价对比 SMA(225) 决定，用于挑选输出中显示哪一组 `{regime}_*` 列。

## 项目文件

| 文件 | 说明 |
|------|------|
| `screener.py` | 全市场扫描工具（每日入口） |
| `backtest.py` | 历史回测引擎（`run_backtest`、`backtest_strategy1/2`） |
| `indicators.py` | jojo 指标计算（纯 pandas/numpy，无 I/O） |
| `generate_report.py` | 批量回测报告生成（默认推 GitHub + S3） |
| `fund_backtest.py` | 基金组合回测（内部工具，支持 `--universe sp500/sp500+/report/custom`、`--historical` 反生存者偏差、`--compare` 多配置对比；输出 `fund_equity.csv` / `fund_trades.csv`） |
| `compare_ranking.py` | 比较 4 种基金选股排名方法 |
| `test_logic.py` | 回测逻辑断言测试（项目唯一测试入口） |
| `validate.py` | 与 TradingView CSV 对账 |
| `debug_indicators.py` | 子指标分量诊断 |
| `jojo.pine` | TradingView Pine Script 版本 |

## 依赖

```bash
pip install -r requirements.txt
```

### 外部依赖

- **GitHub**: `generate_report.py` 默认 `git add/commit/push`，需要本地 git 凭证已配置。
- **S3**: 报告默认上传到 `s3://staking-ledger-bpt/jojo_quant/reports/`（路径硬编码），需要 AWS CLI 与对应 IAM 凭证。两者都可用 `--no-push --no-s3` 跳过。
- **yfinance**: 匿名访问，无需 API key。
- **FMP**（公司 profile）: 有速率限制，失败时降级为空 profile，不会阻塞流程。
