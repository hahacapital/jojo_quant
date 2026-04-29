# jojo_quant (韭韭量化) - jojo指标选股工具

> [English README](README.md) · 中文文档（本页）

每日扫描 NASDAQ + NYSE 全部股票和商品期货，基于 jojo 指标的两种策略选股：

- **策略1 (超买动量)**：上穿 76 买入，回落下穿 68 卖出（ATR%≥2.0 过滤，20%止损）
- **策略2 (超卖反转)**：下穿 28 后拐头向上买入，上穿 51 或再次下穿 28 卖出（20%止损）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 每日选股
python3 src/screener.py --strategy 1 --top 20   # 策略1
python3 src/screener.py --strategy 2 --top 20   # 策略2
python3 src/screener.py                          # 全部策略

# 历史回测
python3 src/backtest.py TSLA NVDA HOOD --years 3
python3 src/backtest.py --csv your_data.csv --use-tv  # 使用 TradingView 导出数据

# 生成回测报告
python3 src/generate_report.py

# 横截面回测（按 9 个 SPX trend × 波动率 regime 排名股票）
python3 src/cross_section.py
python3 src/cross_section.py --strategy 1 --no-push
python3 src/cross_section.py --limit 5 --no-push   # 烟雾测试
```

> 所有 Python 源码已统一放在 `src/` 目录下；项目根目录只保留 `jojo.pine`、文档和数据/报告目录。

## 项目结构

| 路径 | 说明 |
|------|------|
| `jojo.pine` | TradingView Pine Script v6 版本，与 Python 代码完全一致（位于项目根目录） |
| `src/indicators.py` | jojo 指标核心计算（纯 pandas / numpy，无需外部 TA 库） |
| `src/screener.py` | 全市场扫描（股票+期货），附带整体+当前市场环境回测数据 |
| `src/backtest.py` | 历史回测引擎，支持优化版策略（止损+趋势过滤+波动率过滤） |
| `src/generate_report.py` | 批量回测报告生成（牛熊市分别统计，推送 GitHub + S3） |
| `src/cross_section.py` | 横截面回测：按 9 个 SPX 趋势 × 波动率 regime 排名每股 S1/S2 表现（仅推 GitHub） |
| `src/data_loader.py` · `src/download_ohlc.py` | 本地 OHLC 缓存读写（parquet 每股一份） |
| `src/test_logic.py` | 项目唯一测试入口（基于 assert，无 pytest 配置） |
| `data/` · `reports/` · `logs/` | 缓存数据、生成报告、运行日志（均已 gitignore） |
| `CLAUDE.md` · `README.md` · `README.zh.md` | 项目文档（英文 + 中文） |
| `requirements.txt` | Python 依赖 |

## 扫描输出字段

每个信号包含：基本信息（ticker, 英文名, 中文名, 行业, 市值, 收盘价, jojo 指标值）+ 整体历史回测（交易次数, 胜率, 累计收益, 盈亏比, 最大回撤）+ **当前市场环境回测**（根据 SPX vs SMA225 自动判断牛/熊市，只展示对应环境的历史数据）。

## 过滤规则

- 股票：市值 ≥ 1B USD，排除 ETF
- 商品期货：不做市值过滤

## 覆盖范围

- **股票**: NASDAQ + NYSE 全部（约 6000+）
- **商品期货**: 黄金(GC=F), 白银(SI=F), 原油(CL=F), 天然气(NG=F), 铜(HG=F), 铂金(PL=F)

## 横截面回测 (cross_section.py)

跨股票回测 jojo Strategy 1 / 2，按 9 个市场环境（SPX 趋势 × 波动率分位）统计每只股票在每个环境下的表现，找出"哪些股票最适合哪种策略 + 哪种市况"。

**最新报表**：[reports/cross_section_2026-04-29.md](reports/cross_section_2026-04-29.md) · 历史归档见 [reports/](reports/)。

```bash
# 全策略 + 推送 GitHub（默认）
python3 src/cross_section.py

# 仅 Strategy 1，本地运行不推送
python3 src/cross_section.py --strategy 1 --no-push

# 调阈值
python3 src/cross_section.py --top 50 --min-trades 10

# 烟雾测试（前 5 只标的）
python3 src/cross_section.py --limit 5 --no-push
```

### 股票池

`(本地 OHLC 缓存) ∩ (当前 Russell 1000 ∪ S&P 500) ∩ (历史 ≥ 3 年)`，外加 6 只商品期货（不受成分限制）。当前约 960 只 ticker。

3 年阈值故意宽松，让 HOOD / RIVN / RKLB / COIN / SMR 等近年 IPO 进入分析；报表头部对历史 < 5 年的票打 "thin-sample caveat" 警告。

### Regime（9 = 3 趋势 × 3 波动）

- **趋势 (SPX)**：`bull` = close ≥ SMA225 且 SMA50 ≥ SMA200；`bear` = close < SMA225 且 SMA50 < SMA200；其他为 `neutral`。
- **波动率 (SPX)**：30 日年化已实现波动，按 **5 年滚动**百分位排名。`low_vol` ≤ 33%，`mid_vol` 33–67%，`high_vol` > 67%。
- 所有输入只用当日及之前数据 — 无未来函数。`src/test_logic.py` 的 truncation 测试断言截断 SPX 历史后，cutoff 当天 regime 不变。

### 排名

按 `(ticker, strategy, regime)` 分组：

```
score = profit_factor × √trades       # 主排名
过滤: trades ≥ 5
平局: total_pnl 降序 → win_rate 降序
pf = inf 组 → 单独 "perfect-record" 列表
```

### 输出

- `reports/cross_section_<日期>.md` — 每个 regime × 策略 top-30 markdown 表格。
- `reports/cross_section_<日期>.csv` — 完整 `(ticker, strategy, regime)` 聚合（不做 top-N 过滤、不做 `trades ≥ 5` 过滤）。

### 配置说明

- 依赖本地 OHLC 缓存（先跑 `python3 src/download_ohlc.py --init` 生成）
- 首次运行会从 Wikipedia 抓 Russell 1000 + S&P 500 成分到 `data/index_members.json`
- 仅推 GitHub，不推 S3

## 每日提醒 (daily_alert.py)

每个美股交易日收盘后，把当日 jojo Strategy 1 / Strategy 2 信号过滤到 cross-section 当前 9 桶 regime 下 top-30 的票，通过 Telegram 推送。

```bash
# 默认：扫描 + 过滤 + 发 Telegram（无信号则静默退出）
python3 src/daily_alert.py

# 仅本地预览消息，不发 Telegram
python3 src/daily_alert.py --dry-run

# 调整 top-N（默认 30）
python3 src/daily_alert.py --top 50
```

### 配置

1. 通过 `@BotFather` 创建 Telegram bot，加入目标群。
2. 复制 `.env.example` 为 `.env` 并填入 `TELEGRAM_BOT_TOKEN` 与 `TELEGRAM_CHAT_ID`（`.env` 已 gitignored）。
3. 确认 `reports/cross_section_*.csv` 存在；脚本读最新一个。
4. 可选：cron 配置为每个美股收盘后跑。机器时区 = `Asia/Shanghai` 时北京 09:00 Tue–Sat：

   ```
   0 9 * * 2-6 cd /home/yixiang/jojo_quant && /usr/bin/python3 src/daily_alert.py >> logs/daily_alert.log 2>&1
   ```

   机器时区 = UTC 时改为 `0 1 * * 2-6`（UTC 01:00 = 北京 09:00）。

### 行为说明

- 若 yfinance 当日 SPX 数据未更新，脚本以退出码 1 中止，不发消息——稍后手动重跑或等下一次 cron。
- 若 S1 和 S2 都没有符合 top-30 的信号，脚本静默退出（不发"无信号"消息——安静日就让它安静）。
- 公司信息实时从 FMP 拉取（每个 alert ticker 一次请求）；商品期货使用本地硬编码名称回退。

## `reports/` 维护策略

`reports/` 进入 git，作为对外可浏览的归档。

- **只 commit 全 universe 跑出来的报表**，不 commit `--limit` 烟雾测试或调试片段。
- **更新节奏**：月度。每月跑一次完整版，新文件以日期命名（旧文件可保留或替换）。
- 烟雾测试用 `--no-push`，或 commit 前删干净。
- 如果误 commit 了部分 / 烟雾报表，删掉并 push 删除 — 数据可从 cache 重新生成。

## jojo 指标详解

jojo 指标是一个**复合动量震荡指标**，将 6 个子指标归一化到 0–100 后加权合成，再用 EMA 平滑。最终输出值在 0–100 之间：

- **> 76**：超买区域（策略1买入线）
- **68**：策略1卖出线
- **51**：中线（策略2卖出线）
- **< 28**：超卖区域（策略2买入区）

### 计算公式

```
index_raw = RSI × 0.1 + WR × 0.2 + CMO × 0.1 + KD × 0.3 + TSI × 0.2 + ADXRSI × 0.1
jojo = EMA(index_raw, 3)
```

### 6 个子指标

#### 1. RSI — 相对强弱指数 (Relative Strength Index)

- **权重**：10%
- **参数**：`length = 14`
- **范围**：0 – 100
- **含义**：衡量近期涨幅与跌幅的相对强度。RSI > 70 通常视为超买，< 30 视为超卖。
- **公式**：
  ```
  RS = RMA(涨幅, 14) / RMA(跌幅, 14)
  RSI = 100 - 100 / (1 + RS)
  ```
  其中 RMA 是 Wilder 平滑（指数移动平均的变体，alpha = 1/length）。

#### 2. WR — 威廉指标 (Williams %R)

- **权重**：20%
- **参数**：`length = 14`
- **原始范围**：-100 – 0（标准 Williams %R）
- **归一化**：加 100 后变为 0 – 100
- **含义**：衡量收盘价在近期最高价和最低价之间的位置。值越高表示越接近区间顶部（偏强）。
- **公式**：
  ```
  WR = -100 × (最高价 - 收盘价) / (最高价 - 最低价) + 100
  ```
  其中最高价/最低价取过去 14 根 K 线的滚动最大/最小值。

> **注意**：归一化后 WR 与 Stochastic %K 的公式完全相同：`100 × (close - lowest) / (highest - lowest)`。

#### 3. CMO — 钱德动量震荡指标 (Chande Momentum Oscillator)

- **权重**：10%
- **参数**：`length = 14`
- **原始范围**：-100 – 100
- **归一化**：`(CMO + 100) / 2` 变为 0 – 100
- **含义**：类似 RSI，但直接用涨跌幅度差值与总量的比率，对动量变化更敏感。
- **公式**：
  ```
  sum_gain = SUM(涨幅, 14)
  sum_loss = SUM(跌幅, 14)
  CMO = 100 × (sum_gain - sum_loss) / (sum_gain + sum_loss)
  归一化 = (CMO + 100) / 2
  ```

#### 4. KD — 随机指标 %K (Stochastic %K)

- **权重**：30%（最高权重）
- **参数**：`length = 14`
- **范围**：0 – 100
- **含义**：衡量收盘价在近期价格通道中的相对位置。%K > 80 表示接近通道顶部（强势），< 20 表示接近底部（弱势）。这是 jojo 中权重最大的子指标，对短期价格位置变化最敏感。
- **公式**：
  ```
  %K = 100 × (收盘价 - 14日最低价) / (14日最高价 - 14日最低价)
  ```

#### 5. TSI — 真实强度指数 (True Strength Index)

- **权重**：20%
- **参数**：`short_length = 7, long_length = 14`
- **原始范围**：-1 – 1（Pine Script 的 `ta.tsi()` 返回值）
- **归一化**：`(TSI + 1) / 2 × 100` 变为 0 – 100
- **含义**：双重 EMA 平滑的动量指标，比 RSI 更平滑，能更好地反映趋势方向和强度，同时过滤掉短期噪音。
- **公式**：
  ```
  diff = close - close[1]                      # 每日价格变动
  double_smooth    = EMA(EMA(diff, 14), 7)     # 变动的双重平滑
  abs_double_smooth = EMA(EMA(|diff|, 14), 7)  # 绝对变动的双重平滑
  TSI_raw = double_smooth / abs_double_smooth   # [-1, 1]
  TSI = (TSI_raw + 1) / 2 × 100                # [0, 100]
  ```

#### 6. ADXRSI — ADX 的 RSI（方向性过滤器）

- **权重**：10%
- **参数**：`DI length = 14, ADX smoothing = 18, RSI length = 14`
- **范围**：0 – 100
- **含义**：先计算 ADX（平均趋向指数，衡量趋势强度），再对 ADX 取 RSI，最后根据 K 线方向（阳线/阴线）调整符号。这使得 jojo 能区分上涨趋势和下跌趋势中的 ADX 强度。
- **公式**：
  ```
  # 1. DMI (方向运动指标)
  +DM = max(high - high[1], 0)  当 high变动 > low变动 时
  -DM = max(low[1] - low, 0)    当 low变动 > high变动 时
  ATR = RMA(TrueRange, 14)
  +DI = 100 × RMA(+DM, 14) / ATR
  -DI = 100 × RMA(-DM, 14) / ATR

  # 2. ADX (平均趋向指数)
  DX  = 100 × |+DI - -DI| / (+DI + -DI)
  ADX = RMA(DX, 18)

  # 3. ADXRSI (方向性调整)
  sign = +1（阳线）或 -1（阴线）
  ADXRSI = (RSI(ADX, 14) × sign + 100) / 2
  ```

### 平滑方法

jojo 中使用了两种平滑方法，均与 TradingView 的实现完全一致：

| 方法 | 公式 | 用途 |
|------|------|------|
| **RMA (Wilder 平滑)** | `rma[i] = val × (1/n) + rma[i-1] × (1 - 1/n)` | RSI 内部的涨跌幅平滑、ATR、DI、ADX |
| **EMA (指数移动平均)** | `ema[i] = val × (2/(n+1)) + ema[i-1] × (1 - 2/(n+1))` | TSI 内部的双重平滑、最终 index 的 EMA(3) |

两者均以前 N 个值的 **SMA（简单平均）** 作为种子值初始化，这是匹配 TradingView 计算结果的关键。

## 依赖

- **yfinance** — 从 Yahoo Finance 批量下载 OHLC 数据
- **pandas** — 数据处理与时间序列操作
- **numpy** — 数值计算
- **requests** — 从 NASDAQ 网站获取股票列表
- **lxml** — Wikipedia 表格解析（`pd.read_html` 后端）

> 无需 `pandas_ta` 或其他技术分析库，所有指标均从头实现。
