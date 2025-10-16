# 配置系统说明

## 概述

配置系统将策略特定的配置从全局配置中分离出来，每个策略都有自己的配置文件，提供了更好的隔离性和灵活性。配置系统基于 `FrozenConfig` 类，支持属性跟踪和动态配置更新。

## 系统架构
```
frozen/
├── basis/config/
│   └── default_config.yaml  # 仅系统级默认配置
├── demo/strategy/
│   ├── momentum/
│   │   ├── config.yaml      # 动量策略配置
│   │   └── strategy.py
│   ├── mean_reversion/
│   │   ├── config.yaml      # 均值回归策略配置
│   │   └── strategy.py
```

## 使用方法

### 1. 在代码中使用策略配置

```python
from frozen.basis.constant import get_strategy_config

# 加载特定策略的配置（传入策略文件路径）
config = get_strategy_config("demo/strategy/momentum/momentum.py")

# 访问配置参数（通过属性）
print(f"策略基准: {config.benchmark}")
print(f"开始日期: {config.start_date}")
print(f"资产范围: {config.asset_range}")
print(f"优化器: {config.optimizer}")
print(f"初始资金: {config.init_capital}")

# 动态更新配置
config.update_config('start_date', '20250101')
config.update_config('portfolio_config.optimizer', 'equal-weight')

# 获取嵌套配置
risk_config = config.get_strategy_specific_config('risk_config')
```

### 2. 命令行工具使用

#### 创建新策略
```bash
python -m frozen.basis.cli create my_new_strategy
```

#### 列出所有策略
```bash
python -m frozen.basis.cli list
```

#### 验证策略配置
```bash
python -m frozen.basis.cli validate momentum
```

#### 显示策略配置详情
```bash
python -m frozen.basis.cli show momentum
```

#### 复制已有策略
```bash
python -m frozen.basis.cli copy momentum momentum_v2
```

### 3. 编程方式管理配置

```python
from frozen.basis.manager import ConfigManager

# 创建新策略配置模板
ConfigManager.create_strategy_config_template(
    "demo/strategy/my_strategy", 
    "my_strategy"
)

# 验证配置
validation = ConfigManager.validate_strategy_config("demo/strategy/momentum")
if validation['valid']:
    print("配置有效")
else:
    print("配置错误:", validation['errors'])

# 列出所有可用策略
strategies = ConfigManager.list_available_strategies()
for name, info in strategies.items():
    print(f"{name}: {info['description']}")
```

## 配置文件结构

### 策略配置文件示例 (`config.yaml`)

```yaml
# 策略元数据
strategy_meta:
  name: "momentum"
  description: "动量策略"
  version: "1.0.0"
  author: "策略团队"

# 基础配置
base_config:
  database: "duckdb"

# 运行配置
run_config:
  region: "cn"
  trade_unit: 100
  limit_threshold: 0.1
  sell_rule: "currentBar"
  seed: 42
  enable_gif: false

# 客户端配置
client_config:
  init_capital: 1000000
  slippage:
    slippage_type: "byRate"
    slippage_rate: 0.0001
  commission: 0.0003
  stamp_duty: 0.001
  min_cost: 5

# 策略配置
strategy_config:
  benchmark: "000300.SH"
  index_code: "000300.SH"
  start_date: "20200101"
  end_date: "20240702"
  asset_range: "(0, 30)"
  date_rule: "W-WED"

# 组合配置
portfolio_config:
  optimizer: "mean-variance"
  cov_est:
    cov_method: "custom"
    cov_window: 60
  opt_func: "sharpe"
  long_short: false

# 风险管理
risk_config:
  take_profit: .inf
  stop_loss: -.inf
  max_position_size: 0.1
  sector_concentration: 0.3
  rebalance_threshold: 0.05

# 因子配置
factor_config:
  calc:
    lookback_window: 60
  filter:
    universe:
      include_SH: true
      include_SZ: true
      include_BJ: false
      include_GEM: false
      include_STAR: false
      include_ST: false
      include_delist: false
      min_list_days: 100
```

## 配置属性访问

### 属性式访问（推荐）
```python
config = get_strategy_config("demo/strategy/momentum/momentum.py")

# 直接访问属性
start_date = config.start_date
benchmark = config.benchmark
init_capital = config.init_capital

# 修改属性（自动同步到all_config）
config.start_date = "20250101"
config.max_position_size = 0.15
```

### 路径式访问
```python
# 获取嵌套配置值
cov_method = config.get_config('portfolio_config.cov_est.cov_method')
include_SH = config.get_config('factor_config.filter.universe.include_SH')

# 更新嵌套配置值
config.update_config('portfolio_config.cov_est.cov_window', 90)
config.update_config('risk_config.take_profit', 0.2)
```

## 迁移现有策略

### 步骤1: 创建策略文件夹
```bash
mkdir -p demo/strategy/your_strategy_name
```

### 步骤2: 创建配置文件
```bash
python -m frozen.basis.cli create your_strategy_name
```

### 步骤3: 修改配置参数
编辑 `demo/strategy/your_strategy_name/config.yaml` 文件，根据你的策略需求调整参数。

### 步骤4: 更新策略代码
```python
from frozen.basis.constant import get_strategy_config
import os

class YourStrategy:
    def __init__(self):
        # 获取当前文件路径
        current_file = os.path.abspath(__file__)
        self.config = get_strategy_config(current_file)
        
    def run(self):
        # 使用配置参数
        print(f"策略运行期间: {self.config.start_date} 到 {self.config.end_date}")
        print(f"基准指数: {self.config.benchmark}")
        print(f"资产范围: {self.config.asset_range}")
        
        # 动态调整配置
        if some_condition:
            self.config.update_config('rebalance_threshold', 0.08)
```

## 配置验证和监控

### 自动验证
```python
# 配置变更时自动触发相关计算
config.start_date = "20250101"  # 自动重新计算 start_date_extend 等
config.cov_window = 90          # 自动重新计算依赖参数
```

### 手动验证
```python
# 获取所有配置参数
all_params = config.get_all_params()

# 验证配置
validation_result = config.validate()
```

## 向后兼容性

- 全局 `frozen_config` 对象仍然可用，使用系统默认配置
- 现有代码无需立即修改，但建议逐步迁移到策略特定配置

## 好处

1. **属性跟踪**: 配置变更自动同步，支持依赖计算
2. **类型安全**: 通过属性访问提供更好的IDE支持
3. **动态更新**: 支持运行时配置修改
4. **隔离性**: 每个策略都有独立的配置，互不影响
5. **版本控制**: 策略配置与策略代码一起进行版本控制
6. **可复用性**: 可以轻松复制和修改策略配置
7. **灵活性**: 不同策略可以使用完全不同的参数设置
8. **可维护性**: 配置文件结构清晰，易于理解和维护

## 注意事项

- 确保策略文件夹中有 `config.yaml` 文件
- 配置文件必须遵循正确的YAML格式
- 日期格式必须是 `YYYYMMDD`
- 使用 `get_strategy_config()` 时传入策略文件的完整路径
- 使用命令行工具验证配置文件的正确性
- 某些配置变更会触发重新计算（如日期、窗口参数等） 