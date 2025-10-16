## Factor Management System

This module maintains a factor inventory that provides functionalities including factor registration, dependency management, sequential execution, data storage, and intelligent backfilling.

### Core Functionalities

#### 1. Factor Registration Mechanism

- Management of factor metadata (name, description, category, tag, etc.)
- Definition and verification of dependencies
- Detection of cyclic dependencies

#### 2. Dependency Management

- Dependency management based on Directed Acyclic Graph (DAG)
- Automatic topological sorting to determine execution order
- Support for complex multi-level dependencies

#### 3. Sequential Execution

- Automatic determination of factor calculation order based on dependencies
- Support for incremental calculation and forced recalculation
- Automatic transfer of dependent factor data

#### 4. Factor Storage Functionality

- Support for two types of databases: MongoDB and DuckDB
- Unified storage interface for easy expansion
- Support for batch updates and parallel processing

#### 5. Intelligent Historical Data Backfilling

- Automatic detection of missing data
- On-demand backfilling of historical data within a specified time range
- Intelligent judgment on whether backfilling is required

### System Architecture

```
frozen/factor/library/
├── inventory.py                   # main file
├── handlers/                      # database handlers
│   ├── __init__.py
│   ├── base.py                    # abstract base class
│   ├── mongo.py                   # MongoDB handler
│   └── duck.py                    # DuckDB handler
└── README.md                      # this file

unit_test/
└── test_factor_management.py      # test script
```

### Quick Start

### 基本使用流程

```python
from frozen.factor.library.inventory import FactorManager
from frozen.data.database import DatabaseTypes

# 1. 初始化管理器
manager = FactorManager(DatabaseTypes.DUCKDB)

# 2. 注册因子
manager.register_factor(
    name="price",
    description="股票收盘价",
    category="basic"
)

manager.register_factor(
    name="returns", 
    description="日收益率",
    dependencies=["price"],
    category="derived"
)

# 3. 定义计算函数
def compute_price():
    # 你的价格数据计算逻辑
    return Factor(price_data)

def compute_returns(price=None):
    # 基于price计算returns
    returns_data = price.data.pct_change()
    return Factor(returns_data)

factor_functions = {
    'price': compute_price,
    'returns': compute_returns
}

# 4. 执行计算和存储
manager.compute_and_store_factors(
    factor_functions=factor_functions,
    table_name='my_factors'
)

# 5. 读取数据
factors = manager.get_factors(
    table_name='my_factors',
    factor_names=['price', 'returns']
)
```

### 运行演示

```bash
cd /Users/lig/Documents/GitHub/Frozen
python frozen/factor/library/factor_management_demo.py
```

### 运行测试

```bash
cd /Users/lig/Documents/GitHub/Frozen
python unit_test/test_factor_management.py
```

## 详细功能说明

### 因子注册

```python
# 基础因子注册
manager.register_factor(
    name="volume",           # 因子名称（必填）
    description="成交量",    # 因子描述
    category="basic",        # 因子类别
    tags=["volume", "basic"] # 标签列表
)

# 有依赖的因子注册
manager.register_factor(
    name="vwap",
    description="成交量加权平均价格", 
    dependencies=["price", "volume"],  # 依赖列表
    category="composite"
)
```

### 查看因子信息

```python
# 查看所有因子
all_factors = manager.registry.list_factors()

# 按类别查看
basic_factors = manager.registry.list_factors(category="basic")

# 按标签查看
volume_factors = manager.registry.list_factors(tags=["volume"])

# 查看执行顺序
execution_order = manager.registry.get_execution_order()

# 查看具体因子信息
factor_info = manager.registry.get_factor_info("vwap")
print(f"因子描述: {factor_info.description}")
print(f"依赖关系: {factor_info.dependencies}")
```

### 因子计算函数

因子计算函数需要遵循以下规范：

```python
def compute_factor_name(dependency1=None, dependency2=None, **kwargs):
    """
    因子计算函数
    
    Args:
        dependency1: 依赖因子1 (Factor对象)
        dependency2: 依赖因子2 (Factor对象) 
        **kwargs: 其他参数
    
    Returns:
        Factor: 计算结果，必须是Factor对象
    """
    # 检查依赖
    if dependency1 is None:
        raise ValueError("dependency1 is required")
    
    # 计算逻辑
    result_data = some_calculation(dependency1.data, dependency2.data)
    
    # 返回Factor对象
    return Factor(result_data)
```

### 数据读取

```python
# 读取单个因子
price_factor = manager.handler.read_factor('table_name', 'price')

# 读取多个因子
factors = manager.get_factors(
    table_name='table_name',
    factor_names=['price', 'volume', 'returns']
)

# 按时间范围读取
factors = manager.get_factors(
    table_name='table_name', 
    factor_names=['price'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

### 智能回补

```python
# 回补所有因子的历史数据（默认252个交易日）
manager.smart_backfill('table_name')

# 回补特定因子
manager.smart_backfill(
    table_name='table_name',
    target_days=500,  # 回补500天
    factor_names=['price', 'volume']
)
```

### 向后兼容性

原有的`FactorBase`类仍然可以使用：

```python
from frozen.factor.library.inventory import FactorBase

# 传统用法
factor_base = FactorBase("duckdb")
factor_base.factor2db(factor, 'factor_name', 'table_name')
loaded_factor = factor_base.read_factor('table_name', 'factor_name')
```

## 技术特点

### 依赖管理
- 使用networkx库实现图算法（可选依赖）
- 支持无networkx环境下的简单拓扑排序
- 自动检测和防止循环依赖

### 数据库支持
- 采用工厂模式，易于扩展新的数据库类型
- 统一的接口设计，隐藏数据库差异
- 支持MongoDB的并行写入优化

### 错误处理
- 完善的异常处理机制
- 详细的错误信息和日志记录
- 支持部分失败的容错处理

### 性能优化
- 批量数据处理
- 增量计算支持
- 并行数据库操作（MongoDB）

## 设计原则

1. **可扩展性**: 采用抽象基类和工厂模式，易于添加新功能
2. **向后兼容**: 保留原有接口，支持渐进式升级
3. **易用性**: 提供简洁的API和丰富的文档
4. **可靠性**: 完善的错误处理和测试覆盖
5. **性能**: 优化的数据处理和存储策略

## 文件说明

### 核心实现文件

- `frozen/factor/library/inventory.py`: 主要实现文件，包含所有核心类
- `frozen/factor/library/handlers/base.py`: 抽象基类
- `frozen/factor/library/handlers/mongo.py`: MongoDB数据库处理器
- `frozen/factor/library/handlers/duck.py`: DuckDB数据库处理器

### 文档和示例

- `frozen/factor/library/factor_management_demo.py`: 完整功能演示
- `unit_test/test_factor_management.py`: 单元测试

## 依赖关系

### 必需依赖
- pandas: 数据处理
- numpy: 数值计算
- pymongo: MongoDB连接
- duckdb: DuckDB连接

### 可选依赖
- networkx: 图算法支持（推荐安装）
- tqdm: 进度条显示

## 最佳实践

### 1. 因子命名规范
- 使用小写字母和下划线
- 名称要有意义，便于理解
- 避免使用SQL关键字

### 2. 依赖关系设计
- 合理规划因子层次结构
- 避免过深的依赖链
- 基础数据因子不应有依赖

### 3. 计算函数编写
- 添加参数检查
- 处理异常情况
- 确保返回Factor对象
- 添加适当的文档说明

### 4. 数据管理
- 定期清理不需要的因子
- 合理设置回补时间范围
- 根据数据规模选择合适的数据库类型

## 故障排除

### 常见问题

1. **循环依赖错误**
   - 检查因子依赖关系
   - 重新设计依赖结构

2. **因子计算失败**
   - 检查依赖因子是否存在
   - 验证计算函数逻辑
   - 查看错误日志

3. **数据库连接问题**
   - 检查数据库配置
   - 确认数据库服务状态
   - 验证连接参数

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.INFO)

# 查看执行顺序
print(manager.registry.get_execution_order())

# 检查因子是否存在
print(manager.handler.check_factor_exists('table_name', 'factor_name'))

# 查看因子数据范围
start, end = manager.handler.get_factor_date_range('table_name', 'factor_name')
print(f"数据范围: {start} 到 {end}")
```

## 使用建议

### 数据库选择
- **开发环境**: 推荐DuckDB，轻量级，易于使用
- **生产环境**: 大数据推荐MongoDB，支持高并发和分布式

### 注意事项

1. 确保数据库服务正常运行
2. 计算函数必须返回Factor对象
3. 避免创建循环依赖关系
4. 合理设置回补时间范围

## 后续扩展

### 计划功能
- [ ] 因子性能监控
- [ ] 分布式计算支持
- [ ] 因子版本管理
- [ ] Web界面管理
- [ ] 更多数据库支持（ClickHouse、PostgreSQL等）

### 扩展点
- 新增数据库支持：继承`FactorHandler`实现新的处理器
- 自定义计算引擎：扩展计算函数调用机制
- 监控和告警：在关键节点添加监控逻辑

