# Backtesting Framework Extensions

This directory contains multiple backtesting frameworks integrated into the Frozen platform. Each framework has its own strengths and characteristics, making them suitable for different types of quantitative trading strategies and analysis needs.

## Framework Overview

### 1. Backtesting.py
**Version**: Latest  
**License**: AGPL 3.0  
**Best for**: Simple, fast backtesting with clean API

**Key Features:**
- Lightweight and blazing fast backtesting framework
- Simple API with minimal learning curve
- Built-in interactive visualization with Bokeh
- Parameter optimization and heatmap analysis
- Machine learning integration support
- Works with individual tradeable assets (single-asset strategies)

**Use Cases:**
- Technical indicator-based strategies
- Moving average crossovers
- Position entrance/exit signal optimization
- Quick strategy prototyping

### 2. Backtrader
**Version**: 1.9.78.123  
**License**: GPL 3.0  
**Best for**: Comprehensive backtesting with extensive features

**Key Features:**
- Full-featured backtesting platform with rich ecosystem
- Extensive built-in indicators and analyzers
- Multiple data feed support (CSV, Yahoo Finance, Interactive Brokers, etc.)
- Advanced order management and position sizing
- Real-time trading capabilities
- Comprehensive visualization tools
- Support for multiple timeframes and data resampling

**Use Cases:**
- Complex multi-asset strategies
- Portfolio management
- Real-time trading systems
- Research and development of sophisticated strategies

### 3. PyBroker
**Version**: 1.2.11  
**License**: Apache 2.0 with Commons Clause  
**Best for**: Machine learning-driven strategies

**Key Features:**
- Built-in machine learning model training and evaluation
- Advanced caching system for data and models
- Bootstrap metrics for robust performance evaluation
- Custom data source integration
- Position sizing and ranking systems
- Stop-loss and take-profit management
- Vectorized operations for performance

**Use Cases:**
- ML-based trading strategies
- Factor-based investing
- Model-driven portfolio construction
- Statistical arbitrage strategies

### 4. VectorBT
**Version**: 0.28.0  
**License**: Apache 2.0 with Commons Clause  
**Best for**: High-performance vectorized backtesting

**Key Features:**
- Extremely fast vectorized operations using Numba
- Built-in portfolio optimization tools
- Advanced signal generation and analysis
- 3D visualization capabilities
- Walk-forward optimization
- Pairs trading and statistical arbitrage support
- High-frequency data handling

**Use Cases:**
- High-frequency trading strategies
- Portfolio optimization
- Pairs trading
- Statistical arbitrage
- Large-scale strategy testing

### 5. Zipline
**Version**: Latest  
**License**: Apache 2.0  
**Best for**: Institutional-grade backtesting

**Key Features:**
- Originally developed by Quantopian
- Pipeline-based data processing
- Built-in risk management tools
- Calendar-aware trading
- Extensive financial data integration
- Professional-grade portfolio analytics
- Support for complex order types

**Use Cases:**
- Institutional portfolio management
- Factor research and development
- Risk-adjusted strategy evaluation
- Professional quantitative research

## Framework Comparison

| Feature | Backtesting.py | Backtrader | PyBroker | VectorBT | Zipline |
|---------|----------------|------------|----------|----------|---------|
| **Learning Curve** | Easy | Medium | Medium | Hard | Hard |
| **Performance** | Fast | Medium | Fast | Very Fast | Medium |
| **ML Integration** | Basic | No | Advanced | Basic | No |
| **Visualization** | Excellent | Good | Basic | Advanced | Basic |
| **Data Sources** | Manual | Extensive | Built-in | Manual | Extensive |
| **Real-time Trading** | No | Yes | No | No | No |
| **Portfolio Management** | Basic | Advanced | Advanced | Advanced | Advanced |
| **Documentation** | Good | Excellent | Good | Good | Good |

## Getting Started

### VSCode Configuration
Add the following config to VSCode `settings.json` for pylance to recognize all frameworks:

```json
"python.analysis.extraPaths": [
    "frozen/engine/backtest/extensions"
]
```

### Zipline Cython Build Setup

Zipline requires Cython compilation for optimal performance. Follow these steps to build the Cython extensions:

#### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install cython numpy setuptools
```

#### Build Steps

1. **Navigate to the extensions directory:**
   ```bash
   cd frozen/engine/backtest/extensions
   ```

2. **Clean existing compiled files (optional):**
   ```bash
   # On macOS/Linux
   ./rebuild-cython.sh
   
   # Or manually clean and rebuild
   find zipline -name "*.c" -delete
   find zipline -name "*.so" -delete
   find zipline -name "*.html" -delete
   ```

3. **Build Cython extensions:**
   ```bash
   python setup.py build_ext --inplace
   ```

4. **Verify the build:**
   ```python
   import zipline
   print("Zipline successfully imported with Cython extensions")
   ```

#### Troubleshooting

- **Import errors**: Ensure all Cython extensions are properly compiled
- **Performance issues**: Rebuild extensions if you notice slower performance
- **Platform-specific issues**: The build process may vary slightly between operating systems

#### Alternative Build Method
If you encounter issues with the standard build process, you can also use:
```bash
python -m pip install -e . --no-deps
```

### Framework Selection Guide

**Choose Backtesting.py if:**
- You're new to backtesting
- You need quick results and simple strategies
- You want beautiful visualizations
- You're working with single assets

**Choose Backtrader if:**
- You need comprehensive features
- You want real-time trading capabilities
- You're building complex multi-asset strategies
- You need extensive indicator libraries

**Choose PyBroker if:**
- You're using machine learning in your strategies
- You need robust statistical evaluation
- You want built-in data caching
- You're doing factor-based research

**Choose VectorBT if:**
- You need maximum performance
- You're doing high-frequency analysis
- You want advanced portfolio optimization
- You're working with large datasets

**Choose Zipline if:**
- You need institutional-grade features
- You're doing professional research
- You need advanced risk management
- You're building production systems

## Examples and Tutorials

Each framework includes comprehensive examples and tutorials in the `demo/resources/` directory:

- `backtesting/` - Quick start guides and strategy examples
- `backtrader/` - Comprehensive tutorials covering all aspects
- `pybroker/` - ML-focused examples and data source tutorials
- `vectorbt/` - High-performance examples and optimization guides
- `quantopian/` - Professional-grade examples and best practices

## Contributing

When adding new features or fixing bugs in any framework, please ensure:
1. Maintain compatibility with the existing Frozen platform
2. Follow the framework's original coding standards
3. Add appropriate tests and documentation
4. Update this README if adding new frameworks