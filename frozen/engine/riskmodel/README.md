Portfolio Optimization
By default, use equal-weighted portfolio allocation as the baseline strategy.

Frozen adopts modern portfolio theory (MPT) principles to obtain rebalancing positions. Specifically, the system implements a risk model, where Mean-Variance Optimization (MVO) is used to achieve optimal asset allocation weights.

MVO requires an optimization target and key components:

**Optimization Objectives**

* Maximize risk-adjusted returns (Maximum Sharpe Ratio Portfolio)
* Minimize portfolio volatility (Global Minimum Variance Portfolio)

**Covariance Matrix Estimation**

* Sample covariance
* Shrinkage estimation
* Factor-based models
* Robust statistical estimators

**Real-world constraints**

* Position limits and bounds
* Turnover restrictions
* Transaction costs
* Sector exposure limits
* Liquidity requirements

**Efficient Frontier Analysis**

* Generates efficient frontier curve
* Maps risk-return relationships
* Identifies optimal portfolios
* Visualizes risk-return trade-offs
* Enables portfolio selection based on risk tolerance

**Constraint Incorporation**

* Long-only constraints
* Asset weight bounds
* Group constraints
* Turnover constraints
* Risk factor constraints
* Transaction cost constraints

**Global Minimum Variance Portfolio**

* Constructs lowest possible risk portfolio
* Optimization features:
  * No return assumptions required
  * Pure risk minimization
  * Robust to estimation errors
* Implementation considerations:
  * Covariance matrix stability
  * Diversification requirements
  * Rebalancing frequency
  * Transaction cost impact

**Advanced Features**

* Dynamic rebalancing triggers
* Adaptive risk targeting
* Multi-period optimization
* Risk factor decomposition
* Performance attribution analysis
* Stress testing capabilities

This comprehensive framework ensures robust portfolio construction while maintaining practical applicability in real-world market conditions.

### Barra Risk Model

The Barra risk model is now fully implemented and integrated into the Frozen framework. This implementation provides a comprehensive multi-factor risk model for portfolio optimization and risk management.

#### Key Features

**Multi-Factor Framework**
- **Style Factors**: Size, Value, Growth, Momentum, Quality, Volatility, Liquidity, Leverage, Profitability, Beta
- **Industry Factors**: Support for multiple classification standards (SHENWAN, CITIC, custom)
- **Risk Decomposition**: Separates systematic risk (factor) from idiosyncratic risk (specific)

**Mathematical Framework**
The model decomposes stock returns as:
```
r_i = α_i + β_i1 * f_1 + β_i2 * f_2 + ... + β_iK * f_K + ε_i
```

Where:
- `r_i`: Stock i's return
- `β_ik`: Stock i's exposure to factor k
- `f_k`: Factor k's return
- `ε_i`: Stock i's specific return

**Covariance Matrix Decomposition**
```
Σ = B * F * B^T + Δ
```

Where:
- `B`: Factor exposure matrix
- `F`: Factor covariance matrix
- `Δ`: Specific risk diagonal matrix

#### Implementation Details

**1. Factor Exposure Calculation**
- Style factor exposures computed from market data
- Industry factor exposures based on classification standards
- Exposure standardization and winsorization

**2. Factor Return Estimation**
- Cross-sectional regression estimation
- Weighted least squares regression
- Time-series factor return modeling

**3. Risk Modeling**
- Exponentially weighted moving average (EWMA) for time decay
- Newey-West adjustment for autocorrelation
- Volatility regime adjustments

**4. Advanced Analytics**
- Risk attribution analysis
- Stress testing capabilities
- Factor contribution analysis
- Portfolio optimization integration

#### Usage Example

```python
from frozen.engine.riskmodel.covariance.barra import BarraRiskModel
from frozen.engine.riskmodel.optimizer import PortOpt

# Create Barra risk model
barra_model = BarraRiskModel(
    style_factors=['size', 'value', 'momentum', 'quality'],
    industry_classification='sw_l1',
    factor_return_window=252,
    factor_cov_window=252,
    halflife_factor=90,
    halflife_specific=60
)

# Calculate covariance matrix
covariance_matrix = barra_model.predict(returns_data)

# Risk attribution analysis
risk_attribution = barra_model.risk_attribution(portfolio_weights)

# Stress testing
stress_scenarios = {
    'market_crash': {'size': -0.03, 'value': -0.02},
    'value_rotation': {'value': 0.02, 'growth': -0.02}
}
stress_results = barra_model.stress_test(portfolio_weights, stress_scenarios)
```

#### Configuration

```yaml
# Barra risk model configuration
barra_config:
  style_factors:
    - size
    - value
    - growth
    - momentum
    - quality
    - volatility
    - liquidity
    - leverage
    - profitability
    - beta
  industry_classification: sw_l1
  factor_return_window: 252
  factor_cov_window: 252
  halflife_factor: 90
  halflife_specific: 60
  newey_west_lags: 5
  volatility_regime_adjust: true
```

#### Portfolio Optimization Integration

```python
# Use Barra model in portfolio optimization
config.portfolio_config.cov_method = 'barra'
optimizer = PortOpt(config)

# Calculate optimal weights with Barra risk model
optimal_weights = optimizer.calc_portfolio_weights(
    n=n_assets, 
    X=returns_data,
    barra_params={'style_factors': ['size', 'value', 'momentum']}
)

# Get risk attribution for optimized portfolio
risk_attribution = optimizer.get_barra_risk_attribution(
    optimal_weights, returns_data
)
```

#### Benefits

1. **Comprehensive Risk Analysis**: Decomposes portfolio risk into systematic and idiosyncratic components
2. **Factor-Based Insights**: Identifies specific risk factors driving portfolio performance
3. **Stress Testing**: Evaluates portfolio performance under various market scenarios
4. **Optimization Enhancement**: Provides more accurate risk estimates for portfolio optimization
5. **Industry Standards**: Follows established Barra methodology used by institutional investors

#### Demo and Testing

- **Demo**: `/demo/strategy/barra_risk_model_demo/`
- **Tests**: `/frozen/engine/riskmodel/test_barra.py`
- **Documentation**: Comprehensive README with usage examples

The Barra risk model implementation provides institutional-grade risk management capabilities, enabling sophisticated portfolio construction and risk analysis within the Frozen framework.

### Future Enhancements

1. **Enhanced Factor Models**
   - Deep learning-based factor construction
   - Alternative data integration
   - Regime-dependent factor models

2. **Advanced Risk Metrics**
   - Value at Risk (VaR) and Expected Shortfall
   - Maximum Drawdown estimation
   - Tail risk measures

3. **Performance Attribution**
   - Detailed factor contribution analysis
   - Time-varying attribution
   - Benchmark-relative attribution

4. **Integration Improvements**
   - Real-time risk monitoring
   - Automated rebalancing triggers
   - Enhanced visualization tools

Note: Development timeline and specific features may be adjusted based on requirements and priorities.
