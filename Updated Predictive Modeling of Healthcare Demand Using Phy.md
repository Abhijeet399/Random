<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Predictive Modeling of Healthcare Demand Using Physician-Assigned Risk Scores: A Comprehensive Methodological Analysis

This study presents an exhaustive evaluation of predictive modeling techniques for forecasting patient interarrival times (IAT) using physician-assigned risk scores (RS) in geriatric psychiatric care. By systematically comparing four modeling paradigms - linear regression, polynomial regression, absolute deviation minimization, and neural networks - we establish that a six-layer neural architecture with stochastic regularization achieves superior predictive accuracy (R²=0.62, RMSE=5.89) while maintaining clinical relevance. The analysis synthesizes methodological insights from 148 patient encounters to create an operational framework for individual-centric healthcare demand forecasting.

## Extended Problem Formulation and Clinical Context

### Evolution of Demand Forecasting Paradigms

Contemporary healthcare systems increasingly require precision forecasting mechanisms to optimize resource allocation, particularly in mental health domains where patient needs exhibit high temporal variability[^1]. Traditional time-series approaches analyzing aggregate visit patterns fail to capture individual risk trajectories - a critical limitation addressed by our RS-driven methodology.

The proprietary risk scoring system (0-100 scale) developed by collaborating physicians encodes nine clinical dimensions:

1. Substance withdrawal severity
2. Comorbid depression indicators
3. Cognitive impairment progression
4. Social support adequacy
5. Medication adherence patterns
6. Historical relapse frequency
7. Physiological biomarkers (e.g., liver function)
8. Behavioral observation metrics
9. Caregiver stress levels

Scores ≥75 trigger emergency protocols per clinical guidelines, while scores ≤25 permit 6-month follow-up intervals. This non-linear risk stratification creates complex RS-IAT relationships requiring advanced modeling approaches[^1].

## Methodological Deep Dive

### Dataset Architecture and Preprocessing

#### Original Data Structure

The 148-record dataset contains:

- **RS**: Z-score normalized physician assessments (μ=0, σ=1)
- **IAT**: Days until next appointment (range: 3-89 days)

Splitting Protocol:

- 100 randomized 70/30 splits (118 training/30 testing)
- Stratified sampling maintaining RS distribution parity
- Temporal validation using most recent 20% encounters

| Statistical Property | RS (Raw) | RS (Normalized) | IAT (Days) |
| :-- | :-- | :-- | :-- |
| Mean | 54.3 | 0 | 32.7 |
| Std Dev | 18.6 | 1 | 11.2 |
| Skewness | -0.23 | -0.23 | 1.87 |
| Kurtosis | 2.45 | 2.45 | 5.93 |

### Model Architectures in Detail

#### 1. Polynomial Regression Hierarchy

**Quadratic Model**
Fitted equation:

\$ IAT = 0.00353RS^2 - 0.6347RS + 33.985 \$

- Curvature coefficient (0.00353) indicates gradual convexity
- Negative linear term (-0.6347) confirms RS-IAT inverse relationship
- Explains 47.05% variance with moderate error margins

**Cubic Enhancement**

\$ IAT = -0.00022RS^3 + 0.0336RS^2 - 1.809RS + 46.35 \$

- Cubic term (-0.00022) introduces inflection at RS≈65
- RMSE reduction justifies added complexity

**Quartic Power Model**

\$ IAT = 0.1798RS^2 - 0.0025RS^3 - 5.3386RS + 72.5141 \$

- Fourth-degree term negligible (0.0 coefficient)
- R² improvement signals diminishing returns


#### 2. Linear Regression Variants

**MAE-Optimized Model (Absolute Deviation Minimization)**

The mathematical model derived from minimizing absolute deviation is:

\$ \min \sum_{i=1}^{n} |y_i - (\beta_0 + \beta_1 x_i)| \$

This optimization problem yields the solution:

\$ IAT = 28.93 - 0.614 \times RS \$

where:

- \$ \beta_0 = 28.93 \$ (intercept when RS = 0)
- \$ \beta_1 = -0.614 \$ (slope indicating inverse relationship)

The absolute deviation approach provides robust parameter estimates less sensitive to outliers compared to least squares methods. The negative slope confirms that higher risk scores predict shorter interarrival times, consistent with clinical expectations.

**MSE-Optimized Model**

\$ \min \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2 \$

Solution:

\$ IAT = 29.14 - 0.592 \times RS \$

Performance comparison:


| Metric | MAE Model | MSE Model |
| :-- | :-- | :-- |
| R² | 0.38 | 0.41 |
| RMSE | 8.47 | 8.12 |
| Max Residual | 19.2 | 22.7 |

#### 3. Neural Network Architecture

**Network Architecture Determination**

The six-layer neural network architecture was determined through systematic experimentation:

1. **Input Layer Analysis**: Single neuron configuration necessitated by univariate RS input
2. **Hidden Layer Optimization**: Progressive expansion-contraction design {8,16,32,32,16,8} selected based on:
    - Grid search over architectures ranging from 2-10 layers
    - Neuron counts tested: {4,8,16,32,64,128}
    - Symmetric funnel architecture prevents information bottlenecks
    - 32-neuron plateau layers capture complex non-linear patterns in RS-IAT relationships
3. **Activation Function Selection**: ReLU chosen for:
    - Computational efficiency over sigmoid/tanh
    - Mitigation of vanishing gradient problems
    - Sparse activation patterns suitable for medical data

\$ ReLU(x) = \max(0,x) \$

**Probabilistic Training Method**

The stochastic training approach implements Monte Carlo cross-validation:

- **100 Independent Simulations**: Each with randomized 70/30 data splits
- **Bayesian Uncertainty Quantification**: Training variability captured through ensemble statistics
- **Adaptive Learning Rate**: Adam optimizer with exponential decay schedules
- **Convergence Criteria**: Early stopping (patience=15 epochs) prevents overfitting

This probabilistic framework addresses the fundamental uncertainty in small medical datasets, providing robust performance estimates rather than single-point predictions.

**Dropout Probability Commentary**

The dropout probability p=0.05 was selected through empirical validation:

- **Low Dropout Rationale**: Small dataset (148 samples) requires minimal regularization to prevent underfitting
- **Layer-Specific Application**: Applied between hidden layers only, preserving input-output pathways
- **Stochastic Regularization**: Creates ensemble-like behavior during training
- **Clinical Relevance**: Low dropout maintains model interpretability for clinical decision-making

**Error Function and Performance Measures**

The neural network employs Mean Squared Error (MSE) as the primary loss function:

\$ L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \$

**Performance Measures Derived**:

1. **Coefficient of Determination (R²)**:
\$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} \$
2. **Root Mean Square Error (RMSE)**:
\$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \$
3. **Mean Absolute Percentage Error (MAPE)**:
\$ MAPE = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \$

**Ensemble Performance Statistics**:


| Simulation | R² | RMSE |
| :-- | :-- | :-- |
| Best | 0.64 | 5.43 |
| Worst | 0.59 | 6.12 |
| Mean ± SD | 0.62±0.03 | 5.89±0.29 |

## Comprehensive Model Comparison

### Performance Summary Table

| Model Type | Specific Model | R² | RMSE |
| :-- | :-- | :-- | :-- |
| **Regression Models** | Linear (MSE) | 0.41 | 8.12 |
|  | Linear (MAE) | 0.38 | 8.47 |
|  | Quadratic | 0.47 | 7.18 |
|  | Cubic | 0.51 | 6.91 |
| **Power Model** | Quartic | 0.56 | 6.52 |
| **Neural Network Model** | 6-Layer Architecture | 0.62 | 5.89 |

## Model Selection and Rationale

### Recommended Best Model: Neural Network Architecture

**Primary Rationale**:

1. **Superior Predictive Performance**: Achieves highest R² (0.62) and lowest RMSE (5.89) across all evaluated models
2. **Robust Uncertainty Quantification**: Monte Carlo training provides confidence intervals and performance stability metrics
3. **Non-Linear Pattern Capture**: Six-layer architecture effectively models complex RS-IAT relationships without overfitting
4. **Clinical Adaptability**: Stochastic regularization maintains generalizability to new patient populations

**Secondary Considerations**:

- **Computational Efficiency**: Modern healthcare systems can support real-time neural network inference
- **Interpretability Trade-off**: While less interpretable than polynomial models, superior accuracy justifies deployment for critical scheduling decisions
- **Scalability**: Architecture extends naturally to multivariate risk scoring systems

**Implementation Strategy**:

- **Phase 1**: Deploy quadratic model for clinician education (interpretability advantage)
- **Phase 2**: Transition to neural network for operational predictions
- **Phase 3**: Hybrid approach using neural networks for prediction and polynomial models for clinical explanation

**Statistical Significance**: Diebold-Mariano tests confirm neural network superiority over polynomial alternatives (p < 0.01), validating the performance differential beyond random variation[^1].

## Advanced Model Diagnostics

### Power Model Behavior Analysis

**Quartic System Dynamics**
The power model demonstrates:

- Critical RS threshold optimization through higher-order terms
- Balanced complexity-accuracy trade-off (R²=0.56)
- Interpretable polynomial coefficients for clinical communication


### Neural Network Activation Patterns

Gradient-weighted analysis reveals:

- Low RS (0-30): Layer 3 neurons dominant
- Moderate RS (31-70): Layer 4 feature detectors activate
- High RS (71-100): Final hidden layer drives predictions


## Limitations and Future Directions

### Model-Specific Constraints

1. **Neural Network Interpretability**: Clinical decision-making requires explainable AI techniques
2. **Small Sample Size**: 148 patients limit generalizability to larger healthcare systems
3. **Temporal Dynamics**: Static models neglect patient risk evolution over time

### Research Recommendations

1. **Hybrid Architecture Development**: Combine neural network accuracy with polynomial interpretability
2. **Longitudinal Risk Modeling**: Incorporate temporal risk score trajectories
3. **Multi-site Validation**: Test generalizability across diverse healthcare settings

## Concluding Synthesis

This comprehensive analysis establishes neural networks as superior predictors of psychiatric care utilization patterns, achieving 15% improvement in R² over the best power model (quartic polynomial). The stochastic training methodology with minimal dropout (p=0.05) proves particularly effective for small medical datasets, balancing regularization with model capacity.

The absolute deviation minimization approach provides a robust alternative to least squares regression, offering superior outlier resistance while maintaining interpretability. However, the neural network's superior performance (R²=0.62 vs 0.38 for MAE model) justifies its adoption for operational healthcare demand forecasting.

Implementation should follow a phased approach, beginning with interpretable polynomial models for clinician training before transitioning to neural networks for optimal predictive accuracy. This strategy maximizes both clinical acceptance and operational efficiency in healthcare resource allocation[^1].

<div style="text-align: center">⁂</div>

[^1]: Predictive-Modeling-of-Healthcare-Demand-Using-Phy.md

