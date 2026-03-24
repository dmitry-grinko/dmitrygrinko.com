# Statistical Inference

Statistical inference is the process of drawing conclusions about populations based on sample data. It's a fundamental pillar of data science and machine learning, enabling us to make informed decisions under uncertainty and validate our models' performance. This post explores the key concepts and methods of statistical inference.

## 1. What is Statistical Inference?

Statistical inference allows us to:
- **Estimate population parameters** from sample statistics
- **Test hypotheses** about population characteristics
- **Quantify uncertainty** in our estimates and predictions
- **Make data-driven decisions** with known confidence levels

### Key Components

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy import stats

# Sample data for demonstration
np.random.seed(42)
sample_data = np.random.normal(100, 15, 50)  # Sample of 50 from population with μ=100, σ=15
```

## 2. Confidence Intervals

Confidence intervals provide a range of plausible values for a population parameter.

### Understanding Confidence Intervals

A 95% confidence interval means:
- If we repeated our sampling process many times
- 95% of the intervals would contain the true population parameter

```python
# Calculate confidence interval for population mean
def confidence_interval_mean(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    
    # t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    margin_error = t_value * std_err
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return mean, (ci_lower, ci_upper)

# Calculate 95% CI
sample_mean, ci = confidence_interval_mean(sample_data)
print(f"Sample mean: {sample_mean:.2f}")
print(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

### Different Types of Confidence Intervals

```python
# For population proportion
def confidence_interval_proportion(successes, n, confidence=0.95):
    p_hat = successes / n
    z_value = stats.norm.ppf((1 + confidence) / 2)
    
    margin_error = z_value * np.sqrt(p_hat * (1 - p_hat) / n)
    ci_lower = max(0, p_hat - margin_error)
    ci_upper = min(1, p_hat + margin_error)
    
    return p_hat, (ci_lower, ci_upper)

# Example: 45 successes out of 100 trials
prop, prop_ci = confidence_interval_proportion(45, 100)
print(f"Proportion: {prop:.3f}")
print(f"95% CI for proportion: ({prop_ci[0]:.3f}, {prop_ci[1]:.3f})")
```

## 3. Hypothesis Testing Framework

Hypothesis testing provides a structured approach to making decisions about population parameters.

### The Process

1. **Formulate Hypotheses**
   - Null hypothesis (H₀): The status quo
   - Alternative hypothesis (H₁): What we want to test

2. **Choose Significance Level (α)**
   - Common choices: 0.05, 0.01, 0.10

3. **Calculate Test Statistic**
4. **Determine p-value**
5. **Make Decision**

```python
# One-sample t-test
def one_sample_ttest(data, mu0, alpha=0.05):
    """
    Test if sample mean differs significantly from hypothesized population mean
    H0: μ = μ0
    H1: μ ≠ μ0
    """
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    
    # Calculate t-statistic
    t_stat = (sample_mean - mu0) / (sample_std / np.sqrt(n))
    
    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # Decision
    reject_null = p_value < alpha
    
    return {
        'statistic': t_stat,
        'p_value': p_value,
        'reject_null': reject_null,
        'sample_mean': sample_mean,
        'hypothesized_mean': mu0
    }

# Test if our sample came from population with mean = 95
result = one_sample_ttest(sample_data, mu0=95)
print(f"t-statistic: {result['statistic']:.3f}")
print(f"p-value: {result['p_value']:.3f}")
print(f"Reject null hypothesis: {result['reject_null']}")
```

## 4. Types of Errors and Statistical Power

Understanding errors in hypothesis testing is crucial for proper interpretation.

### Type I and Type II Errors

```python
def visualize_errors():
    """Visualize Type I and Type II errors"""
    x = np.linspace(-4, 8, 1000)
    
    # Null hypothesis distribution (H0: μ = 0)
    h0_dist = stats.norm.pdf(x, 0, 1)
    
    # Alternative hypothesis distribution (H1: μ = 3)
    h1_dist = stats.norm.pdf(x, 3, 1)
    
    # Critical value for α = 0.05 (one-tailed test)
    critical_value = stats.norm.ppf(0.95)
    
    plt.figure(figsize=(12, 6))
    
    # Plot distributions
    plt.plot(x, h0_dist, 'b-', label='H0: μ = 0', linewidth=2)
    plt.plot(x, h1_dist, 'r-', label='H1: μ = 3', linewidth=2)
    
    # Shade Type I error (α)
    x_alpha = x[x >= critical_value]
    plt.fill_between(x_alpha, 0, stats.norm.pdf(x_alpha, 0, 1), 
                     alpha=0.3, color='blue', label='Type I Error (α)')
    
    # Shade Type II error (β)
    x_beta = x[x <= critical_value]
    plt.fill_between(x_beta, 0, stats.norm.pdf(x_beta, 3, 1), 
                     alpha=0.3, color='red', label='Type II Error (β)')
    
    # Critical value line
    plt.axvline(critical_value, color='black', linestyle='--', 
                label=f'Critical Value: {critical_value:.2f}')
    
    plt.xlabel('Test Statistic')
    plt.ylabel('Probability Density')
    plt.title('Type I and Type II Errors in Hypothesis Testing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Calculate statistical power
def calculate_power(mu0, mu1, sigma, n, alpha=0.05):
    """Calculate statistical power for one-sample t-test"""
    # Critical value under null hypothesis
    critical_value = mu0 + stats.norm.ppf(1 - alpha) * (sigma / np.sqrt(n))
    
    # Power = P(reject H0 | H1 is true)
    power = 1 - stats.norm.cdf(critical_value, mu1, sigma / np.sqrt(n))
    
    return power

# Example: Power analysis
powers = []
sample_sizes = range(10, 101, 10)

for n in sample_sizes:
    power = calculate_power(mu0=100, mu1=105, sigma=15, n=n)
    powers.append(power)

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, powers, 'bo-', linewidth=2, markersize=6)
plt.axhline(y=0.8, color='r', linestyle='--', label='Desired Power (0.8)')
plt.xlabel('Sample Size')
plt.ylabel('Statistical Power')
plt.title('Power Analysis: Effect of Sample Size on Statistical Power')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

## 5. Common Statistical Tests

### Two-Sample Tests

```python
# Independent samples t-test
def independent_ttest(sample1, sample2, alpha=0.05):
    """Compare means of two independent samples"""
    
    # Using scipy's built-in function
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    
    # Manual calculation for understanding
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    # Pooled variance
    pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
    
    # Standard error
    se = np.sqrt(pooled_var * (1/n1 + 1/n2))
    
    # t-statistic
    t_manual = (mean1 - mean2) / se
    
    return {
        'statistic': t_stat,
        'p_value': p_value,
        'reject_null': p_value < alpha,
        'mean_diff': mean1 - mean2,
        'manual_t': t_manual
    }

# Generate two samples
sample1 = np.random.normal(100, 15, 30)
sample2 = np.random.normal(105, 15, 35)

result = independent_ttest(sample1, sample2)
print(f"Two-sample t-test results:")
print(f"t-statistic: {result['statistic']:.3f}")
print(f"p-value: {result['p_value']:.3f}")
print(f"Mean difference: {result['mean_diff']:.2f}")
```

### Chi-Square Test

```python
# Chi-square test for independence
def chi_square_test(observed_freq):
    """Test independence between categorical variables"""
    
    chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(observed_freq)
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected_freq
    }

# Example: Testing association between treatment and outcome
observed = np.array([[20, 15],   # Treatment A: Success, Failure
                     [25, 10]])  # Treatment B: Success, Failure

result = chi_square_test(observed)
print(f"Chi-square test results:")
print(f"Chi-square statistic: {result['chi2_statistic']:.3f}")
print(f"p-value: {result['p_value']:.3f}")
print(f"Degrees of freedom: {result['degrees_of_freedom']}")
```

## 6. p-Values and Their Interpretation

### Understanding p-Values

```python
def p_value_simulation(true_mean, n_samples=1000, sample_size=30):
    """Simulate p-value distribution under null hypothesis"""
    
    p_values = []
    
    for _ in range(n_samples):
        # Generate sample under null hypothesis (mean = 0)
        sample = np.random.normal(0, 1, sample_size)
        
        # Test against alternative hypothesis
        t_stat, p_val = stats.ttest_1samp(sample, true_mean)
        p_values.append(p_val)
    
    return np.array(p_values)

# Simulate p-values under null hypothesis
p_vals_null = p_value_simulation(true_mean=0)

# Simulate p-values under alternative hypothesis
p_vals_alt = p_value_simulation(true_mean=0.5)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(p_vals_null, bins=20, alpha=0.7, density=True, label='H0 true')
plt.axhline(y=1, color='r', linestyle='--', label='Expected uniform')
plt.xlabel('p-value')
plt.ylabel('Density')
plt.title('p-value Distribution Under H0')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(p_vals_alt, bins=20, alpha=0.7, density=True, label='H1 true')
plt.xlabel('p-value')
plt.ylabel('Density')
plt.title('p-value Distribution Under H1')
plt.legend()

plt.tight_layout()
plt.show()
```

## 7. Applications in Machine Learning

### Model Validation

```python
# A/B testing for model comparison
def ab_test_models(model_a_accuracy, model_b_accuracy, n_tests):
    """Compare two models using statistical testing"""
    
    # Convert accuracies to successes
    successes_a = int(model_a_accuracy * n_tests)
    successes_b = int(model_b_accuracy * n_tests)
    
    # Two-proportion z-test
    p1, p2 = successes_a / n_tests, successes_b / n_tests
    p_pooled = (successes_a + successes_b) / (2 * n_tests)
    
    se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / n_tests))
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        'model_a_accuracy': p1,
        'model_b_accuracy': p2,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant_difference': p_value < 0.05
    }

# Example: Compare two ML models
result = ab_test_models(0.85, 0.87, 1000)
print(f"Model A accuracy: {result['model_a_accuracy']:.3f}")
print(f"Model B accuracy: {result['model_b_accuracy']:.3f}")
print(f"z-statistic: {result['z_statistic']:.3f}")
print(f"p-value: {result['p_value']:.3f}")
print(f"Significant difference: {result['significant_difference']}")
```

### Feature Selection with Statistical Tests

```python
# Feature selection using statistical tests
def feature_importance_test(X, y, feature_names):
    """Test statistical significance of features"""
    
    results = []
    
    for i, feature_name in enumerate(feature_names):
        # For continuous features: correlation test
        corr_coef, p_value = stats.pearsonr(X[:, i], y)
        
        results.append({
            'feature': feature_name,
            'correlation': corr_coef,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    return sorted(results, key=lambda x: x['p_value'])

# Example with synthetic data
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
y = 2*X[:, 0] + 0.5*X[:, 1] + np.random.randn(n_samples)  # Only first two features matter

feature_names = [f'Feature_{i+1}' for i in range(n_features)]
results = feature_importance_test(X, y, feature_names)

print("Feature importance (sorted by p-value):")
for result in results:
    print(f"{result['feature']}: r={result['correlation']:.3f}, "
          f"p={result['p_value']:.3f}, significant={result['significant']}")
```

## 8. Multiple Testing Correction

When performing multiple statistical tests, we need to adjust for multiple comparisons.

```python
from scipy.stats import false_discovery_control

# Multiple testing correction
def multiple_testing_correction(p_values, method='bonferroni', alpha=0.05):
    """Apply multiple testing correction"""
    
    if method == 'bonferroni':
        # Bonferroni correction
        corrected_alpha = alpha / len(p_values)
        significant = p_values < corrected_alpha
        
    elif method == 'benjamini_hochberg':
        # Benjamini-Hochberg (FDR control)
        significant = false_discovery_control(p_values, alpha=alpha)
        corrected_alpha = None
    
    return {
        'original_p_values': p_values,
        'significant': significant,
        'corrected_alpha': corrected_alpha,
        'method': method
    }

# Example: Multiple feature tests
p_values = np.array([0.001, 0.02, 0.04, 0.06, 0.08])

bonferroni_result = multiple_testing_correction(p_values, 'bonferroni')
bh_result = multiple_testing_correction(p_values, 'benjamini_hochberg')

print("Original p-values:", p_values)
print("Bonferroni significant:", bonferroni_result['significant'])
print("Benjamini-Hochberg significant:", bh_result['significant'])
```

## 9. Practical Guidelines

### Choosing the Right Test

```python
def test_selection_guide():
    """Guide for selecting appropriate statistical tests"""
    
    guide = {
        'One sample, continuous': {
            'Normal data': 'One-sample t-test',
            'Non-normal data': 'Wilcoxon signed-rank test'
        },
        'Two samples, continuous': {
            'Independent, normal': 'Independent t-test',
            'Independent, non-normal': 'Mann-Whitney U test',
            'Paired samples': 'Paired t-test'
        },
        'Categorical data': {
            'One variable': 'Chi-square goodness of fit',
            'Two variables': 'Chi-square test of independence',
            'Two proportions': 'Two-proportion z-test'
        },
        'Correlation': {
            'Linear relationship': 'Pearson correlation',
            'Monotonic relationship': 'Spearman correlation'
        }
    }
    
    return guide

# Print the guide
guide = test_selection_guide()
for category, tests in guide.items():
    print(f"\n{category}:")
    for condition, test in tests.items():
        print(f"  {condition}: {test}")
```

## 10. Common Pitfalls and Best Practices

### Best Practices Checklist

1. **Check Assumptions**
   - Normality (if required)
   - Independence of observations
   - Equal variances (if required)

2. **Effect Size Matters**
   - Statistical significance ≠ practical significance
   - Report confidence intervals
   - Consider Cohen's d, correlation coefficients

3. **Multiple Testing**
   - Adjust p-values when testing multiple hypotheses
   - Consider the family-wise error rate

4. **Sample Size Planning**
   - Conduct power analysis before collecting data
   - Ensure adequate sample size for meaningful results

```python
# Effect size calculation
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + 
                         (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Interpretation
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return d, interpretation

# Example
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(105, 15, 50)

d, interpretation = cohens_d(group1, group2)
print(f"Cohen's d: {d:.3f} ({interpretation} effect)")
```

## Conclusion

Statistical inference provides the mathematical foundation for making data-driven decisions under uncertainty. Key takeaways:

1. **Confidence intervals** quantify uncertainty in our estimates
2. **Hypothesis testing** provides a framework for decision-making
3. **p-values** indicate evidence against the null hypothesis
4. **Effect sizes** measure practical significance
5. **Multiple testing corrections** prevent inflated Type I error rates

In machine learning, these concepts are essential for:
- Model validation and comparison
- Feature selection and engineering
- A/B testing and experimentation
- Understanding model uncertainty
- Making robust, evidence-based decisions

Remember: Statistical significance is just one piece of the puzzle. Always consider practical significance, effect sizes, and the broader context of your analysis. 