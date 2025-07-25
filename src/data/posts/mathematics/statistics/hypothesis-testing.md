# Hypothesis Testing

Hypothesis testing is the statistical method that lets us make decisions with uncertainty. It's the bridge between descriptive statistics and actionable insights, enabling us to determine if observed differences are real or just random noise. From A/B testing to model validation, hypothesis testing provides the mathematical framework for confident decision-making in machine learning.

## 1. What Is Hypothesis Testing?

**Hypothesis testing** is a systematic method for making statistical decisions about population parameters based on sample data. It helps us answer questions like:
- Is the new website design actually better?
- Does this drug really work?
- Are these two machine learning models significantly different?
- Has user behavior changed after the update?

### The Logic:
1. **Assume nothing changed** (null hypothesis)
2. **Collect evidence** (sample data)
3. **Calculate probability** of seeing this evidence if nothing changed
4. **Make decision** based on how unlikely the evidence is

## 2. The Hypothesis Testing Framework

### 2.1 Setting Up Hypotheses

**Null Hypothesis (H₀)**: The "status quo" - assumes no effect, no difference, no change
**Alternative Hypothesis (H₁ or Hₐ)**: What we're trying to prove

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, t, chi2

def hypothesis_examples():
    """Examples of null and alternative hypotheses"""
    
    examples = {
        "A/B Testing": {
            "H0": "New design has same conversion rate as old design (μ_new = μ_old)",
            "H1": "New design has different conversion rate (μ_new ≠ μ_old)",
            "Type": "Two-tailed test"
        },
        "Drug Efficacy": {
            "H0": "Drug has no effect (μ_drug = μ_placebo)", 
            "H1": "Drug is effective (μ_drug > μ_placebo)",
            "Type": "One-tailed test (right)"
        },
        "Quality Control": {
            "H0": "Process meets specification (μ = 100)",
            "H1": "Process doesn't meet specification (μ ≠ 100)", 
            "Type": "Two-tailed test"
        },
        "Cost Reduction": {
            "H0": "New process costs same as old (μ_new ≥ μ_old)",
            "H1": "New process costs less (μ_new < μ_old)",
            "Type": "One-tailed test (left)"
        }
    }
    
    print("Hypothesis Testing Examples:")
    print("=" * 30)
    
    for scenario, details in examples.items():
        print(f"\n{scenario}:")
        print(f"  H₀: {details['H0']}")
        print(f"  H₁: {details['H1']}")
        print(f"  Test Type: {details['Type']}")

hypothesis_examples()
```

### 2.2 Types of Errors

```python
def error_types_explanation():
    """Explain Type I and Type II errors"""
    
    print("Types of Errors in Hypothesis Testing:")
    print("=" * 40)
    
    print("\n                Reality")
    print("              H₀ True    H₀ False")
    print("Decision H₀  ✓ Correct  Type II Error (β)")
    print("Reject   H₁  Type I Error (α)  ✓ Correct")
    
    print(f"\nType I Error (False Positive):")
    print(f"• Reject H₀ when it's actually true")
    print(f"• Example: Conclude drug works when it doesn't")
    print(f"• Probability = α (significance level)")
    print(f"• Typically set α = 0.05 (5%)")
    
    print(f"\nType II Error (False Negative):")
    print(f"• Fail to reject H₀ when it's actually false")
    print(f"• Example: Conclude drug doesn't work when it does")
    print(f"• Probability = β")
    print(f"• Power = 1 - β (probability of correctly rejecting false H₀)")
    
    # Simulation of error rates
    np.random.seed(42)
    
    # Simulate when H0 is true (no effect)
    null_true_data = np.random.normal(0, 1, 10000)
    alpha = 0.05
    critical_value = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    
    type_i_errors = np.abs(null_true_data) > critical_value
    type_i_rate = np.mean(type_i_errors)
    
    print(f"\nSimulation Results:")
    print(f"When H₀ is true:")
    print(f"  Type I error rate: {type_i_rate:.3f} (should be ≈ {alpha})")
    
    # Simulate when H0 is false (true effect = 0.5)
    true_effect = 0.5
    alt_true_data = np.random.normal(true_effect, 1, 10000)
    
    type_ii_errors = np.abs(alt_true_data) <= critical_value
    type_ii_rate = np.mean(type_ii_errors)
    power = 1 - type_ii_rate
    
    print(f"When H₁ is true (effect = {true_effect}):")
    print(f"  Type II error rate: {type_ii_rate:.3f}")
    print(f"  Power: {power:.3f}")

error_types_explanation()
```

## 3. The P-Value: What Does It Really Mean?

### 3.1 Understanding P-Values
```python
def p_value_explanation():
    """Explain what p-values actually mean"""
    
    print("Understanding P-Values:")
    print("=" * 23)
    
    print(f"\nP-value = Probability of observing data at least as extreme")
    print(f"as what we observed, assuming H₀ is true")
    
    print(f"\nCommon Misinterpretations:")
    print(f"❌ P-value is NOT the probability that H₀ is true")
    print(f"❌ P-value is NOT the probability of making a mistake")
    print(f"❌ Small p-value does NOT mean large effect size")
    
    print(f"\nCorrect Interpretation:")
    print(f"✅ P-value measures surprise under H₀")
    print(f"✅ Small p-value suggests data is inconsistent with H₀")
    print(f"✅ P-value depends on sample size and effect size")
    
    # Simulation to show p-value distribution under null
    np.random.seed(42)
    n_simulations = 10000
    sample_size = 30
    
    p_values = []
    
    for _ in range(n_simulations):
        # Generate data under null hypothesis (mean = 0)
        sample = np.random.normal(0, 1, sample_size)
        
        # Perform one-sample t-test against 0
        _, p_value = stats.ttest_1samp(sample, 0)
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    
    # Under null hypothesis, p-values should be uniformly distributed
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(p_values, bins=20, density=True, alpha=0.7, edgecolor='black')
    plt.axhline(y=1, color='red', linestyle='--', label='Expected (uniform)')
    plt.xlabel('P-value')
    plt.ylabel('Density')
    plt.title('P-value Distribution Under H₀')
    plt.legend()
    
    # Show false positive rate
    alpha_levels = [0.01, 0.05, 0.10]
    
    plt.subplot(1, 2, 2)
    for alpha in alpha_levels:
        false_positive_rate = np.mean(p_values <= alpha)
        plt.bar(alpha, false_positive_rate, alpha=0.7, 
               label=f'α={alpha}: {false_positive_rate:.3f}')
    
    plt.xlabel('Significance Level (α)')
    plt.ylabel('Observed False Positive Rate')
    plt.title('False Positive Rates')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSimulation Results (H₀ true):")
    for alpha in alpha_levels:
        false_positive_rate = np.mean(p_values <= alpha)
        print(f"α = {alpha}: False positive rate = {false_positive_rate:.3f}")

p_value_explanation()
```

### 3.2 Effect Size vs Statistical Significance
```python
def effect_size_vs_significance():
    """Demonstrate the difference between effect size and significance"""
    
    print("Effect Size vs Statistical Significance:")
    print("=" * 40)
    
    # Small effect, large sample
    np.random.seed(42)
    n_large = 10000
    small_effect = 0.1
    
    group1_large = np.random.normal(0, 1, n_large)
    group2_large = np.random.normal(small_effect, 1, n_large)
    
    _, p_small_effect = stats.ttest_ind(group1_large, group2_large)
    cohen_d_small = (np.mean(group2_large) - np.mean(group1_large)) / np.sqrt(
        (np.var(group1_large, ddof=1) + np.var(group2_large, ddof=1)) / 2
    )
    
    print(f"\nSmall Effect, Large Sample (n={n_large}):")
    print(f"Effect size (Cohen's d): {cohen_d_small:.3f}")
    print(f"P-value: {p_small_effect:.2e}")
    print(f"Statistically significant: {p_small_effect < 0.05}")
    print(f"Practically significant: Probably not")
    
    # Large effect, small sample
    n_small = 20
    large_effect = 1.0
    
    group1_small = np.random.normal(0, 1, n_small)
    group2_small = np.random.normal(large_effect, 1, n_small)
    
    _, p_large_effect = stats.ttest_ind(group1_small, group2_small)
    cohen_d_large = (np.mean(group2_small) - np.mean(group1_small)) / np.sqrt(
        (np.var(group1_small, ddof=1) + np.var(group2_small, ddof=1)) / 2
    )
    
    print(f"\nLarge Effect, Small Sample (n={n_small}):")
    print(f"Effect size (Cohen's d): {cohen_d_large:.3f}")
    print(f"P-value: {p_large_effect:.3f}")
    print(f"Statistically significant: {p_large_effect < 0.05}")
    print(f"Practically significant: Likely yes")
    
    print(f"\nCohen's d Interpretation:")
    print(f"• 0.2: Small effect")
    print(f"• 0.5: Medium effect") 
    print(f"• 0.8: Large effect")
    
    print(f"\nKey Takeaway: With large samples, even tiny differences")
    print(f"become statistically significant. Always consider effect size!")

effect_size_vs_significance()
```

## 4. Common Statistical Tests

### 4.1 One-Sample Tests
```python
def one_sample_tests():
    """Demonstrate one-sample statistical tests"""
    
    print("One-Sample Tests:")
    print("=" * 17)
    
    # Example: Testing if mean response time meets SLA
    np.random.seed(42)
    response_times = np.random.gamma(2, 100)  # Generate response times in ms
    sla_target = 180  # SLA target: 180ms
    
    print(f"\nScenario: Testing if response times meet SLA target")
    print(f"SLA target: {sla_target}ms")
    print(f"Sample size: {len(response_times)}")
    print(f"Sample mean: {np.mean(response_times):.1f}ms")
    print(f"Sample std: {np.std(response_times, ddof=1):.1f}ms")
    
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(response_times, sla_target)
    
    print(f"\nOne-Sample T-Test:")
    print(f"H₀: μ = {sla_target}ms (meets SLA)")
    print(f"H₁: μ > {sla_target}ms (violates SLA)")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value (two-tailed): {p_value:.3f}")
    print(f"P-value (one-tailed): {p_value/2:.3f}")
    
    alpha = 0.05
    reject_null = (p_value/2) < alpha and t_stat > 0  # One-tailed test
    
    print(f"Decision: {'Reject H₀' if reject_null else 'Fail to reject H₀'}")
    
    if reject_null:
        print(f"Conclusion: Response times significantly exceed SLA target")
    else:
        print(f"Conclusion: No significant evidence that SLA is violated")
    
    # Confidence interval
    confidence_level = 0.95
    degrees_freedom = len(response_times) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    
    margin_error = t_critical * (np.std(response_times, ddof=1) / np.sqrt(len(response_times)))
    ci_lower = np.mean(response_times) - margin_error
    ci_upper = np.mean(response_times) + margin_error
    
    print(f"\n95% Confidence Interval: [{ci_lower:.1f}, {ci_upper:.1f}]ms")
    
    return response_times, t_stat, p_value

response_data, t_stat, p_val = one_sample_tests()
```

### 4.2 Two-Sample Tests
```python
def two_sample_tests():
    """Demonstrate two-sample statistical tests"""
    
    print("Two-Sample Tests:")
    print("=" * 17)
    
    # A/B Testing Example
    np.random.seed(42)
    
    # Control group (existing feature)
    control_conversions = np.random.binomial(1, 0.12, 1000)  # 12% conversion
    control_revenue = np.random.gamma(2, 50, 1000)  # Revenue per user
    
    # Treatment group (new feature) 
    treatment_conversions = np.random.binomial(1, 0.14, 950)  # 14% conversion
    treatment_revenue = np.random.gamma(2, 55, 950)  # Slightly higher revenue
    
    print(f"\nA/B Testing Scenario:")
    print(f"Control group size: {len(control_conversions)}")
    print(f"Treatment group size: {len(treatment_conversions)}")
    
    # Test 1: Conversion rates (proportions)
    control_rate = np.mean(control_conversions)
    treatment_rate = np.mean(treatment_conversions)
    
    print(f"\n1. Testing Conversion Rates:")
    print(f"Control rate: {control_rate:.1%}")
    print(f"Treatment rate: {treatment_rate:.1%}")
    
    # Two-proportion z-test
    count1, count2 = np.sum(control_conversions), np.sum(treatment_conversions)
    n1, n2 = len(control_conversions), len(treatment_conversions)
    
    p_pooled = (count1 + count2) / (n1 + n2)
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    z_stat = (treatment_rate - control_rate) / se_pooled
    p_value_prop = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed
    
    print(f"Two-proportion Z-test:")
    print(f"  Z-statistic: {z_stat:.3f}")
    print(f"  P-value: {p_value_prop:.3f}")
    print(f"  Significant: {p_value_prop < 0.05}")
    
    # Test 2: Revenue per user (continuous)
    print(f"\n2. Testing Revenue per User:")
    print(f"Control mean: ${np.mean(control_revenue):.2f}")
    print(f"Treatment mean: ${np.mean(treatment_revenue):.2f}")
    
    # Two-sample t-test (assuming unequal variances)
    t_stat, p_value_rev = stats.ttest_ind(treatment_revenue, control_revenue, 
                                         equal_var=False)
    
    print(f"Two-sample T-test (Welch's):")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value_rev:.3f}")
    print(f"  Significant: {p_value_rev < 0.05}")
    
    # Effect sizes
    cohen_d = (np.mean(treatment_revenue) - np.mean(control_revenue)) / np.sqrt(
        (np.var(treatment_revenue, ddof=1) + np.var(control_revenue, ddof=1)) / 2
    )
    
    print(f"\nEffect Sizes:")
    print(f"Conversion rate difference: {(treatment_rate - control_rate):.3f}")
    print(f"Revenue difference: ${np.mean(treatment_revenue) - np.mean(control_revenue):.2f}")
    print(f"Cohen's d (revenue): {cohen_d:.3f}")
    
    return control_conversions, treatment_conversions, control_revenue, treatment_revenue

ab_results = two_sample_tests()
```

### 4.3 Chi-Square Tests
```python
def chi_square_tests():
    """Demonstrate chi-square tests for categorical data"""
    
    print("Chi-Square Tests:")
    print("=" * 16)
    
    # Example: User engagement by device type
    observed_data = np.array([
        [150, 100, 50],   # Mobile: High, Medium, Low engagement
        [80, 120, 70],    # Desktop: High, Medium, Low engagement  
        [45, 60, 85]      # Tablet: High, Medium, Low engagement
    ])
    
    device_types = ['Mobile', 'Desktop', 'Tablet']
    engagement_levels = ['High', 'Medium', 'Low']
    
    print(f"\nContingency Table: Device Type vs Engagement Level")
    df_observed = pd.DataFrame(observed_data, 
                              index=device_types, 
                              columns=engagement_levels)
    print(df_observed)
    
    # Chi-square test of independence
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed_data)
    
    print(f"\nChi-Square Test of Independence:")
    print(f"H₀: Device type and engagement level are independent")
    print(f"H₁: Device type and engagement level are associated")
    print(f"Chi-square statistic: {chi2_stat:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.3f}")
    print(f"Significant: {p_value < 0.05}")
    
    # Expected frequencies under independence
    print(f"\nExpected frequencies (under independence):")
    df_expected = pd.DataFrame(expected, 
                              index=device_types, 
                              columns=engagement_levels)
    print(df_expected.round(1))
    
    # Check assumptions
    min_expected = np.min(expected)
    print(f"\nAssumption check:")
    print(f"Minimum expected frequency: {min_expected:.1f}")
    print(f"Assumption met (≥5): {min_expected >= 5}")
    
    # Cramér's V (effect size for chi-square)
    n_total = np.sum(observed_data)
    cramers_v = np.sqrt(chi2_stat / (n_total * min(len(device_types)-1, len(engagement_levels)-1)))
    
    print(f"\nEffect size (Cramér's V): {cramers_v:.3f}")
    print(f"Interpretation: {'Small' if cramers_v < 0.3 else 'Medium' if cramers_v < 0.5 else 'Large'}")
    
    return observed_data, chi2_stat, p_value

chi_square_results = chi_square_tests()
```

## 5. Multiple Testing Problem

### 5.1 The Problem
```python
def multiple_testing_problem():
    """Demonstrate the multiple testing problem"""
    
    print("Multiple Testing Problem:")
    print("=" * 26)
    
    # Simulate multiple A/B tests
    np.random.seed(42)
    n_tests = 20
    alpha = 0.05
    
    print(f"Scenario: Running {n_tests} A/B tests simultaneously")
    print(f"Each test uses α = {alpha}")
    print(f"All tests have NO real effect (H₀ is true for all)")
    
    p_values = []
    significant_results = []
    
    for test_num in range(n_tests):
        # Generate data with no real effect
        control = np.random.normal(0, 1, 1000)
        treatment = np.random.normal(0, 1, 1000)  # Same distribution
        
        _, p_value = stats.ttest_ind(control, treatment)
        p_values.append(p_value)
        significant_results.append(p_value < alpha)
    
    n_significant = sum(significant_results)
    
    print(f"\nResults:")
    print(f"Number of significant tests: {n_significant}/{n_tests}")
    print(f"False positive rate: {n_significant/n_tests:.1%}")
    
    # Expected vs observed false positives
    expected_false_positives = n_tests * alpha
    print(f"Expected false positives: {expected_false_positives:.1f}")
    
    # Family-wise error rate
    fwer = 1 - (1 - alpha)**n_tests
    print(f"Family-wise error rate: {fwer:.1%}")
    print(f"(Probability of at least one false positive)")
    
    # Show p-values
    print(f"\nP-values from all tests:")
    for i, (p, sig) in enumerate(zip(p_values, significant_results)):
        marker = "***" if sig else ""
        print(f"Test {i+1:2d}: p = {p:.4f} {marker}")

multiple_testing_problem()
```

### 5.2 Multiple Testing Corrections
```python
def multiple_testing_corrections():
    """Demonstrate multiple testing correction methods"""
    
    print("Multiple Testing Corrections:")
    print("=" * 30)
    
    # Simulate multiple tests with some real effects
    np.random.seed(42)
    n_tests = 10
    real_effects = [0, 0, 0.5, 0, 0, 1.0, 0, 0, 0.3, 0]  # Some tests have real effects
    
    p_values = []
    
    for i, effect in enumerate(real_effects):
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(effect, 1, 100)
        
        _, p_value = stats.ttest_ind(control, treatment)
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    alpha = 0.05
    
    print(f"Raw p-values and true effects:")
    for i, (p, effect) in enumerate(zip(p_values, real_effects)):
        print(f"Test {i+1:2d}: p = {p:.4f}, true effect = {effect}")
    
    # 1. No correction
    uncorrected_significant = p_values < alpha
    print(f"\n1. No correction (α = {alpha}):")
    print(f"Significant tests: {np.where(uncorrected_significant)[0] + 1}")
    
    # 2. Bonferroni correction
    bonferroni_alpha = alpha / n_tests
    bonferroni_significant = p_values < bonferroni_alpha
    print(f"\n2. Bonferroni correction (α = {bonferroni_alpha:.4f}):")
    print(f"Significant tests: {np.where(bonferroni_significant)[0] + 1}")
    
    # 3. Holm-Bonferroni (step-down)
    sorted_indices = np.argsort(p_values)
    holm_significant = np.zeros(n_tests, dtype=bool)
    
    for i, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (n_tests - i)
        if p_values[idx] < adjusted_alpha:
            holm_significant[idx] = True
        else:
            break  # Stop at first non-significant test
    
    print(f"\n3. Holm-Bonferroni correction:")
    print(f"Significant tests: {np.where(holm_significant)[0] + 1}")
    
    # 4. False Discovery Rate (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    
    fdr_corrected = multipletests(p_values, alpha=alpha, method='fdr_bh')[0]
    print(f"\n4. FDR (Benjamini-Hochberg) correction:")
    print(f"Significant tests: {np.where(fdr_corrected)[0] + 1}")
    
    # Summary table
    results_df = pd.DataFrame({
        'Test': range(1, n_tests + 1),
        'P_value': p_values,
        'True_Effect': real_effects,
        'Uncorrected': uncorrected_significant,
        'Bonferroni': bonferroni_significant,
        'Holm': holm_significant,
        'FDR': fdr_corrected
    })
    
    print(f"\nSummary (True positives should match non-zero effects):")
    print(results_df.to_string(index=False, float_format='%.4f'))

multiple_testing_corrections()
```

## 6. Power Analysis

### 6.1 Understanding Statistical Power
```python
def power_analysis_demo():
    """Demonstrate statistical power and sample size calculations"""
    
    print("Statistical Power Analysis:")
    print("=" * 28)
    
    # Power as a function of effect size
    effect_sizes = np.linspace(0, 1.5, 100)
    alpha = 0.05
    n_per_group = 50
    
    powers = []
    
    for effect_size in effect_sizes:
        # Calculate power for two-sample t-test
        # Using approximation: power ≈ Φ(z - z_α/2) where z = effect_size * √(n/2)
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(n_per_group / 2)
        power = 1 - stats.norm.cdf(z_alpha - z_beta)
        powers.append(power)
    
    plt.figure(figsize=(12, 8))
    
    # Power curve
    plt.subplot(2, 2, 1)
    plt.plot(effect_sizes, powers, 'b-', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% power')
    plt.axvline(x=0.8, color='g', linestyle='--', label='Large effect')
    plt.xlabel('Effect Size (Cohen\'s d)')
    plt.ylabel('Statistical Power')
    plt.title(f'Power vs Effect Size (n={n_per_group} per group)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Power as function of sample size
    sample_sizes = np.arange(10, 200, 5)
    effect_size = 0.5  # Medium effect
    
    powers_n = []
    for n in sample_sizes:
        z_beta = effect_size * np.sqrt(n / 2)
        power = 1 - stats.norm.cdf(z_alpha - z_beta)
        powers_n.append(power)
    
    plt.subplot(2, 2, 2)
    plt.plot(sample_sizes, powers_n, 'g-', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% power')
    plt.xlabel('Sample Size per Group')
    plt.ylabel('Statistical Power')
    plt.title(f'Power vs Sample Size (effect size = {effect_size})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Sample size calculation
    def calculate_sample_size(effect_size, power=0.8, alpha=0.05):
        """Calculate required sample size for two-sample t-test"""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size)**2
        return int(np.ceil(n))
    
    plt.subplot(2, 2, 3)
    effect_range = np.linspace(0.2, 1.5, 50)
    sample_sizes_needed = [calculate_sample_size(effect) for effect in effect_range]
    
    plt.plot(effect_range, sample_sizes_needed, 'purple', linewidth=2)
    plt.xlabel('Effect Size (Cohen\'s d)')
    plt.ylabel('Required Sample Size per Group')
    plt.title('Sample Size for 80% Power')
    plt.grid(True, alpha=0.3)
    
    # Power analysis table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Create a summary table
    effect_sizes_table = [0.2, 0.5, 0.8]
    power_levels = [0.8, 0.9]
    
    table_data = []
    for effect in effect_sizes_table:
        row = [f"{effect}"]
        for power in power_levels:
            n_needed = calculate_sample_size(effect, power)
            row.append(f"{n_needed}")
        table_data.append(row)
    
    table = plt.table(cellText=table_data,
                     colLabels=['Effect Size', '80% Power', '90% Power'],
                     rowLabels=['Small', 'Medium', 'Large'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Sample Size Requirements', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    print(f"\nKey Insights:")
    print(f"• Small effects (d=0.2) require large samples: {calculate_sample_size(0.2)} per group")
    print(f"• Medium effects (d=0.5) require moderate samples: {calculate_sample_size(0.5)} per group")
    print(f"• Large effects (d=0.8) require smaller samples: {calculate_sample_size(0.8)} per group")
    print(f"• Higher power requirements increase sample size needs")

power_analysis_demo()
```

### 6.2 A/B Testing Sample Size Calculator
```python
def ab_test_sample_size_calculator():
    """Practical sample size calculator for A/B testing"""
    
    def calculate_ab_sample_size(baseline_rate, min_detectable_effect, 
                                power=0.8, alpha=0.05, two_sided=True):
        """
        Calculate sample size for A/B testing conversion rates
        
        Parameters:
        - baseline_rate: Current conversion rate (e.g., 0.1 for 10%)
        - min_detectable_effect: Minimum effect to detect (e.g., 0.02 for 2 percentage points)
        - power: Statistical power (default 0.8)
        - alpha: Type I error rate (default 0.05)
        - two_sided: Whether to use two-sided test (default True)
        """
        
        # Z-scores
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Conversion rates
        p1 = baseline_rate
        p2 = baseline_rate + min_detectable_effect
        
        # Pooled standard deviation
        p_avg = (p1 + p2) / 2
        
        # Sample size formula for proportions
        numerator = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) + 
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
        
        denominator = (p2 - p1)**2
        
        n_per_group = numerator / denominator
        
        return int(np.ceil(n_per_group))
    
    print("A/B Testing Sample Size Calculator:")
    print("=" * 36)
    
    # Example scenarios
    scenarios = [
        {
            "name": "E-commerce conversion",
            "baseline": 0.05,  # 5% baseline
            "effects": [0.01, 0.02, 0.03],  # 1%, 2%, 3% improvements
            "power": 0.8
        },
        {
            "name": "Email click-through",
            "baseline": 0.15,  # 15% baseline
            "effects": [0.02, 0.04, 0.06],  # 2%, 4%, 6% improvements
            "power": 0.8
        },
        {
            "name": "High-stakes test",
            "baseline": 0.10,  # 10% baseline
            "effects": [0.02],  # 2% improvement
            "power": 0.9  # Higher power requirement
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"Baseline rate: {scenario['baseline']:.1%}")
        print(f"Power: {scenario['power']:.0%}")
        
        for effect in scenario['effects']:
            n_needed = calculate_ab_sample_size(
                scenario['baseline'], effect, scenario['power']
            )
            
            relative_lift = effect / scenario['baseline']
            total_users = n_needed * 2
            
            print(f"  {effect:+.1%} absolute ({relative_lift:+.1%} relative): {n_needed:,} per group ({total_users:,} total)")
    
    # Test duration calculator
    def calculate_test_duration(sample_size_per_group, daily_visitors, 
                               allocation_ratio=0.5):
        """Calculate how long the test needs to run"""
        total_sample_needed = sample_size_per_group * 2
        daily_test_users = daily_visitors * allocation_ratio
        
        if daily_test_users > 0:
            days_needed = total_sample_needed / daily_test_users
            return days_needed
        return float('inf')
    
    print(f"\nTest Duration Examples:")
    print(f"For 1,000 users per group (2,000 total):")
    
    traffic_scenarios = [
        ("Low traffic site", 100),
        ("Medium traffic site", 1000), 
        ("High traffic site", 10000)
    ]
    
    for name, daily_visitors in traffic_scenarios:
        days = calculate_test_duration(1000, daily_visitors)
        print(f"  {name} ({daily_visitors:,} daily visitors): {days:.1f} days")

ab_test_sample_size_calculator()
```

## 7. Common Mistakes and Best Practices

### 7.1 Common Mistakes
```python
def common_testing_mistakes():
    """Demonstrate common mistakes in hypothesis testing"""
    
    print("Common Hypothesis Testing Mistakes:")
    print("=" * 37)
    
    # Mistake 1: P-hacking / Data dredging
    print("\n1. P-Hacking / Data Dredging:")
    print("❌ Keep testing until you find significance")
    print("❌ Test multiple variables without correction")
    print("❌ Stop early when results look good")
    
    np.random.seed(42)
    
    # Simulate p-hacking
    n_variables = 20
    p_values_phacking = []
    
    for _ in range(n_variables):
        # All variables have no real effect
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0, 1, 100)
        _, p = stats.ttest_ind(control, treatment)
        p_values_phacking.append(p)
    
    significant_phacking = sum(p < 0.05 for p in p_values_phacking)
    print(f"Result: {significant_phacking}/{n_variables} 'significant' results by chance")
    
    # Mistake 2: Peeking at results
    print(f"\n2. Peeking at Results (Sequential Testing Without Adjustment):")
    
    # Simulate continuous monitoring without adjustment
    np.random.seed(42)
    max_n = 1000
    check_every = 50
    
    control_stream = np.random.normal(0, 1, max_n)
    treatment_stream = np.random.normal(0, 1, max_n)  # No real effect
    
    peek_points = range(check_every, max_n + 1, check_every)
    early_stops = []
    
    for n in peek_points:
        _, p = stats.ttest_ind(control_stream[:n], treatment_stream[:n])
        if p < 0.05:
            early_stops.append(n)
    
    print(f"Without adjustment: {len(early_stops)} 'significant' results found")
    print(f"First 'significant' result at n={early_stops[0] if early_stops else 'none'}")
    
    # Mistake 3: Ignoring assumptions
    print(f"\n3. Ignoring Test Assumptions:")
    print("❌ Using t-test with very skewed data")
    print("❌ Assuming equal variances when they're very different")
    print("❌ Using parametric tests with small samples from non-normal distributions")
    
    # Example: t-test with very skewed data
    skewed_control = np.random.exponential(1, 50)
    skewed_treatment = np.random.exponential(1.2, 50)
    
    # Wrong: t-test on skewed data
    _, p_ttest = stats.ttest_ind(skewed_control, skewed_treatment)
    
    # Better: Mann-Whitney U test (non-parametric)
    _, p_mannwhitney = stats.mannwhitneyu(skewed_control, skewed_treatment, 
                                         alternative='two-sided')
    
    print(f"Skewed data example:")
    print(f"  T-test p-value: {p_ttest:.3f}")
    print(f"  Mann-Whitney p-value: {p_mannwhitney:.3f}")
    
    # Mistake 4: Confusing statistical and practical significance
    print(f"\n4. Confusing Statistical and Practical Significance:")
    
    # Large sample, tiny effect
    huge_n = 100000
    tiny_effect = 0.01
    
    control_huge = np.random.normal(0, 1, huge_n)
    treatment_huge = np.random.normal(tiny_effect, 1, huge_n)
    
    _, p_huge = stats.ttest_ind(control_huge, treatment_huge)
    
    print(f"Huge sample (n={huge_n:,}), tiny effect ({tiny_effect}):")
    print(f"  P-value: {p_huge:.2e} (highly significant!)")
    print(f"  Effect size: {tiny_effect} (practically meaningless)")

common_testing_mistakes()
```

### 7.2 Best Practices
```python
def testing_best_practices():
    """Demonstrate best practices for hypothesis testing"""
    
    print("Hypothesis Testing Best Practices:")
    print("=" * 35)
    
    print(f"\n1. Pre-register Your Analysis:")
    print("✅ Define hypotheses before seeing data")
    print("✅ Specify your statistical test in advance")
    print("✅ Set sample size based on power analysis") 
    print("✅ Define stopping criteria")
    
    print(f"\n2. Check Test Assumptions:")
    print("✅ Normality (Q-Q plots, Shapiro-Wilk test)")
    print("✅ Equal variances (Levene's test)")
    print("✅ Independence of observations")
    print("✅ Appropriate sample size")
    
    # Example: Assumption checking
    def check_test_assumptions(group1, group2):
        """Check assumptions for two-sample t-test"""
        
        print(f"\nAssumption Checking Example:")
        print(f"Group 1: n={len(group1)}, mean={np.mean(group1):.2f}")
        print(f"Group 2: n={len(group2)}, mean={np.mean(group2):.2f}")
        
        # Normality tests
        _, p_norm1 = stats.shapiro(group1)
        _, p_norm2 = stats.shapiro(group2)
        
        print(f"\nNormality tests (Shapiro-Wilk):")
        print(f"  Group 1 p-value: {p_norm1:.3f} {'✅' if p_norm1 > 0.05 else '❌'}")
        print(f"  Group 2 p-value: {p_norm2:.3f} {'✅' if p_norm2 > 0.05 else '❌'}")
        
        # Equal variances test
        _, p_var = stats.levene(group1, group2)
        print(f"\nEqual variances test (Levene):")
        print(f"  P-value: {p_var:.3f} {'✅' if p_var > 0.05 else '❌'}")
        
        # Recommendations
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            if p_var > 0.05:
                print(f"\nRecommendation: Use standard t-test")
            else:
                print(f"\nRecommendation: Use Welch's t-test (unequal variances)")
        else:
            print(f"\nRecommendation: Use Mann-Whitney U test (non-parametric)")
    
    # Example with normal data
    np.random.seed(42)
    normal_group1 = np.random.normal(10, 2, 50)
    normal_group2 = np.random.normal(12, 2, 50)
    
    check_test_assumptions(normal_group1, normal_group2)
    
    print(f"\n3. Report Effect Sizes:")
    print("✅ Cohen's d for t-tests")
    print("✅ Odds ratios for categorical data")
    print("✅ Confidence intervals for all estimates")
    print("✅ Practical significance thresholds")
    
    print(f"\n4. Handle Multiple Testing:")
    print("✅ Plan your analysis to minimize multiple tests")
    print("✅ Use appropriate corrections (Bonferroni, FDR)")
    print("✅ Consider hierarchical testing strategies")
    print("✅ Focus on primary endpoints")
    
    print(f"\n5. Proper A/B Testing:")
    print("✅ Randomize users properly")
    print("✅ Run until predetermined sample size")
    print("✅ Check for novelty effects")
    print("✅ Monitor for external factors")
    print("✅ Consider practical significance")

testing_best_practices()
```

## 8. Modern Approaches and Extensions

### 8.1 Bayesian vs Frequentist Testing
```python
def bayesian_vs_frequentist():
    """Compare Bayesian and frequentist approaches to hypothesis testing"""
    
    print("Bayesian vs Frequentist Hypothesis Testing:")
    print("=" * 44)
    
    # Example: A/B testing conversion rates
    np.random.seed(42)
    
    # Observed data
    control_conversions = 45
    control_total = 1000
    treatment_conversions = 55
    treatment_total = 1000
    
    print(f"Observed Data:")
    print(f"Control: {control_conversions}/{control_total} = {control_conversions/control_total:.1%}")
    print(f"Treatment: {treatment_conversions}/{treatment_total} = {treatment_conversions/treatment_total:.1%}")
    
    # Frequentist approach
    print(f"\n1. Frequentist Approach:")
    
    # Two-proportion z-test
    p1 = control_conversions / control_total
    p2 = treatment_conversions / treatment_total
    p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
    z_stat = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print(f"  Z-statistic: {z_stat:.3f}")
    print(f"  P-value: {p_value:.3f}")
    print(f"  Conclusion: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
    
    # Bayesian approach
    print(f"\n2. Bayesian Approach:")
    
    # Beta priors (uniform)
    alpha_prior, beta_prior = 1, 1
    
    # Posterior distributions
    alpha_control = alpha_prior + control_conversions
    beta_control = beta_prior + control_total - control_conversions
    
    alpha_treatment = alpha_prior + treatment_conversions
    beta_treatment = beta_prior + treatment_total - treatment_conversions
    
    # Monte Carlo simulation to calculate P(treatment > control)
    n_samples = 100000
    control_samples = np.random.beta(alpha_control, beta_control, n_samples)
    treatment_samples = np.random.beta(alpha_treatment, beta_treatment, n_samples)
    
    prob_treatment_better = np.mean(treatment_samples > control_samples)
    
    # Credible intervals
    control_ci = np.percentile(control_samples, [2.5, 97.5])
    treatment_ci = np.percentile(treatment_samples, [2.5, 97.5])
    
    print(f"  P(Treatment > Control): {prob_treatment_better:.3f}")
    print(f"  Control 95% CI: [{control_ci[0]:.3f}, {control_ci[1]:.3f}]")
    print(f"  Treatment 95% CI: [{treatment_ci[0]:.3f}, {treatment_ci[1]:.3f}]")
    
    # Expected loss
    diff_samples = treatment_samples - control_samples
    expected_diff = np.mean(diff_samples)
    expected_loss_if_wrong = np.mean(np.abs(diff_samples[diff_samples < 0]))
    
    print(f"  Expected difference: {expected_diff:.4f}")
    print(f"  Expected loss if treatment is worse: {expected_loss_if_wrong:.4f}")
    
    print(f"\n3. Comparison:")
    print(f"Frequentist: Binary decision based on p-value threshold")
    print(f"Bayesian: Probability statement about which is better")
    print(f"Frequentist: Confidence intervals (may or may not contain true value)")
    print(f"Bayesian: Credible intervals (95% probability true value is inside)")

bayesian_vs_frequentist()
```

### 8.2 Sequential Testing and Early Stopping
```python
def sequential_testing():
    """Demonstrate proper sequential testing methods"""
    
    print("Sequential Testing with Proper Error Control:")
    print("=" * 45)
    
    # Alpha spending function approach
    def alpha_spending_pocock(t, alpha=0.05):
        """Pocock alpha spending function"""
        return alpha * np.log(1 + (np.e - 1) * t)
    
    def alpha_spending_obrien_fleming(t, alpha=0.05):
        """O'Brien-Fleming alpha spending function"""
        return 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - alpha/2) / np.sqrt(t)))
    
    # Simulate sequential A/B test
    np.random.seed(42)
    max_n_per_group = 1000
    check_points = [100, 200, 400, 600, 800, 1000]
    true_effect = 0.02  # 2 percentage point improvement
    
    # Generate all data upfront
    control_data = np.random.binomial(1, 0.10, max_n_per_group)  # 10% baseline
    treatment_data = np.random.binomial(1, 0.10 + true_effect, max_n_per_group)
    
    print(f"Sequential Testing Simulation:")
    print(f"True effect size: +{true_effect:.1%}")
    print(f"Planned analyses at: {check_points}")
    
    # Test at each checkpoint
    results = []
    
    for i, n in enumerate(check_points):
        # Current data
        control_current = control_data[:n]
        treatment_current = treatment_data[:n]
        
        # Standard test
        p1 = np.mean(control_current)
        p2 = np.mean(treatment_current)
        p_pooled = (np.sum(control_current) + np.sum(treatment_current)) / (2 * n)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / n))
        
        if se > 0:
            z_stat = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            p_value = 1.0
        
        # Alpha spending boundaries
        t = (i + 1) / len(check_points)  # Information fraction
        
        alpha_spent_pocock = alpha_spending_pocock(t)
        alpha_spent_of = alpha_spending_obrien_fleming(t)
        
        # Incremental alpha for this test
        if i == 0:
            alpha_increment_pocock = alpha_spent_pocock
            alpha_increment_of = alpha_spent_of
        else:
            alpha_increment_pocock = alpha_spent_pocock - alpha_spending_pocock((i) / len(check_points))
            alpha_increment_of = alpha_spent_of - alpha_spending_obrien_fleming((i) / len(check_points))
        
        results.append({
            'n': n,
            'p_value': p_value,
            'control_rate': p1,
            'treatment_rate': p2,
            'alpha_pocock': alpha_increment_pocock,
            'alpha_of': alpha_increment_of,
            'significant_pocock': p_value < alpha_increment_pocock,
            'significant_of': p_value < alpha_increment_of
        })
        
        print(f"\nAnalysis {i+1} (n={n} per group):")
        print(f"  Control rate: {p1:.3f}")
        print(f"  Treatment rate: {p2:.3f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Pocock boundary: {alpha_increment_pocock:.4f} {'✓' if p_value < alpha_increment_pocock else '✗'}")
        print(f"  O'Brien-Fleming boundary: {alpha_increment_of:.4f} {'✓' if p_value < alpha_increment_of else '✗'}")
    
    # Summary
    pocock_stops = [r for r in results if r['significant_pocock']]
    of_stops = [r for r in results if r['significant_of']]
    
    print(f"\nSummary:")
    print(f"Pocock: {'Early stop at n=' + str(pocock_stops[0]['n']) if pocock_stops else 'No early stop'}")
    print(f"O'Brien-Fleming: {'Early stop at n=' + str(of_stops[0]['n']) if of_stops else 'No early stop'}")
    
    print(f"\nKey differences:")
    print(f"• Pocock: More likely to stop early, uses alpha more evenly")
    print(f"• O'Brien-Fleming: Conservative early, liberal late")

sequential_testing()
```

## 9. Conclusion

Hypothesis testing provides the statistical framework for making confident decisions under uncertainty:

### **Core Concepts Mastered:**

#### **Testing Framework:**
- **Null hypothesis**: Assumes no effect (status quo)
- **Alternative hypothesis**: What we're trying to prove
- **P-value**: Probability of observing data this extreme under H₀
- **Significance level**: Threshold for rejecting H₀

#### **Types of Tests:**
- **One-sample**: Compare sample to known value
- **Two-sample**: Compare two groups
- **Paired**: Compare before/after measurements
- **Chi-square**: Test relationships in categorical data

#### **Error Types:**
- **Type I (α)**: False positive - reject true H₀
- **Type II (β)**: False negative - fail to reject false H₀
- **Power (1-β)**: Probability of detecting true effect

### **ML Applications:**

1. **A/B Testing**: Compare product variants statistically
2. **Model Validation**: Test if models perform differently
3. **Feature Selection**: Test feature importance
4. **Quality Control**: Monitor system performance
5. **Causal Inference**: Test for treatment effects

### **Best Practices:**
- **Pre-register analysis plans** to avoid p-hacking
- **Check test assumptions** before applying tests
- **Report effect sizes** alongside p-values  
- **Correct for multiple testing** when appropriate
- **Use proper sequential methods** for early stopping
- **Consider practical significance** not just statistical

### **Common Pitfalls to Avoid:**
- **P-hacking**: Testing until finding significance
- **Peeking**: Checking results before planned endpoint
- **Ignoring assumptions**: Using wrong test for data type
- **Multiple testing**: Not correcting for many comparisons
- **Overinterpreting**: Confusing statistical and practical significance

### **Modern Approaches:**
- **Bayesian methods**: Provide probability statements
- **Sequential testing**: Allow early stopping with error control
- **Effect size estimation**: Focus on magnitude, not just significance
- **Meta-analysis**: Combine results across studies

**Next in Statistics**: **Correlation vs Causation** - understanding the critical difference between association and causation in data analysis!

Hypothesis testing is the bridge between data and decisions. Master these concepts, and you'll have the tools to make statistically sound conclusions in any machine learning or data science project. 