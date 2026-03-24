# Descriptive Statistics

Descriptive statistics are the foundation of data analysis - they help us understand what our data looks like before we build models or make predictions. Whether you're exploring a new dataset, performing feature engineering, or communicating findings to stakeholders, descriptive statistics provide the essential tools for summarizing and understanding data patterns.

## 1. What Are Descriptive Statistics?

**Descriptive statistics** summarize and describe the main features of a dataset. They help us answer questions like:
- What's the typical value in our data?
- How spread out are the values?
- Are there outliers?
- What's the shape of the distribution?
- How do different variables relate to each other?

### Two Main Categories:
1. **Measures of Central Tendency** - Where is the "center" of the data?
2. **Measures of Variability** - How spread out is the data?

## 2. Measures of Central Tendency

### 2.1 Mean (Arithmetic Average)
The sum of all values divided by the number of values.

**Formula**: `μ = Σx_i / n`

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def mean_examples():
    """Demonstrate mean calculation and properties"""
    
    # Example: Customer purchase amounts
    purchases = [25.50, 45.00, 12.75, 89.25, 33.50, 67.00, 28.75, 156.00, 41.25, 52.50]
    
    mean_purchase = np.mean(purchases)
    
    print("Mean (Arithmetic Average):")
    print(f"Purchase amounts: {purchases}")
    print(f"Mean purchase: ${mean_purchase:.2f}")
    
    # Properties of the mean
    print(f"\nProperties of the mean:")
    print(f"Sum of deviations from mean: {sum(x - mean_purchase for x in purchases):.10f}")
    
    # Effect of outliers
    purchases_with_outlier = purchases + [500.00]  # Add expensive purchase
    mean_with_outlier = np.mean(purchases_with_outlier)
    
    print(f"\nOutlier sensitivity:")
    print(f"Original mean: ${mean_purchase:.2f}")
    print(f"With $500 outlier: ${mean_with_outlier:.2f}")
    print(f"Increase: ${mean_with_outlier - mean_purchase:.2f}")
    
    return purchases, mean_purchase

purchases, mean_val = mean_examples()
```

### 2.2 Median
The middle value when data is sorted (50th percentile).

```python
def median_examples():
    """Demonstrate median calculation and robustness"""
    
    # Same purchase data
    purchases = [25.50, 45.00, 12.75, 89.25, 33.50, 67.00, 28.75, 156.00, 41.25, 52.50]
    
    median_purchase = np.median(purchases)
    
    print("Median (Middle Value):")
    print(f"Sorted purchases: {sorted(purchases)}")
    print(f"Median purchase: ${median_purchase:.2f}")
    
    # Robustness to outliers
    purchases_with_outlier = purchases + [500.00]
    median_with_outlier = np.median(purchases_with_outlier)
    
    print(f"\nOutlier robustness:")
    print(f"Original median: ${median_purchase:.2f}")
    print(f"With $500 outlier: ${median_with_outlier:.2f}")
    print(f"Change: ${median_with_outlier - median_purchase:.2f}")
    
    # Even vs odd number of observations
    even_data = [10, 20, 30, 40]
    odd_data = [10, 20, 30, 40, 50]
    
    print(f"\nEven number of values {even_data}: median = {np.median(even_data)}")
    print(f"Odd number of values {odd_data}: median = {np.median(odd_data)}")
    
    return median_purchase

median_val = median_examples()
```

### 2.3 Mode
The most frequently occurring value(s).

```python
def mode_examples():
    """Demonstrate mode calculation for different data types"""
    
    # Numerical data with clear mode
    scores = [85, 92, 78, 85, 90, 85, 88, 92, 85, 79]
    mode_result = stats.mode(scores, keepdims=True)
    
    print("Mode (Most Frequent Value):")
    print(f"Test scores: {scores}")
    print(f"Mode: {mode_result.mode[0]} (appears {mode_result.count[0]} times)")
    
    # Categorical data
    categories = ['Premium', 'Standard', 'Premium', 'Basic', 'Standard', 'Premium', 'Premium']
    category_counts = pd.Series(categories).value_counts()
    
    print(f"\nCategorical data: {categories}")
    print(f"Category counts:\n{category_counts}")
    print(f"Mode: {category_counts.index[0]}")
    
    # Bimodal distribution
    bimodal_data = [1, 2, 2, 3, 5, 5, 6]
    print(f"\nBimodal data: {bimodal_data}")
    print(f"Modes: 2 and 5 (both appear twice)")
    
    return mode_result

mode_val = mode_examples()
```

### 2.4 When to Use Each Measure

```python
def central_tendency_comparison():
    """Compare different measures of central tendency"""
    
    # Different distribution types
    distributions = {
        'Normal': np.random.normal(50, 10, 1000),
        'Right Skewed': np.random.exponential(2, 1000),
        'Left Skewed': 100 - np.random.exponential(2, 1000),
        'With Outliers': np.concatenate([np.random.normal(50, 5, 950), [150, 160, 170, 180, 190]])
    }
    
    print("Central Tendency Comparison:")
    print("=" * 40)
    
    for name, data in distributions.items():
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        # Mode for continuous data (approximate using histogram)
        hist, bins = np.histogram(data, bins=20)
        mode_bin_idx = np.argmax(hist)
        mode_approx = (bins[mode_bin_idx] + bins[mode_bin_idx + 1]) / 2
        
        print(f"\n{name} Distribution:")
        print(f"  Mean:   {mean_val:.2f}")
        print(f"  Median: {median_val:.2f}")
        print(f"  Mode:   {mode_approx:.2f} (approx)")
        
        # Skewness indicator
        if abs(mean_val - median_val) < 1:
            print("  → Symmetric distribution")
        elif mean_val > median_val:
            print("  → Right-skewed (positive skew)")
        else:
            print("  → Left-skewed (negative skew)")

central_tendency_comparison()
```

## 3. Measures of Variability

### 3.1 Range
The difference between maximum and minimum values.

```python
def range_examples():
    """Demonstrate range calculation and limitations"""
    
    # Two datasets with same mean but different ranges
    dataset1 = [48, 49, 50, 51, 52]  # Low variability
    dataset2 = [10, 30, 50, 70, 90]  # High variability
    
    print("Range (Max - Min):")
    
    for i, data in enumerate([dataset1, dataset2], 1):
        data_range = np.max(data) - np.min(data)
        data_mean = np.mean(data)
        
        print(f"\nDataset {i}: {data}")
        print(f"  Mean: {data_mean}")
        print(f"  Range: {data_range}")
        print(f"  Min: {np.min(data)}, Max: {np.max(data)}")
    
    # Limitation: sensitive to outliers
    data_with_outlier = [48, 49, 50, 51, 52, 200]
    original_range = np.max([48, 49, 50, 51, 52]) - np.min([48, 49, 50, 51, 52])
    outlier_range = np.max(data_with_outlier) - np.min(data_with_outlier)
    
    print(f"\nOutlier sensitivity:")
    print(f"  Original range: {original_range}")
    print(f"  With outlier: {outlier_range}")

range_examples()
```

### 3.2 Variance and Standard Deviation
Measures of how much data points deviate from the mean.

```python
def variance_std_examples():
    """Demonstrate variance and standard deviation"""
    
    # Sample data: daily website visitors
    daily_visitors = [1200, 1350, 1180, 1420, 1290, 1380, 1250, 1320, 1400, 1180]
    
    # Manual calculation
    mean_visitors = np.mean(daily_visitors)
    deviations = [x - mean_visitors for x in daily_visitors]
    squared_deviations = [d**2 for d in deviations]
    
    # Population vs sample variance
    population_variance = np.mean(squared_deviations)  # Divide by n
    sample_variance = np.sum(squared_deviations) / (len(daily_visitors) - 1)  # Divide by n-1
    
    population_std = np.sqrt(population_variance)
    sample_std = np.sqrt(sample_variance)
    
    print("Variance and Standard Deviation:")
    print(f"Daily visitors: {daily_visitors}")
    print(f"Mean visitors: {mean_visitors:.1f}")
    print()
    
    print("Manual calculation:")
    print(f"Deviations from mean: {[round(d, 1) for d in deviations]}")
    print(f"Squared deviations: {[round(d, 1) for d in squared_deviations]}")
    print()
    
    print("Population statistics (divide by n):")
    print(f"  Variance: {population_variance:.1f}")
    print(f"  Standard deviation: {population_std:.1f}")
    print()
    
    print("Sample statistics (divide by n-1):")
    print(f"  Variance: {sample_variance:.1f}")
    print(f"  Standard deviation: {sample_std:.1f}")
    
    # Using NumPy
    print(f"\nNumPy calculations:")
    print(f"  np.var (population): {np.var(daily_visitors):.1f}")
    print(f"  np.var (sample): {np.var(daily_visitors, ddof=1):.1f}")
    print(f"  np.std (population): {np.std(daily_visitors):.1f}")
    print(f"  np.std (sample): {np.std(daily_visitors, ddof=1):.1f}")
    
    return daily_visitors, sample_std

visitors, std_val = variance_std_examples()
```

### 3.3 Coefficient of Variation
Standard deviation relative to the mean (useful for comparing variability across different scales).

```python
def coefficient_of_variation():
    """Demonstrate coefficient of variation for comparing datasets"""
    
    # Compare variability across different scales
    datasets = {
        'Stock Price ($)': [100, 105, 98, 112, 95, 108, 102],
        'Stock Returns (%)': [2.5, 1.8, -1.2, 3.4, 0.9, 2.1, 1.5],
        'Website Traffic': [15000, 16200, 14800, 17500, 15800, 16900, 15600]
    }
    
    print("Coefficient of Variation (CV = std/mean):")
    print("=" * 45)
    
    for name, data in datasets.items():
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        cv = std_val / mean_val
        
        print(f"\n{name}:")
        print(f"  Mean: {mean_val:.2f}")
        print(f"  Std Dev: {std_val:.2f}")
        print(f"  CV: {cv:.4f} ({cv*100:.2f}%)")
    
    print(f"\nInterpretation:")
    print(f"• CV < 0.1 (10%): Low variability")
    print(f"• 0.1 ≤ CV ≤ 0.3: Moderate variability") 
    print(f"• CV > 0.3 (30%): High variability")

coefficient_of_variation()
```

## 4. Measures of Distribution Shape

### 4.1 Skewness
Measures the asymmetry of the distribution.

```python
def skewness_examples():
    """Demonstrate skewness calculation and interpretation"""
    
    # Generate different distributions
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 1000)
    right_skewed = np.random.exponential(2, 1000)
    left_skewed = -np.random.exponential(2, 1000)
    
    distributions = {
        'Normal (Symmetric)': normal_data,
        'Right Skewed (Positive)': right_skewed,
        'Left Skewed (Negative)': left_skewed
    }
    
    print("Skewness Analysis:")
    print("=" * 20)
    
    for name, data in distributions.items():
        skew_val = stats.skew(data)
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        print(f"\n{name}:")
        print(f"  Skewness: {skew_val:.3f}")
        print(f"  Mean: {mean_val:.2f}")
        print(f"  Median: {median_val:.2f}")
        print(f"  Mean - Median: {mean_val - median_val:.2f}")
        
        # Interpretation
        if abs(skew_val) < 0.5:
            interpretation = "Approximately symmetric"
        elif 0.5 <= abs(skew_val) < 1:
            interpretation = "Moderately skewed"
        else:
            interpretation = "Highly skewed"
        
        print(f"  Interpretation: {interpretation}")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    for i, (name, data) in enumerate(distributions.items(), 1):
        plt.subplot(1, 3, i)
        plt.hist(data, bins=30, density=True, alpha=0.7)
        plt.axvline(np.mean(data), color='red', linestyle='--', label='Mean')
        plt.axvline(np.median(data), color='blue', linestyle='--', label='Median')
        plt.title(f'{name}\nSkewness: {stats.skew(data):.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

skewness_examples()
```

### 4.2 Kurtosis
Measures the "tailedness" of the distribution.

```python
def kurtosis_examples():
    """Demonstrate kurtosis calculation and interpretation"""
    
    np.random.seed(42)
    
    # Different kurtosis examples
    normal_data = np.random.normal(0, 1, 1000)
    heavy_tail = np.random.laplace(0, 1, 1000)  # Heavier tails
    light_tail = np.random.uniform(-2, 2, 1000)  # Lighter tails
    
    distributions = {
        'Normal (Mesokurtic)': normal_data,
        'Heavy Tails (Leptokurtic)': heavy_tail,
        'Light Tails (Platykurtic)': light_tail
    }
    
    print("Kurtosis Analysis:")
    print("=" * 18)
    
    for name, data in distributions.items():
        kurt_val = stats.kurtosis(data)  # Excess kurtosis (normal = 0)
        kurt_raw = stats.kurtosis(data, fisher=False)  # Raw kurtosis (normal = 3)
        
        print(f"\n{name}:")
        print(f"  Excess Kurtosis: {kurt_val:.3f}")
        print(f"  Raw Kurtosis: {kurt_raw:.3f}")
        
        # Interpretation
        if kurt_val < -0.5:
            interpretation = "Platykurtic (light tails)"
        elif -0.5 <= kurt_val <= 0.5:
            interpretation = "Mesokurtic (normal-like tails)"
        else:
            interpretation = "Leptokurtic (heavy tails)"
        
        print(f"  Interpretation: {interpretation}")

kurtosis_examples()
```

## 5. Percentiles and Quartiles

### 5.1 Quartiles and IQR
```python
def quartiles_iqr_examples():
    """Demonstrate quartiles and interquartile range"""
    
    # Student test scores
    scores = [72, 85, 91, 68, 79, 88, 92, 75, 82, 89, 94, 77, 83, 90, 86, 78, 84, 93, 76, 81]
    
    # Calculate quartiles
    q1 = np.percentile(scores, 25)
    q2 = np.percentile(scores, 50)  # Same as median
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    
    print("Quartiles and IQR:")
    print(f"Sorted scores: {sorted(scores)}")
    print(f"Q1 (25th percentile): {q1}")
    print(f"Q2 (50th percentile/median): {q2}")
    print(f"Q3 (75th percentile): {q3}")
    print(f"IQR (Q3 - Q1): {iqr}")
    
    # Five-number summary
    minimum = np.min(scores)
    maximum = np.max(scores)
    
    print(f"\nFive-number summary:")
    print(f"  Minimum: {minimum}")
    print(f"  Q1: {q1}")
    print(f"  Median: {q2}")
    print(f"  Q3: {q3}")
    print(f"  Maximum: {maximum}")
    
    # Outlier detection using IQR
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    outliers = [score for score in scores if score < lower_fence or score > upper_fence]
    
    print(f"\nOutlier detection (1.5 × IQR rule):")
    print(f"  Lower fence: {lower_fence:.1f}")
    print(f"  Upper fence: {upper_fence:.1f}")
    print(f"  Outliers: {outliers}")
    
    return scores, q1, q2, q3, iqr

scores, q1, q2, q3, iqr = quartiles_iqr_examples()
```

### 5.2 Percentiles in Practice
```python
def percentiles_practical():
    """Practical applications of percentiles"""
    
    # Website response times
    response_times = np.random.exponential(0.2, 10000) * 1000  # Convert to milliseconds
    
    # Key percentiles for performance monitoring
    percentiles = [50, 90, 95, 99, 99.9]
    
    print("Website Performance Percentiles:")
    print("=" * 35)
    
    for p in percentiles:
        value = np.percentile(response_times, p)
        print(f"P{p:4.1f}: {value:7.1f} ms")
    
    print(f"\nInterpretation:")
    print(f"• P50 (median): Half of requests are faster than this")
    print(f"• P90: 90% of requests are faster than this")
    print(f"• P95: 95% of requests are faster than this") 
    print(f"• P99: Only 1% of requests are slower than this")
    print(f"• P99.9: Only 0.1% of requests are slower than this")
    
    # SLA example
    sla_threshold = 500  # 500ms SLA
    performance = (response_times <= sla_threshold).mean() * 100
    
    print(f"\nSLA Performance:")
    print(f"• Threshold: {sla_threshold} ms")
    print(f"• Performance: {performance:.2f}% of requests meet SLA")

percentiles_practical()
```

## 6. Data Visualization for Descriptive Statistics

### 6.1 Histograms and Distribution Plots
```python
def visualization_examples():
    """Create visualizations for descriptive statistics"""
    
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 1000)
    skewed_data = np.random.exponential(2, 1000) * 50
    
    plt.figure(figsize=(15, 10))
    
    # Histogram with statistics
    plt.subplot(2, 3, 1)
    plt.hist(normal_data, bins=30, density=True, alpha=0.7, color='skyblue')
    plt.axvline(np.mean(normal_data), color='red', linestyle='--', label=f'Mean: {np.mean(normal_data):.1f}')
    plt.axvline(np.median(normal_data), color='blue', linestyle='--', label=f'Median: {np.median(normal_data):.1f}')
    plt.title('Normal Distribution')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    # Box plot
    plt.subplot(2, 3, 2)
    plt.boxplot([normal_data], labels=['Normal Data'])
    plt.title('Box Plot')
    plt.ylabel('Value')
    
    # Q-Q plot
    plt.subplot(2, 3, 3)
    stats.probplot(normal_data, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal)')
    plt.grid(True)
    
    # Skewed data histogram
    plt.subplot(2, 3, 4)
    plt.hist(skewed_data, bins=30, density=True, alpha=0.7, color='lightcoral')
    plt.axvline(np.mean(skewed_data), color='red', linestyle='--', label=f'Mean: {np.mean(skewed_data):.1f}')
    plt.axvline(np.median(skewed_data), color='blue', linestyle='--', label=f'Median: {np.median(skewed_data):.1f}')
    plt.title('Skewed Distribution')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    # Box plot comparison
    plt.subplot(2, 3, 5)
    plt.boxplot([normal_data, skewed_data], labels=['Normal', 'Skewed'])
    plt.title('Box Plot Comparison')
    plt.ylabel('Value')
    
    # Violin plot
    plt.subplot(2, 3, 6)
    data_for_violin = [normal_data, skewed_data]
    parts = plt.violinplot(data_for_violin, positions=[1, 2])
    plt.xticks([1, 2], ['Normal', 'Skewed'])
    plt.title('Violin Plot')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()

visualization_examples()
```

### 6.2 Summary Statistics Tables
```python
def summary_statistics_table():
    """Create comprehensive summary statistics table"""
    
    # Multi-variable dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 12, n_samples).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'satisfaction': np.random.beta(2, 1, n_samples) * 10,
        'purchases': np.random.poisson(5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Built-in summary
    print("Built-in Pandas describe():")
    print("=" * 30)
    print(df.describe())
    
    # Custom summary with additional statistics
    def custom_summary(series):
        return pd.Series({
            'count': series.count(),
            'mean': series.mean(),
            'median': series.median(),
            'mode': series.mode().iloc[0] if len(series.mode()) > 0 else np.nan,
            'std': series.std(),
            'var': series.var(),
            'min': series.min(),
            'max': series.max(),
            'range': series.max() - series.min(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'cv': series.std() / series.mean() if series.mean() != 0 else np.nan
        })
    
    print(f"\n\nCustom Summary Statistics:")
    print("=" * 30)
    summary_df = df.apply(custom_summary)
    print(summary_df.round(3))
    
    return df, summary_df

df, summary = summary_statistics_table()
```

## 7. Machine Learning Applications

### 7.1 Exploratory Data Analysis (EDA)
```python
def eda_with_descriptive_stats():
    """Use descriptive statistics for EDA in ML projects"""
    
    # Simulate customer churn dataset
    np.random.seed(42)
    n_customers = 1000
    
    # Features that influence churn
    tenure = np.random.exponential(2, n_customers)
    monthly_charges = np.random.normal(65, 20, n_customers).clip(20, 150)
    total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_customers)
    
    # Churn probability based on features
    churn_prob = 1 / (1 + np.exp(-(0.5 - 0.1 * tenure + 0.01 * monthly_charges)))
    churned = np.random.binomial(1, churn_prob, n_customers)
    
    # Create dataset
    customer_data = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churned': churned
    })
    
    print("EDA with Descriptive Statistics:")
    print("=" * 35)
    
    # Overall statistics
    print("\n1. Overall Dataset Summary:")
    print(customer_data.describe())
    
    # Group statistics by target variable
    print(f"\n2. Statistics by Churn Status:")
    churn_stats = customer_data.groupby('churned').agg({
        'tenure': ['mean', 'median', 'std'],
        'monthly_charges': ['mean', 'median', 'std'],
        'total_charges': ['mean', 'median', 'std']
    }).round(2)
    
    print(churn_stats)
    
    # Identify potential issues
    print(f"\n3. Data Quality Checks:")
    print(f"Missing values:\n{customer_data.isnull().sum()}")
    print(f"\nOutliers (using IQR method):")
    
    for col in ['tenure', 'monthly_charges', 'total_charges']:
        q1, q3 = customer_data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        outliers = customer_data[(customer_data[col] < lower_fence) | 
                               (customer_data[col] > upper_fence)]
        
        print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(customer_data)*100:.1f}%)")
    
    return customer_data

customer_data = eda_with_descriptive_stats()
```

### 7.2 Feature Engineering with Statistics
```python
def feature_engineering_stats():
    """Use descriptive statistics for feature engineering"""
    
    # Continue with customer dataset
    data = customer_data.copy()
    
    print("Feature Engineering with Descriptive Statistics:")
    print("=" * 48)
    
    # 1. Standardization (z-score normalization)
    print("\n1. Standardization (Z-score):")
    for col in ['tenure', 'monthly_charges', 'total_charges']:
        mean_val = data[col].mean()
        std_val = data[col].std()
        data[f'{col}_standardized'] = (data[col] - mean_val) / std_val
        
        print(f"{col}:")
        print(f"  Original: mean={mean_val:.2f}, std={std_val:.2f}")
        print(f"  Standardized: mean={data[f'{col}_standardized'].mean():.2f}, std={data[f'{col}_standardized'].std():.2f}")
    
    # 2. Min-Max normalization
    print(f"\n2. Min-Max Normalization:")
    for col in ['tenure', 'monthly_charges', 'total_charges']:
        min_val = data[col].min()
        max_val = data[col].max()
        data[f'{col}_minmax'] = (data[col] - min_val) / (max_val - min_val)
        
        print(f"{col}:")
        print(f"  Original range: [{min_val:.2f}, {max_val:.2f}]")
        print(f"  Normalized range: [{data[f'{col}_minmax'].min():.2f}, {data[f'{col}_minmax'].max():.2f}]")
    
    # 3. Outlier-based features
    print(f"\n3. Outlier-based Features:")
    for col in ['tenure', 'monthly_charges', 'total_charges']:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        data[f'{col}_is_outlier'] = ((data[col] < lower_fence) | 
                                   (data[col] > upper_fence)).astype(int)
        
        outlier_rate = data[f'{col}_is_outlier'].mean()
        print(f"{col}_is_outlier: {outlier_rate:.1%} flagged as outliers")
    
    # 4. Percentile-based features
    print(f"\n4. Percentile-based Features:")
    for col in ['tenure', 'monthly_charges', 'total_charges']:
        data[f'{col}_percentile'] = data[col].rank(pct=True)
        
        print(f"{col}_percentile: converted to percentile ranks")
    
    print(f"\nFinal dataset shape: {data.shape}")
    print(f"Original features: 4")
    print(f"Engineered features: {data.shape[1] - 4}")
    
    return data

engineered_data = feature_engineering_stats()
```

## 8. Common Mistakes and Best Practices

### 8.1 Common Pitfalls
```python
def common_mistakes():
    """Demonstrate common mistakes in descriptive statistics"""
    
    print("Common Mistakes in Descriptive Statistics:")
    print("=" * 42)
    
    # Mistake 1: Using mean for skewed data
    print("\n1. Using Mean for Skewed Data:")
    skewed_salaries = [30000, 32000, 35000, 38000, 40000, 42000, 45000, 48000, 50000, 200000]
    
    mean_salary = np.mean(skewed_salaries)
    median_salary = np.median(skewed_salaries)
    
    print(f"Salaries: {skewed_salaries}")
    print(f"Mean: ${mean_salary:,.0f} (misleading due to outlier)")
    print(f"Median: ${median_salary:,.0f} (better representation)")
    
    # Mistake 2: Ignoring sample vs population
    print(f"\n2. Sample vs Population Variance:")
    sample_data = [85, 90, 78, 92, 88, 79, 94, 86]
    
    pop_var = np.var(sample_data)  # Population variance (n)
    sample_var = np.var(sample_data, ddof=1)  # Sample variance (n-1)
    
    print(f"Data: {sample_data}")
    print(f"Population variance (biased): {pop_var:.2f}")
    print(f"Sample variance (unbiased): {sample_var:.2f}")
    print("Always use sample statistics for inference!")
    
    # Mistake 3: Not checking for outliers
    print(f"\n3. Not Checking for Outliers:")
    data_with_error = [10, 12, 11, 13, 10, 12, 14, 11, 1100]  # Data entry error
    
    print(f"Data: {data_with_error}")
    print(f"Mean (with outlier): {np.mean(data_with_error):.1f}")
    print(f"Mean (without outlier): {np.mean(data_with_error[:-1]):.1f}")
    print("Always visualize and check for outliers!")
    
    # Mistake 4: Comparing different scales without normalization
    print(f"\n4. Comparing Different Scales:")
    feature1 = [100, 150, 120, 180, 110]  # Age in months
    feature2 = [0.02, 0.05, 0.03, 0.08, 0.01]  # Conversion rate
    
    print(f"Feature 1 (age): std = {np.std(feature1):.1f}")
    print(f"Feature 2 (rate): std = {np.std(feature2):.3f}")
    print("Use coefficient of variation or standardization!")

common_mistakes()
```

### 8.2 Best Practices
```python
def best_practices():
    """Demonstrate best practices for descriptive statistics"""
    
    print("Best Practices for Descriptive Statistics:")
    print("=" * 42)
    
    # Practice 1: Always visualize first
    print("\n1. Always Visualize Your Data:")
    print("• Create histograms, box plots, scatter plots")
    print("• Look for patterns, outliers, and distributions")
    print("• Don't rely only on summary statistics")
    
    # Practice 2: Report multiple measures
    print(f"\n2. Report Multiple Measures of Central Tendency:")
    mixed_data = [1, 2, 2, 3, 4, 5, 5, 5, 6, 100]
    
    print(f"Data: {mixed_data}")
    print(f"Mean: {np.mean(mixed_data):.1f}")
    print(f"Median: {np.median(mixed_data):.1f}")
    print(f"Mode: {stats.mode(mixed_data).mode[0]}")
    print("Provides complete picture of central tendency")
    
    # Practice 3: Use robust statistics for outliers
    print(f"\n3. Use Robust Statistics When Appropriate:")
    from scipy.stats import trim_mean
    
    outlier_data = [10, 12, 11, 13, 10, 12, 14, 11, 50, 8]
    
    regular_mean = np.mean(outlier_data)
    trimmed_mean = trim_mean(outlier_data, 0.1)  # Trim 10% from each end
    median_val = np.median(outlier_data)
    
    print(f"Data with outliers: {outlier_data}")
    print(f"Regular mean: {regular_mean:.1f}")
    print(f"Trimmed mean (10%): {trimmed_mean:.1f}")
    print(f"Median: {median_val:.1f}")
    
    # Practice 4: Consider the context
    print(f"\n4. Consider the Context and Purpose:")
    print("• Income data: Use median (skewed distribution)")
    print("• Test scores: Mean may be appropriate (often normal)")
    print("• Response times: Use percentiles (P95, P99)")
    print("• Error rates: Mean and variance both important")

best_practices()
```

## 9. Real-World Applications

### 9.1 A/B Testing Analysis
```python
def ab_testing_descriptive_stats():
    """Use descriptive statistics for A/B testing analysis"""
    
    # Simulate A/B test data
    np.random.seed(42)
    
    # Control group (existing design)
    control_conversions = np.random.binomial(1, 0.12, 1000)  # 12% conversion rate
    control_revenue = control_conversions * np.random.gamma(2, 50)  # Revenue when converted
    
    # Treatment group (new design) 
    treatment_conversions = np.random.binomial(1, 0.14, 1000)  # 14% conversion rate
    treatment_revenue = treatment_conversions * np.random.gamma(2, 55)  # Slightly higher revenue
    
    print("A/B Testing Analysis with Descriptive Statistics:")
    print("=" * 50)
    
    # Conversion rate analysis
    control_rate = np.mean(control_conversions)
    treatment_rate = np.mean(treatment_conversions)
    
    print(f"\n1. Conversion Rate Analysis:")
    print(f"Control conversion rate: {control_rate:.1%}")
    print(f"Treatment conversion rate: {treatment_rate:.1%}")
    print(f"Relative improvement: {(treatment_rate - control_rate)/control_rate:.1%}")
    
    # Revenue analysis
    control_revenue_per_user = np.mean(control_revenue)
    treatment_revenue_per_user = np.mean(treatment_revenue)
    
    print(f"\n2. Revenue per User Analysis:")
    print(f"Control revenue per user: ${control_revenue_per_user:.2f}")
    print(f"Treatment revenue per user: ${treatment_revenue_per_user:.2f}")
    print(f"Revenue improvement: ${treatment_revenue_per_user - control_revenue_per_user:.2f}")
    
    # Variability analysis
    control_revenue_std = np.std(control_revenue)
    treatment_revenue_std = np.std(treatment_revenue)
    
    print(f"\n3. Variability Analysis:")
    print(f"Control revenue std: ${control_revenue_std:.2f}")
    print(f"Treatment revenue std: ${treatment_revenue_std:.2f}")
    print(f"Control CV: {control_revenue_std/control_revenue_per_user:.2f}")
    print(f"Treatment CV: {treatment_revenue_std/treatment_revenue_per_user:.2f}")

ab_testing_descriptive_stats()
```

### 9.2 Performance Monitoring
```python
def performance_monitoring():
    """Use descriptive statistics for system performance monitoring"""
    
    # Simulate system metrics over time
    np.random.seed(42)
    hours = 24
    measurements_per_hour = 60
    
    # Generate realistic performance data
    base_response_time = 200  # Base 200ms
    time_trend = np.linspace(0, 2*np.pi, hours * measurements_per_hour)
    daily_pattern = 50 * np.sin(time_trend) + 30 * np.sin(2 * time_trend)  # Daily pattern
    
    response_times = (base_response_time + daily_pattern + 
                     np.random.exponential(20, len(time_trend)))
    
    # Add some performance incidents
    incident_times = [300, 800, 1200]  # Incident at these measurements
    for incident in incident_times:
        if incident < len(response_times):
            response_times[incident:incident+30] *= 3  # 3x slower during incident
    
    print("Performance Monitoring with Descriptive Statistics:")
    print("=" * 52)
    
    # Overall statistics
    print(f"\n1. Overall Performance (24 hours):")
    print(f"Mean response time: {np.mean(response_times):.1f} ms")
    print(f"Median response time: {np.median(response_times):.1f} ms")
    print(f"95th percentile: {np.percentile(response_times, 95):.1f} ms")
    print(f"99th percentile: {np.percentile(response_times, 99):.1f} ms")
    print(f"Maximum: {np.max(response_times):.1f} ms")
    
    # SLA analysis
    sla_threshold = 500  # 500ms SLA
    sla_compliance = (response_times <= sla_threshold).mean()
    
    print(f"\n2. SLA Analysis (500ms threshold):")
    print(f"SLA compliance: {sla_compliance:.1%}")
    print(f"SLA violations: {(1-sla_compliance):.1%}")
    
    # Hourly breakdown
    hourly_stats = []
    for hour in range(hours):
        start_idx = hour * measurements_per_hour
        end_idx = (hour + 1) * measurements_per_hour
        hour_data = response_times[start_idx:end_idx]
        
        hourly_stats.append({
            'hour': hour,
            'mean': np.mean(hour_data),
            'p95': np.percentile(hour_data, 95),
            'max': np.max(hour_data),
            'sla_compliance': (hour_data <= sla_threshold).mean()
        })
    
    hourly_df = pd.DataFrame(hourly_stats)
    
    print(f"\n3. Peak Performance Hours:")
    worst_hours = hourly_df.nlargest(3, 'p95')
    print(worst_hours[['hour', 'mean', 'p95', 'sla_compliance']].to_string(index=False))

performance_monitoring()
```

## 10. Conclusion

Descriptive statistics provide the foundation for understanding data in machine learning:

### **Key Concepts Mastered:**

#### **Central Tendency:**
- **Mean**: Best for symmetric distributions, sensitive to outliers
- **Median**: Robust to outliers, good for skewed data
- **Mode**: Useful for categorical data and multimodal distributions

#### **Variability:**
- **Range**: Simple but outlier-sensitive
- **Standard Deviation**: Most common, interpretable in original units
- **IQR**: Robust alternative for spread measurement
- **Coefficient of Variation**: Enables comparison across different scales

#### **Distribution Shape:**
- **Skewness**: Measures asymmetry, indicates tail direction
- **Kurtosis**: Measures tail heaviness, identifies outlier-prone distributions

#### **Position Measures:**
- **Percentiles**: Powerful for performance monitoring and SLA analysis
- **Quartiles**: Essential for box plots and outlier detection

### **ML Applications:**

1. **Exploratory Data Analysis**: Understanding data before modeling
2. **Feature Engineering**: Creating meaningful features from statistics
3. **Outlier Detection**: Identifying anomalous data points
4. **Performance Monitoring**: System health and SLA compliance
5. **A/B Testing**: Comparing experimental groups
6. **Data Quality**: Assessing completeness and consistency

### **Best Practices:**
- **Always visualize** data alongside statistics
- **Report multiple measures** for complete understanding
- **Consider context** when choosing appropriate statistics
- **Check for outliers** before making decisions
- **Use robust statistics** when dealing with messy real-world data

### **Common Tools:**
- **NumPy/Pandas**: Core statistical functions
- **SciPy**: Advanced statistical methods
- **Matplotlib/Seaborn**: Visualization
- **Descriptive statistics**: Foundation for all advanced analytics

**Next in Statistics**: **Hypothesis Testing** - moving from describing data to making statistical inferences and testing claims about populations!

Understanding descriptive statistics gives you the essential tools to explore, understand, and communicate insights from data - the critical first step in any machine learning project. 