# Probability Distributions

Probability distributions are the mathematical models that describe how probability is spread across different possible outcomes. They're everywhere in machine learning - from modeling user behavior to describing neural network weights, from generating synthetic data to quantifying prediction uncertainty. Understanding distributions is crucial for choosing the right statistical tools and building effective ML models.

## 1. What Are Probability Distributions?

A **probability distribution** describes how probability is distributed over a set of possible outcomes. It tells us:
- What outcomes are possible
- How likely each outcome is
- The shape and characteristics of the randomness

### Types of Distributions:

#### Discrete Distributions
Defined over countable outcomes (integers, categories)
- **Examples**: Number of clicks, classification labels, word counts

#### Continuous Distributions  
Defined over uncountable outcomes (real numbers)
- **Examples**: Height, temperature, prediction confidence scores

## 2. Discrete Distributions

### 2.1 Bernoulli Distribution
**Models**: Single trial with two outcomes (success/failure)

**Parameters**: p (probability of success)

**PMF**: P(X = 1) = p, P(X = 0) = 1-p

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def bernoulli_examples():
    """Demonstrate Bernoulli distribution applications"""
    
    # Example 1: Coin flip
    p_heads = 0.6  # Biased coin
    bernoulli_coin = stats.bernoulli(p_heads)
    
    # Generate samples
    flips = bernoulli_coin.rvs(size=1000)
    
    print("Bernoulli Distribution - Coin Flip:")
    print(f"P(Heads) = {p_heads}")
    print(f"Theoretical mean: {bernoulli_coin.mean()}")
    print(f"Empirical mean: {np.mean(flips):.3f}")
    print(f"Theoretical variance: {bernoulli_coin.var()}")
    print(f"Empirical variance: {np.var(flips):.3f}")
    
    # Example 2: Click-through rate
    print("\nApplication: Email Click-Through Rate")
    ctr = 0.05  # 5% click rate
    n_emails = 10000
    
    clicks = stats.bernoulli(ctr).rvs(size=n_emails)
    total_clicks = np.sum(clicks)
    
    print(f"Sent {n_emails} emails with {ctr:.1%} CTR")
    print(f"Expected clicks: {n_emails * ctr}")
    print(f"Actual clicks: {total_clicks}")
    
    return flips, clicks

bernoulli_examples()
```

### 2.2 Binomial Distribution
**Models**: Number of successes in n independent Bernoulli trials

**Parameters**: n (trials), p (success probability)

**PMF**: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

```python
def binomial_examples():
    """Demonstrate Binomial distribution applications"""
    
    # Example 1: A/B test analysis
    n_visitors = 1000
    conversion_rate = 0.12
    
    binomial_conversions = stats.binom(n_visitors, conversion_rate)
    
    # Probability calculations
    prob_exactly_100 = binomial_conversions.pmf(100)
    prob_at_least_150 = 1 - binomial_conversions.cdf(149)
    prob_between_100_140 = binomial_conversions.cdf(140) - binomial_conversions.cdf(99)
    
    print("Binomial Distribution - A/B Test:")
    print(f"n={n_visitors}, p={conversion_rate}")
    print(f"Expected conversions: {binomial_conversions.mean():.1f}")
    print(f"Standard deviation: {binomial_conversions.std():.1f}")
    print(f"P(exactly 100 conversions) = {prob_exactly_100:.4f}")
    print(f"P(at least 150 conversions) = {prob_at_least_150:.4f}")
    print(f"P(100-140 conversions) = {prob_between_100_140:.4f}")
    
    # Simulate multiple experiments
    n_experiments = 1000
    results = binomial_conversions.rvs(size=n_experiments)
    
    print(f"\nSimulation of {n_experiments} experiments:")
    print(f"Mean conversions: {np.mean(results):.1f}")
    print(f"95% of experiments had between {np.percentile(results, 2.5):.0f} and {np.percentile(results, 97.5):.0f} conversions")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=30, alpha=0.7, density=True, label='Simulated')
    
    x = np.arange(80, 160)
    plt.plot(x, binomial_conversions.pmf(x), 'ro-', alpha=0.8, label='Theoretical PMF')
    
    plt.axvline(binomial_conversions.mean(), color='red', linestyle='--', label='Mean')
    plt.xlabel('Number of Conversions')
    plt.ylabel('Probability')
    plt.title('Binomial Distribution: A/B Test Conversions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results

binomial_results = binomial_examples()
```

### 2.3 Poisson Distribution
**Models**: Number of events occurring in a fixed interval

**Parameters**: λ (rate parameter)

**PMF**: P(X = k) = (λ^k × e^(-λ)) / k!

```python
def poisson_examples():
    """Demonstrate Poisson distribution applications"""
    
    # Example 1: Website visits per hour
    avg_visits_per_hour = 45
    poisson_visits = stats.poisson(avg_visits_per_hour)
    
    print("Poisson Distribution - Website Visits:")
    print(f"Average visits per hour: {avg_visits_per_hour}")
    print(f"P(exactly 50 visits) = {poisson_visits.pmf(50):.4f}")
    print(f"P(more than 60 visits) = {1 - poisson_visits.cdf(60):.4f}")
    print(f"P(fewer than 30 visits) = {poisson_visits.cdf(29):.4f}")
    
    # Example 2: Email arrivals
    emails_per_minute = 2.5
    poisson_emails = stats.poisson(emails_per_minute)
    
    # Simulate one day (1440 minutes)
    daily_emails = []
    for minute in range(1440):
        emails_this_minute = poisson_emails.rvs()
        daily_emails.append(emails_this_minute)
    
    total_emails = sum(daily_emails)
    max_emails_per_minute = max(daily_emails)
    
    print(f"\nEmail Simulation (24 hours):")
    print(f"Expected emails per day: {emails_per_minute * 1440}")
    print(f"Actual emails: {total_emails}")
    print(f"Maximum emails in one minute: {max_emails_per_minute}")
    
    # Busy periods (more than 5 emails per minute)
    busy_minutes = sum(1 for emails in daily_emails if emails > 5)
    print(f"Minutes with >5 emails: {busy_minutes}")
    
    return daily_emails

poisson_results = poisson_examples()
```

### 2.4 Categorical Distribution
**Models**: Single trial with multiple outcomes

**Parameters**: p₁, p₂, ..., pₖ (probabilities for k categories)

```python
def categorical_examples():
    """Demonstrate Categorical distribution applications"""
    
    # Example: Product recommendation system
    product_probs = [0.4, 0.3, 0.2, 0.1]  # Probabilities for 4 products
    product_names = ['Product A', 'Product B', 'Product C', 'Product D']
    
    categorical_products = stats.multinomial(1, product_probs)
    
    # Simulate user preferences
    n_users = 10000
    user_choices = []
    
    for _ in range(n_users):
        choice = categorical_products.rvs()
        chosen_product = np.argmax(choice)
        user_choices.append(chosen_product)
    
    # Analyze results
    choice_counts = np.bincount(user_choices)
    choice_frequencies = choice_counts / n_users
    
    print("Categorical Distribution - Product Recommendations:")
    for i, (name, true_prob, observed_freq) in enumerate(zip(product_names, product_probs, choice_frequencies)):
        print(f"{name}: True prob = {true_prob:.3f}, Observed freq = {observed_freq:.3f}")
    
    # Revenue analysis
    product_revenues = [50, 75, 100, 150]  # Revenue per product
    expected_revenue = sum(prob * revenue for prob, revenue in zip(product_probs, product_revenues))
    
    actual_revenues = [choice_counts[i] * product_revenues[i] for i in range(len(product_revenues))]
    total_actual_revenue = sum(actual_revenues)
    
    print(f"\nRevenue Analysis:")
    print(f"Expected revenue per user: ${expected_revenue:.2f}")
    print(f"Actual average revenue per user: ${total_actual_revenue / n_users:.2f}")
    
    return user_choices

categorical_results = categorical_examples()
```

## 3. Continuous Distributions

### 3.1 Normal (Gaussian) Distribution
**Models**: Many natural phenomena, measurement errors, aggregated effects

**Parameters**: μ (mean), σ² (variance)

**PDF**: f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

```python
def normal_distribution_examples():
    """Demonstrate Normal distribution applications"""
    
    # Example 1: Student test scores
    mean_score = 75
    std_score = 12
    normal_scores = stats.norm(mean_score, std_score)
    
    print("Normal Distribution - Test Scores:")
    print(f"Mean: {mean_score}, Std Dev: {std_score}")
    print(f"P(score > 90) = {1 - normal_scores.cdf(90):.3f}")
    print(f"P(score < 60) = {normal_scores.cdf(60):.3f}")
    print(f"P(70 < score < 80) = {normal_scores.cdf(80) - normal_scores.cdf(70):.3f}")
    
    # Grade boundaries (percentiles)
    print(f"Top 10% threshold: {normal_scores.ppf(0.9):.1f}")
    print(f"Bottom 10% threshold: {normal_scores.ppf(0.1):.1f}")
    
    # Example 2: Neural network weights initialization
    def initialize_weights(layer_size, init_method='xavier'):
        """Initialize neural network weights using different distributions"""
        
        if init_method == 'xavier':
            # Xavier/Glorot initialization
            fan_in, fan_out = layer_size
            limit = np.sqrt(6 / (fan_in + fan_out))
            weights = np.random.uniform(-limit, limit, layer_size)
        elif init_method == 'he':
            # He initialization (good for ReLU)
            fan_in = layer_size[0]
            std = np.sqrt(2 / fan_in)
            weights = np.random.normal(0, std, layer_size)
        else:  # normal
            weights = np.random.normal(0, 0.1, layer_size)
        
        return weights
    
    # Compare initialization methods
    layer_sizes = [(784, 128), (128, 64), (64, 10)]
    
    print("\nNeural Network Weight Initialization:")
    for i, size in enumerate(layer_sizes):
        xavier_weights = initialize_weights(size, 'xavier')
        he_weights = initialize_weights(size, 'he')
        normal_weights = initialize_weights(size, 'normal')
        
        print(f"Layer {i+1} ({size[0]}→{size[1]}):")
        print(f"  Xavier - Mean: {np.mean(xavier_weights):.4f}, Std: {np.std(xavier_weights):.4f}")
        print(f"  He     - Mean: {np.mean(he_weights):.4f}, Std: {np.std(he_weights):.4f}")
        print(f"  Normal - Mean: {np.mean(normal_weights):.4f}, Std: {np.std(normal_weights):.4f}")
    
    return normal_scores

normal_examples = normal_distribution_examples()
```

### 3.2 Exponential Distribution
**Models**: Time between events, lifetime/survival analysis

**Parameters**: λ (rate parameter)

**PDF**: f(x) = λe^(-λx) for x ≥ 0

```python
def exponential_examples():
    """Demonstrate Exponential distribution applications"""
    
    # Example 1: Server response times
    avg_response_time = 0.5  # 500ms average
    rate_param = 1 / avg_response_time
    exponential_response = stats.expon(scale=avg_response_time)
    
    print("Exponential Distribution - Server Response Times:")
    print(f"Average response time: {avg_response_time:.1f}s")
    print(f"P(response < 0.2s) = {exponential_response.cdf(0.2):.3f}")
    print(f"P(response > 1.0s) = {1 - exponential_response.cdf(1.0):.3f}")
    print(f"95th percentile: {exponential_response.ppf(0.95):.3f}s")
    
    # Example 2: Customer service analysis
    # Time between customer arrivals
    avg_time_between_customers = 3  # minutes
    exponential_arrivals = stats.expon(scale=avg_time_between_customers)
    
    # Simulate 8-hour day (480 minutes)
    current_time = 0
    arrival_times = []
    
    while current_time < 480:
        time_to_next = exponential_arrivals.rvs()
        current_time += time_to_next
        if current_time < 480:
            arrival_times.append(current_time)
    
    n_customers = len(arrival_times)
    
    print(f"\nCustomer Arrival Simulation (8 hours):")
    print(f"Expected customers: {480 / avg_time_between_customers:.0f}")
    print(f"Actual customers: {n_customers}")
    
    # Busy periods (less than 1 minute between arrivals)
    inter_arrival_times = np.diff([0] + arrival_times)
    busy_periods = sum(1 for t in inter_arrival_times if t < 1)
    
    print(f"Busy periods (<1 min between arrivals): {busy_periods}")
    
    return arrival_times

exponential_results = exponential_examples()
```

### 3.3 Beta Distribution
**Models**: Probabilities, proportions, Bayesian priors

**Parameters**: α, β (shape parameters)

**PDF**: f(x) = (x^(α-1) × (1-x)^(β-1)) / B(α,β) for 0 ≤ x ≤ 1

```python
def beta_distribution_examples():
    """Demonstrate Beta distribution applications"""
    
    # Example 1: Bayesian A/B testing
    def beta_ab_test(control_data, treatment_data):
        """Analyze A/B test using Beta distributions"""
        
        # Prior: Beta(1,1) - uniform distribution
        prior_alpha, prior_beta = 1, 1
        
        # Control group posterior
        control_alpha = prior_alpha + control_data['conversions']
        control_beta = prior_beta + control_data['visitors'] - control_data['conversions']
        control_posterior = stats.beta(control_alpha, control_beta)
        
        # Treatment group posterior
        treatment_alpha = prior_alpha + treatment_data['conversions']
        treatment_beta = prior_beta + treatment_data['visitors'] - treatment_data['conversions']
        treatment_posterior = stats.beta(treatment_alpha, treatment_beta)
        
        return control_posterior, treatment_posterior
    
    # A/B test data
    control = {'conversions': 45, 'visitors': 1000}
    treatment = {'conversions': 55, 'visitors': 1000}
    
    control_dist, treatment_dist = beta_ab_test(control, treatment)
    
    print("Beta Distribution - A/B Testing:")
    print(f"Control: {control['conversions']}/{control['visitors']} = {control['conversions']/control['visitors']:.3f}")
    print(f"Treatment: {treatment['conversions']}/{treatment['visitors']} = {treatment['conversions']/treatment['visitors']:.3f}")
    
    print(f"\nPosterior distributions:")
    print(f"Control: Beta({control_dist.args[0]:.0f}, {control_dist.args[1]:.0f})")
    print(f"Treatment: Beta({treatment_dist.args[0]:.0f}, {treatment_dist.args[1]:.0f})")
    
    print(f"\nPosterior means:")
    print(f"Control: {control_dist.mean():.3f}")
    print(f"Treatment: {treatment_dist.mean():.3f}")
    
    # Probability that treatment is better
    n_samples = 100000
    control_samples = control_dist.rvs(n_samples)
    treatment_samples = treatment_dist.rvs(n_samples)
    prob_treatment_better = np.mean(treatment_samples > control_samples)
    
    print(f"\nP(Treatment > Control) = {prob_treatment_better:.3f}")
    
    # Example 2: Modeling user engagement scores
    print("\nExample 2: User Engagement Modeling")
    
    # Different user segments with different engagement patterns
    segments = {
        'new_users': stats.beta(2, 8),      # Low engagement
        'regular_users': stats.beta(5, 5),  # Moderate engagement  
        'power_users': stats.beta(8, 2)     # High engagement
    }
    
    for segment_name, dist in segments.items():
        mean_engagement = dist.mean()
        engagement_scores = dist.rvs(1000)
        
        print(f"{segment_name.replace('_', ' ').title()}:")
        print(f"  Mean engagement: {mean_engagement:.3f}")
        print(f"  P(engagement > 0.7): {1 - dist.cdf(0.7):.3f}")
        print(f"  P(engagement < 0.3): {dist.cdf(0.3):.3f}")
    
    return control_dist, treatment_dist

beta_results = beta_distribution_examples()
```

## 4. Central Limit Theorem in Practice

```python
def central_limit_theorem_demo():
    """Demonstrate Central Limit Theorem with different distributions"""
    
    def sample_means(distribution, n_samples, sample_size):
        """Generate sample means from any distribution"""
        means = []
        for _ in range(n_samples):
            sample = distribution.rvs(sample_size)
            means.append(np.mean(sample))
        return np.array(means)
    
    # Test with different source distributions
    distributions = {
        'Uniform': stats.uniform(0, 1),
        'Exponential': stats.expon(scale=2),
        'Bernoulli': stats.bernoulli(0.3),
        'Poisson': stats.poisson(5)
    }
    
    sample_sizes = [5, 30, 100]
    n_samples = 1000
    
    print("Central Limit Theorem Demonstration:")
    print("=" * 50)
    
    for dist_name, distribution in distributions.items():
        print(f"\nSource distribution: {dist_name}")
        print(f"Population mean: {distribution.mean():.3f}")
        print(f"Population std: {distribution.std():.3f}")
        
        for sample_size in sample_sizes:
            means = sample_means(distribution, n_samples, sample_size)
            
            # Theoretical CLT predictions
            theoretical_mean = distribution.mean()
            theoretical_std = distribution.std() / np.sqrt(sample_size)
            
            print(f"  Sample size {sample_size}:")
            print(f"    Sample mean of means: {np.mean(means):.3f} (theory: {theoretical_mean:.3f})")
            print(f"    Sample std of means: {np.std(means):.3f} (theory: {theoretical_std:.3f})")
            
            # Test normality (Shapiro-Wilk test)
            if len(means) <= 5000:  # Shapiro-Wilk has sample size limits
                stat, p_value = stats.shapiro(means[:1000])
                print(f"    Normality test p-value: {p_value:.4f}")

central_limit_theorem_demo()
```

## 5. Distribution Selection for ML

### 5.1 Choosing the Right Distribution
```python
def distribution_selection_guide():
    """Guide for selecting appropriate distributions in ML"""
    
    selection_guide = {
        'Binary Classification': {
            'Labels': 'Bernoulli',
            'Probabilities': 'Beta (for priors)',
            'Use case': 'Spam detection, medical diagnosis'
        },
        'Count Data': {
            'Fixed trials': 'Binomial', 
            'Rate-based': 'Poisson',
            'Use case': 'Click counts, word frequencies'
        },
        'Continuous Values': {
            'Symmetric, unbounded': 'Normal',
            'Positive only': 'Exponential, Gamma, Log-normal',
            'Bounded [0,1]': 'Beta',
            'Use case': 'Sensor readings, response times, probabilities'
        },
        'Multi-class': {
            'Single trial': 'Categorical',
            'Multiple trials': 'Multinomial',
            'Use case': 'Image classification, topic modeling'
        }
    }
    
    print("Distribution Selection Guide for ML:")
    print("=" * 40)
    
    for category, info in selection_guide.items():
        print(f"\n{category}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

distribution_selection_guide()
```

### 5.2 Distribution Fitting
```python
def fit_distributions_to_data():
    """Fit different distributions to data and compare"""
    
    # Generate synthetic data (mixture of distributions)
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(10, 2, 300),    # Normal component
        np.random.exponential(3, 200)    # Exponential component
    ])
    
    # Candidate distributions
    distributions = [
        stats.norm,      # Normal
        stats.expon,     # Exponential
        stats.gamma,     # Gamma
        stats.lognorm,   # Log-normal
    ]
    
    print("Distribution Fitting Results:")
    print("=" * 30)
    
    best_distribution = None
    best_aic = np.inf
    
    for distribution in distributions:
        try:
            # Fit distribution
            params = distribution.fit(data)
            
            # Calculate log-likelihood
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            
            # Calculate AIC (Akaike Information Criterion)
            k = len(params)  # Number of parameters
            aic = 2 * k - 2 * log_likelihood
            
            print(f"{distribution.name}:")
            print(f"  Parameters: {params}")
            print(f"  Log-likelihood: {log_likelihood:.2f}")
            print(f"  AIC: {aic:.2f}")
            
            if aic < best_aic:
                best_aic = aic
                best_distribution = (distribution, params)
            
        except Exception as e:
            print(f"{distribution.name}: Failed to fit ({e})")
    
    print(f"\nBest distribution: {best_distribution[0].name}")
    print(f"Best AIC: {best_aic:.2f}")
    
    return data, best_distribution

data, best_fit = fit_distributions_to_data()
```

## 6. Generating Synthetic Data

### 6.1 Data Augmentation with Distributions
```python
def synthetic_data_generation():
    """Generate synthetic data for ML training"""
    
    def generate_user_behavior_data(n_users=1000):
        """Generate synthetic user behavior dataset"""
        
        # User demographics
        ages = np.random.normal(35, 12, n_users).clip(18, 80)
        
        # Income (log-normal distribution)
        log_incomes = np.random.normal(10.5, 0.5, n_users)  # log scale
        incomes = np.exp(log_incomes)
        
        # Engagement score (Beta distribution)
        engagement_alpha = 2 + incomes / 20000  # Higher income → higher engagement
        engagement_beta = 3
        engagement_scores = np.random.beta(engagement_alpha, engagement_beta)
        
        # Purchase probability (logistic function of features)
        linear_combination = (
            -3 +                                    # Intercept
            0.02 * ages +                          # Age effect
            0.00001 * incomes +                    # Income effect  
            2 * engagement_scores                   # Engagement effect
        )
        purchase_probabilities = 1 / (1 + np.exp(-linear_combination))
        purchases = np.random.binomial(1, purchase_probabilities)
        
        return {
            'age': ages,
            'income': incomes,
            'engagement': engagement_scores,
            'purchase_probability': purchase_probabilities,
            'purchased': purchases
        }
    
    # Generate dataset
    synthetic_data = generate_user_behavior_data(5000)
    
    print("Synthetic User Behavior Dataset:")
    print(f"Number of users: {len(synthetic_data['age'])}")
    print(f"Average age: {np.mean(synthetic_data['age']):.1f}")
    print(f"Average income: ${np.mean(synthetic_data['income']):,.0f}")
    print(f"Average engagement: {np.mean(synthetic_data['engagement']):.3f}")
    print(f"Purchase rate: {np.mean(synthetic_data['purchased']):.3f}")
    
    # Verify relationships
    purchasers = synthetic_data['purchased'] == 1
    non_purchasers = synthetic_data['purchased'] == 0
    
    print("\nPurchaser vs Non-purchaser comparison:")
    print(f"Age - Purchasers: {np.mean(synthetic_data['age'][purchasers]):.1f}, Non-purchasers: {np.mean(synthetic_data['age'][non_purchasers]):.1f}")
    print(f"Income - Purchasers: ${np.mean(synthetic_data['income'][purchasers]):,.0f}, Non-purchasers: ${np.mean(synthetic_data['income'][non_purchasers]):,.0f}")
    print(f"Engagement - Purchasers: {np.mean(synthetic_data['engagement'][purchasers]):.3f}, Non-purchasers: {np.mean(synthetic_data['engagement'][non_purchasers]):.3f}")
    
    return synthetic_data

synthetic_dataset = synthetic_data_generation()
```

## 7. Advanced Applications

### 7.1 Mixture Models
```python
def gaussian_mixture_model_demo():
    """Demonstrate Gaussian Mixture Models"""
    
    from sklearn.mixture import GaussianMixture
    
    # Generate data from mixture of Gaussians
    np.random.seed(42)
    
    # Component 1: Young users
    young_users = np.random.multivariate_normal([25, 30000], [[25, 1000], [1000, 50000000]], 300)
    
    # Component 2: Middle-aged users
    middle_users = np.random.multivariate_normal([45, 60000], [[36, 2000], [2000, 100000000]], 400)
    
    # Component 3: Senior users
    senior_users = np.random.multivariate_normal([65, 45000], [[64, 1500], [1500, 75000000]], 200)
    
    # Combine data
    data = np.vstack([young_users, middle_users, senior_users])
    true_labels = np.array([0]*300 + [1]*400 + [2]*200)
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(data)
    
    # Predict cluster assignments
    predicted_labels = gmm.predict(data)
    predicted_probs = gmm.predict_proba(data)
    
    print("Gaussian Mixture Model Results:")
    print(f"Component weights: {gmm.weights_}")
    print(f"Component means:")
    for i, mean in enumerate(gmm.means_):
        print(f"  Component {i}: Age={mean[0]:.1f}, Income=${mean[1]:,.0f}")
    
    # Cluster interpretation
    for i in range(3):
        cluster_data = data[predicted_labels == i]
        print(f"\nCluster {i} characteristics:")
        print(f"  Size: {len(cluster_data)}")
        print(f"  Average age: {np.mean(cluster_data[:, 0]):.1f}")
        print(f"  Average income: ${np.mean(cluster_data[:, 1]):,.0f}")
    
    return gmm, data, predicted_labels

gmm_results = gaussian_mixture_model_demo()
```

### 7.2 Anomaly Detection with Distributions
```python
def anomaly_detection_with_distributions():
    """Use probability distributions for anomaly detection"""
    
    # Generate normal behavior data
    np.random.seed(42)
    normal_response_times = np.random.gamma(2, 0.1, 10000)  # Typical server response times
    
    # Fit distribution to normal data
    fitted_gamma = stats.gamma.fit(normal_response_times, floc=0)
    
    # Define anomaly threshold (99.9th percentile)
    threshold = stats.gamma.ppf(0.999, *fitted_gamma)
    
    print("Anomaly Detection with Gamma Distribution:")
    print(f"Fitted parameters: shape={fitted_gamma[0]:.3f}, scale={fitted_gamma[2]:.3f}")
    print(f"Anomaly threshold (99.9th percentile): {threshold:.3f}")
    
    # Test with new data (including some anomalies)
    test_data = np.concatenate([
        np.random.gamma(2, 0.1, 100),     # Normal data
        np.random.gamma(2, 0.5, 10),      # Slower responses (anomalies)
        [2.0, 3.0, 1.8]                   # Obvious anomalies
    ])
    
    # Detect anomalies
    anomaly_scores = -stats.gamma.logpdf(test_data, *fitted_gamma)  # Negative log-likelihood
    is_anomaly = test_data > threshold
    
    print(f"\nTest results:")
    print(f"Total test samples: {len(test_data)}")
    print(f"Detected anomalies: {np.sum(is_anomaly)}")
    print(f"Anomaly rate: {np.mean(is_anomaly):.1%}")
    
    # Show worst anomalies
    worst_anomalies = test_data[np.argsort(-anomaly_scores)[:5]]
    print(f"Top 5 anomalies: {worst_anomalies}")
    
    return test_data, is_anomaly, anomaly_scores

anomaly_results = anomaly_detection_with_distributions()
```

## 8. Common Pitfalls and Best Practices

### 8.1 Distribution Assumptions
```python
def validate_distribution_assumptions():
    """Validate key assumptions about data distributions"""
    
    def test_normality(data, alpha=0.05):
        """Test if data follows normal distribution"""
        from scipy.stats import shapiro, normaltest
        
        # Shapiro-Wilk test (good for small samples)
        if len(data) <= 5000:
            stat_sw, p_sw = shapiro(data)
            normal_sw = p_sw > alpha
        else:
            stat_sw, p_sw = None, None
            normal_sw = None
        
        # D'Agostino test (good for larger samples) 
        stat_da, p_da = normaltest(data)
        normal_da = p_da > alpha
        
        return {
            'shapiro_wilk': {'statistic': stat_sw, 'p_value': p_sw, 'is_normal': normal_sw},
            'dagostino': {'statistic': stat_da, 'p_value': p_da, 'is_normal': normal_da}
        }
    
    def test_independence(data):
        """Test if data points are independent (simplified)"""
        # Ljung-Box test for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        result = acorr_ljungbox(data, lags=10, return_df=True)
        is_independent = result['lb_pvalue'].iloc[-1] > 0.05
        
        return {'is_independent': is_independent, 'p_value': result['lb_pvalue'].iloc[-1]}
    
    # Test different datasets
    datasets = {
        'Normal': np.random.normal(0, 1, 1000),
        'Uniform': np.random.uniform(0, 1, 1000), 
        'Exponential': np.random.exponential(1, 1000),
        'Autocorrelated': None  # Will generate below
    }
    
    # Generate autocorrelated data
    autocorr_data = np.zeros(1000)
    autocorr_data[0] = np.random.normal()
    for i in range(1, 1000):
        autocorr_data[i] = 0.7 * autocorr_data[i-1] + np.random.normal(0, 0.5)
    datasets['Autocorrelated'] = autocorr_data
    
    print("Distribution Assumption Testing:")
    print("=" * 40)
    
    for name, data in datasets.items():
        print(f"\n{name} Data:")
        
        # Test normality
        normality_results = test_normality(data)
        print(f"  Normality (Shapiro-Wilk): {normality_results['shapiro_wilk']['is_normal']} (p={normality_results['shapiro_wilk']['p_value']:.4f})")
        print(f"  Normality (D'Agostino): {normality_results['dagostino']['is_normal']} (p={normality_results['dagostino']['p_value']:.4f})")
        
        # Test independence  
        independence_result = test_independence(data)
        print(f"  Independence: {independence_result['is_independent']} (p={independence_result['p_value']:.4f})")

validate_distribution_assumptions()
```

### 8.2 Parameter Estimation Pitfalls
```python
def parameter_estimation_pitfalls():
    """Common mistakes in parameter estimation"""
    
    print("Parameter Estimation Pitfalls:")
    print("=" * 30)
    
    # Pitfall 1: Small sample bias
    print("\n1. Small Sample Bias:")
    true_mean, true_std = 10, 2
    
    sample_sizes = [5, 20, 100, 1000]
    n_experiments = 1000
    
    for n in sample_sizes:
        estimated_means = []
        estimated_stds = []
        
        for _ in range(n_experiments):
            sample = np.random.normal(true_mean, true_std, n)
            estimated_means.append(np.mean(sample))
            estimated_stds.append(np.std(sample))  # Biased estimator
        
        mean_bias = np.mean(estimated_means) - true_mean
        std_bias = np.mean(estimated_stds) - true_std
        
        print(f"  n={n:4d}: Mean bias={mean_bias:+.3f}, Std bias={std_bias:+.3f}")
    
    # Pitfall 2: Outlier sensitivity
    print("\n2. Outlier Sensitivity:")
    base_data = np.random.normal(5, 1, 100)
    
    # Add outliers
    data_with_outliers = np.concatenate([base_data, [20, 25, -10]])
    
    print(f"  Without outliers: mean={np.mean(base_data):.2f}, std={np.std(base_data):.2f}")
    print(f"  With outliers:    mean={np.mean(data_with_outliers):.2f}, std={np.std(data_with_outliers):.2f}")
    
    # Robust alternatives
    from scipy.stats import trim_mean
    robust_mean = trim_mean(data_with_outliers, 0.1)  # Trim 10% from each end
    print(f"  Robust mean (10% trimmed): {robust_mean:.2f}")
    
    # Pitfall 3: Distribution misspecification
    print("\n3. Distribution Misspecification:")
    
    # Generate data from mixture, try to fit single distribution
    mixture_data = np.concatenate([
        np.random.normal(2, 0.5, 500),
        np.random.normal(8, 0.5, 500)
    ])
    
    # Fit normal distribution
    fitted_mean = np.mean(mixture_data)
    fitted_std = np.std(mixture_data)
    
    print(f"  Mixture data fitted as normal: mean={fitted_mean:.2f}, std={fitted_std:.2f}")
    print(f"  True modes are at 2 and 8, but fitted mean is {fitted_mean:.2f}")

parameter_estimation_pitfalls()
```

## 9. Conclusion

Probability distributions are the mathematical foundation for modeling uncertainty and variability in machine learning:

### Key Takeaways:

#### Distribution Types:
- **Discrete**: Bernoulli, Binomial, Poisson, Categorical
- **Continuous**: Normal, Exponential, Beta, Gamma

#### ML Applications:
- **Data Generation**: Synthetic datasets and data augmentation
- **Model Assumptions**: Choosing appropriate loss functions
- **Uncertainty Quantification**: Bayesian inference and confidence intervals
- **Anomaly Detection**: Identifying outliers using probability thresholds

#### Best Practices:
- **Validate assumptions**: Test normality, independence, and distribution fit
- **Handle small samples**: Be aware of estimation bias
- **Consider robustness**: Use robust estimators when outliers are present
- **Visualize distributions**: Always plot your data before modeling

#### Common Distributions in ML:
- **Classification**: Bernoulli/Categorical for labels, Beta for priors
- **Count data**: Poisson for events, Binomial for fixed trials
- **Continuous features**: Normal for symmetric data, Exponential for waiting times
- **Neural networks**: Normal for weights, various distributions for activations

Understanding probability distributions empowers you to:
- Choose appropriate statistical models
- Generate realistic synthetic data
- Quantify model uncertainty
- Detect anomalies and outliers
- Make better assumptions about your data

**Next up**: **Central Limit Theorem** - the mathematical principle that explains why normal distributions appear everywhere and underlies much of statistical inference!

Mastering probability distributions gives you a powerful toolkit for understanding and modeling the randomness inherent in real-world data. Whether you're building recommendation systems, analyzing user behavior, or developing new ML algorithms, distributions provide the mathematical foundation for reasoning about uncertainty and variability. 