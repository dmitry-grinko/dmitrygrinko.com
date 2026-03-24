# Bayes' Theorem and Applications

Bayes' theorem is the mathematical heart of modern machine learning. From spam filters to medical diagnosis, from recommendation systems to autonomous vehicles, Bayes' theorem provides the fundamental framework for updating beliefs with new evidence. Understanding this theorem deeply will unlock your understanding of how AI systems learn and make decisions under uncertainty.

## 1. Bayes' Theorem: The Foundation

### Mathematical Form:
`P(A|B) = P(B|A) × P(A) / P(B)`

### Components:
- **P(A|B)**: **Posterior** - what we want to know (updated belief)
- **P(B|A)**: **Likelihood** - how well A explains the evidence B
- **P(A)**: **Prior** - our initial belief about A
- **P(B)**: **Evidence** - probability of observing B (normalizing constant)

### Intuitive Meaning:
"Given new evidence B, how should we update our belief about A?"

## 2. The Bayesian Mindset

Bayesian thinking follows a cycle:
1. **Start with prior beliefs** (based on domain knowledge)
2. **Observe new evidence** 
3. **Update beliefs** using Bayes' theorem
4. **Make decisions** based on updated beliefs
5. **Repeat** as new evidence arrives

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def bayesian_update_demo():
    """Demonstrate the Bayesian update process"""
    
    # Scenario: Estimating bias of a coin
    # Prior belief: coin is fair (centered around 0.5)
    
    # Prior: Beta distribution (conjugate prior for binomial)
    alpha_prior, beta_prior = 1, 1  # Uniform prior (no initial bias)
    
    # Observations: sequence of coin flips
    observations = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]  # 1=heads, 0=tails
    
    # Track beliefs over time
    beliefs = []
    
    alpha, beta = alpha_prior, beta_prior
    
    for i, obs in enumerate(observations):
        # Bayesian update (for Beta-Binomial conjugacy)
        if obs == 1:  # heads
            alpha += 1
        else:  # tails
            beta += 1
        
        # Current belief distribution
        belief_dist = stats.beta(alpha, beta)
        beliefs.append({
            'alpha': alpha,
            'beta': beta,
            'mean': alpha / (alpha + beta),
            'mode': (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else 0.5
        })
        
        print(f"After {i+1} flips: P(heads) ≈ {belief_dist.mean():.3f}")
    
    return beliefs

beliefs = bayesian_update_demo()
```

## 3. Classic Examples

### 3.1 Medical Diagnosis
```python
def medical_diagnosis_bayes():
    """Medical diagnosis using Bayes' theorem"""
    
    # Scenario: Testing for a rare disease
    disease_prevalence = 0.001      # 0.1% of population has disease
    test_sensitivity = 0.99         # 99% true positive rate
    test_specificity = 0.95         # 95% true negative rate
    
    # Patient tests positive - what's the probability they have the disease?
    
    # Prior probabilities
    p_disease = disease_prevalence
    p_healthy = 1 - disease_prevalence
    
    # Likelihoods
    p_positive_given_disease = test_sensitivity
    p_positive_given_healthy = 1 - test_specificity
    
    # Evidence (total probability of testing positive)
    p_positive = (p_positive_given_disease * p_disease + 
                 p_positive_given_healthy * p_healthy)
    
    # Posterior (Bayes' theorem)
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    
    print("Medical Diagnosis with Bayes' Theorem:")
    print(f"Disease prevalence: {p_disease:.1%}")
    print(f"Test sensitivity: {test_sensitivity:.1%}")
    print(f"Test specificity: {test_specificity:.1%}")
    print(f"P(positive test) = {p_positive:.3f}")
    print(f"P(disease | positive test) = {p_disease_given_positive:.1%}")
    print(f"\nCounterinterior: Even with 99% accurate test,")
    print(f"positive result only means {p_disease_given_positive:.1%} chance of disease!")
    
    return p_disease_given_positive

medical_diagnosis_bayes()
```

### 3.2 Spam Email Detection
```python
def naive_bayes_spam_filter():
    """Implement a simple Naive Bayes spam filter"""
    
    # Training data: word frequencies
    # Format: {word: {'spam': count, 'ham': count}}
    word_counts = {
        'free': {'spam': 100, 'ham': 5},
        'money': {'spam': 80, 'ham': 10},
        'meeting': {'spam': 5, 'ham': 60},
        'lunch': {'spam': 2, 'ham': 40},
        'click': {'spam': 70, 'ham': 8},
        'report': {'spam': 10, 'ham': 50}
    }
    
    # Total emails in training
    total_spam = 500
    total_ham = 1000
    total_emails = total_spam + total_ham
    
    # Prior probabilities
    p_spam = total_spam / total_emails
    p_ham = total_ham / total_emails
    
    def classify_email(words):
        """Classify email using Naive Bayes"""
        
        # Start with prior probabilities (log space for numerical stability)
        log_prob_spam = np.log(p_spam)
        log_prob_ham = np.log(p_ham)
        
        for word in words:
            if word in word_counts:
                # Laplace smoothing to handle unseen words
                spam_count = word_counts[word]['spam'] + 1
                ham_count = word_counts[word]['ham'] + 1
                
                # P(word | spam) and P(word | ham)
                p_word_given_spam = spam_count / (total_spam + len(word_counts))
                p_word_given_ham = ham_count / (total_ham + len(word_counts))
                
                # Add log probabilities
                log_prob_spam += np.log(p_word_given_spam)
                log_prob_ham += np.log(p_word_given_ham)
        
        # Convert back to probabilities and normalize
        prob_spam = np.exp(log_prob_spam)
        prob_ham = np.exp(log_prob_ham)
        
        total_prob = prob_spam + prob_ham
        prob_spam_normalized = prob_spam / total_prob
        
        return {
            'spam_probability': prob_spam_normalized,
            'ham_probability': 1 - prob_spam_normalized,
            'classification': 'spam' if prob_spam_normalized > 0.5 else 'ham'
        }
    
    # Test emails
    test_emails = [
        ['free', 'money', 'click'],           # Likely spam
        ['meeting', 'lunch', 'report'],       # Likely ham
        ['free', 'lunch'],                    # Mixed signals
    ]
    
    print("Naive Bayes Spam Filter Results:")
    for i, email in enumerate(test_emails):
        result = classify_email(email)
        print(f"Email {i+1} {email}:")
        print(f"  Spam probability: {result['spam_probability']:.3f}")
        print(f"  Classification: {result['classification']}")
        print()
    
    return classify_email

spam_classifier = naive_bayes_spam_filter()
```

## 4. Bayesian Machine Learning

### 4.1 Bayesian Linear Regression
```python
def bayesian_linear_regression():
    """Bayesian approach to linear regression"""
    
    # Generate synthetic data
    np.random.seed(42)
    true_slope = 2.0
    true_intercept = 1.0
    noise_std = 0.5
    
    X = np.linspace(0, 5, 20)
    y_true = true_intercept + true_slope * X
    y = y_true + np.random.normal(0, noise_std, len(X))
    
    # Bayesian linear regression (simplified)
    # Prior beliefs about parameters
    prior_mean = np.array([0, 0])  # [intercept, slope]
    prior_cov = np.array([[10, 0], [0, 10]])  # High uncertainty initially
    
    # Likelihood precision (inverse of noise variance)
    likelihood_precision = 1 / (noise_std ** 2)
    
    # Design matrix
    X_design = np.column_stack([np.ones(len(X)), X])
    
    # Posterior calculation (conjugate prior)
    posterior_precision = np.linalg.inv(prior_cov) + likelihood_precision * (X_design.T @ X_design)
    posterior_cov = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_cov @ (np.linalg.inv(prior_cov) @ prior_mean + 
                                    likelihood_precision * (X_design.T @ y))
    
    print("Bayesian Linear Regression:")
    print(f"True parameters: intercept={true_intercept}, slope={true_slope}")
    print(f"Posterior mean: intercept={posterior_mean[0]:.3f}, slope={posterior_mean[1]:.3f}")
    print(f"Posterior std: intercept={np.sqrt(posterior_cov[0,0]):.3f}, slope={np.sqrt(posterior_cov[1,1]):.3f}")
    
    # Prediction with uncertainty
    X_test = np.array([3.0])
    X_test_design = np.array([[1, X_test[0]]])
    
    # Predictive mean and variance
    pred_mean = X_test_design @ posterior_mean
    pred_var = X_test_design @ posterior_cov @ X_test_design.T + noise_std**2
    pred_std = np.sqrt(pred_var)
    
    print(f"\nPrediction at X=3.0:")
    print(f"Mean: {pred_mean[0]:.3f}")
    print(f"95% confidence interval: [{pred_mean[0] - 1.96*pred_std[0]:.3f}, {pred_mean[0] + 1.96*pred_std[0]:.3f}]")
    
    return posterior_mean, posterior_cov

bayesian_linear_regression()
```

### 4.2 Bayesian Neural Networks (Concept)
```python
def bayesian_neural_network_concept():
    """Conceptual example of Bayesian neural networks"""
    
    # In Bayesian neural networks, weights are distributions, not point estimates
    
    class BayesianLayer:
        def __init__(self, input_size, output_size):
            # Weight means and log variances (parameters to learn)
            self.weight_mean = np.random.normal(0, 0.1, (input_size, output_size))
            self.weight_log_var = np.random.normal(-3, 0.1, (input_size, output_size))
            
        def sample_weights(self):
            """Sample weights from their posterior distributions"""
            weight_std = np.exp(0.5 * self.weight_log_var)
            return np.random.normal(self.weight_mean, weight_std)
        
        def forward_uncertainty(self, x, n_samples=100):
            """Forward pass with uncertainty quantification"""
            outputs = []
            
            for _ in range(n_samples):
                weights = self.sample_weights()
                output = x @ weights
                outputs.append(output)
            
            outputs = np.array(outputs)
            
            return {
                'mean': np.mean(outputs, axis=0),
                'std': np.std(outputs, axis=0),
                'samples': outputs
            }
    
    # Example usage
    layer = BayesianLayer(5, 3)
    x = np.random.randn(10, 5)  # Batch of 10 samples
    
    result = layer.forward_uncertainty(x)
    
    print("Bayesian Neural Network Layer:")
    print(f"Input shape: {x.shape}")
    print(f"Output mean shape: {result['mean'].shape}")
    print(f"Output std shape: {result['std'].shape}")
    print(f"Mean uncertainty (avg std): {np.mean(result['std']):.3f}")

bayesian_neural_network_concept()
```

## 5. A/B Testing with Bayes

### 5.1 Bayesian A/B Testing
```python
def bayesian_ab_testing():
    """Bayesian approach to A/B testing"""
    
    # Observed data
    control = {'conversions': 120, 'visitors': 2000}
    treatment = {'conversions': 135, 'visitors': 2000}
    
    # Beta priors (uniform: Beta(1,1))
    prior_alpha, prior_beta = 1, 1
    
    # Posterior parameters (Beta-Binomial conjugacy)
    control_alpha = prior_alpha + control['conversions']
    control_beta = prior_beta + control['visitors'] - control['conversions']
    
    treatment_alpha = prior_alpha + treatment['conversions']
    treatment_beta = prior_beta + treatment['visitors'] - treatment['conversions']
    
    # Posterior distributions
    control_posterior = stats.beta(control_alpha, control_beta)
    treatment_posterior = stats.beta(treatment_alpha, treatment_beta)
    
    # Monte Carlo estimation of P(treatment > control)
    n_samples = 100000
    control_samples = control_posterior.rvs(n_samples)
    treatment_samples = treatment_posterior.rvs(n_samples)
    
    prob_treatment_better = np.mean(treatment_samples > control_samples)
    
    # Expected lift
    expected_lift = (treatment_posterior.mean() - control_posterior.mean()) / control_posterior.mean()
    
    print("Bayesian A/B Testing Results:")
    print(f"Control conversion rate: {control['conversions']/control['visitors']:.3f}")
    print(f"Treatment conversion rate: {treatment['conversions']/treatment['visitors']:.3f}")
    print(f"Posterior mean - Control: {control_posterior.mean():.3f}")
    print(f"Posterior mean - Treatment: {treatment_posterior.mean():.3f}")
    print(f"P(Treatment > Control): {prob_treatment_better:.3f}")
    print(f"Expected lift: {expected_lift:.1%}")
    
    # Credible intervals
    control_ci = control_posterior.interval(0.95)
    treatment_ci = treatment_posterior.interval(0.95)
    
    print(f"Control 95% CI: [{control_ci[0]:.3f}, {control_ci[1]:.3f}]")
    print(f"Treatment 95% CI: [{treatment_ci[0]:.3f}, {treatment_ci[1]:.3f}]")
    
    return prob_treatment_better, expected_lift

bayesian_ab_testing()
```

### 5.2 Multi-Armed Bandit with Thompson Sampling
```python
def thompson_sampling_bandit():
    """Thompson Sampling for multi-armed bandit problems"""
    
    class ThompsonSamplingBandit:
        def __init__(self, n_arms):
            self.n_arms = n_arms
            # Beta distribution parameters for each arm
            self.alpha = np.ones(n_arms)  # Success count + 1
            self.beta = np.ones(n_arms)   # Failure count + 1
            
        def select_arm(self):
            """Select arm using Thompson Sampling"""
            # Sample from each arm's posterior
            samples = [np.random.beta(self.alpha[i], self.beta[i]) 
                      for i in range(self.n_arms)]
            return np.argmax(samples)
        
        def update(self, arm, reward):
            """Update posterior after observing reward"""
            if reward == 1:
                self.alpha[arm] += 1
            else:
                self.beta[arm] += 1
        
        def get_arm_stats(self):
            """Get current statistics for each arm"""
            means = self.alpha / (self.alpha + self.beta)
            variances = (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
            return means, variances
    
    # Simulate bandit environment
    true_conversion_rates = [0.05, 0.03, 0.07, 0.04]  # Arm 2 is best
    n_arms = len(true_conversion_rates)
    n_rounds = 1000
    
    bandit = ThompsonSamplingBandit(n_arms)
    
    # Track results
    arm_counts = np.zeros(n_arms)
    cumulative_rewards = 0
    
    for round_num in range(n_rounds):
        # Select arm
        chosen_arm = bandit.select_arm()
        
        # Generate reward
        reward = np.random.binomial(1, true_conversion_rates[chosen_arm])
        
        # Update bandit
        bandit.update(chosen_arm, reward)
        
        # Track statistics
        arm_counts[chosen_arm] += 1
        cumulative_rewards += reward
        
        # Print progress every 200 rounds
        if (round_num + 1) % 200 == 0:
            means, _ = bandit.get_arm_stats()
            print(f"Round {round_num + 1}:")
            print(f"  Arm selection %: {arm_counts / (round_num + 1) * 100}")
            print(f"  Estimated rates: {means}")
            print(f"  True rates: {true_conversion_rates}")
            print()
    
    print(f"Final cumulative reward: {cumulative_rewards}")
    print(f"Optimal strategy would have given: {n_rounds * max(true_conversion_rates):.0f}")
    print(f"Regret: {n_rounds * max(true_conversion_rates) - cumulative_rewards:.0f}")

thompson_sampling_bandit()
```

## 6. Bayesian Optimization

### 6.1 Hyperparameter Optimization
```python
def bayesian_hyperparameter_optimization():
    """Bayesian optimization for hyperparameter tuning"""
    
    # Simulated objective function (model performance vs hyperparameter)
    def objective_function(x):
        """Simulated validation accuracy as function of learning rate"""
        # Peak around x=0.01, with noise
        return -(x - 0.01)**2 * 1000 + 0.95 + np.random.normal(0, 0.01)
    
    # Bayesian optimization simulation
    class GaussianProcess:
        def __init__(self):
            self.X_observed = []
            self.y_observed = []
        
        def fit(self, X, y):
            self.X_observed = X
            self.y_observed = y
        
        def predict(self, X_new):
            """Simplified GP prediction (normally would use proper kernel)"""
            if not self.X_observed:
                return np.zeros(len(X_new)), np.ones(len(X_new))
            
            # Very simplified prediction
            means = []
            stds = []
            
            for x in X_new:
                # Find nearest observed point
                distances = [abs(x - x_obs) for x_obs in self.X_observed]
                nearest_idx = np.argmin(distances)
                nearest_dist = distances[nearest_idx]
                
                # Simple interpolation
                mean = self.y_observed[nearest_idx]
                std = 0.1 + nearest_dist * 0.5  # Higher uncertainty further from data
                
                means.append(mean)
                stds.append(std)
            
            return np.array(means), np.array(stds)
    
    def acquisition_function(means, stds, best_observed):
        """Expected Improvement acquisition function"""
        z = (means - best_observed) / stds
        ei = (means - best_observed) * stats.norm.cdf(z) + stds * stats.norm.pdf(z)
        return ei
    
    # Bayesian optimization loop
    gp = GaussianProcess()
    X_observed = []
    y_observed = []
    
    # Search space
    search_space = np.logspace(-4, -1, 100)  # Learning rates from 0.0001 to 0.1
    
    print("Bayesian Hyperparameter Optimization:")
    
    for iteration in range(10):
        if iteration == 0:
            # Random initial point
            next_x = np.random.choice(search_space)
        else:
            # Use acquisition function to select next point
            gp.fit(X_observed, y_observed)
            means, stds = gp.predict(search_space)
            
            best_observed = max(y_observed) if y_observed else 0
            ei_values = acquisition_function(means, stds, best_observed)
            
            next_x = search_space[np.argmax(ei_values)]
        
        # Evaluate objective function
        y_new = objective_function(next_x)
        
        # Update observations
        X_observed.append(next_x)
        y_observed.append(y_new)
        
        best_so_far = max(y_observed)
        print(f"Iteration {iteration + 1}: x={next_x:.6f}, y={y_new:.4f}, best={best_so_far:.4f}")
    
    best_idx = np.argmax(y_observed)
    print(f"\nBest hyperparameter found: {X_observed[best_idx]:.6f}")
    print(f"Best performance: {y_observed[best_idx]:.4f}")

bayesian_hyperparameter_optimization()
```

## 7. Practical Implementation Tips

### 7.1 Handling Numerical Stability
```python
def log_space_bayes():
    """Implementing Bayes' theorem in log space for numerical stability"""
    
    def safe_bayes_log(log_likelihood, log_prior, log_evidence):
        """Compute posterior in log space"""
        log_posterior = log_likelihood + log_prior - log_evidence
        return log_posterior
    
    def normalize_log_probabilities(log_probs):
        """Normalize log probabilities to sum to 1"""
        max_log_prob = np.max(log_probs)
        normalized_log_probs = log_probs - max_log_prob
        probs = np.exp(normalized_log_probs)
        probs = probs / np.sum(probs)
        return probs
    
    # Example: very small probabilities that would underflow
    log_likelihoods = np.array([-50, -45, -52, -48])  # Very small likelihoods
    log_priors = np.array([-1.5, -1.2, -1.8, -1.3])  # Log of small priors
    
    # Compute log evidence (log sum of likelihood * prior)
    log_evidence = np.logaddexp.reduce(log_likelihoods + log_priors)
    
    # Compute posteriors in log space
    log_posteriors = log_likelihoods + log_priors - log_evidence
    
    # Convert back to probabilities
    posteriors = np.exp(log_posteriors)
    
    print("Numerical Stability with Log Space:")
    print(f"Log posteriors: {log_posteriors}")
    print(f"Posteriors: {posteriors}")
    print(f"Sum of posteriors: {np.sum(posteriors):.10f}")

log_space_bayes()
```

### 7.2 Conjugate Priors
```python
def conjugate_priors_examples():
    """Examples of conjugate prior distributions"""
    
    # Beta-Binomial conjugacy
    def beta_binomial_update(prior_alpha, prior_beta, successes, trials):
        """Update Beta prior with Binomial likelihood"""
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + trials - successes
        return posterior_alpha, posterior_beta
    
    # Normal-Normal conjugacy (known variance)
    def normal_normal_update(prior_mean, prior_var, data, likelihood_var):
        """Update Normal prior with Normal likelihood"""
        n = len(data)
        data_mean = np.mean(data)
        
        # Posterior parameters
        posterior_precision = 1/prior_var + n/likelihood_var
        posterior_var = 1/posterior_precision
        posterior_mean = posterior_var * (prior_mean/prior_var + n*data_mean/likelihood_var)
        
        return posterior_mean, posterior_var
    
    # Gamma-Poisson conjugacy
    def gamma_poisson_update(prior_shape, prior_rate, data):
        """Update Gamma prior with Poisson likelihood"""
        n = len(data)
        data_sum = np.sum(data)
        
        posterior_shape = prior_shape + data_sum
        posterior_rate = prior_rate + n
        
        return posterior_shape, posterior_rate
    
    print("Conjugate Prior Examples:")
    
    # Example 1: Beta-Binomial
    alpha, beta = beta_binomial_update(1, 1, 7, 10)  # 7 successes in 10 trials
    print(f"Beta-Binomial: posterior Beta({alpha}, {beta})")
    
    # Example 2: Normal-Normal
    data = [1.2, 1.5, 0.8, 1.1, 1.3]
    mean, var = normal_normal_update(0, 1, data, 0.1)
    print(f"Normal-Normal: posterior N({mean:.3f}, {var:.3f})")
    
    # Example 3: Gamma-Poisson
    poisson_data = [3, 5, 2, 4, 6, 3, 4]  # Event counts
    shape, rate = gamma_poisson_update(1, 1, poisson_data)
    print(f"Gamma-Poisson: posterior Gamma({shape}, {rate})")

conjugate_priors_examples()
```

## 8. Common Pitfalls and Solutions

### 8.1 Prior Sensitivity Analysis
```python
def prior_sensitivity_analysis():
    """Analyze how sensitive results are to prior choices"""
    
    # Scenario: estimating conversion rate
    data = {'successes': 15, 'trials': 100}
    
    # Different priors
    priors = {
        'uniform': (1, 1),           # Uniform prior
        'pessimistic': (1, 9),       # Expect low conversion
        'optimistic': (9, 1),        # Expect high conversion  
        'informative': (5, 45),      # Expect ~10% conversion
    }
    
    print("Prior Sensitivity Analysis:")
    print(f"Observed: {data['successes']}/{data['trials']} = {data['successes']/data['trials']:.3f}")
    print()
    
    for name, (alpha_prior, beta_prior) in priors.items():
        # Posterior
        alpha_post = alpha_prior + data['successes']
        beta_post = beta_prior + data['trials'] - data['successes']
        
        posterior_mean = alpha_post / (alpha_post + beta_post)
        posterior_mode = (alpha_post - 1) / (alpha_post + beta_post - 2) if alpha_post > 1 and beta_post > 1 else None
        
        # 95% credible interval
        ci = stats.beta(alpha_post, beta_post).interval(0.95)
        
        print(f"{name.capitalize()} prior Beta({alpha_prior}, {beta_prior}):")
        print(f"  Posterior mean: {posterior_mean:.3f}")
        if posterior_mode:
            print(f"  Posterior mode: {posterior_mode:.3f}")
        print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        print()

prior_sensitivity_analysis()
```

### 8.2 Model Comparison
```python
def bayesian_model_comparison():
    """Compare models using Bayes factors"""
    
    # Generate data that follows one of two models
    np.random.seed(42)
    true_model = 'linear'
    
    if true_model == 'linear':
        X = np.linspace(0, 10, 20)
        y = 2 * X + 1 + np.random.normal(0, 1, 20)
    else:  # quadratic
        X = np.linspace(0, 10, 20)
        y = 0.5 * X**2 + np.random.normal(0, 1, 20)
    
    def compute_marginal_likelihood(X, y, model_type):
        """Simplified marginal likelihood computation"""
        if model_type == 'linear':
            # Linear model: y = a + bx
            X_design = np.column_stack([np.ones(len(X)), X])
            n_params = 2
        else:  # quadratic
            # Quadratic model: y = a + bx + cx^2
            X_design = np.column_stack([np.ones(len(X)), X, X**2])
            n_params = 3
        
        # Simplified BIC approximation to marginal likelihood
        n = len(y)
        
        # Fit model
        theta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        y_pred = X_design @ theta
        
        # RSS and BIC
        rss = np.sum((y - y_pred)**2)
        bic = n * np.log(rss/n) + n_params * np.log(n)
        
        # Approximate log marginal likelihood
        log_marginal_likelihood = -0.5 * bic
        
        return log_marginal_likelihood, rss, theta
    
    # Compare models
    log_ml_linear, rss_linear, theta_linear = compute_marginal_likelihood(X, y, 'linear')
    log_ml_quad, rss_quad, theta_quad = compute_marginal_likelihood(X, y, 'quadratic')
    
    # Bayes factor
    log_bayes_factor = log_ml_linear - log_ml_quad
    bayes_factor = np.exp(log_bayes_factor)
    
    print("Bayesian Model Comparison:")
    print(f"True model: {true_model}")
    print(f"Linear model RSS: {rss_linear:.3f}")
    print(f"Quadratic model RSS: {rss_quad:.3f}")
    print(f"Log marginal likelihood - Linear: {log_ml_linear:.3f}")
    print(f"Log marginal likelihood - Quadratic: {log_ml_quad:.3f}")
    print(f"Log Bayes factor (Linear vs Quadratic): {log_bayes_factor:.3f}")
    print(f"Bayes factor: {bayes_factor:.3f}")
    
    if bayes_factor > 1:
        print("Evidence favors linear model")
    else:
        print("Evidence favors quadratic model")

bayesian_model_comparison()
```

## 9. Advanced Applications

### 9.1 Bayesian Deep Learning
```python
def bayesian_deep_learning_concept():
    """Conceptual framework for Bayesian deep learning"""
    
    print("Bayesian Deep Learning Concepts:")
    print("1. Weight Uncertainty: Treat neural network weights as distributions")
    print("2. Variational Inference: Approximate intractable posteriors")
    print("3. Monte Carlo Dropout: Simple approximation to Bayesian NN")
    print("4. Uncertainty Quantification: Know when model is uncertain")
    print()
    
    # Example: Monte Carlo Dropout for uncertainty
    def mc_dropout_prediction(model_output_samples):
        """Simulate MC Dropout predictions"""
        # Simulate multiple forward passes with dropout
        predictions = model_output_samples
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_pred
        
        return mean_pred, epistemic_uncertainty
    
    # Simulate predictions from multiple dropout samples
    n_samples = 100
    n_classes = 5
    predictions = np.random.dirichlet([1]*n_classes, n_samples)  # Softmax outputs
    
    mean_pred, uncertainty = mc_dropout_prediction(predictions)
    
    print("Monte Carlo Dropout Results:")
    print(f"Mean prediction: {mean_pred}")
    print(f"Uncertainty: {uncertainty}")
    print(f"Predicted class: {np.argmax(mean_pred)}")
    print(f"Confidence: {np.max(mean_pred):.3f} ± {uncertainty[np.argmax(mean_pred)]:.3f}")

bayesian_deep_learning_concept()
```

### 9.2 Bayesian Reinforcement Learning
```python
def bayesian_reinforcement_learning():
    """Thompson Sampling in reinforcement learning"""
    
    class BayesianBandit:
        """Multi-armed bandit with Bayesian updates"""
        
        def __init__(self, n_arms, prior_mean=0, prior_precision=1):
            self.n_arms = n_arms
            self.prior_mean = prior_mean
            self.prior_precision = prior_precision
            
            # Posterior parameters (Normal with known variance)
            self.posterior_means = np.full(n_arms, prior_mean)
            self.posterior_precisions = np.full(n_arms, prior_precision)
            self.n_pulls = np.zeros(n_arms)
        
        def select_arm(self):
            """Thompson Sampling: sample from posterior and choose best"""
            # Sample expected reward for each arm
            sampled_means = []
            for i in range(self.n_arms):
                posterior_var = 1 / self.posterior_precisions[i]
                sampled_mean = np.random.normal(self.posterior_means[i], np.sqrt(posterior_var))
                sampled_means.append(sampled_mean)
            
            return np.argmax(sampled_means)
        
        def update(self, arm, reward):
            """Bayesian update after observing reward"""
            # Update posterior (conjugate Normal-Normal)
            old_precision = self.posterior_precisions[arm]
            old_mean = self.posterior_means[arm]
            
            # Assume unit variance for likelihood
            likelihood_precision = 1.0
            
            new_precision = old_precision + likelihood_precision
            new_mean = (old_precision * old_mean + likelihood_precision * reward) / new_precision
            
            self.posterior_means[arm] = new_mean
            self.posterior_precisions[arm] = new_precision
            self.n_pulls[arm] += 1
    
    # Simulate environment
    true_means = [0.1, 0.3, 0.2, 0.5]  # Arm 3 is best
    bandit = BayesianBandit(len(true_means))
    
    n_rounds = 500
    cumulative_reward = 0
    
    for round_num in range(n_rounds):
        arm = bandit.select_arm()
        reward = np.random.normal(true_means[arm], 1.0)  # Noisy reward
        bandit.update(arm, reward)
        cumulative_reward += reward
        
        if (round_num + 1) % 100 == 0:
            print(f"Round {round_num + 1}:")
            print(f"  Pulls: {bandit.n_pulls}")
            print(f"  Posterior means: {bandit.posterior_means}")
            print(f"  True means: {true_means}")
            print()
    
    print(f"Total reward: {cumulative_reward:.2f}")
    print(f"Best arm discovered: {np.argmax(bandit.posterior_means)}")

bayesian_reinforcement_learning()
```

## 10. Conclusion

Bayes' theorem is the mathematical foundation of learning from data:

### Key Insights:
1. **Prior + Evidence → Posterior**: The fundamental update mechanism
2. **Uncertainty Quantification**: Not just predictions, but confidence
3. **Sequential Learning**: Update beliefs as new data arrives
4. **Model Comparison**: Compare hypotheses using evidence
5. **Decision Making**: Act optimally under uncertainty

### ML Applications:
- **Naive Bayes**: Text classification and spam detection
- **Bayesian Optimization**: Efficient hyperparameter tuning
- **A/B Testing**: Comparing interventions with uncertainty
- **Thompson Sampling**: Exploration-exploitation in bandits
- **Bayesian Neural Networks**: Deep learning with uncertainty

### Practical Guidelines:
- Choose appropriate priors based on domain knowledge
- Use conjugate priors when possible for analytical solutions
- Implement in log space for numerical stability
- Perform sensitivity analysis for prior choice
- Consider computational complexity vs. approximation trade-offs

**Real-world Impact:**
- **Google**: PageRank algorithm uses Bayesian principles
- **Netflix**: Recommendation systems with uncertainty
- **Medical AI**: Diagnosis with confidence intervals
- **Autonomous Vehicles**: Decision making under uncertainty
- **Financial Trading**: Portfolio optimization with risk assessment

Understanding Bayes' theorem deeply transforms how you approach machine learning problems. Instead of just getting point predictions, you learn to quantify uncertainty, update beliefs systematically, and make optimal decisions when data is incomplete or noisy.

**Next up:** **Probability Distributions** - the mathematical tools that model the patterns and variability we see in real-world data! 