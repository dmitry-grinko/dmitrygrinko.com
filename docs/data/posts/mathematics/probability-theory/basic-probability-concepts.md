# Basic Probability Concepts

Probability is the mathematical language of uncertainty, and uncertainty is everywhere in machine learning. From predicting user behavior to handling noisy data, understanding probability is essential for making sense of ML algorithms and making informed decisions when data is incomplete or uncertain.

## 1. What Is Probability?

Probability quantifies **uncertainty** - how likely an event is to occur. It provides a mathematical framework for reasoning about randomness and making predictions when we don't have complete information.

### Definition:
For an event A, probability P(A) is a number between 0 and 1:
- **P(A) = 0**: Event A never happens
- **P(A) = 1**: Event A always happens  
- **0 < P(A) < 1**: Event A sometimes happens

### Interpretation:
- **Frequency interpretation**: If we repeat an experiment many times, P(A) is the fraction of times A occurs
- **Belief interpretation**: P(A) represents our degree of belief that A is true

## 2. Sample Spaces and Events

### 2.1 Sample Space (Ω)
The set of all possible outcomes of an experiment.

**Examples:**
- Coin flip: `Ω = {Heads, Tails}`
- Die roll: `Ω = {1, 2, 3, 4, 5, 6}`
- Email classification: `Ω = {Spam, Not Spam}`
- Image recognition: `Ω = {Cat, Dog, Bird, ...}`

### 2.2 Events
Subsets of the sample space.

```python
import numpy as np
from collections import Counter

# Example: Die roll
sample_space = {1, 2, 3, 4, 5, 6}

# Events
even_numbers = {2, 4, 6}
odd_numbers = {1, 3, 5}
greater_than_4 = {5, 6}
```

## 3. Axioms of Probability

Probability must satisfy three fundamental axioms:

### Axiom 1: Non-negativity
`P(A) ≥ 0` for any event A

### Axiom 2: Normalization  
`P(Ω) = 1` (something must happen)

### Axiom 3: Additivity
For mutually exclusive events A and B:
`P(A ∪ B) = P(A) + P(B)`

```python
def validate_probability_function(probabilities):
    """Check if a function satisfies probability axioms"""
    
    # Axiom 1: Non-negativity
    non_negative = all(p >= 0 for p in probabilities.values())
    
    # Axiom 2: Normalization
    total_prob = sum(probabilities.values())
    normalized = abs(total_prob - 1.0) < 1e-10
    
    return {
        'non_negative': non_negative,
        'normalized': normalized,
        'total_probability': total_prob,
        'valid': non_negative and normalized
    }

# Example: Valid probability distribution
die_probs = {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
print("Die probabilities:", validate_probability_function(die_probs))

# Example: Invalid probability distribution
invalid_probs = {1: 0.3, 2: 0.4, 3: 0.5}  # Sums to 1.2 > 1
print("Invalid probabilities:", validate_probability_function(invalid_probs))
```

## 4. Computing Probabilities

### 4.1 Classical Probability (Equally Likely Outcomes)
`P(A) = |A| / |Ω|`

Where |A| is the number of outcomes in A.

```python
def classical_probability(favorable_outcomes, total_outcomes):
    """Compute probability for equally likely outcomes"""
    return len(favorable_outcomes) / len(total_outcomes)

# Example: Probability of rolling an even number
sample_space = {1, 2, 3, 4, 5, 6}
even_outcomes = {2, 4, 6}
prob_even = classical_probability(even_outcomes, sample_space)
print(f"P(even) = {prob_even}")  # 0.5
```

### 4.2 Empirical Probability (From Data)
`P(A) ≈ count(A) / total_count`

```python
def empirical_probability(data, event_condition):
    """Estimate probability from observed data"""
    total_count = len(data)
    event_count = sum(1 for x in data if event_condition(x))
    return event_count / total_count

# Example: Estimate probability from coin flips
np.random.seed(42)
coin_flips = np.random.choice(['H', 'T'], size=1000)

prob_heads = empirical_probability(coin_flips, lambda x: x == 'H')
print(f"Empirical P(Heads) = {prob_heads:.3f}")

# As sample size increases, empirical probability approaches true probability
sample_sizes = [10, 100, 1000, 10000]
for n in sample_sizes:
    flips = np.random.choice(['H', 'T'], size=n)
    prob = empirical_probability(flips, lambda x: x == 'H')
    print(f"n={n:5d}: P(Heads) = {prob:.3f}")
```

## 5. Set Operations and Probability Rules

### 5.1 Union (OR): A ∪ B
Event A **or** B occurs (or both).

### 5.2 Intersection (AND): A ∩ B  
Event A **and** B both occur.

### 5.3 Complement: A'
Event A does **not** occur.

```python
def demonstrate_set_operations():
    """Demonstrate probability with set operations"""
    
    # Sample space: rolling two dice
    sample_space = [(i, j) for i in range(1, 7) for j in range(1, 7)]
    total_outcomes = len(sample_space)
    
    # Events
    first_die_even = {(i, j) for i, j in sample_space if i % 2 == 0}
    sum_equals_7 = {(i, j) for i, j in sample_space if i + j == 7}
    
    # Probabilities
    p_first_even = len(first_die_even) / total_outcomes
    p_sum_7 = len(sum_equals_7) / total_outcomes
    
    # Union: first die even OR sum equals 7
    union = first_die_even | sum_equals_7
    p_union = len(union) / total_outcomes
    
    # Intersection: first die even AND sum equals 7
    intersection = first_die_even & sum_equals_7
    p_intersection = len(intersection) / total_outcomes
    
    print(f"P(first die even) = {p_first_even:.3f}")
    print(f"P(sum = 7) = {p_sum_7:.3f}")
    print(f"P(first die even OR sum = 7) = {p_union:.3f}")
    print(f"P(first die even AND sum = 7) = {p_intersection:.3f}")
    
    # Verify addition rule: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    addition_rule = p_first_even + p_sum_7 - p_intersection
    print(f"Addition rule verification: {addition_rule:.3f} = {p_union:.3f}")

demonstrate_set_operations()
```

## 6. Conditional Probability

**Definition**: Probability of A given that B has occurred:
`P(A|B) = P(A ∩ B) / P(B)` (when P(B) > 0)

This is crucial in machine learning for updating beliefs with new information.

### 6.1 Intuition
Conditional probability restricts the sample space to only cases where B occurred.

```python
def conditional_probability_example():
    """Medical diagnosis example"""
    
    # Data: 1000 patients
    # Disease present: 50 patients (5%)
    # Test positive: 100 patients
    # True positives: 45 patients
    
    total_patients = 1000
    disease_present = 50
    test_positive = 100
    true_positives = 45
    
    # Basic probabilities
    p_disease = disease_present / total_patients
    p_positive = test_positive / total_patients
    p_positive_and_disease = true_positives / total_patients
    
    # Conditional probabilities
    p_positive_given_disease = true_positives / disease_present  # Sensitivity
    p_disease_given_positive = true_positives / test_positive   # Precision
    
    print("Medical Test Analysis:")
    print(f"P(disease) = {p_disease:.3f}")
    print(f"P(positive test) = {p_positive:.3f}")
    print(f"P(positive | disease) = {p_positive_given_disease:.3f} (Sensitivity)")
    print(f"P(disease | positive) = {p_disease_given_positive:.3f} (Precision)")
    
    # What doctors really want to know!
    print(f"\nIf test is positive, probability of having disease: {p_disease_given_positive:.1%}")

conditional_probability_example()
```

### 6.2 Chain Rule
For multiple events: `P(A ∩ B ∩ C) = P(A) × P(B|A) × P(C|A ∩ B)`

```python
def chain_rule_example():
    """Drawing cards without replacement"""
    
    # Drawing 3 cards from deck without replacement
    # What's the probability all 3 are hearts?
    
    total_cards = 52
    hearts = 13
    
    # First card is heart
    p_first_heart = hearts / total_cards
    
    # Second card is heart, given first was heart
    p_second_heart_given_first = (hearts - 1) / (total_cards - 1)
    
    # Third card is heart, given first two were hearts
    p_third_heart_given_first_two = (hearts - 2) / (total_cards - 2)
    
    # All three are hearts (using chain rule)
    p_all_hearts = p_first_heart * p_second_heart_given_first * p_third_heart_given_first_two
    
    print("Drawing 3 hearts without replacement:")
    print(f"P(1st heart) = {p_first_heart:.3f}")
    print(f"P(2nd heart | 1st heart) = {p_second_heart_given_first:.3f}")
    print(f"P(3rd heart | 1st & 2nd hearts) = {p_third_heart_given_first_two:.3f}")
    print(f"P(all 3 hearts) = {p_all_hearts:.4f}")

chain_rule_example()
```

## 7. Independence

Events A and B are **independent** if:
`P(A|B) = P(A)` or equivalently `P(A ∩ B) = P(A) × P(B)`

Independence means knowing B doesn't change the probability of A.

```python
def test_independence(event_a, event_b, sample_space):
    """Test if two events are independent"""
    
    total = len(sample_space)
    
    # Count occurrences
    count_a = len(event_a)
    count_b = len(event_b)
    count_both = len(event_a & event_b)
    
    # Probabilities
    p_a = count_a / total
    p_b = count_b / total
    p_both = count_both / total
    
    # Test independence: P(A ∩ B) = P(A) × P(B)?
    independence_product = p_a * p_b
    
    print(f"P(A) = {p_a:.3f}")
    print(f"P(B) = {p_b:.3f}")
    print(f"P(A ∩ B) = {p_both:.3f}")
    print(f"P(A) × P(B) = {independence_product:.3f}")
    print(f"Independent? {abs(p_both - independence_product) < 1e-10}")
    
    return abs(p_both - independence_product) < 1e-10

# Example: Two dice rolls
sample_space = {(i, j) for i in range(1, 7) for j in range(1, 7)}
first_die_even = {(i, j) for i, j in sample_space if i % 2 == 0}
second_die_odd = {(i, j) for i, j in sample_space if j % 2 == 1}

print("Testing independence of two dice:")
test_independence(first_die_even, second_die_odd, sample_space)
```

## 8. Law of Total Probability

If B₁, B₂, ..., Bₙ partition the sample space, then:
`P(A) = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + ... + P(A|Bₙ)P(Bₙ)`

```python
def law_of_total_probability_example():
    """Email spam detection example"""
    
    # Partition: emails come from 3 sources
    sources = ['work', 'personal', 'promotional']
    
    # Prior probabilities (base rates)
    p_source = {'work': 0.6, 'personal': 0.3, 'promotional': 0.1}
    
    # Conditional probabilities: P(spam | source)
    p_spam_given_source = {'work': 0.01, 'personal': 0.05, 'promotional': 0.8}
    
    # Law of total probability: P(spam)
    p_spam = sum(p_spam_given_source[source] * p_source[source] 
                 for source in sources)
    
    print("Email Spam Analysis:")
    for source in sources:
        print(f"P({source}) = {p_source[source]:.2f}")
        print(f"P(spam | {source}) = {p_spam_given_source[source]:.2f}")
    
    print(f"\nOverall P(spam) = {p_spam:.3f}")
    
    return p_spam, p_source, p_spam_given_source

law_of_total_probability_example()
```

## 9. Bayes' Theorem

**The most important formula in machine learning!**

`P(A|B) = P(B|A) × P(A) / P(B)`

Bayes' theorem lets us **update** our beliefs when we get new evidence.

### Components:
- **P(A|B)**: Posterior probability (what we want)
- **P(B|A)**: Likelihood (how well A explains B)
- **P(A)**: Prior probability (initial belief)
- **P(B)**: Evidence (normalizing constant)

```python
def bayes_theorem_spam_filter():
    """Naive Bayes spam filter example"""
    
    # Prior probabilities
    p_spam = 0.3
    p_ham = 0.7
    
    # Likelihoods: P(word | class)
    # Word: "free"
    p_free_given_spam = 0.8
    p_free_given_ham = 0.1
    
    # Evidence: P(free) using law of total probability
    p_free = p_free_given_spam * p_spam + p_free_given_ham * p_ham
    
    # Bayes' theorem: P(spam | "free")
    p_spam_given_free = (p_free_given_spam * p_spam) / p_free
    
    print("Naive Bayes Spam Filter:")
    print(f"Prior P(spam) = {p_spam:.2f}")
    print(f"Prior P(ham) = {p_ham:.2f}")
    print(f"P('free' | spam) = {p_free_given_spam:.2f}")
    print(f"P('free' | ham) = {p_free_given_ham:.2f}")
    print(f"P('free') = {p_free:.2f}")
    print(f"\nPosterior P(spam | 'free') = {p_spam_given_free:.3f}")
    
    return p_spam_given_free

bayes_theorem_spam_filter()
```

## 10. Machine Learning Applications

### 10.1 Classification Metrics
```python
def classification_metrics_probability():
    """Understanding precision, recall, F1 through probability"""
    
    # Confusion matrix data
    true_positives = 85
    false_positives = 15
    false_negatives = 10
    true_negatives = 90
    
    total = true_positives + false_positives + false_negatives + true_negatives
    
    # Probabilities
    p_positive_prediction = (true_positives + false_positives) / total
    p_actual_positive = (true_positives + false_negatives) / total
    
    # Metrics as conditional probabilities
    precision = true_positives / (true_positives + false_positives)  # P(actual=1 | pred=1)
    recall = true_positives / (true_positives + false_negatives)     # P(pred=1 | actual=1)
    specificity = true_negatives / (true_negatives + false_positives) # P(pred=0 | actual=0)
    
    print("Classification Metrics as Probabilities:")
    print(f"Precision = P(actual positive | predicted positive) = {precision:.3f}")
    print(f"Recall = P(predicted positive | actual positive) = {recall:.3f}")
    print(f"Specificity = P(predicted negative | actual negative) = {specificity:.3f}")

classification_metrics_probability()
```

### 10.2 A/B Testing
```python
def ab_test_analysis():
    """A/B test analysis using probability"""
    
    # Test data
    control_group = {'conversions': 250, 'total': 5000}
    treatment_group = {'conversions': 280, 'total': 5000}
    
    # Conversion rates (empirical probabilities)
    p_control = control_group['conversions'] / control_group['total']
    p_treatment = treatment_group['conversions'] / treatment_group['total']
    
    # Difference
    lift = (p_treatment - p_control) / p_control
    
    print("A/B Test Analysis:")
    print(f"Control conversion rate: {p_control:.1%}")
    print(f"Treatment conversion rate: {p_treatment:.1%}")
    print(f"Relative lift: {lift:.1%}")
    
    # Statistical significance requires more advanced probability theory
    # (confidence intervals, hypothesis testing)

ab_test_analysis()
```

### 10.3 Uncertainty in Predictions
```python
def model_uncertainty_example():
    """Expressing model uncertainty through probabilities"""
    
    # Ensemble predictions from 5 models
    model_predictions = [
        [0.8, 0.2],  # Model 1: [P(class 0), P(class 1)]
        [0.7, 0.3],  # Model 2
        [0.9, 0.1],  # Model 3
        [0.6, 0.4],  # Model 4
        [0.8, 0.2],  # Model 5
    ]
    
    # Average probabilities (ensemble prediction)
    ensemble_pred = np.mean(model_predictions, axis=0)
    
    # Uncertainty measures
    std_dev = np.std(model_predictions, axis=0)
    confidence = 1 - np.max(std_dev)  # Simple confidence measure
    
    print("Ensemble Model Uncertainty:")
    print(f"Ensemble prediction: {ensemble_pred}")
    print(f"Standard deviation: {std_dev}")
    print(f"Confidence: {confidence:.3f}")
    
    # Decision making with uncertainty
    threshold = 0.8
    high_confidence = ensemble_pred[1] > threshold and std_dev[1] < 0.1
    
    print(f"High confidence prediction? {high_confidence}")

model_uncertainty_example()
```

## 11. Common Probability Distributions (Preview)

### 11.1 Discrete Distributions
```python
def demonstrate_discrete_distributions():
    """Examples of common discrete distributions"""
    
    # Bernoulli: single trial (coin flip)
    def bernoulli(p):
        return np.random.choice([0, 1], p=[1-p, p])
    
    # Binomial: multiple Bernoulli trials
    def binomial_simulation(n, p, trials=1000):
        results = [sum(bernoulli(p) for _ in range(n)) for _ in range(trials)]
        return results
    
    # Example: 10 coin flips, probability of getting exactly k heads
    n, p = 10, 0.5
    results = binomial_simulation(n, p)
    
    print("Binomial Distribution Simulation (10 coin flips):")
    for k in range(11):
        probability = sum(1 for r in results if r == k) / len(results)
        print(f"P(exactly {k} heads) ≈ {probability:.3f}")

demonstrate_discrete_distributions()
```

### 11.2 Continuous Distributions
```python
def demonstrate_continuous_distributions():
    """Examples of continuous distributions"""
    
    # Normal distribution (Gaussian)
    np.random.seed(42)
    normal_samples = np.random.normal(0, 1, 1000)
    
    # Empirical probabilities for ranges
    prob_positive = np.mean(normal_samples > 0)
    prob_within_1_std = np.mean(np.abs(normal_samples) <= 1)
    prob_within_2_std = np.mean(np.abs(normal_samples) <= 2)
    
    print("Normal Distribution Properties:")
    print(f"P(X > 0) ≈ {prob_positive:.3f} (should be ≈ 0.5)")
    print(f"P(|X| ≤ 1) ≈ {prob_within_1_std:.3f} (should be ≈ 0.68)")
    print(f"P(|X| ≤ 2) ≈ {prob_within_2_std:.3f} (should be ≈ 0.95)")

demonstrate_continuous_distributions()
```

## 12. Common Mistakes and Pitfalls

### 12.1 Confusing P(A|B) with P(B|A)
```python
def prosecutor_fallacy_example():
    """The prosecutor's fallacy - confusing conditional probabilities"""
    
    # DNA evidence scenario
    p_match_given_innocent = 1e-6  # P(DNA match | innocent)
    p_innocent = 0.999             # Prior probability of innocence
    
    # WRONG reasoning: "DNA matches, so P(innocent) = 1e-6"
    # This confuses P(evidence | innocent) with P(innocent | evidence)
    
    # Correct Bayes' theorem calculation
    p_match_given_guilty = 1.0     # Assume perfect match if guilty
    p_guilty = 1 - p_innocent
    
    # P(DNA match) using law of total probability
    p_match = p_match_given_innocent * p_innocent + p_match_given_guilty * p_guilty
    
    # Correct posterior probability
    p_innocent_given_match = (p_match_given_innocent * p_innocent) / p_match
    
    print("Prosecutor's Fallacy Example:")
    print(f"P(DNA match | innocent) = {p_match_given_innocent:.2e}")
    print(f"WRONG: P(innocent | DNA match) = {p_match_given_innocent:.2e}")
    print(f"CORRECT: P(innocent | DNA match) = {p_innocent_given_match:.6f}")

prosecutor_fallacy_example()
```

### 12.2 Base Rate Neglect
```python
def base_rate_neglect_example():
    """Ignoring prior probabilities (base rates)"""
    
    # Medical test scenario
    disease_prevalence = 0.001  # 0.1% of population has disease
    test_sensitivity = 0.99     # 99% true positive rate
    test_specificity = 0.95     # 95% true negative rate
    
    # Many people think: "99% accurate test + positive result = 99% chance of disease"
    # This ignores the base rate!
    
    # Correct calculation using Bayes' theorem
    p_positive_given_disease = test_sensitivity
    p_positive_given_healthy = 1 - test_specificity
    
    p_positive = (p_positive_given_disease * disease_prevalence + 
                 p_positive_given_healthy * (1 - disease_prevalence))
    
    p_disease_given_positive = (p_positive_given_disease * disease_prevalence) / p_positive
    
    print("Base Rate Neglect Example:")
    print(f"Disease prevalence: {disease_prevalence:.1%}")
    print(f"Test sensitivity: {test_sensitivity:.1%}")
    print(f"Test specificity: {test_specificity:.1%}")
    print(f"INTUITION: P(disease | positive) ≈ 99%")
    print(f"REALITY: P(disease | positive) = {p_disease_given_positive:.1%}")

base_rate_neglect_example()
```

## 13. Practical Tips

### 13.1 Probability Estimation from Data
```python
def robust_probability_estimation(data, event_condition, smoothing=1):
    """Estimate probability with Laplace smoothing"""
    
    event_count = sum(1 for x in data if event_condition(x))
    total_count = len(data)
    
    # Laplace smoothing to avoid zero probabilities
    smoothed_prob = (event_count + smoothing) / (total_count + 2 * smoothing)
    
    return {
        'raw_probability': event_count / total_count if total_count > 0 else 0,
        'smoothed_probability': smoothed_prob,
        'event_count': event_count,
        'total_count': total_count
    }

# Example
data = ['A', 'B', 'A', 'C', 'A', 'B']
prob_A = robust_probability_estimation(data, lambda x: x == 'A')
print("Probability estimation for event A:")
for key, value in prob_A.items():
    print(f"{key}: {value}")
```

### 13.2 Debugging Probability Calculations
```python
def debug_probability_calculation():
    """Tips for debugging probability calculations"""
    
    # Always check that probabilities sum to 1
    probs = {'A': 0.3, 'B': 0.4, 'C': 0.2}
    total = sum(probs.values())
    print(f"Total probability: {total}")
    
    if abs(total - 1.0) > 1e-10:
        print("WARNING: Probabilities don't sum to 1!")
        # Normalize
        normalized = {k: v/total for k, v in probs.items()}
        print(f"Normalized: {normalized}")
    
    # Check for impossible events (negative probabilities)
    if any(p < 0 for p in probs.values()):
        print("ERROR: Negative probabilities detected!")
    
    # Sanity check conditional probabilities
    # P(A|B) should be between 0 and 1, and if B implies A, then P(A|B) = 1

debug_probability_calculation()
```

## 14. Conclusion

Probability provides the mathematical foundation for reasoning about uncertainty in machine learning:

### Key Concepts:
1. **Sample spaces and events** - defining what can happen
2. **Probability axioms** - fundamental rules that must be satisfied
3. **Conditional probability** - updating beliefs with new information
4. **Independence** - when events don't affect each other
5. **Bayes' theorem** - the engine of machine learning inference

### ML Applications:
- **Classification**: Predicting class probabilities
- **Naive Bayes**: Text classification and spam filtering  
- **A/B Testing**: Measuring experiment significance
- **Uncertainty quantification**: Expressing model confidence
- **Bayesian inference**: Updating model parameters

### Practical Guidelines:
- Always check that probabilities sum to 1
- Be careful not to confuse P(A|B) with P(B|A)
- Consider base rates when interpreting test results
- Use Laplace smoothing for small datasets
- Validate your probability calculations with simulations

**Next up:** **Bayes' Theorem and Applications** - diving deeper into the mathematical tool that powers modern machine learning inference!

Understanding basic probability concepts gives you the foundation to grasp more advanced topics like probability distributions, statistical inference, and the mathematical principles behind machine learning algorithms. Probability is the language of uncertainty, and in a world full of noisy data and incomplete information, it's an essential tool for any data scientist or machine learning practitioner. 