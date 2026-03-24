# MLxtend

MLxtend (Machine Learning Extensions) is a Python library of useful tools for day-to-day data science tasks. It provides a collection of machine learning extensions and utilities that complement existing libraries like scikit-learn — filling gaps in model evaluation, feature selection, visualization, and ensemble methods.

## Key Features

- **Model evaluation** utilities and techniques
- **Feature selection** algorithms
- **Plotting utilities** for machine learning
- **Ensemble methods** and stacking
- **Frequent pattern mining** algorithms

## 1. Installation

```bash
pip install mlxtend
```

```python
import mlxtend
print(mlxtend.__version__)
```

## 2. Stacking and Ensemble Methods

### 2.1 StackingClassifier

Stacking combines multiple base models with a meta-learner that learns how to best combine their predictions.

```python
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base classifiers
base_classifiers = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    SVC(kernel='rbf', probability=True, random_state=42)
]

# Meta-learner
meta_learner = LogisticRegression(max_iter=1000)

# Stack them
stack = StackingClassifier(
    classifiers=base_classifiers,
    meta_classifier=meta_learner,
    use_probas=True,        # Pass probabilities to meta-learner
    use_features_in_secondary=True  # Also pass original features
)

stack.fit(X_train, y_train)
print(f"Stacking accuracy: {stack.score(X_test, y_test):.4f}")

# Compare with individual models
for clf in base_classifiers:
    clf.fit(X_train, y_train)
    print(f"{clf.__class__.__name__}: {clf.score(X_test, y_test):.4f}")
```

### 2.2 StackingRegressor

```python
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

stack_reg = StackingRegressor(
    regressors=[
        RandomForestRegressor(n_estimators=100, random_state=42),
        Ridge(alpha=1.0)
    ],
    meta_regressor=LinearRegression()
)

stack_reg.fit(X_train, y_train)
print(f"Stacking R²: {stack_reg.score(X_test, y_test):.4f}")
```

## 3. Feature Selection

### 3.1 Sequential Feature Selector

Greedy forward or backward feature selection.

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# Forward selection: start with 0 features, add one at a time
sfs_forward = SFS(
    knn,
    k_features=5,          # Select 5 best features
    forward=True,
    floating=False,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

sfs_forward.fit(X_train, y_train)
print("Selected features:", sfs_forward.k_feature_idx_)
print("Best CV score:", sfs_forward.k_score_)

# Transform data to selected features
X_train_selected = sfs_forward.transform(X_train)
X_test_selected  = sfs_forward.transform(X_test)
```

### 3.2 Exhaustive Feature Selector

Tries all possible feature combinations (only feasible for small feature sets).

```python
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

efs = EFS(
    KNeighborsClassifier(n_neighbors=3),
    min_features=2,
    max_features=4,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

efs.fit(X_train[:, :8], y_train)  # Limit to 8 features for feasibility
print("Best feature subset:", efs.best_idx_)
print("Best score:", efs.best_score_)
```

## 4. Plotting Utilities

### 4.1 Decision Region Plot

Visualize decision boundaries for 2D data.

```python
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Use 2 features for visualization
X_2d = X[:, :2]
clf = SVC(kernel='rbf', random_state=42)
clf.fit(X_2d, y)

plot_decision_regions(X_2d, y, clf=clf, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Regions')
plt.tight_layout()
plt.show()
```

### 4.2 Confusion Matrix

```python
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = stack.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plot_confusion_matrix(
    conf_mat=cm,
    class_names=['Malignant', 'Benign'],
    colorbar=True,
    show_normed=True
)
plt.tight_layout()
plt.show()
```

### 4.3 Learning Curves

```python
from mlxtend.plotting import plot_learning_curves
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)

plot_learning_curves(
    X_train, y_train,
    X_test, y_test,
    clf,
    print_model=True,
    style='ggplot'
)
plt.show()
```

## 5. Model Evaluation

### 5.1 Bias-Variance Decomposition

```python
from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
    tree,
    X_train, y_train,
    X_test, y_test,
    loss='0-1_loss',
    random_seed=42,
    num_rounds=200
)

print(f"Average expected loss: {avg_expected_loss:.4f}")
print(f"Average bias:          {avg_bias:.4f}")
print(f"Average variance:      {avg_var:.4f}")
```

### 5.2 McNemar's Test

Statistical test to compare two classifiers.

```python
from mlxtend.evaluate import mcnemar, mcnemar_table

# Get predictions from two models
y_pred1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train).predict(X_test)
y_pred2 = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train).predict(X_test)

# Build contingency table
tb = mcnemar_table(y_target=y_test, y_model1=y_pred1, y_model2=y_pred2)
print("McNemar table:\n", tb)

# Run test
chi2, p_value = mcnemar(ary=tb, corrected=True)
print(f"chi-squared: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < 0.05)
```

### 5.3 Paired t-test for Cross-Validation

```python
from mlxtend.evaluate import paired_ttest_5x2cv

clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)

t, p = paired_ttest_5x2cv(
    estimator1=clf1,
    estimator2=clf2,
    X=X, y=y,
    scoring='accuracy',
    random_seed=42
)

print(f"t statistic: {t:.4f}")
print(f"p-value: {p:.4f}")
```

## 6. Frequent Pattern Mining

### 6.1 Apriori Algorithm

Find frequent itemsets in transaction data.

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Example: market basket data (one-hot encoded)
dataset = pd.DataFrame({
    'milk':   [1, 1, 0, 1, 0],
    'bread':  [1, 0, 1, 1, 1],
    'butter': [0, 1, 1, 1, 0],
    'eggs':   [1, 1, 0, 0, 1],
    'cheese': [0, 1, 1, 0, 0]
}, dtype=bool)

# Find frequent itemsets
frequent_itemsets = apriori(dataset, min_support=0.4, use_colnames=True)
print("Frequent itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
rules = rules.sort_values('lift', ascending=False)
print("\nAssociation rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
```

## 7. Utilities

### 7.1 Column Selector (for Pipelines)

```python
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Select specific columns in a pipeline
pipeline = Pipeline([
    ('select', ColumnSelector(cols=(0, 1, 2))),  # Select first 3 columns
    ('scale', StandardScaler()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
print("Pipeline accuracy:", pipeline.score(X_test, y_test))
```

### 7.2 DenseTransformer

Converts sparse matrices to dense (useful in pipelines).

```python
from mlxtend.preprocessing import DenseTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('to_dense', DenseTransformer()),  # Convert sparse to dense
    ('clf', RandomForestClassifier())
])
```

## 8. Conclusion

MLxtend fills important gaps in the scikit-learn ecosystem. Key takeaways:

- **`StackingClassifier`** is the easiest way to implement model stacking in Python
- **`SequentialFeatureSelector`** provides greedy feature selection with cross-validation
- **`plot_decision_regions`** is great for visualizing classifier boundaries
- **`bias_variance_decomp`** helps diagnose underfitting vs overfitting
- **`apriori`** and `association_rules` make market basket analysis straightforward

MLxtend is a practical toolkit — reach for it when scikit-learn doesn't have exactly what you need.
