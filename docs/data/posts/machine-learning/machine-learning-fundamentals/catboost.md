# CatBoost

CatBoost is a gradient boosting library developed by Yandex that excels at handling categorical features without requiring extensive preprocessing. It provides state-of-the-art results and is particularly known for its ability to work with categorical data out of the box.

## Key Features

- **Categorical features support** — no manual encoding required
- **Fast and scalable** — GPU and multi-GPU support
- **Accurate** — state-of-the-art results on many datasets
- **Easy to use** — minimal parameter tuning required
- **Robust** — ordered boosting reduces overfitting automatically

## 1. Installation

```bash
pip install catboost
```

```python
import catboost
print(catboost.__version__)
```

## 2. What Makes CatBoost Unique

### Ordered Boosting
Traditional gradient boosting computes gradients on the same data used to build the tree, causing target leakage. CatBoost uses an ordered (permutation-based) approach that computes gradients on a separate subset, reducing overfitting.

### Ordered Target Statistics for Categoricals
Instead of one-hot encoding, CatBoost computes target statistics for each category using an ordered scheme — preventing target leakage while capturing category-target relationships.

### Symmetric Trees
CatBoost builds symmetric (oblivious) trees where the same split condition is applied at each level. This makes prediction fast and reduces overfitting.

## 3. Basic Usage

### 3.1 Classification

```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=50
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=20
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
```

### 3.2 Regression

```python
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import numpy as np

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

reg.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)

y_pred = reg.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

## 4. Handling Categorical Features

This is where CatBoost truly shines — pass categorical column indices and it handles everything.

```python
import pandas as pd
from catboost import CatBoostClassifier, Pool

# Create a dataset with categorical features
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 28, 33],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD'],
    'income': [50000, 60000, 90000, 70000, 55000, 95000],
    'target': [0, 1, 1, 0, 1, 1]
})

X = df.drop('target', axis=1)
y = df['target']

# Specify categorical feature indices or names
cat_features = ['city', 'education']

# Using Pool (CatBoost's data structure)
train_pool = Pool(X, y, cat_features=cat_features)

model = CatBoostClassifier(iterations=100, depth=4, verbose=0)
model.fit(train_pool)

# Predict on new data with same categorical columns
test_data = pd.DataFrame({
    'age': [27, 45],
    'city': ['NYC', 'Chicago'],
    'education': ['Master', 'Bachelor'],
    'income': [58000, 75000]
})
print(model.predict(test_data, cat_features=cat_features))
```

## 5. Key Hyperparameters

```python
model = CatBoostClassifier(
    # Boosting
    iterations=1000,        # Number of trees
    learning_rate=0.03,     # Step size (auto-set if not specified)
    depth=6,                # Tree depth (1-16, default 6)

    # Regularization
    l2_leaf_reg=3.0,        # L2 regularization coefficient
    min_data_in_leaf=1,     # Min samples in a leaf
    random_strength=1.0,    # Randomness for scoring splits

    # Sampling
    subsample=0.8,          # Row sampling (for Bernoulli bootstrap)
    rsm=0.8,                # Random subspace method (feature sampling)

    # Categorical
    one_hot_max_size=2,     # Use one-hot for categoricals with <= N values

    # System
    task_type='GPU',        # 'CPU' or 'GPU'
    thread_count=-1,        # Use all CPU cores
    random_seed=42,
    verbose=0
)
```

## 6. CatBoost Pool

`Pool` is CatBoost's optimized data structure, similar to XGBoost's `DMatrix`.

```python
from catboost import Pool

# Create pools
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features,
    feature_names=list(X_train.columns)
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=cat_features
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=20)
```

## 7. Cross-Validation

```python
from catboost import cv, Pool

pool = Pool(X, y, cat_features=cat_features)

params = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.05,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0
}

cv_results = cv(
    pool,
    params,
    fold_count=5,
    early_stopping_rounds=20,
    seed=42
)

print(f"Best AUC: {cv_results['test-AUC-mean'].max():.4f}")
print(f"Best iteration: {cv_results['test-AUC-mean'].idxmax()}")
```

## 8. Feature Importance

```python
# Get feature importances
importances = model.get_feature_importance()
feature_names = model.feature_names_

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

# Plot
model.plot_tree(tree_idx=0)  # Visualize a single tree

# SHAP values for explainability
shap_values = model.get_feature_importance(
    data=test_pool,
    type='ShapValues'
)
print("SHAP values shape:", shap_values.shape)
```

## 9. Saving and Loading

```python
# Save model
model.save_model('catboost_model.cbm')          # CatBoost binary format
model.save_model('catboost_model.json', format='json')

# Load model
loaded = CatBoostClassifier()
loaded.load_model('catboost_model.cbm')

# Export to other formats
model.save_model('model.onnx', format='onnx')   # ONNX for deployment
model.save_model('model.cpp', format='cpp')     # C++ code for embedding
```

## 10. CatBoost vs XGBoost vs LightGBM

| Aspect | CatBoost | XGBoost | LightGBM |
|---|---|---|---|
| Categorical features | Native (best) | Requires encoding | Native |
| Default performance | Often best out-of-box | Good | Good |
| Training speed | Slower | Medium | Fastest |
| Overfitting resistance | Ordered boosting | Standard | Standard |
| GPU support | Excellent | Good | Good |
| Tuning required | Minimal | Moderate | Moderate |

## 11. Conclusion

CatBoost is the easiest gradient boosting library to get started with, especially when your data has categorical features. Key takeaways:

- **No preprocessing needed** for categorical features — just pass column names
- **Ordered boosting** reduces overfitting without extensive tuning
- **`Pool`** is the most efficient way to pass data, especially with categoricals
- **Early stopping** is still important — always use a validation set
- **SHAP values** are built-in for model explainability

When your dataset has many categorical columns, CatBoost is often the best first choice.
