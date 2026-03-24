# LightGBM

LightGBM is a gradient boosting framework developed by Microsoft that uses tree-based learning algorithms. It's designed to be distributed and efficient with faster training speed, higher efficiency, lower memory usage, and better accuracy than many other boosting frameworks.

## Key Features

- **Faster training speed** and higher efficiency via GOSS and EFB
- **Lower memory usage** — histogram-based algorithm
- **Better accuracy** on many benchmark datasets
- **Support for parallel, distributed, and GPU learning**
- **Capable of handling large-scale data** with millions of rows

## 1. Installation

```bash
pip install lightgbm
```

```python
import lightgbm as lgb
print(lgb.__version__)
```

## 2. What Makes LightGBM Fast

### Gradient-based One-Side Sampling (GOSS)
Instead of using all data instances to estimate information gain, GOSS keeps instances with large gradients (high error) and randomly samples instances with small gradients. This reduces data size while maintaining accuracy.

### Exclusive Feature Bundling (EFB)
Mutually exclusive features (rarely non-zero simultaneously) are bundled together, reducing the number of features without losing information.

### Leaf-wise Tree Growth
Unlike XGBoost's level-wise growth, LightGBM grows the leaf with the maximum delta loss — leading to lower loss with fewer splits, but requiring `max_depth` or `num_leaves` to prevent overfitting.

## 3. Basic Usage

### 3.1 Classification

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(
    n_estimators=500,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
```

### 3.2 Regression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import numpy as np

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = lgb.LGBMRegressor(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

reg.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(100)]
)

y_pred = reg.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

## 4. Key Hyperparameters

```python
model = lgb.LGBMClassifier(
    # Tree structure
    num_leaves=31,          # Max leaves per tree (main complexity control)
    max_depth=-1,           # -1 = no limit (use num_leaves instead)
    min_child_samples=20,   # Min samples in a leaf

    # Learning
    n_estimators=1000,
    learning_rate=0.05,
    min_split_gain=0.0,     # Min gain to split a node

    # Sampling
    subsample=0.8,          # Row sampling per tree
    subsample_freq=1,       # Frequency of row sampling
    colsample_bytree=0.8,   # Feature sampling per tree

    # Regularization
    reg_alpha=0.0,          # L1 regularization
    reg_lambda=0.0,         # L2 regularization

    # Speed
    n_jobs=-1,
    device='gpu',           # Use GPU (if available)
    verbose=-1
)
```

## 5. Native API with Dataset

```python
# Create LightGBM Dataset
dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
dvalid = lgb.Dataset(X_test,  label=y_test,  reference=dtrain)

params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'num_leaves': 31,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbose': -1,
    'seed': 42
}

callbacks = [
    lgb.early_stopping(stopping_rounds=20),
    lgb.log_evaluation(period=50)
]

bst = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dvalid],
    valid_names=['train', 'valid'],
    callbacks=callbacks
)

print(f"Best iteration: {bst.best_iteration}")
y_prob = bst.predict(X_test, num_iteration=bst.best_iteration)
```

## 6. Categorical Features

LightGBM handles categorical features natively — no need for one-hot encoding.

```python
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'city': ['NYC', 'LA', 'NYC', 'Chicago'],
    'income': [50000, 60000, 70000, 80000],
    'target': [0, 1, 1, 0]
})

# Specify categorical columns
categorical_features = ['city']
dtrain = lgb.Dataset(
    df[['age', 'city', 'income']],
    label=df['target'],
    categorical_feature=categorical_features
)

# Or with sklearn API
model = lgb.LGBMClassifier(verbose=-1)
model.fit(
    df[['age', 'city', 'income']],
    df['target'],
    categorical_feature=categorical_features
)
```

## 7. Cross-Validation

```python
dtrain = lgb.Dataset(X, label=y)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

cv_results = lgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    callbacks=[lgb.early_stopping(20)],
    seed=42
)

best_rounds = len(cv_results['valid auc-mean'])
best_auc = max(cv_results['valid auc-mean'])
print(f"Best AUC: {best_auc:.4f} at round {best_rounds}")
```

## 8. Feature Importance

```python
import matplotlib.pyplot as plt

# Plot feature importance
lgb.plot_importance(model, importance_type='gain', max_num_features=15)
plt.tight_layout()
plt.show()

# As array
importances = model.feature_importances_
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
print(importance_df.head(10))
```

## 9. Saving and Loading

```python
# Save model
bst.save_model('lgbm_model.txt')

# Load model
bst_loaded = lgb.Booster(model_file='lgbm_model.txt')
y_pred = bst_loaded.predict(X_test)

# Sklearn API
model.booster_.save_model('lgbm_sklearn.txt')
```

## 10. LightGBM vs XGBoost

| Aspect | LightGBM | XGBoost |
|---|---|---|
| Tree growth | Leaf-wise | Level-wise |
| Speed | Faster (GOSS + EFB) | Slightly slower |
| Memory | Lower | Higher |
| Categorical features | Native support | Requires encoding |
| Small datasets | Can overfit | More stable |
| Large datasets | Excellent | Good |

## 11. Conclusion

LightGBM is often the fastest gradient boosting option for large datasets. Key takeaways:

- **`num_leaves`** is the primary complexity control — keep it reasonable (31–255)
- **Leaf-wise growth** is powerful but can overfit on small datasets — use `min_child_samples`
- **Native categorical support** saves preprocessing time
- **Early stopping** is essential — always use a validation set
- **GOSS and EFB** are the secret sauce behind LightGBM's speed advantage

For large-scale tabular data, LightGBM is often the first algorithm to try.
