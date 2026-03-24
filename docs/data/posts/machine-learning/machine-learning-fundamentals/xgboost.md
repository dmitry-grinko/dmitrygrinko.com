# XGBoost

XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting framework that has become the go-to choice for many machine learning competitions and real-world applications. Known for its performance and speed, XGBoost implements machine learning algorithms under the Gradient Boosting framework.

## Key Features

- **High Performance** - Optimized for speed and model performance
- **Flexibility** - Supports regression, classification, ranking, and custom objectives
- **Portability** - Runs on Windows, Linux, and macOS
- **Regularization** - Built-in L1 and L2 regularization to prevent overfitting
- **Parallel processing** - Column block structure enables parallelized tree building

## 1. Installation

```bash
pip install xgboost
```

```python
import xgboost as xgb
print(xgb.__version__)
```

## 2. How Gradient Boosting Works

XGBoost builds an ensemble of decision trees sequentially. Each new tree corrects the errors of the previous ones by fitting the residuals (negative gradients of the loss function).

```
Final prediction = tree_1(x) + tree_2(x) + ... + tree_n(x)
```

Each tree is a "weak learner" — slightly better than random. Together they form a strong model.

## 3. Basic Usage

### 3.1 Classification

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
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

reg = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
```

## 4. Key Hyperparameters

```python
model = xgb.XGBClassifier(
    # Tree structure
    n_estimators=500,       # Number of trees
    max_depth=6,            # Max tree depth (3-10 typical)
    min_child_weight=1,     # Min sum of instance weight in a leaf

    # Learning
    learning_rate=0.05,     # Step size shrinkage (eta)
    gamma=0,                # Min loss reduction to split a node

    # Sampling (reduces overfitting)
    subsample=0.8,          # Fraction of samples per tree
    colsample_bytree=0.8,   # Fraction of features per tree
    colsample_bylevel=1.0,  # Fraction of features per level

    # Regularization
    reg_alpha=0,            # L1 regularization
    reg_lambda=1,           # L2 regularization

    # System
    n_jobs=-1,              # Use all CPU cores
    tree_method='hist',     # Fast histogram-based algorithm
    device='cuda',          # Use GPU (if available)
)
```

## 5. Early Stopping

Early stopping prevents overfitting by stopping training when validation performance stops improving.

```python
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=20,  # Stop if no improvement for 20 rounds
    eval_metric='auc'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score:.4f}")
```

## 6. Native DMatrix API

The native XGBoost API with `DMatrix` is faster and more memory-efficient.

```python
# Create DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_names)

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

evals_result = {}
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=50
)

# Predict
y_prob = bst.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)
```

## 7. Feature Importance

```python
import matplotlib.pyplot as plt

# Built-in importance types
# 'weight'  — number of times a feature is used in splits
# 'gain'    — average gain of splits using the feature (most informative)
# 'cover'   — average coverage of splits using the feature

importances = model.get_booster().get_score(importance_type='gain')
xgb.plot_importance(model, importance_type='gain', max_num_features=15)
plt.tight_layout()
plt.show()

# As a dict
print(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5])
```

## 8. Cross-Validation

```python
dtrain = xgb.DMatrix(X, label=y)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,
    'eta': 0.05
}

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    nfold=5,
    early_stopping_rounds=20,
    seed=42,
    verbose_eval=50
)

print(f"Best AUC: {cv_results['test-auc-mean'].max():.4f}")
print(f"Best round: {cv_results['test-auc-mean'].idxmax()}")
```

## 9. Handling Imbalanced Data

```python
# scale_pos_weight balances positive/negative class weights
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()

model = xgb.XGBClassifier(
    scale_pos_weight=neg_count / pos_count,  # e.g., 10 for 10:1 imbalance
    eval_metric='aucpr'  # Area under precision-recall curve (better for imbalance)
)
```

## 10. Saving and Loading

```python
# Save model
model.save_model('xgb_model.json')  # JSON format (recommended)
model.save_model('xgb_model.ubj')   # Binary format (smaller)

# Load model
loaded = xgb.XGBClassifier()
loaded.load_model('xgb_model.json')

# Native booster
bst.save_model('booster.json')
bst_loaded = xgb.Booster()
bst_loaded.load_model('booster.json')
```

## 11. Conclusion

XGBoost remains one of the most effective algorithms for structured/tabular data. Key takeaways:

- **Gradient boosting** builds trees sequentially, each correcting the previous one's errors
- **Early stopping** is essential — use it with a validation set to find the optimal number of trees
- **`learning_rate` and `n_estimators`** are inversely related — lower LR needs more trees
- **`subsample` and `colsample_bytree`** add randomness that reduces overfitting
- **Feature importance** (especially `gain`) helps identify which features drive predictions

For tabular data competitions and production ML, XGBoost (alongside LightGBM and CatBoost) is consistently among the top-performing algorithms.
