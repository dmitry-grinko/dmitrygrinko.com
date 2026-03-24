# Scikit-learn

Scikit-learn is the most popular machine learning library in Python, providing simple and efficient tools for data mining and data analysis. Built on NumPy, SciPy, and matplotlib, it offers a consistent interface for a wide range of machine learning algorithms.

## Key Features

- **Simple and efficient tools** for predictive data analysis
- **Accessible to everybody** and reusable in various contexts
- **Built on NumPy, SciPy, and matplotlib**
- **Open source** and commercially usable under BSD license

## 1. Installation

```bash
pip install scikit-learn
```

```python
import sklearn
print(sklearn.__version__)
```

## 2. The Estimator API

Scikit-learn's unified API means every model follows the same pattern: `fit`, `predict`, `score`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## 3. Preprocessing

### 3.1 Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardize (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training stats on test set

# Scale to [0, 1]
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X_train)

# Robust to outliers (uses median and IQR)
robust = RobustScaler()
X_robust = robust.fit_transform(X_train)
```

### 3.2 Encoding Categorical Features

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import numpy as np

# Label encoding (for ordinal categories)
le = LabelEncoder()
y_encoded = le.fit_transform(['cat', 'dog', 'cat', 'bird'])
print(y_encoded)  # [1, 2, 1, 0]

# One-hot encoding
ohe = OneHotEncoder(sparse_output=False)
X_cat = np.array([['red'], ['blue'], ['green'], ['red']])
X_ohe = ohe.fit_transform(X_cat)
print(X_ohe)
```

### 3.3 Handling Missing Values

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Fill with mean/median/most_frequent
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_with_nans)

# KNN-based imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_knn = knn_imputer.fit_transform(X_with_nans)
```

## 4. Supervised Learning

### 4.1 Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100),
    'SVM':                 SVC(kernel='rbf', probability=True),
    'KNN':                 KNeighborsClassifier(n_neighbors=5)
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    print(f"{name}: {clf.score(X_test, y_test):.4f}")
```

### 4.2 Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("R² score:", lr.score(X_test, y_test))

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1 regularization — performs feature selection)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("Non-zero features:", (lasso.coef_ != 0).sum())
```

## 5. Pipelines

Pipelines chain preprocessing and modeling steps, preventing data leakage.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Define preprocessing for numeric and categorical columns
numeric_features = [0, 1, 2]
categorical_features = [3, 4]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
print("Pipeline accuracy:", pipeline.score(X_test, y_test))
```

## 6. Model Evaluation

### 6.1 Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 6.2 Classification Metrics

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# For binary classification
# print("ROC-AUC:", roc_auc_score(y_test, y_prob[:, 1]))
```

### 6.3 Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = lr.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

## 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Random search (faster for large spaces)
from scipy.stats import randint
param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(3, 20)}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,
    cv=5,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

## 8. Unsupervised Learning

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
print("Cluster centers shape:", kmeans.cluster_centers_.shape)

# DBSCAN (density-based, no need to specify k)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
print("Clusters found:", len(set(labels)) - (1 if -1 in labels else 0))

# PCA dimensionality reduction
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
print("Explained variance:", pca.explained_variance_ratio_.sum())
```

## 9. Feature Importance and Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.inspection import permutation_importance

# Select top k features by statistical test
selector = SelectKBest(f_classif, k=3)
X_selected = selector.fit_transform(X_train, y_train)

# Recursive feature elimination
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50), n_features_to_select=3)
rfe.fit(X_train, y_train)
print("Selected features:", rfe.support_)

# Feature importances from tree models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
print("Feature importances:", importances)
```

## 10. Conclusion

Scikit-learn's consistent API makes it easy to experiment with many algorithms quickly. Key takeaways:

- **Pipelines** prevent data leakage and make code cleaner
- **Cross-validation** gives more reliable performance estimates than a single train/test split
- **`GridSearchCV` / `RandomizedSearchCV`** automate hyperparameter tuning
- **Preprocessing** (scaling, encoding, imputation) is critical — always fit on training data only
- **Feature importance** helps you understand and simplify your models

Scikit-learn is the go-to library for classical ML. For deep learning, pair it with PyTorch or TensorFlow.
