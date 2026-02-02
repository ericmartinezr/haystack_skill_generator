---
name: machine_learning
description: Comprehensive toolkit for building, training, and deploying machine learning models. Covers data preprocessing, model selection, training pipelines, evaluation metrics, and deployment strategies using popular ML libraries.
license: Proprietary. LICENSE.txt has complete terms
---

# Machine Learning Development Guide

## Overview

This guide covers essential machine learning operations from data preparation to model deployment. For deep learning, neural architectures, and advanced techniques, see reference.md.

## Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
```

## Data Preprocessing

### pandas - Data Manipulation

#### Load and Explore Data

```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

# Quick exploration
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
```

#### Handle Missing Values

```python
# Check missing values
print(df.isnull().sum())

# Fill missing values
df['column'].fillna(df['column'].mean(), inplace=True)  # Mean
df['column'].fillna(df['column'].median(), inplace=True)  # Median
df['column'].fillna(df['column'].mode()[0], inplace=True)  # Mode
df['column'].fillna('Unknown', inplace=True)  # Constant

# Drop rows with missing values
df.dropna(inplace=True)
```

#### Feature Engineering

```python
# Create new features
df['new_feature'] = df['feature1'] / df['feature2']
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month

# Binning numerical features
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], 
                         labels=['Child', 'Young', 'Middle', 'Senior'])

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)
```

### scikit-learn - Preprocessing Pipeline

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Numerical features preprocessing
numeric_features = ['age', 'income', 'score']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # or MinMaxScaler()
])

# Categorical features preprocessing
categorical_features = ['gender', 'city', 'category']
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit pipeline
pipeline.fit(X_train, y_train)
```

## Model Selection

### Classification Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Quick comparison
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results[name] = score
    print(f"{name}: {score:.3f}")
```

### Regression Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(),
    'SVR': SVR()
}
```

## Model Training

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

# Basic cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Stratified K-Fold (for classification)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```

### Hyperparameter Tuning

```python
# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Randomized Search (faster for large search spaces)
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
```

## Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)

# Basic metrics
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # For binary classification

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

# Detailed report
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ROC-AUC (binary classification)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {auc:.3f}")
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Feature Importance
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = X.columns if hasattr(X, 'columns') else range(len(importances))
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
```

## Model Persistence

```python
import joblib
import pickle

# Save model
joblib.dump(model, 'model.pkl')
# or
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
model = joblib.load('model.pkl')
# or
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Save complete pipeline
joblib.dump(pipeline, 'pipeline.pkl')
```

## Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[rfe.support_]
print(f"Selected features: {selected_features}")
```

## Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight

# Resampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
model = RandomForestClassifier(class_weight='balanced')
```

## Model Deployment

### Flask API

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': probability[0].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    return {
        "prediction": int(prediction[0]),
        "probability": probability[0].tolist()
    }
```

## Common Tasks

### Time Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

### Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('lr', LogisticRegression())
    ],
    voting='soft'
)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    final_estimator=LogisticRegression()
)
```

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)

# t-SNE (for visualization)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
```

## Quick Reference

| Task | Library/Method | Code |
|------|---------------|------|
| Load data | pandas | `pd.read_csv('data.csv')` |
| Handle missing values | pandas | `df.fillna(), df.dropna()` |
| Encode categoricals | sklearn | `OneHotEncoder(), LabelEncoder()` |
| Scale features | sklearn | `StandardScaler(), MinMaxScaler()` |
| Train/test split | sklearn | `train_test_split()` |
| Cross-validation | sklearn | `cross_val_score()` |
| Hyperparameter tuning | sklearn | `GridSearchCV(), RandomizedSearchCV()` |
| Classification | sklearn | `RandomForestClassifier()` |
| Regression | sklearn | `RandomForestRegressor()` |
| Evaluation | sklearn | `accuracy_score(), f1_score(), r2_score()` |
| Save model | joblib | `joblib.dump(model, 'file.pkl')` |
| Load model | joblib | `joblib.load('file.pkl')` |
| Handle imbalance | imblearn | `SMOTE(), class_weight` |
| Feature selection | sklearn | `SelectKBest(), RFE()` |
| Dimensionality reduction | sklearn | `PCA(), TSNE()` |

## Next Steps

- For deep learning with TensorFlow/PyTorch, see reference.md
- For natural language processing, see reference.md
- For computer vision, see reference.md
- For MLOps and model monitoring, see reference.md
- For advanced ensemble methods, see reference.md
