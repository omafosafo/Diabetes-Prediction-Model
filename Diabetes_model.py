import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert to DataFrame for convenience
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y

# Optional: One-hot encode 'sex' if needed (already numerical in this dataset)
# But here for demonstration
# df = pd.get_dummies(df, columns=['sex'], drop_first=True)

# Separate features and target again
X = df.drop('target', axis=1).values
y = df['target'].values

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values (unnecessary for this dataset, but left for robustness)
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)  # <- Note: transform, not fit_transform

# Standardize the data
X_train = scale(X_train)
X_test = scale(X_test)

# Define hyperparameter grid
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5]  # Degree is only used with 'poly' kernel
}

# Grid Search
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_

# Final model
final_model = SVR(kernel=best_params['kernel'], degree=best_params.get('degree', 3))
final_model.fit(X_train, y_train)

# Predict & evaluate
y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Mean Squared Error on Test Data: {mse:.2f}")
print(f"Mean Absolute Error on Test Data: {mae:.2f}")
