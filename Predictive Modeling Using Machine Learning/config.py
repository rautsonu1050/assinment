"""
Configuration settings for the project
"""

# Model configurations
MODELS = {
    'Logistic Regression': {
        'model': 'LogisticRegression',
        'param_grid': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'Decision Tree': {
        'model': 'DecisionTreeClassifier',
        'param_grid': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Random Forest': {
        'model': 'RandomForestClassifier',
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

# Visualization colors (solid colors - no gradients)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
SOLID_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Dataset URL
DATASET_URL = "https://raw.githubusercontent.com/dataprofessor/data/master/indian_liver_patient.csv"

# Random seed for reproducibility
RANDOM_SEED = 42

# Test size
TEST_SIZE = 0.2

# Cross-validation folds
CV_FOLDS = 5