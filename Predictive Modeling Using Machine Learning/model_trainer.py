"""
Model training and evaluation functions
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


def get_model_instance(model_name):
    """Get model instance by name"""
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42)
    }
    return models.get(model_name)


def train_models(X_train, y_train, X_train_scaled, model_configs):
    """Train multiple models with hyperparameter tuning"""
    best_models = {}
    best_scores = {}
    
    for name, config in model_configs.items():
        print(f"\n🔄 Training {name} on Indian data...")
        
        model = get_model_instance(config['model'])
        grid_search = GridSearchCV(
            model,
            config['param_grid'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        # Use scaled data for Logistic Regression
        if name == 'Logistic Regression':
            grid_search.fit(X_train_scaled, y_train)
        else:
            grid_search.fit(X_train, y_train)
        
        best_models[name] = grid_search.best_estimator_
        best_scores[name] = grid_search.best_score_
        
        print(f"   ✅ Best parameters: {grid_search.best_params_}")
        print(f"   ✅ Cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return best_models, best_scores


def evaluate_models(models, X_test, y_test, X_test_scaled):
    """Evaluate trained models on test data"""
    results = []
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            X_test_use = X_test_scaled
        else:
            X_test_use = X_test
        
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model
        })
    
    return results


def print_results(results):
    """Print evaluation results"""
    print("\n[5] Evaluating models on Indian test data...")
    print("-"*60)
    
    for res in results:
        print(f"\n📊 {res['Model']} on Indian data:")
        print(f"   Accuracy:  {res['Accuracy']:.4f}")
        print(f"   Precision: {res['Precision']:.4f}")
        print(f"   Recall:    {res['Recall']:.4f}")
        print(f"   F1-Score:  {res['F1-Score']:.4f}")
        print(f"   ROC-AUC:   {res['ROC-AUC']:.4f}")
    
    # Summary
    print("\n[6] Performance Summary for Indian Dataset")
    print("="*60)
    summary_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 
                                       'Recall', 'F1-Score', 'ROC-AUC']]
    print(summary_df.round(4).to_string(index=False))
    
    best_model = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Model']
    best_acc = summary_df['Accuracy'].max()
    print(f"\n🏆 Best performing model: {best_model} (Accuracy: {best_acc:.4f})")
    
    return summary_df, best_model, best_acc