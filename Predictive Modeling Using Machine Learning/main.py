"""
Main execution script - Predictive Modeling with Indian Liver Dataset
Due Date: 15 Jun 2026
"""

from config import *
from data_loader import load_indian_liver_data, preprocess_data, get_data_info
from model_trainer import train_models, evaluate_models, print_results
from visualizer import generate_all_visualizations
from sklearn.metrics import classification_report
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PREDICTIVE MODELING USING MACHINE LEARNING")
print("Indian Liver Patient Dataset")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1] Loading Indian Dataset...")
df = load_indian_liver_data(DATASET_URL)
get_data_info(df)

# ============================================
# 2. PREPROCESS DATA
# ============================================
print("\n[2] Preprocessing Indian Data...")
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_data(df)
print(f"✅ Data split complete:")
print(f"   - Training set: {X_train.shape[0]} samples")
print(f"   - Test set: {X_test.shape[0]} samples")

# ============================================
# 3. TRAIN MODELS
# ============================================
print("\n[3] Setting up models for training...")
print("\n[4] Training models with hyperparameter tuning...")
print("-"*60)
best_models, best_scores = train_models(X_train, y_train, X_train_scaled, MODELS)

# ============================================
# 4. EVALUATE MODELS
# ============================================
results = evaluate_models(best_models, X_test, y_test, X_test_scaled)
summary_df, best_model_name, best_accuracy = print_results(results)

# ============================================
# 5. VISUALIZE RESULTS
# ============================================
generate_all_visualizations(results, y_test, summary_df, df, 
                           X_train.columns, COLORS)

# ============================================
# 6. CLASSIFICATION REPORT
# ============================================
print("\n[8] Detailed Classification Report for Best Model")
print("="*60)

best_model = best_models[best_model_name]
if best_model_name == 'Logistic Regression':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

print(f"\n📋 Classification Report for {best_model_name} on Indian Data:")
print(classification_report(y_test, y_pred_best, 
                           target_names=['No Disease', 'Liver Disease']))

# ============================================
# 7. SAVE RESULTS
# ============================================
print("\n[9] Saving Indian data results...")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_best,
    'Probability': best_model.predict_proba(X_test_scaled if best_model_name == 'Logistic Regression' else X_test)[:, 1]
})
predictions_df.to_csv('indian_liver_predictions.csv', index=False)
print("   ✅ Saved: indian_liver_predictions.csv")

# Save summary
summary_df.to_csv('indian_liver_performance_summary.csv', index=False)
print("   ✅ Saved: indian_liver_performance_summary.csv")

# Save feature importance
if 'Random Forest' in best_models:
    rf_model = best_models['Random Forest']
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance_df.to_csv('indian_liver_feature_importance.csv', index=False)
    print("   ✅ Saved: indian_liver_feature_importance.csv")

# ============================================
# 8. FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("✅ TASK COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📌 EXPECTED OUTCOMES ACHIEVED:")
print("   ✅ Applied supervised learning to Indian healthcare data")
print("   ✅ Used Logistic Regression, Decision Tree, Random Forest")
print("   ✅ Trained and tested models with cross-validation")
print("   ✅ Visualized performance using confusion matrices and ROC curves")
print("   ✅ NO GRADIENT COLORS used in any visualization")
print("\n📁 FILES GENERATED:")
print("   1. indian_liver_confusion_roc.png")
print("   2. indian_liver_feature_importance.png")
print("   3. indian_liver_model_comparison.png")
print("   4. indian_liver_feature_distribution.png")
print("   5. indian_liver_performance_radar.png")
print("   6. indian_liver_predictions.csv")
print("   7. indian_liver_performance_summary.csv")
print("   8. indian_liver_feature_importance.csv")
print("\n📊 Key Insights:")
print(f"   • Best model: {best_model_name} with {best_accuracy:.2%} accuracy")
if 'Random Forest' in best_models:
    importances = best_models['Random Forest'].feature_importances_
    indices = importances.argsort()[::-1][:3]
    top_features = [X_train.columns[i] for i in indices]
    print(f"   • Most important features: {', '.join(top_features)}")
print(f"   • Dataset represents Indian patient population")
print("\n" + "="*60)
print("🎯 DUE DATE: 15 Jun 2026 - COMPLETED")
print("🇮🇳 Indian Healthcare Analytics")
print("="*60)