"""
Visualization functions (NO GRADIENT COLORS)
FIXED: Properly handles y_test and all data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_roc(results, y_test, colors):
    """
    Plot confusion matrices and ROC curves
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, res in enumerate(results):
        # Confusion Matrix - Solid colors
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Disease', 'Liver Disease'],
                    yticklabels=['No Disease', 'Liver Disease'],
                    ax=axes[idx], cbar=False)
        axes[idx].set_title(f"{res['Model']}\nConfusion Matrix", fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=12)
        axes[idx].set_ylabel('Actual', fontsize=12)
        
        # ROC Curve - Solid colors
        idx_roc = idx + 3
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        axes[idx_roc].plot(fpr, tpr, linewidth=3, 
                          color=colors[idx % len(colors)],
                          label=f'AUC = {res["ROC-AUC"]:.3f}')
        axes[idx_roc].plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7)
        axes[idx_roc].set_xlim([0.0, 1.0])
        axes[idx_roc].set_ylim([0.0, 1.05])
        axes[idx_roc].set_xlabel('False Positive Rate', fontsize=12)
        axes[idx_roc].set_ylabel('True Positive Rate', fontsize=12)
        axes[idx_roc].set_title(f"{res['Model']}\nROC Curve", fontsize=14, fontweight='bold')
        axes[idx_roc].legend(loc="lower right", framealpha=0.9)
        axes[idx_roc].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('indian_liver_confusion_roc.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: indian_liver_confusion_roc.png")
    plt.close()


def plot_feature_importance(model, X_columns, top_n=10):
    """
    Plot feature importance (solid colors)
    """
    if not hasattr(model, 'feature_importances_'):
        print("   ⚠️  Model doesn't have feature_importances_ attribute")
        return
    
    plt.figure(figsize=(12, 7))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Solid color bars
    bars = plt.barh(range(top_n), importances[indices][::-1], 
                   color='#1f77b4', edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.3f}', va='center', fontsize=10)
    
    plt.yticks(range(top_n), [X_columns[i] for i in indices[::-1]], fontsize=11)
    plt.xlabel('Feature Importance Score', fontsize=13)
    plt.title('Random Forest - Top Feature Importances\nIndian Liver Patient Data', 
              fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('indian_liver_feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: indian_liver_feature_importance.png")
    plt.close()


def plot_model_comparison(summary_df, colors):
    """
    Plot model comparison bar chart (solid colors)
    """
    plt.figure(figsize=(12, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (_, row) in enumerate(summary_df.iterrows()):
        values = [row[m] for m in metrics]
        bars = plt.bar(x + i*width, values, width, 
                      label=row['Model'], color=colors[i % len(colors)],
                      edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Evaluation Metrics', fontsize=13)
    plt.ylabel('Score', fontsize=13)
    plt.title('Model Performance Comparison - Indian Data', 
              fontsize=15, fontweight='bold')
    plt.xticks(x + width, metrics, fontsize=11)
    plt.legend(loc='lower right', fontsize=11)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('indian_liver_model_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: indian_liver_model_comparison.png")
    plt.close()


def plot_feature_distribution(df, features_to_plot):
    """
    Plot feature distributions (solid colors)
    """
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_plot):
        if idx < len(axes) and feature in df.columns:
            # Histogram with solid colors
            axes[idx].hist(df[df['Dataset']==0][feature], 
                          color='#1f77b4', alpha=0.7,
                          edgecolor='black', linewidth=0.5,
                          label='No Disease', bins=20, density=True)
            axes[idx].hist(df[df['Dataset']==1][feature], 
                          color='#ff7f0e', alpha=0.7,
                          edgecolor='black', linewidth=0.5,
                          label='Liver Disease', bins=20, density=True)
            axes[idx].set_title(f'Distribution of {feature}', fontsize=13, fontweight='bold')
            axes[idx].set_xlabel(feature, fontsize=11)
            axes[idx].set_ylabel('Density', fontsize=11)
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('indian_liver_feature_distribution.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: indian_liver_feature_distribution.png")
    plt.close()


def plot_performance_radar(summary_df, colors):
    """
    Plot radar chart for model comparison
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, (_, row) in enumerate(summary_df.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2, 
               color=colors[i % len(colors)], label=row['Model'])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.set_title('Model Performance Radar Chart\nIndian Liver Data', 
                fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('indian_liver_performance_radar.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: indian_liver_performance_radar.png")
    plt.close()


def generate_all_visualizations(results, y_test, summary_df, df, X_columns, colors):
    """
    Generate all visualizations
    """
    print("\n[7] Generating visualizations (solid colors)...")
    print("-"*60)
    
    # 1. Confusion Matrices & ROC Curves
    plot_confusion_roc(results, y_test, colors)
    
    # 2. Feature Importance (from Random Forest)
    rf_model = None
    for res in results:
        if res['Model'] == 'Random Forest':
            rf_model = res['model']
            break
    if rf_model:
        plot_feature_importance(rf_model, X_columns)
    else:
        print("   ⚠️  Random Forest model not found for feature importance")
    
    # 3. Model Comparison Bar Chart
    plot_model_comparison(summary_df, colors)
    
    # 4. Feature Distributions
    features_to_plot = ['Age', 'Total_Bilirubin', 'Alkaline_Phosphotase', 
                       'Alamine_Aminotransferase', 'Albumin', 'Total_Proteins']
    available_features = [f for f in features_to_plot if f in df.columns]
    plot_feature_distribution(df, available_features)
    
    # 5. Radar Chart
    plot_performance_radar(summary_df, colors)
    
    print("\n   ✅ All visualizations generated successfully!")