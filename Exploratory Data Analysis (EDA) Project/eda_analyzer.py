# ============================================================
# STATISTICAL ANALYSIS MODULE
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
from eda_config import ALPHA

def perform_statistical_analysis(df):
    """Perform comprehensive statistical analysis"""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    results = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['category']).columns
    
    # 1. Descriptive Statistics
    print("\n📊 DESCRIPTIVE STATISTICS")
    print("-"*50)
    print("\nNumerical Variables:")
    desc_stats = df[numerical_cols].describe().round(2)
    print(desc_stats)
    results['descriptive_stats'] = desc_stats.to_dict()
    
    # Additional statistics
    print("\nAdditional Statistics:")
    for col in numerical_cols:
        print(f"\n{col.capitalize()}:")
        print(f"  Variance: {df[col].var():.2f}")
        print(f"  Std Dev: {df[col].std():.2f}")
        print(f"  Skewness: {df[col].skew():.3f}")
        print(f"  Kurtosis: {df[col].kurtosis():.3f}")
    
    # 2. Correlation Analysis
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        print("\n🔗 CORRELATION ANALYSIS")
        print("-"*50)
        print("\nTop 5 Strongest Correlations:")
        corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1]  # Remove self-correlations
        top_corrs = {}
        for pair, value in corr_pairs.head(5).items():
            print(f"  {pair[0]} ↔ {pair[1]}: {value:.3f}")
            top_corrs[f"{pair[0]}_{pair[1]}"] = float(value)
        results['top_correlations'] = top_corrs
        
        # Correlation with target
        if 'survived' in df.columns:
            print("\n📈 Correlation with Survival:")
            target_corr = corr_matrix['survived'].drop('survived').sort_values(ascending=False)
            for var, corr_val in target_corr.items():
                print(f"  {var}: {corr_val:.3f}")
            results['target_correlations'] = target_corr.to_dict()
    
    # 3. Statistical Tests
    print("\n🔬 STATISTICAL SIGNIFICANCE TESTS")
    print("-"*50)
    results['statistical_tests'] = {}
    
    if 'survived' in df.columns:
        # Chi-square tests for categorical variables
        for col in categorical_cols:
            if col != 'survived' and col != 'age_group':
                contingency = pd.crosstab(df[col], df['survived'])
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                significance = "✅ SIGNIFICANT" if p < ALPHA else "❌ Not Significant"
                print(f"\nChi-square Test ({col} vs Survival):")
                print(f"  χ² = {chi2:.3f}, p = {p:.4f}")
                print(f"  Degrees of freedom: {dof}")
                print(f"  Result: {significance}")
                results['statistical_tests'][f'chi2_{col}'] = {
                    'statistic': float(chi2),
                    'p_value': float(p),
                    'degrees_freedom': int(dof),
                    'significant': bool(p < ALPHA)
                }
        
        # T-tests for numerical variables
        survived = df[df['survived'] == 1]
        not_survived = df[df['survived'] == 0]
        
        for col in numerical_cols:
            if col != 'survived':
                t_stat, p_val = stats.ttest_ind(
                    survived[col].dropna(), 
                    not_survived[col].dropna()
                )
                significance = "✅ SIGNIFICANT" if p_val < ALPHA else "❌ Not Significant"
                print(f"\nT-test ({col}):")
                print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
                print(f"  Result: {significance}")
                results['statistical_tests'][f'ttest_{col}'] = {
                    'statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': bool(p_val < ALPHA)
                }
    
    # 4. Outlier Detection
    print("\n⚠️ OUTLIER DETECTION (IQR Method)")
    print("-"*50)
    results['outliers'] = {}
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        
        if outlier_count > 0:
            print(f"\n{col.capitalize()}:")
            print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  Normal Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Outliers: {outlier_count} ({outlier_count/len(df)*100:.1f}%)")
            results['outliers'][col] = {
                'count': int(outlier_count),
                'percentage': float(outlier_count/len(df)*100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR)
            }
    
    # 5. Group Comparisons
    if 'survived' in df.columns:
        print("\n👥 GROUP COMPARISONS")
        print("-"*50)
        results['group_comparisons'] = {}
        
        # Survival by gender
        if 'sex' in df.columns:
            print("\nSurvival by Gender:")
            gender_survival = df.groupby('sex')['survived'].agg(['mean', 'count'])
            gender_survival['mean'] = gender_survival['mean'] * 100
            print(gender_survival.round(2))
            results['group_comparisons']['gender'] = gender_survival.to_dict()
        
        # Survival by class
        if 'pclass' in df.columns:
            print("\nSurvival by Passenger Class:")
            class_survival = df.groupby('pclass')['survived'].agg(['mean', 'count'])
            class_survival['mean'] = class_survival['mean'] * 100
            print(class_survival.round(2))
            results['group_comparisons']['pclass'] = class_survival.to_dict()
        
        # Survival by age group
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], 
                                   labels=['0-18', '19-30', '31-50', '50+'])
            print("\nSurvival by Age Group:")
            age_survival = df.groupby('age_group')['survived'].agg(['mean', 'count'])
            age_survival['mean'] = age_survival['mean'] * 100
            print(age_survival.round(2))
            results['group_comparisons']['age_group'] = age_survival.to_dict()
    
    return results