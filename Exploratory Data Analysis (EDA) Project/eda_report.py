# ============================================================
# REPORT GENERATION MODULE
# ============================================================

import json
import pandas as pd
import numpy as np
from datetime import datetime
from eda_config import CURRENT_DIR, ALPHA
from scipy import stats

def generate_report(df, results):
    """Generate comprehensive analysis report"""
    print("\n" + "="*70)
    print("GENERATING ANALYSIS REPORT")
    print("="*70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['category']).columns
    
    report = []
    report.append("="*70)
    report.append("EXPLORATORY DATA ANALYSIS REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"File Location: {CURRENT_DIR}")
    
    # Dataset Overview
    report.append("\n1. DATASET OVERVIEW")
    report.append("-"*50)
    report.append(f"Total Records: {len(df)}")
    report.append(f"Total Features: {len(df.columns)}")
    report.append(f"Numerical Features: {len(numerical_cols)}")
    report.append(f"Categorical Features: {len(categorical_cols)}")
    
    # Missing Values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        report.append(f"\nMissing Values: {missing.sum()} total")
        for col, count in missing.items():
            if count > 0:
                report.append(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        report.append("\n✅ No Missing Values Found")
    
    # Key Statistics
    report.append("\n2. KEY STATISTICS")
    report.append("-"*50)
    
    if 'survived' in df.columns:
        survival_rate = df['survived'].mean() * 100
        report.append(f"Overall Survival Rate: {survival_rate:.1f}%")
    
    if 'age' in df.columns:
        report.append(f"Average Age: {df['age'].mean():.1f} years")
        report.append(f"Age Range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
        report.append(f"Median Age: {df['age'].median():.1f} years")
    
    if 'fare' in df.columns:
        report.append(f"Average Fare: ${df['fare'].mean():.2f}")
        report.append(f"Fare Range: ${df['fare'].min():.2f} - ${df['fare'].max():.2f}")
        report.append(f"Median Fare: ${df['fare'].median():.2f}")
    
    # Survival Insights
    if 'survived' in df.columns:
        report.append("\n3. SURVIVAL INSIGHTS")
        report.append("-"*50)
        
        if 'sex' in df.columns:
            male_survival = df[df['sex'] == 'male']['survived'].mean() * 100
            female_survival = df[df['sex'] == 'female']['survived'].mean() * 100
            report.append(f"\nGender Impact:")
            report.append(f"  • Female Survival Rate: {female_survival:.1f}%")
            report.append(f"  • Male Survival Rate: {male_survival:.1f}%")
            report.append(f"  • Difference: {female_survival - male_survival:.1f} percentage points")
            report.append(f"  • Female-to-Male Ratio: {female_survival/male_survival:.2f}x")
        
        if 'pclass' in df.columns:
            report.append(f"\nClass Impact:")
            for pclass in sorted(df['pclass'].unique()):
                class_survival = df[df['pclass'] == pclass]['survived'].mean() * 100
                class_count = len(df[df['pclass'] == pclass])
                report.append(f"  • Class {pclass}: {class_survival:.1f}% survival rate ({class_count} passengers)")
        
        if 'age' in df.columns:
            children = df[df['age'] < 18]['survived'].mean() * 100
            adults = df[df['age'] >= 18]['survived'].mean() * 100
            report.append(f"\nAge Impact:")
            report.append(f"  • Children (<18): {children:.1f}% survival rate")
            report.append(f"  • Adults (≥18): {adults:.1f}% survival rate")
            report.append(f"  • Difference: {children - adults:.1f} percentage points")
            report.append(f"  • Children-to-Adult Ratio: {children/adults:.2f}x")
    
    # Statistical Significance
    report.append("\n4. STATISTICAL SIGNIFICANCE")
    report.append("-"*50)
    
    if 'survived' in df.columns:
        categorical_cols = df.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            if col != 'survived' and col != 'age_group':
                contingency = pd.crosstab(df[col], df['survived'])
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                significance = "✅ SIGNIFICANT" if p < ALPHA else "❌ Not Significant"
                report.append(f"\n{col.capitalize()} vs Survival:")
                report.append(f"  χ² = {chi2:.3f}, p = {p:.4f}")
                report.append(f"  Degrees of Freedom: {dof}")
                report.append(f"  Result: {significance}")
    
    # Key Findings
    report.append("\n5. KEY FINDINGS")
    report.append("-"*50)
    
    if 'survived' in df.columns:
        # Find most important factors
        if 'sex' in df.columns:
            female_survival = df[df['sex'] == 'female']['survived'].mean() * 100
            male_survival = df[df['sex'] == 'male']['survived'].mean() * 100
            report.append(f"\n• Gender is a strong predictor: Females had {female_survival - male_survival:.1f}% higher survival rate")
        
        if 'pclass' in df.columns:
            class1_survival = df[df['pclass'] == 1]['survived'].mean() * 100
            class3_survival = df[df['pclass'] == 3]['survived'].mean() * 100
            report.append(f"• Class is a strong predictor: Class 1 had {class1_survival - class3_survival:.1f}% higher survival rate than Class 3")
        
        if 'age' in df.columns:
            child_survival = df[df['age'] < 18]['survived'].mean() * 100
            adult_survival = df[df['age'] >= 18]['survived'].mean() * 100
            report.append(f"• Age is a factor: Children had {child_survival - adult_survival:.1f}% higher survival rate than adults")
    
    # Recommendations
    report.append("\n6. RECOMMENDATIONS")
    report.append("-"*50)
    recommendations = [
        "• Build predictive models"]