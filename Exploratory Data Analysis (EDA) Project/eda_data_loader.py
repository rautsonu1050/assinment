# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
from eda_config import CURRENT_DIR

def load_and_clean_data():
    """Load and clean the dataset"""
    print("\n" + "="*70)
    print("DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    # Load dataset (using Titanic as example)
    df = sns.load_dataset('titanic')
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display first few rows
    print("\nFirst 5 rows of data:")
    print(df.head())
    
    # Handle missing values
    if 'age' in df.columns:
        df['age'] = df.groupby('pclass')['age'].transform(
            lambda x: x.fillna(x.median())
        )
        print(f"✅ Age missing values filled with median by class")
    
    if 'embarked' in df.columns:
        df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
        print(f"✅ Embarked missing values filled with mode")
    
    if 'deck' in df.columns:
        df = df.drop('deck', axis=1)
        print(f"✅ Deck column dropped (too many missing values)")
    
    if 'embark_town' in df.columns:
        df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
        print(f"✅ Embark_town missing values filled with mode")
    
    # Convert categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].nunique() < 20:
            df[col] = df[col].astype('category')
    
    print("\n✅ Data cleaned successfully")
    print(f"   Numerical features: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   Categorical features: {len(df.select_dtypes(include=['category']).columns)}")
    print(f"   Total features: {len(df.columns)}")
    
    # Save cleaned data
    csv_path = f"{CURRENT_DIR}/cleaned_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Cleaned dataset saved: {csv_path}")
    
    return df

def get_data_summary(df):
    """Get comprehensive data summary"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['category']).columns
    
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'numerical_cols': list(numerical_cols),
        'categorical_cols': list(categorical_cols),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Rows: {summary['shape'][0]}")
    print(f"Columns: {summary['shape'][1]}")
    print(f"Numerical Features: {len(summary['numerical_cols'])}")
    print(f"Categorical Features: {len(summary['categorical_cols'])}")
    
    print("\nColumn Names and Data Types:")
    for col, dtype in summary['dtypes'].items():
        print(f"  {col}: {dtype}")
    
    missing = {k: v for k, v in summary['missing_values'].items() if v > 0}
    if missing:
        print(f"\nMissing Values:")
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/summary['shape'][0]*100:.1f}%)")
    else:
        print("\n✅ No missing values found")
    
    return summary