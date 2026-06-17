"""
Data loading and preprocessing functions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_indian_liver_data(url):
    """Load Indian Liver Patient Dataset"""
    try:
        df = pd.read_csv(url)
        print("✅ Dataset loaded successfully from UCI Repository!")
        return df
    except:
        print("⚠️  Using generated Indian healthcare data (fallback)...")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic Indian healthcare data as fallback"""
    np.random.seed(42)
    n_samples = 583
    
    data = {
        'Age': np.random.randint(18, 90, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.65, 0.35]),
        'Total_Bilirubin': np.random.gamma(2, 1.5, n_samples),
        'Direct_Bilirubin': np.random.gamma(1.5, 0.8, n_samples),
        'Alkaline_Phosphotase': np.random.gamma(3, 80, n_samples),
        'Alamine_Aminotransferase': np.random.gamma(2, 30, n_samples),
        'Aspartate_Aminotransferase': np.random.gamma(2, 25, n_samples),
        'Total_Proteins': np.random.normal(7, 1, n_samples),
        'Albumin': np.random.normal(4, 0.8, n_samples),
        'Albumin_and_Globulin_Ratio': np.random.normal(1.2, 0.3, n_samples),
    }
    
    df = pd.DataFrame(data)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Generate target
    score = (df['Total_Bilirubin'] > 2) * 0.3 + \
            (df['Alkaline_Phosphotase'] > 250) * 0.3 + \
            (df['Alamine_Aminotransferase'] > 40) * 0.2 + \
            (df['Age'] > 50) * 0.2
    
    df['Dataset'] = (score + np.random.randn(n_samples) * 0.2 > 0.5).astype(int)
    return df.dropna()


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess data and split into train/test sets"""
    # Separate features and target
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']
    
    # Handle categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def get_data_info(df):
    """Print dataset information"""
    print(f"\n📊 Dataset Information:")
    print(f"   - Source: Indian Liver Patient Dataset (UCI)")
    print(f"   - Samples: {df.shape[0]}")
    print(f"   - Features: {df.shape[1]-1}")
    print(f"   - Target: Liver Disease (0=No, 1=Yes)")
    
    print(f"\n📈 Target Distribution:")
    print(df['Dataset'].value_counts().to_string())
    
    print("\n📊 First 5 rows:")
    print(df.head())