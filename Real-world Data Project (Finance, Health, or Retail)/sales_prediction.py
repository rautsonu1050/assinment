import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SalesPredictor:
    def __init__(self, data_path='retail_sales_data.csv'):
        """Initialize the predictor with data"""
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_features(self):
        """Prepare features for modeling"""
        df = self.df.copy()
        
        # Extract date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['store_size', 'category', 'season', 'promotion']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Select features for modeling
        feature_cols = ['store_id', 'product_id', 'price', 'promotion', 
                       'store_size', 'category', 'season', 'month', 
                       'day_of_week', 'is_weekend', 'quarter']
        
        X = df[feature_cols]
        y = df['revenue']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        return X_train_scaled, X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models for comparison"""
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Model configurations
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            self.models[name] = model
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  R² Score: {r2:.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['R²'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        self.best_predictions = results[best_model_name]['predictions']
        
        print(f"\n🏆 Best Model: {best_model_name} (R²: {results[best_model_name]['R²']:.4f})")
        
        return results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print("Tuning Random Forest hyperparameters...")
        rf = RandomForestRegressor(random_state=42)
        
        random_search = RandomizedSearchCV(
            rf, param_grid, n_iter=20, cv=3, 
            scoring='r2', random_state=42, n_jobs=-1
        )
        
        random_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best R² score: {random_search.best_score_:.4f}")
        
        # Train best model with tuned parameters
        tuned_model = random_search.best_estimator_
        tuned_model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate tuned model
        y_pred_tuned = tuned_model.predict(self.X_test_scaled)
        r2_tuned = r2_score(self.y_test, y_pred_tuned)
        mae_tuned = mean_absolute_error(self.y_test, y_pred_tuned)
        
        print(f"\nTuned Model Performance:")
        print(f"  R² Score: {r2_tuned:.4f}")
        print(f"  MAE: ${mae_tuned:.2f}")
        
        # Update best model if tuning improved
        if r2_tuned > self.models[self.best_model_name].score(self.X_test_scaled, self.y_test):
            self.best_model = tuned_model
            self.best_predictions = y_pred_tuned
            print("✅ Tuned model outperforms the original!")
    
    def visualize_predictions(self):
        """Visualize model predictions vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Sales Prediction Results', fontsize=16)
        
        # Actual vs Predicted Scatter Plot
        axes[0, 0].scatter(self.y_test, self.best_predictions, alpha=0.5, s=10)
        axes[0, 0].plot([0, max(self.y_test)], [0, max(self.y_test)], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Revenue ($)')
        axes[0, 0].set_ylabel('Predicted Revenue ($)')
        axes[0, 0].set_title('Actual vs Predicted (Best Model)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals Plot
        residuals = self.y_test - self.best_predictions
        axes[0, 1].scatter(self.best_predictions, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Revenue ($)')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of Errors
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Prediction Error ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Prediction Errors')
        
        # Feature Importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = ['store_id', 'product_id', 'price', 'promotion', 
                           'store_size', 'category', 'season', 'month', 
                           'day_of_week', 'is_weekend', 'quarter']
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            axes[1, 1].barh(range(11), importances[indices][::-1])
            axes[1, 1].set_yticks(range(11))
            axes[1, 1].set_yticklabels([feature_names[i] for i in indices[::-1]])
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Feature Importance (Best Model)')
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_prediction_pipeline(self):
        """Run the complete prediction pipeline"""
        print("\n" + "="*50)
        print("SALES PREDICTION PIPELINE")
        print("="*50)
        
        # Prepare features
        self.prepare_features()
        
        # Train models
        results = self.train_models()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Visualize results
        self.visualize_predictions()
        
        print("\n✅ Prediction pipeline complete!")
        print(f"Best model saved: {self.best_model_name}")
        
        return results

if __name__ == "__main__":
    predictor = SalesPredictor()
    results = predictor.run_prediction_pipeline()