import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RetailDataAnalyzer:
    def __init__(self, data_path='retail_sales_data.csv'):
        """Initialize with data"""
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"Data loaded: {self.df.shape[0]} records")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
    
    def basic_info(self):
        """Print basic dataset information"""
        print("\n" + "="*50)
        print("BASIC DATASET INFORMATION")
        print("="*50)
        print(f"Total records: {len(self.df)}")
        print(f"Number of stores: {self.df['store_id'].nunique()}")
        print(f"Number of products: {self.df['product_id'].nunique()}")
        print(f"Categories: {self.df['category'].unique()}")
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nData types:")
        print(self.df.dtypes)
    
    def summary_statistics(self):
        """Generate summary statistics"""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        numeric_cols = ['price', 'units_sold', 'revenue']
        summary = self.df[numeric_cols].describe()
        print(summary)
        
        # Additional metrics
        print("\n" + "-"*30)
        print("Category-wise Revenue:")
        print(self.df.groupby('category')['revenue'].sum().sort_values(ascending=False))
        
        print("\n" + "-"*30)
        print("Store Performance:")
        store_stats = self.df.groupby('store_id').agg({
            'revenue': ['sum', 'mean'],
            'units_sold': 'sum'
        }).round(2)
        print(store_stats)
    
    def visualize_sales_trends(self):
        """Visualize sales trends over time"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Retail Sales Analysis', fontsize=16)
        
        # Daily revenue trend
        daily_revenue = self.df.groupby('date')['revenue'].sum()
        axes[0, 0].plot(daily_revenue.index, daily_revenue.values, color='blue', alpha=0.7)
        axes[0, 0].set_title('Daily Revenue Trend')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Revenue ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Revenue by category
        category_revenue = self.df.groupby('category')['revenue'].sum().sort_values()
        axes[0, 1].barh(category_revenue.index, category_revenue.values)
        axes[0, 1].set_title('Revenue by Product Category')
        axes[0, 1].set_xlabel('Revenue ($)')
        
        # Units sold by store size
        store_size_sales = self.df.groupby('store_size')['units_sold'].mean()
        axes[1, 0].bar(store_size_sales.index, store_size_sales.values)
        axes[1, 0].set_title('Average Units Sold by Store Size')
        axes[1, 0].set_ylabel('Average Units Sold')
        
        # Promotion impact
        promo_impact = self.df.groupby('promotion')['revenue'].mean()
        axes[1, 1].bar(['No Promotion', 'Promotion'], promo_impact.values)
        axes[1, 1].set_title('Average Revenue: Promotion vs No Promotion')
        axes[1, 1].set_ylabel('Average Revenue ($)')
        
        plt.tight_layout()
        plt.savefig('sales_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_seasonality(self):
        """Analyze seasonal patterns"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Seasonality Analysis', fontsize=14)
        
        # By month
        self.df['month'] = self.df['date'].dt.month
        monthly_revenue = self.df.groupby('month')['revenue'].mean()
        axes[0].bar(monthly_revenue.index, monthly_revenue.values)
        axes[0].set_title('Average Revenue by Month')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Average Revenue ($)')
        
        # By day of week
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_revenue = self.df.groupby('day_of_week')['revenue'].mean()
        axes[1].bar(day_names, daily_revenue.values)
        axes[1].set_title('Average Revenue by Day of Week')
        axes[1].set_xlabel('Day')
        axes[1].set_ylabel('Average Revenue ($)')
        
        plt.tight_layout()
        plt.savefig('seasonality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self):
        """Correlation analysis between numeric variables"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        numeric_cols = ['price', 'units_sold', 'revenue']
        corr = self.df[numeric_cols].corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, ax=ax)
        ax.set_title('Correlation Matrix of Numeric Variables')
        
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_analysis(self):
        """Run all analysis functions"""
        self.basic_info()
        self.summary_statistics()
        self.visualize_sales_trends()
        self.analyze_seasonality()
        self.correlation_analysis()
        print("\n✅ Analysis complete! Visualizations saved as PNG files.")

if __name__ == "__main__":
    analyzer = RetailDataAnalyzer()
    analyzer.run_full_analysis()