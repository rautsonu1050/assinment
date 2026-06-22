import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_retail_data(n_days=365, n_stores=5, n_products=20):
    """
    Generate synthetic retail sales data
    
    Parameters:
    -----------
    n_days : int, number of days of data
    n_stores : int, number of stores
    n_products : int, number of products
    
    Returns:
    --------
    pd.DataFrame
    """
    # Date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Groceries', 'Books', 'Home Goods']
    
    # Generate data
    data = []
    
    for date in dates:
        for store_id in range(1, n_stores + 1):
            # Store size factor
            store_size = random.choice(['Small', 'Medium', 'Large'])
            size_factor = {'Small': 0.7, 'Medium': 1.0, 'Large': 1.4}[store_size]
            
            for product_id in range(1, n_products + 1):
                # Product characteristics
                category = random.choice(categories)
                base_price = random.uniform(5, 100)
                
                # Day of week effect (weekends have higher sales)
                weekday = date.weekday()
                weekday_factor = 1.0 if weekday < 5 else 1.3
                
                # Seasonality effect
                month = date.month
                season_factor = 1.0
                if month in [11, 12]:  # Holiday season
                    season_factor = 1.5
                elif month in [6, 7, 8]:  # Summer
                    season_factor = 1.1
                
                # Random promotions (20% of days)
                promotion = random.random() < 0.2
                promo_factor = 1.4 if promotion else 1.0
                
                # Base units sold
                base_units = random.randint(1, 50)
                
                # Calculate final units sold
                units_sold = int(base_units * size_factor * weekday_factor * 
                               season_factor * promo_factor * random.uniform(0.8, 1.2))
                units_sold = max(1, units_sold)
                
                # Calculate revenue
                price = base_price * (0.85 if promotion else 1.0)
                revenue = units_sold * price
                
                data.append({
                    'date': date,
                    'store_id': store_id,
                    'store_size': store_size,
                    'product_id': product_id,
                    'category': category,
                    'price': round(price, 2),
                    'units_sold': units_sold,
                    'revenue': round(revenue, 2),
                    'promotion': promotion,
                    'season': 'Holiday' if month in [11, 12] else 
                             'Summer' if month in [6, 7, 8] else 'Regular'
                })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating retail sales data...")
    df = generate_retail_data(n_days=365, n_stores=5, n_products=20)
    
    # Save to CSV
    df.to_csv('retail_sales_data.csv', index=False)
    print(f"Dataset saved! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())