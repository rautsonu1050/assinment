#!/usr/bin/env python3
"""
Real-World Retail Data Project
End-to-End Data Analysis and Prediction
"""

import os
import sys
import time
from data_generator import generate_retail_data
from data_analysis import RetailDataAnalyzer
from sales_prediction import SalesPredictor

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def main():
    """Main execution function"""
    print_header("REAL-WORLD RETAIL DATA PROJECT")
    print("Project: Sales Analysis and Prediction")
    print("Domain: Retail")
    
    # Step 1: Generate data (if not exists)
    print_header("STEP 1: DATA GENERATION")
    data_file = 'retail_sales_data.csv'
    
    if not os.path.exists(data_file):
        print("Generating retail sales dataset...")
        df = generate_retail_data(n_days=365, n_stores=5, n_products=20)
        df.to_csv(data_file, index=False)
        print(f"✅ Dataset saved to {data_file}")
        print(f"   Shape: {df.shape}")
    else:
        print(f"✅ Dataset already exists: {data_file}")
    
    # Step 2: Exploratory Data Analysis
    print_header("STEP 2: EXPLORATORY DATA ANALYSIS")
    analyzer = RetailDataAnalyzer(data_file)
    analyzer.run_full_analysis()
    
    # Step 3: Sales Prediction
    print_header("STEP 3: SALES PREDICTION")
    predictor = SalesPredictor(data_file)
    results = predictor.run_prediction_pipeline()
    
    # Step 4: Summary
    print_header("PROJECT COMPLETED SUCCESSFULLY!")
    print("\n📊 Key Insights:")
    print("   - Sales analysis visualizations saved as PNG files")
    print("   - Machine learning models trained for revenue prediction")
    print("   - Best performing model identified and tuned")
    
    print("\n📁 Output Files:")
    print("   - retail_sales_data.csv          : Generated dataset")
    print("   - sales_analysis_visualizations.png : EDA visualizations")
    print("   - seasonality_analysis.png       : Seasonal patterns")
    print("   - correlation_analysis.png       : Feature correlations")
    print("   - prediction_results.png         : Model predictions")
    
    print("\n💡 Recommendations:")
    print("   - Focus on high-performing categories and seasons")
    print("   - Leverage promotions strategically")
    print("   - Consider store expansion in high-demand areas")
    print("   - Use ML predictions for inventory optimization")
    
    print("\n✅ Project completed! Check the generated files for details.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {elapsed_time:.2f} seconds")