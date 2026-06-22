# ============================================================
# MAIN EXECUTION FILE
# ============================================================

from eda_config import CURRENT_DIR
from eda_data_loader import load_and_clean_data, get_data_summary
from eda_visualizer import create_dashboard, create_correlation_analysis, create_pairplot
from eda_analyzer import perform_statistical_analysis
from eda_report import generate_report

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS PROJECT")
    print("="*70)
    print(f"Working Directory: {CURRENT_DIR}")
    print(f"All outputs will be saved in this directory")
    
    try:
        # Step 1: Load and clean data
        df = load_and_clean_data()
        
        # Step 2: Get data summary
        summary = get_data_summary(df)
        
        # Step 3: Generate visualizations
        create_dashboard(df)
        create_correlation_analysis(df)
        create_pairplot(df)
        
        # Step 4: Perform statistical analysis
        results = perform_statistical_analysis(df)
        
        # Step 5: Generate report
        report = generate_report(df, results)
        
        # Step 6: Final summary
        print("\n" + "="*70)
        print("✅ EDA COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 All Files Saved in Current Directory:")
        print(f"   📊 Dashboard: {CURRENT_DIR}/eda_dashboard.png")
        print(f"   📊 Correlation: {CURRENT_DIR}/correlation_analysis.png")
        print(f"   📊 Pairplot: {CURRENT_DIR}/pairplot.png")
        print(f"   📄 Report: {CURRENT_DIR}/analysis_report.txt")
        print(f"   📄 Summary: {CURRENT_DIR}/analysis_summary.json")
        print(f"   💾 Data: {CURRENT_DIR}/cleaned_dataset.csv")
        
        # Print report preview
        print("\n" + "="*70)
        print("REPORT PREVIEW")
        print("="*70)
        report_lines = report.split('\n')
        for i, line in enumerate(report_lines[:20]):
            print(line)
        if len(report_lines) > 20:
            print(f"\n... and {len(report_lines) - 20} more lines")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()