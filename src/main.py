from data_processing import get_processed_data
from visualization import *
from analysis import *
import os

def main():
    print("Starting Video Game Success Analysis...")
    
    os.makedirs('../reports/figures/genre_analysis', exist_ok=True)
    os.makedirs('../reports/figures/publisher_analysis', exist_ok=True)
    os.makedirs('../reports/figures/platform_analysis', exist_ok=True)
    
    print("Loading and processing data...")
    df = get_processed_data()
    
    setup_visuals()
    
    print("Generating visualizations...")
    plot_genre_analysis(df, '../reports/figures/genre_analysis/genre_overview.png')
    plot_critic_analysis(df, '../reports/figures/critic_analysis/critic_vs_sales.png')
    plot_publisher_analysis(df, '../reports/figures/publisher_analysis/publisher_overview.png')
    plot_platform_analysis(df, '../reports/figures/platform_analysis/platform_overview.png')
    plot_regional_analysis(df, '../reports/figures/regional_analysis/regional_overview.png')
    
    print("Performing statistical analysis...")
    correlations = calculate_correlations(df)
    stats_summary = get_summary_statistics(df)
    statistical_tests = perform_statistical_tests(df)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS SUMMARY")
    print("="*60)
    for key, value in stats_summary.items():
        print(f"{key}: {value}")
    
    print(f"\nCritic Score vs Global Sales correlation: {correlations.loc['Critic_Score', 'Global_Sales']:.3f}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()