import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_correlations(df):
    print("Calculating correlations...")
    numeric_columns = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 
                      'Other_Sales', 'Critic_Score', 'User_Score']
    
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    
    correlation_matrix = df[numeric_columns].corr()
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))
    
    return correlation_matrix

def get_summary_statistics(df):
    print("\nGenerating summary statistics...")
    
    stats_dict = {
        'total_games': len(df),
        'successful_games': df['Is_Successful'].sum(),
        'success_rate': (df['Is_Successful'].sum() / len(df)) * 100,
        'avg_global_sales': df['Global_Sales'].mean(),
        'median_global_sales': df['Global_Sales'].median(),
        'total_global_sales': df['Global_Sales'].sum(),
        'top_genre_sales': df.groupby('Genre')['Global_Sales'].mean().idxmax(),
        'top_genre_count': df['Genre'].value_counts().idxmax(),
        'top_publisher_sales': df.groupby('Publisher')['Global_Sales'].sum().idxmax(),
        'top_platform_sales': df.groupby('Platform')['Global_Sales'].sum().idxmax(),
        'avg_critic_score': df['Critic_Score'].mean(),
        'years_covered': f"{int(df['Year_of_Release'].min())}-{int(df['Year_of_Release'].max())}"
    }
    
    print("\nSummary Statistics:")
    for key, value in stats_dict.items():
        print(f"{key}: {value}")
    
    return stats_dict

def perform_statistical_tests(df):
    print("\nPerforming statistical tests...")
    
    results = {}
    
    aaa_sales = df[df['Company_Type'] == 'AAA']['Global_Sales']
    indie_sales = df[df['Company_Type'] == 'Indie/Other']['Global_Sales']
    
    t_stat, p_value = stats.ttest_ind(aaa_sales, indie_sales, equal_var=False)
    results['aaa_vs_indie'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'aaa_mean': aaa_sales.mean(),
        'indie_mean': indie_sales.mean()
    }
    
    print(f"AAA vs Indie T-test: t={t_stat:.3f}, p={p_value:.3f}")
    print(f"AAA mean sales: ${aaa_sales.mean():.2f}M")
    print(f"Indie mean sales: ${indie_sales.mean():.2f}M")
    
    successful_critic = df[df['Is_Successful']]['Critic_Score']
    unsuccessful_critic = df[~df['Is_Successful']]['Critic_Score']
    
    t_stat, p_value = stats.ttest_ind(successful_critic, unsuccessful_critic, equal_var=False)
    results['critic_success'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'successful_mean': successful_critic.mean(),
        'unsuccessful_mean': unsuccessful_critic.mean()
    }
    
    print(f"Successful vs Unsuccessful Critic Score T-test: t={t_stat:.3f}, p={p_value:.3f}")
    print(f"Successful games mean critic score: {successful_critic.mean():.1f}")
    print(f"Unsuccessful games mean critic score: {unsuccessful_critic.mean():.1f}")
    
    return results

def analyze_genre_performance(df):
    print("\nAnalyzing genre performance...")
    
    genre_analysis = df.groupby('Genre').agg({
        'Global_Sales': ['count', 'mean', 'median', 'sum'],
        'Critic_Score': 'mean',
        'Is_Successful': 'mean'
    }).round(3)
    
    genre_analysis.columns = ['count', 'avg_sales', 'median_sales', 'total_sales', 'avg_critic', 'success_rate']
    genre_analysis['success_rate'] = genre_analysis['success_rate'] * 100
    
    print("\nGenre Performance Analysis:")
    print(genre_analysis.sort_values('avg_sales', ascending=False))
    
    return genre_analysis

def analyze_publisher_performance(df):
    print("\nAnalyzing publisher performance...")
    
    top_publishers = df.groupby('Publisher')['Global_Sales'].sum().nlargest(20)
    
    publisher_analysis = df[df['Publisher'].isin(top_publishers.index)].groupby('Publisher').agg({
        'Global_Sales': ['count', 'mean', 'sum'],
        'Critic_Score': 'mean',
        'Is_Successful': 'mean',
        'Company_Type': 'first'
    }).round(3)
    
    publisher_analysis.columns = ['count', 'avg_sales', 'total_sales', 'avg_critic', 'success_rate', 'company_type']
    publisher_analysis['success_rate'] = publisher_analysis['success_rate'] * 100
    publisher_analysis = publisher_analysis.sort_values('total_sales', ascending=False)
    
    print("\nTop Publishers Performance Analysis:")
    print(publisher_analysis)
    
    return publisher_analysis

def analyze_platform_performance(df):
    print("\nAnalyzing platform performance...")
    
    top_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(15)
    
    platform_analysis = df[df['Platform'].isin(top_platforms.index)].groupby('Platform').agg({
        'Global_Sales': ['count', 'mean', 'sum'],
        'Critic_Score': 'mean',
        'Is_Successful': 'mean',
        'Year_of_Release': ['min', 'max']
    }).round(3)
    
    platform_analysis.columns = ['count', 'avg_sales', 'total_sales', 'avg_critic', 'success_rate', 'first_year', 'last_year']
    platform_analysis['success_rate'] = platform_analysis['success_rate'] * 100
    platform_analysis['lifespan'] = platform_analysis['last_year'] - platform_analysis['first_year']
    platform_analysis = platform_analysis.sort_values('total_sales', ascending=False)
    
    print("\nTop Platforms Performance Analysis:")
    print(platform_analysis)
    
    return platform_analysis

def generate_insights(df, stats_dict, correlation_matrix, statistical_tests):
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
    print("="*80)
    
    total_games = stats_dict['total_games']
    success_rate = stats_dict['success_rate']
    top_genre = stats_dict['top_genre_sales']
    top_publisher = stats_dict['top_publisher_sales']
    top_platform = stats_dict['top_platform_sales']
    
    genre_performance = df.groupby('Genre')['Global_Sales'].mean().sort_values(ascending=False)
    best_genre = genre_performance.index[0]
    worst_genre = genre_performance.index[-1]
    
    critic_correlation = correlation_matrix.loc['Critic_Score', 'Global_Sales']
    
    aaa_success = df[df['Company_Type'] == 'AAA']['Is_Successful'].mean() * 100
    indie_success = df[df['Company_Type'] == 'Indie/Other']['Is_Successful'].mean() * 100
    
    print(f"\nMARKET OVERVIEW:")
    print(f"• Analyzed {total_games:,} games from {stats_dict['years_covered']}")
    print(f"• Overall success rate: {success_rate:.1f}% of games sell >1M units")
    print(f"• Total industry sales: ${stats_dict['total_global_sales']:.0f}M")
    
    print(f"\nGENRE STRATEGY:")
    print(f"• Highest performing genre: {best_genre} (${genre_performance[best_genre]:.2f}M average sales)")
    print(f"• Most popular genre: {stats_dict['top_genre_count']}")
    print(f"• Avoid: {worst_genre} has the lowest average sales")
    
    print(f"\nPUBLISHER INSIGHTS:")
    print(f"• Market leader: {top_publisher}")
    print(f"• AAA success rate: {aaa_success:.1f}% vs Indie: {indie_success:.1f}%")
    print(f"• AAA games sell {statistical_tests['aaa_vs_indie']['aaa_mean']/statistical_tests['aaa_vs_indie']['indie_mean']:.1f}x more on average")
    
    print(f"\nPLATFORM STRATEGY:")
    print(f"• Most successful platform: {top_platform}")
    print(f"• Platform choice significantly impacts market reach")
    
    print(f"\nQUALITY VS SUCCESS:")
    print(f"• Critic score correlation with sales: {critic_correlation:.3f}")
    print(f"• Successful games have {statistical_tests['critic_success']['successful_mean']-statistical_tests['critic_success']['unsuccessful_mean']:.1f} higher critic scores on average")
    
    print(f"\nSTRATEGIC RECOMMENDATIONS:")
    print("1. Focus on genres with both high sales potential and high success rates")
    print("2. Partner with established publishers for better market reach")
    print("3. Consider regional preferences when planning global releases")
    print("4. Quality matters - invest in game development for better critic scores")
    print("5. Platform choice is critical - target platforms with high user engagement")
    print("6. Balance between creative innovation and proven market formulas")
    
    print("="*80)
    
    insights = {
        'best_genre': best_genre,
        'worst_genre': worst_genre,
        'critic_correlation': critic_correlation,
        'aaa_vs_indie_ratio': statistical_tests['aaa_vs_indie']['aaa_mean']/statistical_tests['aaa_vs_indie']['indie_mean'],
        'successful_critic_advantage': statistical_tests['critic_success']['successful_mean']-statistical_tests['critic_success']['unsuccessful_mean']
    }
    
    return insights