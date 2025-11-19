import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Patch

def setup_visuals():
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    print("Visualization style configured")

def plot_genre_analysis(df, save_dir='../reports/figures/genre_analysis'):
    print("Creating genre analysis visualizations...")
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Genre Analysis for Video Game Success', fontsize=16, fontweight='bold')
    
    genre_sales = df.groupby('Genre')['Global_Sales'].mean().sort_values(ascending=True)
    axes[0,0].barh(genre_sales.index, genre_sales.values, color='skyblue', alpha=0.8)
    axes[0,0].set_title('Average Global Sales by Genre', fontweight='bold')
    axes[0,0].set_xlabel('Average Global Sales (Millions)')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(genre_sales.values):
        axes[0,0].text(v + 0.1, i, f'${v:.2f}M', va='center', fontweight='bold')

    total_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=True)
    axes[0,1].barh(total_sales.index, total_sales.values, color='lightgreen', alpha=0.8)
    axes[0,1].set_title('Total Global Sales by Genre', fontweight='bold')
    axes[0,1].set_xlabel('Total Global Sales (Millions)')
    axes[0,1].grid(axis='x', alpha=0.3)

    genre_count = df['Genre'].value_counts()
    axes[1,0].bar(genre_count.index, genre_count.values, color='salmon', alpha=0.8)
    axes[1,0].set_title('Number of Games by Genre', fontweight='bold')
    axes[1,0].set_ylabel('Number of Games')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(axis='y', alpha=0.3)

    success_rate = df.groupby('Genre')['Is_Successful'].mean().sort_values(ascending=True)
    axes[1,1].barh(success_rate.index, success_rate.values * 100, color='gold', alpha=0.8)
    axes[1,1].set_title('Success Rate by Genre (%)', fontweight='bold')
    axes[1,1].set_xlabel('Success Rate (%)')
    axes[1,1].grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(success_rate.values * 100):
        axes[1,1].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/genre_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return genre_sales, total_sales, success_rate

def plot_critic_analysis(df, save_dir='../reports/figures/critic_analysis'):
    print("Creating critic score analysis visualizations...")
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Critic Score vs Sales Relationship', fontsize=16, fontweight='bold')

    valid_data = df.dropna(subset=['Critic_Score', 'Global_Sales'])
    scatter = axes[0,0].scatter(valid_data['Critic_Score'], valid_data['Global_Sales'], 
                               alpha=0.6, c=valid_data['Global_Sales'], cmap='viridis')
    axes[0,0].set_xlabel('Critic Score')
    axes[0,0].set_ylabel('Global Sales (Millions)')
    axes[0,0].set_title('Critic Score vs Global Sales', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0,0], label='Global Sales (Millions)')

    z = np.polyfit(valid_data['Critic_Score'], valid_data['Global_Sales'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(valid_data['Critic_Score'], p(valid_data['Critic_Score']), "r--", alpha=0.8, linewidth=2)
    
    critic_bins = pd.cut(df['Critic_Score'], bins=10)
    score_sales = df.groupby(critic_bins)['Global_Sales'].mean()
    axes[0,1].plot(range(len(score_sales)), score_sales.values, 
                  marker='o', linewidth=2, markersize=8, color='green')
    axes[0,1].set_xlabel('Critic Score Range')
    axes[0,1].set_ylabel('Average Global Sales (Millions)')
    axes[0,1].set_title('Average Sales by Critic Score Range', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xticks(range(len(score_sales)))
    axes[0,1].set_xticklabels([f"{int(interval.left)}-{int(interval.right)}" 
                              for interval in score_sales.index], rotation=45)

    successful_games = df[df['Is_Successful']]
    unsuccessful_games = df[~df['Is_Successful']]
    
    box_data = [unsuccessful_games['Critic_Score'].dropna(), 
                successful_games['Critic_Score'].dropna()]
    box_plot = axes[1,0].boxplot(box_data, labels=['Unsuccessful', 'Successful'],
                                patch_artist=True)
    
    colors = ['lightcoral', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1,0].set_ylabel('Critic Score')
    axes[1,0].set_title('Critic Score Distribution: Successful vs Unsuccessful Games', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)

    numeric_columns = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score']
    correlation_matrix = df[numeric_columns].corr()

    im = axes[1,1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(len(numeric_columns)))
    axes[1,1].set_yticks(range(len(numeric_columns)))
    axes[1,1].set_xticklabels(numeric_columns, rotation=45)
    axes[1,1].set_yticklabels(numeric_columns)
    axes[1,1].set_title('Sales and Score Correlation Matrix', fontweight='bold')

    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            color = 'white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black'
            axes[1,1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                          ha='center', va='center', color=color, fontweight='bold')

    plt.colorbar(im, ax=axes[1,1])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/critic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def plot_publisher_analysis(df, save_dir='../reports/figures/publisher_analysis'):
    print("Creating publisher analysis visualizations...")
    os.makedirs(save_dir, exist_ok=True)

    top_publishers = df['Publisher'].value_counts().head(10).index
    top_pub_data = df[df['Publisher'].isin(top_publishers)]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Publisher and Genre Performance Analysis', fontsize=16, fontweight='bold')

    publisher_total_sales = df.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10)
    axes[0,0].barh(range(len(publisher_total_sales)), publisher_total_sales.values, color='lightblue')
    axes[0,0].set_yticks(range(len(publisher_total_sales)))
    axes[0,0].set_yticklabels(publisher_total_sales.index)
    axes[0,0].set_xlabel('Total Global Sales (Millions)')
    axes[0,0].set_title('Top 10 Publishers by Total Sales', fontweight='bold')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(publisher_total_sales.values):
        axes[0,0].text(v + 5, i, f'${v:.0f}M', va='center', fontweight='bold')

    success_by_company = df.groupby('Company_Type')['Is_Successful'].mean() * 100
    bars = axes[0,1].bar(success_by_company.index, success_by_company.values, 
                        color=['steelblue', 'darkorange'], alpha=0.8)
    axes[0,1].set_ylabel('Success Rate (%)')
    axes[0,1].set_title('Success Rate: AAA vs Indie/Other Games', fontweight='bold')
    axes[0,1].grid(axis='y', alpha=0.3)
    
    for bar, v in zip(bars, success_by_company.values):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    publisher_success = df.groupby('Publisher')['Is_Successful'].mean() * 100
    top_publisher_success = publisher_success[publisher_success.index.isin(top_publishers)].sort_values()
    
    axes[1,0].barh(range(len(top_publisher_success)), top_publisher_success.values, color='lightgreen')
    axes[1,0].set_yticks(range(len(top_publisher_success)))
    axes[1,0].set_yticklabels(top_publisher_success.index)
    axes[1,0].set_xlabel('Success Rate (%)')
    axes[1,0].set_title('Success Rate for Top 10 Publishers', fontweight='bold')
    axes[1,0].grid(axis='x', alpha=0.3)

    sales_data = [df[df['Company_Type'] == 'AAA']['Global_Sales'],
                  df[df['Company_Type'] == 'Indie/Other']['Global_Sales']]
    
    box_plot = axes[1,1].boxplot(sales_data, labels=['AAA', 'Indie/Other'], patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1,1].set_ylabel('Global Sales (Millions)')
    axes[1,1].set_title('Sales Distribution: AAA vs Indie/Other', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/publisher_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return publisher_total_sales, success_by_company

def plot_platform_analysis(df, save_dir='../reports/figures/platform_analysis'):
    print("Creating platform analysis visualizations...")
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Platform Performance Analysis', fontsize=16, fontweight='bold')

    platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(15)
    axes[0,0].barh(range(len(platform_sales)), platform_sales.values, color='lightsteelblue')
    axes[0,0].set_yticks(range(len(platform_sales)))
    axes[0,0].set_yticklabels(platform_sales.index)
    axes[0,0].set_xlabel('Total Global Sales (Millions)')
    axes[0,0].set_title('Top 15 Platforms by Total Sales', fontweight='bold')
    axes[0,0].grid(axis='x', alpha=0.3)

    platform_era_sales = df.pivot_table(
        values='Global_Sales', 
        index='Release_Era', 
        columns='Platform', 
        aggfunc='sum',
        fill_value=0
    )

    top_platforms_by_era = platform_era_sales.idxmax(axis=1)
    era_top_sales = platform_era_sales.max(axis=1)
    
    colors = ['red', 'blue', 'green', 'orange']
    bars = axes[0,1].bar(platform_era_sales.index, era_top_sales.values, color=colors, alpha=0.8)
    axes[0,1].set_ylabel('Sales by Dominant Platform (Millions)')
    axes[0,1].set_title('Dominant Platform in Each Era', fontweight='bold')
    axes[0,1].grid(axis='y', alpha=0.3)

    for i, (era, platform) in enumerate(top_platforms_by_era.items()):
        axes[0,1].text(i, era_top_sales[era] + 5, platform, 
                      ha='center', va='bottom', rotation=0, fontweight='bold')

    platform_success = df.groupby('Platform')['Is_Successful'].mean().sort_values(ascending=False).head(10)
    axes[1,0].barh(range(len(platform_success)), platform_success.values * 100, color='lightgreen')
    axes[1,0].set_yticks(range(len(platform_success)))
    axes[1,0].set_yticklabels(platform_success.index)
    axes[1,0].set_xlabel('Success Rate (%)')
    axes[1,0].set_title('Top 10 Platforms by Success Rate', fontweight='bold')
    axes[1,0].grid(axis='x', alpha=0.3)

    platform_counts = df['Platform'].value_counts().head(15)
    axes[1,1].barh(range(len(platform_counts)), platform_counts.values, color='salmon')
    axes[1,1].set_yticks(range(len(platform_counts)))
    axes[1,1].set_yticklabels(platform_counts.index)
    axes[1,1].set_xlabel('Number of Games')
    axes[1,1].set_title('Number of Games Released per Platform (Top 15)', fontweight='bold')
    axes[1,1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/platform_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return platform_sales, platform_success

def plot_regional_analysis(df, save_dir='../reports/figures/regional_analysis'):
    print("Creating regional analysis visualizations...")
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Regional Sales Analysis', fontsize=16, fontweight='bold')

    regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    region_names = ['North America', 'Europe', 'Japan', 'Other Regions']
    colors = ['blue', 'green', 'red', 'orange']

    for i, (region, name, color) in enumerate(zip(regions, region_names, colors)):
        region_data = df[df[region] > 0][region]
        axes[i//2, i%2].hist(region_data, bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[i//2, i%2].set_xlabel(f'Sales in {name} (Millions)')
        axes[i//2, i%2].set_ylabel('Number of Games')
        axes[i//2, i%2].set_title(f'Sales Distribution in {name}', fontweight='bold')
        axes[i//2, i%2].grid(True, alpha=0.3)
        axes[i//2, i%2].set_xlim(0, 5)
        
        mean_val = region_data.mean()
        axes[i//2, i%2].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                               label=f'Mean: {mean_val:.2f}M')
        axes[i//2, i%2].legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/regional_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    region_genre_pref = df.groupby('Genre')[regions].mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Genre Preferences by Region', fontsize=16, fontweight='bold')
    
    for i, (region, name, color) in enumerate(zip(regions, region_names, colors)):
        sorted_data = region_genre_pref[region].sort_values(ascending=True)
        axes[i//2, i%2].barh(sorted_data.index, sorted_data.values, color=color, alpha=0.8)
        axes[i//2, i%2].set_title(f'Average {name} Sales by Genre', fontweight='bold')
        axes[i//2, i%2].set_xlabel('Average Sales (Millions)')
        axes[i//2, i%2].grid(axis='x', alpha=0.3)
        
        for j, v in enumerate(sorted_data.values):
            if v > 0.1: 
                axes[i//2, i%2].text(v + 0.01, j, f'{v:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/regional_genre_preferences.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return region_genre_pref

def plot_temporal_analysis(df, save_dir='../reports/figures/temporal_analysis'):
    print("Creating temporal analysis visualizations...")
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Temporal Trends in Video Game Industry', fontsize=16, fontweight='bold')
    
    games_per_year = df.groupby('Year_of_Release').size()
    axes[0,0].plot(games_per_year.index, games_per_year.values, linewidth=2, marker='o', color='blue')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Number of Games Released')
    axes[0,0].set_title('Games Released Per Year', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)

    sales_per_year = df.groupby('Year_of_Release')['Global_Sales'].mean()
    axes[0,1].plot(sales_per_year.index, sales_per_year.values, linewidth=2, marker='s', color='green')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Average Global Sales (Millions)')
    axes[0,1].set_title('Average Sales Per Year', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)

    success_per_year = df.groupby('Year_of_Release')['Is_Successful'].mean() * 100
    axes[1,0].plot(success_per_year.index, success_per_year.values, linewidth=2, marker='^', color='red')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Success Rate (%)')
    axes[1,0].set_title('Success Rate Over Time', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)

    critic_per_year = df.groupby('Year_of_Release')['Critic_Score'].mean()
    axes[1,1].plot(critic_per_year.index, critic_per_year.values, linewidth=2, marker='d', color='purple')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Average Critic Score')
    axes[1,1].set_title('Average Critic Score Over Time', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return games_per_year, sales_per_year, success_per_year, critic_per_year