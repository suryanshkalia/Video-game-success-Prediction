import pandas as pd
import numpy as np
import os 

def load_raw_data(file_path='data/video_games_sales.csv'):
    df = pd.read_csv(file_path)
    print(f"original dataset shape: {df.shape}")  
    return df

def clean_data(df):
    df_clean = df.copy()
    
    initial_shape = df_clean.shape[0] 
    
    print("\nMissing values before cleaning:")
    missing_data = (df_clean.isnull().sum() / len(df_clean)) * 100
    print(missing_data[missing_data > 0]) 
    
    df_clean['Critic_Score'] = df_clean.groupby(['Platform', 'Genre'])['Critic_Score'].transform(
        lambda x: x.fillna(x.median())
    )
    df_clean['Critic_Score'].fillna(df_clean['Critic_Score'].median(), inplace=True)
    df_clean['Publisher'].fillna('Unknown', inplace=True)
    df_clean = df_clean.dropna(subset=['Year_of_Release', 'Genre', 'Platform'])
    
    print(f"Removed {initial_shape - df_clean.shape[0]} rows with critical missing values")
    
    return df_clean

def create_features(df):
    df_processed = df.copy()
    
    major_publishers = [
        'Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment',
        'Ubisoft', 'Take-Two Interactive', 'THQ', 'Sega', 'Microsoft Game Studios',
        'Capcom', 'Square Enix', 'Bandai Namco Games', 'Konami Digital Entertainment'
    ]
    
    df_processed['Company_Type'] = df_processed['Publisher'].apply(
        lambda x: 'AAA' if x in major_publishers else 'Indie/Other'
    )

    def categorize_era(year):
        if pd.isna(year):
            return 'Unknown'
        year = int(year)
        if year < 1990:
            return '1980s'
        elif year < 2000:
            return '1990s'
        elif year < 2010:
            return '2000s'
        else:
            return '2010s'
    
    df_processed['Release_Era'] = df_processed['Year_of_Release'].apply(categorize_era)

    df_processed['Is_Successful'] = df_processed['Global_Sales'] > 1.0
    df_processed['Sales_Category'] = pd.cut(df_processed['Global_Sales'], 
        bins=[0, 0.1, 1, 5, 10, 100],
        labels=['flop', 'below average', 'average', 'hit', 'blockbuster'])
    
    regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'] 
    for region in regions:
        df_processed[f'{region}_Pct'] = (df_processed[region] / df_processed['Global_Sales'].replace(0, np.nan)) * 100
        df_processed[f'{region}_Pct'].fillna(0, inplace=True)
        
    return df_processed

def save_processed_data(df, file_path='data/processed_sales.csv'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")

def get_processed_data():
    raw_df = load_raw_data('data/video_games_sales.csv')
    cleaned_df = clean_data(raw_df)
    processed_df = create_features(cleaned_df)
    save_processed_data(processed_df)
    
    print(f"\nFinal dataset shape: {processed_df.shape}") 
    print(f"Final dataset columns: {list(processed_df.columns)}")

    return processed_df

if __name__ == "__main__":
    df = get_processed_data()