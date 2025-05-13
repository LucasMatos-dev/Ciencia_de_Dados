import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('seaborn-v0_8') 
sns.set_theme()  

def get_price_data():
    """Fetch price data from the database and return as a DataFrame"""
    conn = sqlite3.connect('cards.db')
    

    query = """
    SELECT 
        cs.set_name,
        AVG(cp.cardmarket_price) as avg_cardmarket_price,
        AVG(cp.tcgplayer_price) as avg_tcgplayer_price,
        AVG(cp.ebay_price) as avg_ebay_price,
        AVG(cp.amazon_price) as avg_amazon_price,
        AVG(cp.coolstuffinc_price) as avg_coolstuffinc_price
    FROM card_sets cs
    JOIN card_prices cp ON cs.card_id = cp.card_id
    GROUP BY cs.set_name
    ORDER BY avg_cardmarket_price DESC
    LIMIT 20
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def plot_prices_by_set():
    """Create visualizations for card prices by set"""
    df = get_price_data()
    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    

    sns.barplot(data=df, x='set_name', y='avg_cardmarket_price', ax=ax1)
    ax1.set_title('Average Cardmarket Prices by Set (Top 20)')
    ax1.set_xlabel('Set Name')
    ax1.set_ylabel('Average Price (USD)')
    ax1.tick_params(axis='x', rotation=45)
    

    price_columns = ['avg_cardmarket_price', 'avg_tcgplayer_price', 
                     'avg_ebay_price', 'avg_amazon_price', 'avg_coolstuffinc_price']
    df_melted = pd.melt(df, value_vars=price_columns, 
                        var_name='Platform', value_name='Price')
    df_melted['Platform'] = df_melted['Platform'].str.replace('avg_', '').str.replace('_price', '')
    
    sns.boxplot(data=df_melted, x='Platform', y='Price', ax=ax2)
    ax2.set_title('Price Distribution Across Different Platforms')
    ax2.set_xlabel('Platform')
    ax2.set_ylabel('Price (USD)')
    

    plt.tight_layout()
    plt.savefig('card_prices_analysis.png')
    plt.close()

if __name__ == "__main__":
    plot_prices_by_set()
    print("Analysis complete! Check 'card_prices_analysis.png' for the results.") 