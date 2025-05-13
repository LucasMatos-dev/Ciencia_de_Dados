import sqlite3
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

DB_NAME = "cards.db"
CARDS_PER_PACK = 9
PACK_PRICE = 22


def fetch_set_prices():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
    SELECT cs.set_name, cp.card_id, cp.cardmarket_price
    FROM card_sets cs
    JOIN card_prices cp ON cs.card_id = cp.card_id
    WHERE cp.cardmarket_price > 0
    ''')

    set_prices = {}
    rows = cursor.fetchall()

    print(f"Found {len(rows)} rows of card set prices.")

    for row in rows:
        set_name, card_id, price = row
        if set_name not in set_prices:
            set_prices[set_name] = []
        set_prices[set_name].append(price)

    conn.close()
    return set_prices


def monte_carlo_simulation(set_prices, simulations=10000):
    set_simulations = {}

    for set_name, prices in tqdm(set_prices.items()):
        pack_values = []

        for _ in list(range(simulations)):
            pack_cards = random.sample(prices, CARDS_PER_PACK)
            pack_value = sum(pack_cards)
            pack_values.append(pack_value)

        average_pack_value = np.mean(pack_values)
        profit = average_pack_value - PACK_PRICE
        profit_margin = (profit / PACK_PRICE) * 100

        set_simulations[set_name] = {
            'avg_cost': PACK_PRICE,
            'avg_value': average_pack_value,
            'profit': profit,
            'profit_margin': profit_margin
        }

    sorted_sets = sorted(set_simulations.items(),
                         key=lambda x: x[1]['avg_value'], reverse=True)

    return sorted_sets


def plot_top_sets(top_sets):
    if not top_sets:
        print("No sets found to plot.")
        return

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    set_names = [set_name for set_name, _ in top_sets]
    avg_values = [metrics['avg_value'] for _, metrics in top_sets]
    profits = [metrics['profit'] for _, metrics in top_sets]
    costs = [metrics['avg_cost'] for _, metrics in top_sets]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))

    barWidth = 0.35

    r1_top = np.arange(len(set_names[:5]))
    r2_top = [x + barWidth for x in r1_top]

    ax1.bar(r1_top, costs[:5], width=barWidth,
            label='Custo do Pacote (USD)', color='red', alpha=0.7)
    ax1.bar(r2_top, profits[:5], width=barWidth,
            label='Lucro (USD)', color='green', alpha=0.7)

    ax1.set_xlabel('Sets')
    ax1.set_ylabel('Valor (USD)')
    ax1.set_title('Top 5 Sets - Comparação de Custo e Lucro')
    ax1.set_xticks([r + barWidth/2 for r in range(len(set_names[:5]))])
    ax1.set_xticklabels(set_names[:5], rotation=45, ha='right')
    ax1.legend()

    r1_bottom = np.arange(len(set_names[-5:]))
    r2_bottom = [x + barWidth for x in r1_bottom]

    ax2.bar(r1_bottom, costs[-5:], width=barWidth,
            label='Custo do Pacote (USD)', color='red', alpha=0.7)
    ax2.bar(r2_bottom, profits[-5:], width=barWidth,
            label='Lucro (USD)', color='green', alpha=0.7)

    ax2.set_xlabel('Sets')
    ax2.set_ylabel('Valor (USD)')
    ax2.set_title('Bottom 5 Sets - Comparação de Custo e Lucro')
    ax2.set_xticks([r + barWidth/2 for r in range(len(set_names[-5:]))])
    ax2.set_xticklabels(set_names[-5:], rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('set_analysis.png', dpi=300, bbox_inches='tight')

    print("\nAnálise Detalhada dos Sets:")
    print("-" * 100)
    print(f"{'Nome do Set':<30} {'Custo (USD)':<15} {
          'Valor Médio (USD)':<15} {'Lucro (USD)':<15} {'Margem (%)':<15}")
    print("-" * 100)

    print("\nTop 5 Sets:")
    print("-" * 100)
    for set_name, metrics in top_sets[:5]:
        print(f"{set_name:<30} {metrics['avg_cost']:<15.2f} {metrics['avg_value']:<15.2f} {
              metrics['profit']:<15.2f} {metrics['profit_margin']:<15.2f}")

    print("\nBottom 5 Sets:")
    print("-" * 100)
    for set_name, metrics in top_sets[-5:]:
        print(f"{set_name:<30} {metrics['avg_cost']:<15.2f} {metrics['avg_value']:<15.2f} {
              metrics['profit']:<15.2f} {metrics['profit_margin']:<15.2f}")


def main():
    set_prices = fetch_set_prices()

    if not set_prices:
        print("No set prices found.")
        return

    top_sets = monte_carlo_simulation(set_prices)

    if not top_sets:
        print("No sets found after simulation.")
        return

    plot_top_sets(top_sets)


if __name__ == "__main__":
    main()
