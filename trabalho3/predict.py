import numpy as np
import pickle
import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt

# Create plots directory if it doesn't exist
plots_dir = "plots_predict"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def load_model(model_dir="model"):
    """Carrega o modelo salvo e seus componentes."""
    # Carrega metadados
    with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Carrega pesos e vieses
    W1 = np.load(os.path.join(model_dir, "W1.npy"))
    b1 = np.load(os.path.join(model_dir, "b1.npy"))
    W2 = np.load(os.path.join(model_dir, "W2.npy"))
    b2 = np.load(os.path.join(model_dir, "b2.npy"))
    W3 = np.load(os.path.join(model_dir, "W3.npy"))
    b3 = np.load(os.path.join(model_dir, "b3.npy"))
    W4 = np.load(os.path.join(model_dir, "W4.npy"))
    b4 = np.load(os.path.join(model_dir, "b4.npy"))
    
    # Carrega normalizadores
    with open(os.path.join(model_dir, "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(model_dir, "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
    
    return (W1, b1, W2, b2, W3, b3, W4, b4), (scaler_X, scaler_y), metadata

def prepare_features(data):
    """Prepara características a partir dos dados brutos da ação."""
    close = data['Close'].values.flatten()
    volume = data['Volume'].values.flatten()
    returns = np.diff(close) / close[:-1]  # retornos diários
    
    def moving_average(arr, window):
        return np.convolve(arr, np.ones(window) / window, mode='valid')
    
    def rolling_std(arr, window):
        result = []
        for i in range(len(arr) - window + 1):
            result.append(np.std(arr[i:i+window]))
        return np.array(result)
    
    ma5 = moving_average(close, 5)
    ma10 = moving_average(close, 10)
    vol5 = rolling_std(returns, 5)
    
    min_len = min(len(ma5), len(ma10), len(vol5))
    ma5 = ma5[-min_len:]
    ma10 = ma10[-min_len:]
    vol5 = vol5[-min_len:]
    volume_aligned = volume[-min_len:]
    returns_aligned = returns[-min_len:]
    
    # Normaliza volume para escala similar aos outros features
    volume_normalized = volume_aligned / np.mean(volume_aligned)
    
    features = np.column_stack([ma5, ma10, vol5, volume_normalized, returns_aligned])
    return features, data.index[-min_len:]

def predict(X, model_params, scalers):
    """Faz previsões usando o modelo carregado."""
    W1, b1, W2, b2, W3, b3, W4, b4 = model_params
    scaler_X, scaler_y = scalers
    
    # Normaliza entrada
    X_scaled = scaler_X.transform(X)
    
    # Passo forward
    z1 = X_scaled @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2 + b2
    a2 = np.maximum(0, z2)  # ReLU
    z3 = a2 @ W3 + b3
    a3 = np.maximum(0, z3)  # ReLU
    y_pred = a3 @ W4 + b4
    
    # Inverte normalização da previsão
    y_pred_actual = scaler_y.inverse_transform(y_pred)
    
    return y_pred_actual

def main():
    # Carrega o modelo
    try:
        model_params, scalers, metadata = load_model()
        print("Modelo carregado com sucesso!")
        print("\nMetadados do modelo:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    # Baixa dados recentes
    ticker = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Obtém dados do último ano
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("Erro ao baixar dados da ação")
        return
    
    print(f"\nDados baixados de {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
    print(f"Total de dias: {len(data)}")
    
    # Prepara características
    features, dates = prepare_features(data)
    
    if len(features) == 0:
        print("Não há dados suficientes para fazer previsões")
        return
    
    # Faz previsões
    predictions = predict(features, model_params, scalers)
    
    # Calcula variações reais
    close_prices = data['Close'].values.flatten()  # Ensure 1D array
    actual_changes = np.diff(close_prices) / close_prices[:-1] * 100  # Variação percentual real
    
    # Alinha as datas e previsões corretamente
    dates = dates[1:]  # Remove primeiro dia pois não temos variação para ele
    predictions = predictions[:-1]  # Remove última previsão pois não temos valor real para ela
    
    # Cria DataFrame com resultados
    results = pd.DataFrame({
        'Data': dates,
        'Variação_Prevista(%)': predictions.flatten(),
        'Variação_Real(%)': actual_changes[-len(predictions):]  # Alinha com as previsões
    })
    
    # Imprime estatísticas básicas
    print("\nEstatísticas das Variações:")
    print("\nVariações Reais:")
    print(results['Variação_Real(%)'].describe())
    print("\nVariações Previstas:")
    print(results['Variação_Prevista(%)'].describe())
    
    # Imprime previsões e comparação
    print("\nPrevisões vs Valores Reais:")
    print(results.to_string(index=False))
    
    # Calcula métricas de erro
    mse = np.mean((results['Variação_Real(%)'] - results['Variação_Prevista(%)'])**2)
    mae = np.mean(np.abs(results['Variação_Real(%)'] - results['Variação_Prevista(%)']))
    print(f"\nMétricas de Erro:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Plota previsões vs valores reais
    # 1. Gráfico Preço Real vs Previsto
    plt.figure(figsize=(14,6))
    plt.plot(results['Data'], results['Variação_Real(%)'], 
             label="Variação Real (%)", linewidth=2)
    plt.plot(results['Data'], results['Variação_Prevista(%)'], 
             label="Variação Prevista (%)", linewidth=2, linestyle='--')
    plt.title("Variação Real vs Prevista do Preço da ação(AAPL)")
    plt.xlabel("Data")
    plt.ylabel("Variação do Preço (%)")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(plots_dir, '1_variacao_real_vs_prevista.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Resíduos e média móvel
    residuals = results['Variação_Real(%)'] - results['Variação_Prevista(%)']
    rolling_window = 5
    residuals_smooth = pd.Series(residuals).rolling(window=rolling_window).mean()
    
    plt.figure(figsize=(14,5))
    plt.plot(results['Data'], residuals, label='Resíduos', color='red', alpha=0.6)
    plt.plot(results['Data'], residuals_smooth, 
             label=f'Média Móvel ({rolling_window} dias)', color='blue', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Resíduos das Previsões ao Longo do Tempo")
    plt.xlabel("Data")
    plt.ylabel("Resíduo (Real - Previsto)")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(plots_dir, '2_residuos_media_movel.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Histograma dos resíduos
    plt.figure(figsize=(8,4))
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title("Distribuição dos Resíduos das Previsões")
    plt.xlabel("Resíduo")
    plt.ylabel("Frequência")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, '3_histograma_residuos.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Gráfico de dispersão Real vs Previsto
    plt.figure(figsize=(8,8))
    plt.scatter(results['Variação_Real(%)'], results['Variação_Prevista(%)'], 
                color='blue', alpha=0.6)
    plt.plot([results['Variação_Real(%)'].min(), results['Variação_Real(%)'].max()],
             [results['Variação_Real(%)'].min(), results['Variação_Real(%)'].max()],
             'r--', label='Linha de Referência')
    plt.title("Dispersão: Valores Reais vs Previstos")
    plt.xlabel("Variação Real (%)")
    plt.ylabel("Variação Prevista (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, '4_dispersao_real_vs_previsto.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Box Plot das Previsões vs Valores Reais
    plt.figure(figsize=(10,6))
    results.boxplot(column=['Variação_Real(%)', 'Variação_Prevista(%)'])
    plt.title("Distribuição das Variações Reais vs Previstas")
    plt.ylabel("Variação (%)")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, '5_boxplot_real_vs_previsto.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Gráfico de Barras das Previsões vs Valores Reais por Dia da Semana
    results['Dia_Semana'] = results['Data'].dt.day_name()
    weekly_avg = results.groupby('Dia_Semana')[['Variação_Real(%)', 'Variação_Prevista(%)']].mean()
    
    plt.figure(figsize=(12,6))
    weekly_avg.plot(kind='bar')
    plt.title("Média das Variações Reais vs Previstas por Dia da Semana")
    plt.xlabel("Dia da Semana")
    plt.ylabel("Variação Média (%)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '6_barras_dia_semana.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Intervalo de Confiança das Previsões
    std_dev = np.std(results['Variação_Prevista(%)'])
    confidence_interval = 1.96 * std_dev  # intervalo de confiança de 95%

    plt.figure(figsize=(14,6))
    plt.plot(results['Data'], results['Variação_Prevista(%)'], 
             label='Variação Prevista', color='blue', linewidth=2)
    plt.fill_between(results['Data'], 
                     results['Variação_Prevista(%)'] - confidence_interval,
                     results['Variação_Prevista(%)'] + confidence_interval,
                     color='blue', alpha=0.2, label='Intervalo de Confiança 95%')
    plt.title("Previsões com Intervalo de Confiança")
    plt.xlabel("Data")
    plt.ylabel("Variação Prevista (%)")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(plots_dir, '7_intervalo_confianca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Tendência das Previsões
    x = np.arange(len(results['Variação_Prevista(%)']))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, results['Variação_Prevista(%)'])
    trend_line = slope * x + intercept

    plt.figure(figsize=(14,6))
    plt.scatter(results['Data'], results['Variação_Prevista(%)'], 
                label='Previsões', color='blue', alpha=0.6)
    plt.plot(results['Data'], trend_line, 
             label=f'Tendência (inclinação: {slope:.4f})', color='red', linewidth=2)
    plt.title("Tendência das Previsões")
    plt.xlabel("Data")
    plt.ylabel("Variação Prevista (%)")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(plots_dir, '8_tendencia_previsoes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Gráfico de Barras com Distribuição de Erros
    plt.figure(figsize=(12, 6))
    error_distribution = pd.cut(
        abs(results['Variação_Real(%)'] - results['Variação_Prevista(%)']),
        bins=[0, 0.5, 1.0, 2.0, float('inf')],
        labels=['≤ 0.5%', '0.5-1%', '1-2%', '> 2%']
    ).value_counts().sort_index()
    
    error_distribution.plot(kind='bar', color=['#2ecc71', '#3498db', '#f1c40f', '#e74c3c'])
    plt.title("Distribuição dos Erros de Previsão")
    plt.xlabel("Magnitude do Erro")
    plt.ylabel("Número de Previsões")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '9_distribuicao_erros.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 