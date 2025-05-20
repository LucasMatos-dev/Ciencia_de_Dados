import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import pickle
import scipy.stats as stats

# Create plots directory if it doesn't exist
plots_dir = "plots_trabalho"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# 1. Download dos dados e limpeza
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
data = data.dropna()

# 2. Engenharia de características
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

features = np.column_stack([ma5, ma10, vol5, volume_aligned, returns_aligned])

# 3. Labels: variação percentual do preço no dia seguinte
price_today = close[-min_len-1:-1]
price_next_day = close[-min_len:]
price_change = (price_next_day - price_today) / price_today

# 4. Normalização
scaler_X = StandardScaler()
X = scaler_X.fit_transform(features)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(price_change.reshape(-1, 1))

# 5. Divisão treino-teste (sem embaralhar para manter ordem temporal)
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=False)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 6. Parâmetros da MLP
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16
output_size = 1
lr = 0.001
momentum = 0.9
epochs = 300

np.random.seed(42)
# Initialize weights and biases for all layers
W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0/input_size)
b1 = np.zeros((1, hidden_size1))
W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0/hidden_size1)
b2 = np.zeros((1, hidden_size2))
W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2.0/hidden_size2)
b3 = np.zeros((1, hidden_size3))
W4 = np.random.randn(hidden_size3, output_size) * np.sqrt(2.0/hidden_size3)
b4 = np.zeros((1, output_size))

# Initialize momentum terms
vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)
vW3 = np.zeros_like(W3)
vb3 = np.zeros_like(b3)
vW4 = np.zeros_like(W4)
vb4 = np.zeros_like(b4)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# 7. Loop de treino
losses = []
best_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(epochs):
    # Forward pass
    z1 = X_train @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = relu(z3)
    y_pred = a3 @ W4 + b4

    loss = mse_loss(y_train, y_pred)
    losses.append(loss)

    # Early stopping check
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Backpropagation with gradient clipping
    dloss = mse_grad(y_train, y_pred)
    
    # Layer 4
    dW4 = a3.T @ dloss
    db4 = np.sum(dloss, axis=0, keepdims=True)
    
    # Layer 3
    da3 = dloss @ W4.T
    dz3 = da3 * relu_deriv(z3)
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0, keepdims=True)
    
    # Layer 2
    da2 = dz3 @ W3.T
    dz2 = da2 * relu_deriv(z2)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    # Layer 1
    da1 = dz2 @ W2.T
    dz1 = da1 * relu_deriv(z1)
    dW1 = X_train.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Gradient clipping
    max_norm = 1.0
    for dW in [dW1, dW2, dW3, dW4]:
        norm = np.linalg.norm(dW)
        if norm > max_norm:
            dW *= max_norm / norm

    # Update with momentum
    vW1 = momentum * vW1 - lr * dW1
    vb1 = momentum * vb1 - lr * db1
    vW2 = momentum * vW2 - lr * dW2
    vb2 = momentum * vb2 - lr * db2
    vW3 = momentum * vW3 - lr * dW3
    vb3 = momentum * vb3 - lr * db3
    vW4 = momentum * vW4 - lr * dW4
    vb4 = momentum * vb4 - lr * db4

    W1 += vW1
    b1 += vb1
    W2 += vW2
    b2 += vb2
    W3 += vW3
    b3 += vb3
    W4 += vW4
    b4 += vb4

    if epoch % 30 == 0:
        print(f"Época {epoch}, MSE Loss: {loss:.6f}")

# 8. Avaliação no conjunto de teste
z1_test = X_test @ W1 + b1
a1_test = relu(z1_test)
z2_test = a1_test @ W2 + b2
a2_test = relu(z2_test)
z3_test = a2_test @ W3 + b3
a3_test = relu(z3_test)
y_test_pred = a3_test @ W4 + b4

test_loss = mse_loss(y_test, y_test_pred)
print(f"\nMSE Loss no teste: {test_loss:.6f}")

# 9. Salvar o modelo treinado
# Criar diretório para salvar o modelo se não existir
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Salvar os parâmetros do modelo
np.save(os.path.join(model_dir, "W1.npy"), W1)
np.save(os.path.join(model_dir, "b1.npy"), b1)
np.save(os.path.join(model_dir, "W2.npy"), W2)
np.save(os.path.join(model_dir, "b2.npy"), b2)
np.save(os.path.join(model_dir, "W3.npy"), W3)
np.save(os.path.join(model_dir, "b3.npy"), b3)
np.save(os.path.join(model_dir, "W4.npy"), W4)
np.save(os.path.join(model_dir, "b4.npy"), b4)

# Salvar os scalers
with open(os.path.join(model_dir, "scaler_X.pkl"), "wb") as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(model_dir, "scaler_y.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)

# Salvar metadados do modelo
model_metadata = {
    "input_size": input_size,
    "hidden_size1": hidden_size1,
    "hidden_size2": hidden_size2,
    "hidden_size3": hidden_size3,
    "output_size": output_size,
    "activation": "relu",
    "loss": "mse",
    "best_loss": float(best_loss),
    "final_test_loss": float(test_loss),
    "losses": [float(loss) for loss in losses]  # Convert losses to float for JSON serialization
}

with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
    json.dump(model_metadata, f, indent=4)

print("\nModelo salvo com sucesso no diretório 'model'")

# 10. Inverter escala para valores originais
y_test_pred_actual = scaler_y.inverse_transform(y_test_pred)
y_test_actual = scaler_y.inverse_transform(y_test)

# 11. Alinhar datas com os labels
dates_all = data.index[-(len(price_change) + 1):-1]

dates_test = dates_all[test_idx]

sorted_indices = np.argsort(dates_test)
dates_test_sorted = dates_test[sorted_indices]
y_test_actual_sorted = y_test_actual[sorted_indices].flatten()
y_test_pred_sorted = y_test_pred_actual[sorted_indices].flatten()

# 12. Gráfico Preço Real vs Previsto
plt.figure(figsize=(14,6))
plt.plot(dates_test_sorted, y_test_actual_sorted, label="Variação Real (%)", linewidth=2)
plt.plot(dates_test_sorted, y_test_pred_sorted, label="Variação Prevista (%)", linewidth=2, linestyle='--')
plt.title("Variação Real vs Prevista do Preço no Dia Seguinte da ação(AAPL)")
plt.xlabel("Data")
plt.ylabel("Variação do Preço (%)")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.savefig(os.path.join(plots_dir, '1_variacao_real_vs_prevista.png'), dpi=300, bbox_inches='tight')
plt.close()

# 13. Resíduos e média móvel
residuals = y_test_actual_sorted - y_test_pred_sorted
rolling_window = 5
residuals_smooth = pd.Series(residuals).rolling(window=rolling_window).mean()

plt.figure(figsize=(14,5))
plt.plot(dates_test_sorted, residuals, label='Resíduos', color='red', alpha=0.6)
plt.plot(dates_test_sorted, residuals_smooth, label=f'Média Móvel ({rolling_window} dias)', color='blue', linewidth=2)
plt.axhline(0, color='black', linestyle='--')
plt.title("Resíduos das Previsões ao Longo do Tempo")
plt.xlabel("Data")
plt.ylabel("Resíduo (Real - Previsto)")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.savefig(os.path.join(plots_dir, '2_residuos_media_movel.png'), dpi=300, bbox_inches='tight')
plt.close()

# 14. Histograma dos resíduos
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=30, color='purple', alpha=0.7)
plt.title("Distribuição dos Resíduos das Previsões")
plt.xlabel("Resíduo")
plt.ylabel("Frequência")
plt.grid(True)
plt.savefig(os.path.join(plots_dir, '3_histograma_residuos.png'), dpi=300, bbox_inches='tight')
plt.close()

# 15. Gráfico da perda durante o treinamento
plt.figure(figsize=(8,4))
plt.plot(losses)
plt.title("Erro Quadrático Médio (MSE) Durante o Treinamento")
plt.xlabel("Época")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig(os.path.join(plots_dir, '4_perda_treinamento.png'), dpi=300, bbox_inches='tight')
plt.close()

# 16. Intervalo de Confiança das Previsões
std_dev = np.std(y_test_pred_sorted)
confidence_interval = 1.96 * std_dev  # intervalo de confiança de 95%

plt.figure(figsize=(14,6))
plt.plot(dates_test_sorted, y_test_pred_sorted, 
         label='Variação Prevista', color='blue', linewidth=2)
plt.fill_between(dates_test_sorted, 
                 y_test_pred_sorted - confidence_interval,
                 y_test_pred_sorted + confidence_interval,
                 color='blue', alpha=0.2, label='Intervalo de Confiança 95%')
plt.title("Previsões com Intervalo de Confiança")
plt.xlabel("Data")
plt.ylabel("Variação Prevista (%)")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.savefig(os.path.join(plots_dir, '5_intervalo_confianca.png'), dpi=300, bbox_inches='tight')
plt.close()

# 17. Tendência das Previsões
x = np.arange(len(y_test_pred_sorted))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_test_pred_sorted)
trend_line = slope * x + intercept

plt.figure(figsize=(14,6))
plt.scatter(dates_test_sorted, y_test_pred_sorted, 
            label='Previsões', color='blue', alpha=0.6)
plt.plot(dates_test_sorted, trend_line, 
         label=f'Tendência (inclinação: {slope:.4f})', color='red', linewidth=2)
plt.title("Tendência das Previsões")
plt.xlabel("Data")
plt.ylabel("Variação Prevista (%)")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.savefig(os.path.join(plots_dir, '6_tendencia_previsoes.png'), dpi=300, bbox_inches='tight')
plt.close()

# 18. Box Plot das Previsões por Dia da Semana
test_results = pd.DataFrame({
    'Data': dates_test_sorted,
    'Variação_Prevista(%)': y_test_pred_sorted
})
test_results['Dia_Semana'] = test_results['Data'].dt.day_name()

# Ensure we have data for all days of the week
plt.figure(figsize=(10,6))
test_results.boxplot(column='Variação_Prevista(%)', by='Dia_Semana', rot=45)
plt.title("Distribuição das Previsões por Dia da Semana")
plt.suptitle("")  # Remove título automático
plt.xlabel("Dia da Semana")
plt.ylabel("Variação Prevista (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '7_boxplot_dia_semana.png'), dpi=300, bbox_inches='tight')
plt.close()

# 19. Gráfico de Dispersão das Previsões
plt.figure(figsize=(8,4))
plt.scatter(range(len(y_test_pred_sorted)), y_test_pred_sorted, 
            color='green', alpha=0.6)
plt.title("Dispersão das Previsões de Variação")
plt.xlabel("Índice")
plt.ylabel("Variação Prevista (%)")
plt.grid(True)
plt.savefig(os.path.join(plots_dir, '8_dispersao_previsoes.png'), dpi=300, bbox_inches='tight')
plt.close()

# 20. Gráfico de Barras das Previsões por Dia da Semana
weekly_avg = test_results.groupby('Dia_Semana')['Variação_Prevista(%)'].mean()
plt.figure(figsize=(10,6))
weekly_avg.plot(kind='bar', color='skyblue')
plt.title("Média das Previsões por Dia da Semana")
plt.xlabel("Dia da Semana")
plt.ylabel("Variação Prevista Média (%)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '9_barras_dia_semana.png'), dpi=300, bbox_inches='tight')
plt.close()

# 21. Gráfico de Barras das Previsões Positivas vs Negativas
positive_predictions = (y_test_pred_sorted > 0).sum()
negative_predictions = (y_test_pred_sorted <= 0).sum()

plt.figure(figsize=(10, 6))
categories = ['Previsões Positivas', 'Previsões Negativas']
values = [positive_predictions, negative_predictions]
colors = ['green', 'red']

bars = plt.bar(categories, values, color=colors)
plt.title("Distribuição de Previsões Positivas e Negativas")
plt.ylabel("Número de Previsões")

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

# Add percentage labels
total = positive_predictions + negative_predictions
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = (height / total) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height/2,
             f'{percentage:.1f}%',
             ha='center', va='center', color='white', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '10_previsoes_positivas_negativas.png'), dpi=300, bbox_inches='tight')
plt.close()
