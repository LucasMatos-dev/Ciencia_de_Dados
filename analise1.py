import pandas as pd
import matplotlib.pyplot as plt

#Leitura e pré-processamento dos dados
df = pd.read_csv("MICRODADOS_ENEM_2023.csv", sep=";", encoding="latin-1")
df['NU_NOTA_LC'] = pd.to_numeric(df['NU_NOTA_LC'], errors='coerce')
df_notas = df['NU_NOTA_LC'].dropna()  # Remove valores nulos

Q1 = df_notas.quantile(0.25)
Q3 = df_notas.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR 
upper_bound = Q3 + 1.5 * IQR 

outliers_baixo = df_notas[df_notas < lower_bound]
outliers_alto = df_notas[df_notas > upper_bound]

plt.figure(figsize=(10, 6))
plt.hist(df_notas, bins=30, color='lightblue', edgecolor='black')
plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=1.5, label='Limite Inferior')
plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=1.5, label='Limite Superior')
plt.title('Distribuição das Notas de Linguagens, Códigos e suas Tecnologias')
plt.xlabel('Nota')
plt.ylabel('Frequência')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("resultados/histograma_notas_lc.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(df_notas, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Boxplot das Notas de Linguagens, Códigos e suas Tecnologias')
plt.xlabel('Nota')
plt.savefig("resultados/boxplot_notas_lc.png")
plt.show()

print("Estatísticas gerais da coluna 'NU_NOTA_LC':")
print(df_notas.describe())

# Valores atípicos ABAIXO da normalidade
print("\nValores atípicos ABAIXO da normalidade:")
print(f"Quantidade: {len(outliers_baixo)}")
if len(outliers_baixo) > 0:
    print(outliers_baixo.describe())

# Valores atípicos ACIMA da normalidade
print("\nValores atípicos ACIMA da normalidade:")
print(f"Quantidade: {len(outliers_alto)}")
if len(outliers_alto) > 0:
    print(outliers_alto.describe())
