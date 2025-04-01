import pandas as pd
import matplotlib.pyplot as plt

#Leitura e pré-processamento dos dados
df = pd.read_csv("MICRODADOS_ENEM_2023.csv", sep=";", encoding="latin-1")

#Converter a nota de Linguagens para numérico e remover registros com NaN
df['NU_NOTA_LC'] = pd.to_numeric(df['NU_NOTA_LC'], errors='coerce')
df = df.dropna(subset=['NU_NOTA_LC'])

renda_col = 'Q006'
ordem_renda = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q']
df[renda_col] = pd.Categorical(df[renda_col], categories=ordem_renda, ordered=True)

df_notas = df['NU_NOTA_LC']
Q1 = df_notas.quantile(0.25)
Q3 = df_notas.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_baixo = df[df['NU_NOTA_LC'] < lower_bound]  # Notas abaixo da normalidade
df_alto = df[df['NU_NOTA_LC'] > upper_bound]    # Notas acima da normalidade

print("Número de alunos com notas ABAIXO da normalidade:", df_baixo.shape[0])
print("Número de alunos com notas ACIMA da normalidade:", df_alto.shape[0])

renda_baixo = df_baixo[renda_col].dropna()

contagem_renda_baixo = renda_baixo.value_counts().sort_index()

plt.figure(figsize=(8, 6))
contagem_renda_baixo.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Distribuição da Renda Mensal Familiar\n(Notas ABAIXO da Normalidade)')
plt.xlabel('Categoria de Renda')
plt.ylabel('Número de Alunos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("resultados/renda_abaixo_categorica.png")
plt.show()

print("Distribuição da Renda Mensal Familiar para alunos com notas ABAIXO da normalidade:")
print(contagem_renda_baixo)

renda_alto = df_alto[renda_col].dropna()

contagem_renda_alto = renda_alto.value_counts().sort_index()

plt.figure(figsize=(8, 6))
contagem_renda_alto.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Distribuição da Renda Mensal Familiar\n(Notas ACIMA da Normalidade)')
plt.xlabel('Categoria de Renda')
plt.ylabel('Número de Alunos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("resultados/renda_acima_categorica.png")
plt.show()

print("Distribuição da Renda Mensal Familiar para alunos com notas ACIMA da normalidade:")
print(contagem_renda_alto)
