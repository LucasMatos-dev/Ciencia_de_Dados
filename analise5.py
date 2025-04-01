import pandas as pd
import matplotlib.pyplot as plt

#Leitura e pré-processamento dos dados
df = pd.read_csv("MICRODADOS_ENEM_2023.csv", sep=";", encoding="latin-1")

df['NU_NOTA_LC'] = pd.to_numeric(df['NU_NOTA_LC'], errors='coerce')
df = df.dropna(subset=['NU_NOTA_LC'])


df_notas = df['NU_NOTA_LC']
Q1 = df_notas.quantile(0.25)
Q3 = df_notas.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_baixo = df[df['NU_NOTA_LC'] < lower_bound]  # Notas ABAIXO da normalidade
df_alto = df[df['NU_NOTA_LC'] > upper_bound]    # Notas ACIMA da normalidade

print("Número de alunos com notas ABAIXO da normalidade:", df_baixo.shape[0])
print("Número de alunos com notas ACIMA da normalidade:", df_alto.shape[0])

acesso_internet_col = 'Q025'
ordem_internet = ['A', 'B']
df[acesso_internet_col] = pd.Categorical(df[acesso_internet_col], categories=ordem_internet, ordered=True)

acesso_baixo = df_baixo[acesso_internet_col].dropna()
contagem_internet_baixo = acesso_baixo.value_counts().sort_index()

plt.figure(figsize=(8, 6))
ax = contagem_internet_baixo.plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Distribuição do Acesso à Internet\n(Notas ABAIXO da Normalidade)")
plt.xlabel("Acesso à Internet")
plt.ylabel("Número de Alunos")
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticklabels(['Não', 'Sim'], rotation=0)
plt.savefig("resultados/internet_abaixo.png")
plt.show()

print("Distribuição do Acesso à Internet para alunos com notas ABAIXO da normalidade:")
print(contagem_internet_baixo)

acesso_alto = df_alto[acesso_internet_col].dropna()
contagem_internet_alto = acesso_alto.value_counts().sort_index()

plt.figure(figsize=(8, 6))
ax = contagem_internet_alto.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title("Distribuição do Acesso à Internet\n(Notas ACIMA da Normalidade)")
plt.xlabel("Acesso à Internet")
plt.ylabel("Número de Alunos")
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticklabels(['Não', 'Sim'], rotation=0)
plt.savefig("resultados/internet_acima.png")
plt.show()

print("Distribuição do Acesso à Internet para alunos com notas ACIMA da normalidade:")
print(contagem_internet_alto)
