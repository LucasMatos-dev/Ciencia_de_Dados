import pandas as pd
import matplotlib.pyplot as plt

#Leitura e pré-processamento dos dados
df = pd.read_csv("MICRODADOS_ENEM_2023.csv", sep=";", encoding="latin-1")
df['NU_NOTA_LC'] = pd.to_numeric(df['NU_NOTA_LC'], errors='coerce')
df = df.dropna(subset=['NU_NOTA_LC'])

ordem_escolaridade = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
df['Q001'] = pd.Categorical(df['Q001'], categories=ordem_escolaridade + ['H'], ordered=True)
df['Q002'] = pd.Categorical(df['Q002'], categories=ordem_escolaridade + ['H'], ordered=True)

def combine_escolaridade(row):
    pai = row['Q001']
    mae = row['Q002']
    if ((pd.isna(pai) or pai == 'H') and (pd.isna(mae) or mae == 'H')):
        return pd.NA
    if pd.isna(pai) or pai == 'H':
        return mae
    if pd.isna(mae) or mae == 'H':
        return pai
    return pai if pai > mae else mae

df['ESCOLARIDADE_PAIS'] = df.apply(combine_escolaridade, axis=1)
df['ESCOLARIDADE_PAIS'] = pd.Categorical(df['ESCOLARIDADE_PAIS'], categories=ordem_escolaridade, ordered=True)

df_notas = df['NU_NOTA_LC']
Q1_nota = df_notas.quantile(0.25)
Q3_nota = df_notas.quantile(0.75)
IQR = Q3_nota - Q1_nota
lower_bound = Q1_nota - 1.5 * IQR
upper_bound = Q3_nota + 1.5 * IQR

df_baixo = df[df['NU_NOTA_LC'] < lower_bound]  # Alunos com notas ABAIXO da normalidade
df_alto = df[df['NU_NOTA_LC'] > upper_bound]    # Alunos com notas ACIMA da normalidade

print("Número de alunos com notas ABAIXO da normalidade:", df_baixo.shape[0])
print("Número de alunos com notas ACIMA da normalidade:", df_alto.shape[0])

escolaridade_baixo = df_baixo['ESCOLARIDADE_PAIS'].dropna()
contagem_escolaridade_baixo = escolaridade_baixo.value_counts().sort_index()

plt.figure(figsize=(8, 6))
contagem_escolaridade_baixo.plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Distribuição da Escolaridade dos Pais\n(Notas ABAIXO da Normalidade)")
plt.xlabel("Categoria de Escolaridade")
plt.ylabel("Número de Alunos")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("resultados/escolaridade_pais_abaixo.png")
plt.show()

print("Distribuição da Escolaridade dos Pais para alunos com notas ABAIXO da normalidade:")
print(contagem_escolaridade_baixo)

escolaridade_alto = df_alto['ESCOLARIDADE_PAIS'].dropna()
contagem_escolaridade_alto = escolaridade_alto.value_counts().sort_index()

plt.figure(figsize=(8, 6))
contagem_escolaridade_alto.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title("Distribuição da Escolaridade dos Pais\n(Notas ACIMA da Normalidade)")
plt.xlabel("Categoria de Escolaridade")
plt.ylabel("Número de Alunos")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("resultados/escolaridade_pais_acima.png")
plt.show()

print("Distribuição da Escolaridade dos Pais para alunos com notas ACIMA da normalidade:")
print(contagem_escolaridade_alto)
