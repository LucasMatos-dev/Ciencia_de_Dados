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

# Alunos com notas ABAIXO da normalidade
df_baixo = df[df['NU_NOTA_LC'] < lower_bound]
# Alunos com notas ACIMA da normalidade
df_alto = df[df['NU_NOTA_LC'] > upper_bound]

faixa_baixo = df_baixo['TP_FAIXA_ETARIA'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
faixa_baixo.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Distribuição da Faixa Etária - Notas ABAIXO da Normalidade')
plt.xlabel('Faixa Etária (TP_FAIXA_ETARIA)')
plt.ylabel('Contagem de Alunos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("resultados/faixa_etaria_abaixo.png")
plt.show()

print("Distribuição da Faixa Etária para alunos com notas ABAIXO da normalidade:")
print(faixa_baixo)

faixa_alto = df_alto['TP_FAIXA_ETARIA'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
faixa_alto.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Distribuição da Faixa Etária - Notas ACIMA da Normalidade')
plt.xlabel('Faixa Etária (TP_FAIXA_ETARIA)')
plt.ylabel('Contagem de Alunos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("resultados/faixa_etaria_acima.png")
plt.show()

print("Distribuição da Faixa Etária para alunos com notas ACIMA da normalidade:")
print(faixa_alto)
