import pandas as pd
import matplotlib.pyplot as plt

# Carregar os microdados do ENEM 2023 (substituir pelo caminho correto do arquivo CSV)
arquivo = "MICRODADOS_ENEM_2023.csv"
df = pd.read_csv(arquivo, sep=';', encoding="latin-1")


# Remover valores nulos
df = df.dropna(subset=["NU_NOTA_LC"])

# Definir outliers usando o método do IQR
Q1 = df["NU_NOTA_LC"].quantile(0.25)
Q3 = df["NU_NOTA_LC"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar alunos com notas atípicas
atipicos_baixo = df[df["NU_NOTA_LC"] < limite_inferior]
atipicos_alto = df[df["NU_NOTA_LC"] > limite_superior]

# Análise 9: Municípios de aplicação da prova entre notas atípicas (Histograma)
# Contar a frequência dos municípios com notas baixas
municipios_baixos = atipicos_baixo["NO_MUNICIPIO_PROVA"].value_counts().head(
    20)

# Plotar gráfico de barras
plt.figure(figsize=(12, 6))
municipios_baixos.plot(kind='bar', color='tomato',
                       edgecolor='black', alpha=0.8)
plt.title("Top 20 Municípios com Mais Notas Baixas em Linguagens")
plt.xlabel("Município")
plt.ylabel("Quantidade de Alunos com Nota Baixa")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("resultados/outliers_baixos_linguagem_municipios.png")
plt.show()

# Contar a frequência dos municípios com notas altas
municipios_altos = atipicos_alto["NO_MUNICIPIO_PROVA"].value_counts().head(20)

# Plotar gráfico de barras
plt.figure(figsize=(12, 6))
municipios_altos.plot(kind='bar', color='seagreen',
                      edgecolor='black', alpha=0.8)
plt.title("Top 20 Municípios com Mais Notas Altas em Linguagens")
plt.xlabel("Município")
plt.ylabel("Quantidade de Alunos com Nota Alta")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("resultados/outliers_altos_linguagem_municipios.png")
plt.show()
