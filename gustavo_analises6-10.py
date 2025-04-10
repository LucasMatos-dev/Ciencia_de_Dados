import numpy as np
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

def plot_boxplot(data, column, title):
    plt.figure(figsize=(10, 5))
    data.boxplot(column="NU_NOTA_LC", by=column, vert=False, grid=False)
    plt.title(title)
    plt.suptitle("")
    plt.xlabel("Nota em Linguagens, Códigos e suas Tecnologias")
    plt.ylabel(column)
    plt.show()

# # Análise 6: Tipo de escola entre notas atípicas
# plot_boxplot(atipicos_baixo, "TP_ESCOLA", "Distribuição do Tipo de Escola - Notas Baixas")
# plot_boxplot(atipicos_alto, "TP_ESCOLA", "Distribuição do Tipo de Escola - Notas Altas")

# # Análise 7: Localização da escola entre notas atípicas
# plot_boxplot(atipicos_baixo, "TP_LOCALIZACAO_ESC", "Distribuição da Localização da Escola - Notas Baixas")
# plot_boxplot(atipicos_alto, "TP_LOCALIZACAO_ESC", "Distribuição da Localização da Escola - Notas Altas")

# # Análise 8: UF da escola entre notas atípicas
# plot_boxplot(atipicos_baixo, "SG_UF_ESC", "Distribuição da UF da Escola - Notas Baixas")
# plot_boxplot(atipicos_alto, "SG_UF_ESC", "Distribuição da UF da Escola - Notas Altas")

# Análise 9: Distribuição da quantidade de alunos com notas atípicas por município (Histograma)

# # Contagem de alunos com notas atípicas por município
# municipios_abaixo = atipicos_baixo["NO_MUNICIPIO_PROVA"].value_counts()
# municipios_acima = atipicos_alto["NO_MUNICIPIO_PROVA"].value_counts()

# # Histograma - Notas Baixas
# plt.figure(figsize=(12, 6))
# plt.hist(municipios_abaixo.values, bins=30, color="salmon", edgecolor="black")
# plt.title("Distribuição de Municípios por Quantidade de Alunos com Notas Atípicas Baixas", fontsize=14)
# plt.xlabel("Quantidade de Alunos com Notas Baixas por Município", fontsize=12)
# plt.ylabel("Número de Municípios", fontsize=12)
# plt.grid(axis="y", linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()

# # Histograma - Notas Altas
# plt.figure(figsize=(12, 6))
# plt.hist(municipios_acima.values, bins=30, color="lightblue", edgecolor="black")
# plt.title("Distribuição de Municípios por Quantidade de Alunos com Notas Atípicas Altas", fontsize=14)
# plt.xlabel("Quantidade de Alunos com Notas Altas por Município", fontsize=12)
# plt.ylabel("Número de Municípios", fontsize=12)
# plt.grid(axis="y", linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()


# Análise 10: Comparação da quantidade de notas atípicas por município (Altas vs Baixas)

# # Contar número de alunos com notas atípicas por município
# contagem_baixo = atipicos_baixo["NO_MUNICIPIO_PROVA"].value_counts()
# contagem_alto = atipicos_alto["NO_MUNICIPIO_PROVA"].value_counts()

# comparacao = pd.DataFrame({
#     "Notas_Baixas": contagem_baixo,
#     "Notas_Altas": contagem_alto
# }).fillna(0)

# # Valores de X e Y
# x = comparacao["Notas_Baixas"]
# y = comparacao["Notas_Altas"]

# # Ajustar linha de tendência (polinomial de 1º grau → reta)
# coef = np.polyfit(x, y, 1)
# linha_tendencia = np.poly1d(coef)

# # Gráfico
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color="mediumseagreen", alpha=0.6, edgecolor="black", label="Municípios")
# plt.plot(x, linha_tendencia(x), color="darkgreen", linewidth=2, linestyle="--", label="Tendência")

# plt.title("Relação entre Notas Atípicas Altas e Baixas por Município", fontsize=14)
# plt.xlabel("Quantidade de Alunos com Notas Baixas", fontsize=12)
# plt.ylabel("Quantidade de Alunos com Notas Altas", fontsize=12)
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()