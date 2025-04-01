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

# Análise 6: Tipo de escola entre notas atípicas
plot_boxplot(atipicos_baixo, "TP_ESCOLA", "Distribuição do Tipo de Escola - Notas Baixas")
plot_boxplot(atipicos_alto, "TP_ESCOLA", "Distribuição do Tipo de Escola - Notas Altas")

# Análise 7: Localização da escola entre notas atípicas
plot_boxplot(atipicos_baixo, "TP_LOCALIZACAO", "Distribuição da Localização da Escola - Notas Baixas")
plot_boxplot(atipicos_alto, "TP_LOCALIZACAO", "Distribuição da Localização da Escola - Notas Altas")

# Análise 8: UF da escola entre notas atípicas
plot_boxplot(atipicos_baixo, "SG_UF_ESC", "Distribuição da UF da Escola - Notas Baixas")
plot_boxplot(atipicos_alto, "SG_UF_ESC", "Distribuição da UF da Escola - Notas Altas")

# Análise 9: Municípios de aplicação da prova entre notas atípicas
plot_boxplot(atipicos_baixo, "NO_MUNICIPIO_PROVA", "Municípios com Mais Notas Baixas")
plot_boxplot(atipicos_alto, "NO_MUNICIPIO_PROVA", "Municípios com Mais Notas Altas")

# Análise 10: Comparação entre municípios
plt.figure(figsize=(10, 5))
municipios_baixo = atipicos_baixo["NO_MUNICIPIO_PROVA"].value_counts()
municipios_alto = atipicos_alto["NO_MUNICIPIO_PROVA"].value_counts()
plt.scatter(municipios_baixo.index, municipios_baixo.values, label="Notas Baixas", color='red', alpha=0.7)
plt.scatter(municipios_alto.index, municipios_alto.values, label="Notas Altas", color='green', alpha=0.7)
plt.xlabel("Município")
plt.ylabel("Quantidade de Alunos")
plt.title("Relação entre Notas Atípicas Altas e Baixas por Município")
plt.xticks(rotation=90, ha='right')
plt.legend()
plt.show()
