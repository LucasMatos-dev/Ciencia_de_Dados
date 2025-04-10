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

tipos_escola = {
    1: "Não Respondeu",
    2: "Pública",
    3: "Privada",
}

# Filtrar alunos com notas atípicas
atipicos_baixo = df[df["NU_NOTA_LC"] < limite_inferior]
atipicos_alto = df[df["NU_NOTA_LC"] > limite_superior]

# Mapear e contar a frequência dos tipos de escola entre notas baixas
frequencia_baixas = atipicos_baixo["TP_ESCOLA"].map(tipos_escola).value_counts().sort_index()

# Plotar gráfico de barras
plt.figure(figsize=(8, 5))
frequencia_baixas.plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title("Distribuição do Tipo de Escola - Notas Baixas (Outliers)")
plt.xlabel("Tipo de Escola")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("resultados/frequencia_outliers_abaixo.png")
plt.show()

# Mapear e contar a frequência dos tipos de escola entre notas baixas
frequencia_altas = atipicos_alto["TP_ESCOLA"].map(tipos_escola).value_counts().sort_index()

# Plotar gráfico de barras
plt.figure(figsize=(8, 5))
frequencia_altas.plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title("Distribuição do Tipo de Escola - Notas Altas (Outliers)")
plt.xlabel("Tipo de Escola")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("resultados/frequencia_outliers_acima.png")
plt.show()
