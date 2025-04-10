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


def plot_municipio_scatter_todos(data, title):
    contagem = data["NO_MUNICIPIO_PROVA"].value_counts().reset_index()
    contagem.columns = ["NO_MUNICIPIO_PROVA", "FREQUENCIA"]
    contagem["IDX"] = range(len(contagem))  # Índice numérico pro eixo X

    plt.figure(figsize=(16, 6))
    plt.scatter(contagem["IDX"], contagem["FREQUENCIA"],
                color='royalblue', s=60, alpha=0.7, edgecolors='black')
    plt.title(title)
    plt.xlabel("Município")
    plt.ylabel("Quantidade de Alunos com Nota Atípica")

    # Se muitos municípios, remover os rótulos ou mostrar alguns poucos
    if len(contagem) <= 30:
        plt.xticks(contagem["IDX"],
                   contagem["NO_MUNICIPIO_PROVA"], rotation=75)
    else:
        # Mostrar apenas alguns rótulos no eixo X
        step = max(1, len(contagem) // 30)  # Mostra ~30 labels
        xticks = contagem["IDX"][::step]
        labels = contagem["NO_MUNICIPIO_PROVA"][::step]
        plt.xticks(xticks, labels, rotation=75)

    plt.tight_layout()
    plt.show()


# Análise 10: Comparação entre municípios (Gráfico de Dispersão)
plot_municipio_scatter_todos(
    atipicos_baixo, "Municípios com Mais Notas Baixas em Linguagens")
plot_municipio_scatter_todos(
    atipicos_alto, "Municípios com Mais Notas Altas em Linguagens")
