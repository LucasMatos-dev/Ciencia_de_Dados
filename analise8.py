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

# Filtrar os dados com UF válida (só os atípicos, se quiser focar neles)
# ou atipicos_alto, ou pd.concat([...]) para juntar os dois
dados_baixos = atipicos_baixo
dados_altos = atipicos_alto

# Agrupar as notas por UF da escola
ufs = sorted(dados_baixos["SG_UF_ESC"].dropna().unique())
notas_por_uf = [dados_baixos[dados_baixos["SG_UF_ESC"] == uf]
                ["NU_NOTA_LC"] for uf in ufs]

# Plotar boxplot
plt.figure(figsize=(14, 6))
plt.boxplot(notas_por_uf, labels=ufs, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(markerfacecolor='orange', marker='o', markersize=5, linestyle='none'))

plt.title("Distribuição das Notas de Linguagens por UF da Escola (Notas Baixas)")
plt.xlabel("UF da Escola")
plt.ylabel("Nota - Linguagens (NU_NOTA_LC)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("resultados/outliers_baixos_linguagem_federação_escola.png")
plt.show()

ufs = sorted(dados_altos["SG_UF_ESC"].dropna().unique())
notas_por_uf = [dados_altos[dados_altos["SG_UF_ESC"] == uf]
                ["NU_NOTA_LC"] for uf in ufs]

# Plotar boxplot
plt.figure(figsize=(14, 6))
plt.boxplot(notas_por_uf, labels=ufs, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(markerfacecolor='orange', marker='o', markersize=5, linestyle='none'))

plt.title("Distribuição das Notas de Linguagens por UF da Escola (Notas Baixas)")
plt.xlabel("UF da Escola")
plt.ylabel("Nota - Linguagens (NU_NOTA_LC)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("resultados/outliers_altos_linguagem_federação_escola.png")
plt.show()
