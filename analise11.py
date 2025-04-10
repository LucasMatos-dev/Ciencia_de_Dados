import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv("MICRODADOS_ENEM_2023.csv", sep=";", encoding="latin-1")

# Converter colunas de notas para numérico
notas = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT"]
for nota in notas:
    df[nota] = pd.to_numeric(df[nota], errors="coerce")

# Criar coluna de média geral das provas objetivas
df["MEDIA_GERAL"] = df[notas].mean(axis=1)

# Filtrar apenas participantes que concluíram (1) ou estão cursando (2) o ensino médio
df = df[df["TP_ST_CONCLUSAO"].isin([1, 2])]
df["SITUACAO_MEDIO"] = df["TP_ST_CONCLUSAO"].map({
    1: "Concluiu o EM",
    2: "Está cursando o EM"
})

# Calcular médias gerais por grupo
media_concluiu = df[df["SITUACAO_MEDIO"] == "Concluiu o EM"]["MEDIA_GERAL"].mean()
media_nao_concluiu = df[df["SITUACAO_MEDIO"] == "Está cursando o EM"]["MEDIA_GERAL"].mean()

# Criar boxplot comparativo
df.boxplot(column="MEDIA_GERAL", by="SITUACAO_MEDIO", grid=False, figsize=(8, 6))
plt.title("Comparação da Média Geral das Notas por Situação no Ensino Médio")
plt.suptitle("")
plt.xlabel("Situação no Ensino Médio")
plt.ylabel("Média das Notas (CN, CH, LC, MT)")

# Adicionar linhas de média de cada grupo
plt.axhline(media_concluiu, color="blue", linestyle="--", label=f"Média Concluiu: {media_concluiu:.1f}")
plt.axhline(media_nao_concluiu, color="red", linestyle="--", label=f"Média Cursando: {media_nao_concluiu:.1f}")

plt.legend()
plt.tight_layout()
plt.show()
