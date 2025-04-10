import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
arquivo = "MICRODADOS_ENEM_2023.csv"
df = pd.read_csv(arquivo, sep=';', encoding='latin-1')

# Converter notas para numérico
colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']
for col in colunas_notas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calcular média das notas objetivas
df['MEDIA_GERAL'] = df[colunas_notas].mean(axis=1)

# Remover ausentes na média e em Q005
df = df.dropna(subset=['MEDIA_GERAL', 'Q005'])

# Aqui usamos diretamente o número da Q005, pois já é numérico
df['NUM_PESSOAS_RESIDENCIA'] = pd.to_numeric(df['Q005'], errors='coerce')

# Criar gráfico
plt.figure(figsize=(12, 6))
df.boxplot(column='MEDIA_GERAL', by='NUM_PESSOAS_RESIDENCIA', grid=False, patch_artist=True,
           boxprops=dict(facecolor='lightgray'))
plt.title("Média Geral das Notas por Quantidade de Pessoas na Residência")
plt.suptitle("")
plt.xlabel("Número de Pessoas na Residência")
plt.ylabel("Média das Notas Objetivas (CN, CH, LC, MT)")
plt.tight_layout()
plt.show()