import pandas as pd
import matplotlib.pyplot as plt

# Carregar dados
arquivo = "MICRODADOS_ENEM_2023.csv"
df = pd.read_csv(arquivo, sep=";", encoding="latin-1")

# Converter colunas de nota para numérico
colunas_objetivas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']
for col in colunas_objetivas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calcular média das provas objetivas
df['MEDIA_OBJETIVAS'] = df[colunas_objetivas].mean(axis=1)

# Remover dados nulos
df = df.dropna(subset=['MEDIA_OBJETIVAS', 'TP_LOCALIZACAO_ESC'])

# Mapear a localização para texto
df['LOCALIZACAO'] = df['TP_LOCALIZACAO_ESC'].map({
    1: 'Urbana',
    2: 'Rural'
})

# Criar o boxplot
plt.figure(figsize=(8, 6))
df.boxplot(column='MEDIA_OBJETIVAS', by='LOCALIZACAO', grid=False, patch_artist=True,
           boxprops=dict(facecolor='lightgreen'))
plt.title("Média das Notas Objetivas por Localização da Escola")
plt.suptitle("")
plt.xlabel("Localização da Escola")
plt.ylabel("Média das Notas (CN, CH, LC, MT)")
plt.tight_layout()
plt.show()
