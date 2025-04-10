import pandas as pd
import matplotlib.pyplot as plt

# Carregamento do arquivo (ajuste o nome se necessário)
arquivo = "MICRODADOS_ENEM_2023.csv"
df = pd.read_csv(arquivo, sep=';', encoding='latin-1')

# Remove linhas com alguma nota ausente
colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']
df = df.dropna(subset=colunas_notas)

# Calcula a média das quatro áreas do conhecimento
df['MEDIA_GERAL'] = df[colunas_notas].mean(axis=1)

# Dicionário para mapear códigos de cor/raça
mapa_cor = {
    0: 'Não declarado',
    1: 'Branca',
    2: 'Preta',
    3: 'Parda',
    4: 'Amarela',
    5: 'Indígena'
}

df['COR_RACA'] = df['TP_COR_RACA'].map(mapa_cor)

# Remove valores não declarados
df = df[df['COR_RACA'] != 'Não declarado']

# Criação do boxplot
plt.figure(figsize=(10, 6))
df.boxplot(column='MEDIA_GERAL', by='COR_RACA', grid=False, patch_artist=True,
           boxprops=dict(facecolor='lightblue'))
plt.title('Distribuição da Média Geral das Notas por Cor/Raça')
plt.suptitle('')
plt.xlabel('Cor/Raça Declarada')
plt.ylabel('Média das Notas (CN, CH, LC, MT)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
