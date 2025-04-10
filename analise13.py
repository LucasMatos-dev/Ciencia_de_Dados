import pandas as pd
import matplotlib.pyplot as plt

# Carregamento do arquivo (ajuste o nome se necessário)
arquivo = "MICRODADOS_ENEM_2023.csv"
df = pd.read_csv(arquivo, sep=';', encoding='latin-1')

# Remover dados faltantes na nota de Linguagens ou na escolha da língua
df = df.dropna(subset=['NU_NOTA_LC', 'TP_LINGUA'])

# Mapear código da língua para texto
mapa_lingua = {
    0: 'Inglês',
    1: 'Espanhol'
}
df['LINGUA'] = df['TP_LINGUA'].map(mapa_lingua)

# Criar o boxplot comparando as notas de LC por língua
plt.figure(figsize=(8, 5))
df.boxplot(column='NU_NOTA_LC', by='LINGUA', patch_artist=True,
           boxprops=dict(facecolor='lightgreen'))
plt.title('Nota em Linguagens por Escolha de Língua Estrangeira')
plt.suptitle('')
plt.xlabel('Língua Estrangeira Escolhida')
plt.ylabel('Nota em Linguagens (NU_NOTA_LC)')
plt.grid(False)
plt.tight_layout()
plt.show()
