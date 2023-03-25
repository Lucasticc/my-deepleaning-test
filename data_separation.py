# 将emotion和pixels像素数据分离
import pandas as pd

# 注意train.csv是在你电脑本地的相对或绝对路劲地址
path = '/Users/lanyiwei/data/train.csv'
# 读取数据
df = pd.read_csv(path)
# 提取emotion数据
df_y = df[['emotion']]
# 提取pixels数据
df_x = df[['pixels']]
# 将emotion写入emotion.csv
df_y.to_csv('/Users/lanyiwei/data/emotion.csv', index=False, header=False)
# 将pixels数据写入pixels.csv
df_x.to_csv('/Users/lanyiwei/data/pixels.csv', index=False, header=False)

