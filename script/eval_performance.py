import csv
import random
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

df_score = pd.read_csv('metadata.csv', sep=',', names=['time', 'tile_out', 'rsq'])
mean = df_score.groupby(by='tile_out').mean()
tile_out_list = df_score[df_score.time == 0000].tile_out.tolist()

df_slope = pd.read_json('weight_matrix.json', orient='records')

df_groups = df_slope.groupby(by='Tile_out')
tile_list = list(df_groups.groups.keys())

cor_dict = {}

for i in np.arange(0,len(tile_list)):
  df_w = df_groups.get_group(tile_list[i])
  set_w = set()
  for index, row in df_w.iterrows():
    set_w = set_w.union(set(row['Tile_in']))
  rsq = mean.iloc[i].rsq
  b_matches = [rsq if i in list(set_w) else 0.0 for i in tile_out_list]
  cor_dict[tile_out_list[i]] = b_matches

column_dict = dict(zip(np.arange(0,len(tile_out_list )), tile_out_list))
cor_df =pd.DataFrame(cor_dict).transpose().rename(columns=column_dict )


sns.set(font_scale = 0.4)
plt.figure()
g = sns.heatmap(cor_df, cmap="YlGnBu")
g.set_xticklabels(labels=cor_df.columns,rotation=70)
plt.savefig('matrix_of_features.png', dpi=150)

