#! /usr/bin/env python3
# Usage: python pixel_train_3_merge.py json_dir/
# json_dir must be a folder containing all the learning script json outputs.

import pandas as pd
import sys
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer

# parse command line
if len(sys.argv) < 2:
  print("Usage :", sys.argv[0], "path/to/json/dir")
  exit(1)
dir_path = sys.argv[1]

print('[Pixel-3-merge_matrix] Starting')

tic = timer()

file_list = [f for f in listdir(dir_path) if (isfile(join(dir_path, f)) and f.endswith('.json'))]

columns_list = ['Hour_in', 'Hour_out', 'Intercept', 'Score', 'Slope', 'Tile_in', 'Tile_out']
df_weights = pd.DataFrame(columns=columns_list)
for i in file_list:
  dir_file = dir_path + '/' + i
  df_item = pd.read_json(dir_file, orient='records')
  df_item.Hour_in = ['{:>04}'.format(i) for i in df_item.Hour_in]
  df_item.Hour_out = ['{:>04}'.format(i) for i in df_item.Hour_out]
  df_weights = df_weights.append(df_item)

df_weights = df_weights.sort_values(by=['Hour_in', 'Hour_out'])
df_weights.to_json('weight_matrix.json',orient='records')

tac = timer()
print('[Pixel-3-Merge_matrix] : {}'.format(tac - tic), flush=True)
