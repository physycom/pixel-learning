import pandas as pd
import numpy as np
import csv
import json
import sys


# Script to convert a single mobile json
# data to single column csv

# parse command line
if len(sys.argv) < 2:
  print("Usage :", sys.argv[0], "path/to/csv/input")
  exit(1)
input_file = sys.argv[1]
file_name = input_file.split('\\')[-1]

# Load csv input
df_input = pd.read_csv(input_file, sep=',')
df_input['Tile'] = [str(i)+'-'+str(j) for i,j in zip(df_input.X, df_input.Y)]
hour_now = file_name.split('_')[1]

# Load matrix of weights and tile info
df_weights = pd.read_json('weight_matrix.json', orient='records')
df_weights.Hour = ['{:>04}'.format(i) for i in df_weights.Hour]
df_maptile = pd.read_csv('maptiletim_latlon.csv', sep=',', names=['X','Y','LatMin','LonMin','LatMax','LonMax'])

df_hour_now = df_weights[df_weights.Hour == hour_now]
with open('prediction.csv','w') as out_file:
  out_file.write('LatMin,LonMin,LatMax,LonMax,P\n')
  for i in np.arange(0, df_hour_now.shape[0]):
    df_model = df_hour_now.iloc[i]
    intercept = df_model.Intercept
    X_out = df_model.Tile_out.split('-')[0]
    Y_out = df_model.Tile_out.split('-')[1]
    tile_out = df_model.Tile_out
    df_mod_in = df_input[df_input.Tile.isin(df_model.Tile_in)]
    if (df_mod_in.shape[0] != len(df_model.Tile_in)):
      print("Problem: you can't do this prediction!!")
    output = (np.array(df_mod_in[hour_now])*np.array(df_model.Slope)).sum()+intercept
    tile_out_cord = df_maptile[(df_maptile.X == int(X_out)) & (df_maptile.Y == int(Y_out))]
    out_file.write('{},{},{},{},{}\n'.format(tile_out_cord.LatMin.values[0], tile_out_cord.LatMax.values[0],
      tile_out_cord.LonMin.values[0], tile_out_cord.LonMax.values[0], int(round(output))))



