import pandas as pd
import numpy as np
import csv
import json
import sys

def receive(file):
  # Load csv input
  df_input = pd.read_csv(file, sep=',')
  return df_input

def next_prediction(df_input):
  df_input['Tile'] = [str(i)+'-'+str(j) for i,j in zip(df_input.X, df_input.Y)]
  tag_hour_now = df_input.columns[2]
  df_hour_now = df_weights[df_weights.Hour == tag_hour_now]
  min_next = int(tag_hour_now[2:])+15
  if min_next == 60:
    hour_next = int(tag_hour_now[:2])+1
    min_next = 0
  else :
    hour_next = int(tag_hour_now[:2])

  tag_hour_next ='{:>02}{:>02}'.format(hour_next, min_next)

  column_list = ['X','Y', tag_hour_next]
  df_output = pd.DataFrame( columns = column_list)
  for i in np.arange(0, df_hour_now.shape[0]):
    df_model = df_hour_now.iloc[i]
    intercept = df_model.Intercept
    X_out = df_model.Tile_out.split('-')[0]
    Y_out = df_model.Tile_out.split('-')[1]
    tile_out = df_model.Tile_out
    df_mod_in = df_input[df_input.Tile.isin(df_model.Tile_in)]
    if df_mod_in.shape[0] != len(df_model.Tile_in):
      print("Problem: you can't do this prediction!!")
    prediction = round((np.array(df_mod_in[tag_hour_now])*np.array(df_model.Slope)).sum()+intercept)
    if prediction < 0:
      print("Negative prediction:")
      #print(np.array(df_mod_in[tag_hour_now]))
      #print(np.array(df_model.Slope))
    df_output = df_output.append(pd.DataFrame([[X_out, Y_out, prediction]], columns = column_list))
  return df_output

def send(df_tosend):
#    tile_out_cord = df_maptile[(df_maptile.X == int(X_out)) & (df_maptile.Y == int(Y_out))]
#    out_file.write('{},{},{},{},{}\n'.format(tile_out_cord.LatMin.values[0], tile_out_cord.LatMax.values[0],
#      tile_out_cord.LonMin.values[0], tile_out_cord.LonMax.values[0], int(round(output))))
  return

# parse command line
if len(sys.argv) < 2:
  print("Usage :", sys.argv[0], "path/to/csv/input")
  exit(1)
input_file = sys.argv[1]

# Load matrix of weights and tile info
df_weights = pd.read_json('weight_matrix.json', orient='records')
df_weights.Hour = ['{:>04}'.format(i) for i in df_weights.Hour]
df_maptile = pd.read_csv('maptiletim_latlon.csv', sep=',', names=['X','Y','LatMin','LonMin','LatMax','LonMax'])

NUM_NEXT_STEP = 8
df_data = receive(input_file)

for i in np.arange(0, NUM_NEXT_STEP):
  print("----- New prediction ------")
  df_data = next_prediction(df_data)





