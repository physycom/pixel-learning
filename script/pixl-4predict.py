import pandas as pd
import numpy as np
import csv
import json
import sys

def receive(file):
  # Load csv input
  df_input = pd.read_csv(file, sep=',')
  return df_input

def make_prediction(df_input):
  df_input['Tile'] = [str(i)+'-'+str(j) for i,j in zip(df_input.X, df_input.Y)]
  tag_hour_now = df_input.columns[2]
  df_hour_now = df_weights[df_weights.Hour_in == tag_hour_now]

  column_list = ['X','Y','Hour_in','Hour_out','P']
  df_output = pd.DataFrame( columns = column_list)
  for i in np.arange(0, df_hour_now.shape[0]):
    df_model = df_hour_now.iloc[i]
    time_in = df_model.Hour_in
    time_out = df_model.Hour_out
    intercept = df_model.Intercept
    X_out = df_model.Tile_out.split('-')[0]
    Y_out = df_model.Tile_out.split('-')[1]

    tile_out = df_model.Tile_out
    df_mod_in = df_input[df_input.Tile.isin(df_model.Tile_in)]
    df_mod_in = df_mod_in.sort_values(by=['X','Y'])

    if df_mod_in.shape[0] != len(df_model.Tile_in):
      print("Problem: you can't do this prediction!!")

    prediction = round((np.array(df_mod_in[tag_hour_now])*np.array(df_model.Slope)).sum()+intercept)

    if prediction < 0:
      print("Negative prediction:")
      exit(4)

    df_output = df_output.append(pd.DataFrame([[X_out, Y_out, time_in, time_out, prediction]], columns = column_list))

  return df_output

def send(df_tosend):
  df_tosend.to_csv(day_name+'_'+df_data.Hour_in.iloc[0]+'.csv', columns=['Hour_out','X','Y','P'], index=False)
  return

# parse command line
if len(sys.argv) < 2:
  print("Usage :", sys.argv[0], "path/to/csv/input", "number/of/prediction/quarter")
  exit(1)
input_file    = sys.argv[1]
day_name     = (input_file.split('\\')[-1]).split('_')[0]

# Load matrix of weights and tile info
df_weights = pd.read_json('weight_matrix.json', orient='records')
df_weights.Hour_in = ['{:>04}'.format(i) for i in df_weights.Hour_in]
df_weights.Hour_out = ['{:>04}'.format(i) for i in df_weights.Hour_out]

# Necessary for lat lon connection
#df_maptile = pd.read_csv('maptiletim_latlon.csv', sep=',', names=['X','Y','LatMin','LonMin','LatMax','LonMax'])


df_data = receive(input_file)
df_data = make_prediction(df_data)
send(df_data)




