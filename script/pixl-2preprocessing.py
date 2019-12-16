#! /usr/bin/env python3

#################################
#### RUN FROM 'script' FOLDER ###
#################################

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import multiprocessing as mp
from timeit import default_timer as timer


def make_input(df_time):
  tag = df_time.IdHour.iloc[0]
  group_df  =  df_time.groupby(['TileX','TileY'])
  tile_list = list(group_df.groups.keys())
  df_in =pd.DataFrame(columns=['Date'])

  for i in tile_list:
    df_tile = group_df.get_group(i)
    df_tile = df_tile.drop(['TileX', 'TileY', 'IdHour','Timestamp'], axis=1)
    tile_name = str(i[0])+'-'+str(i[1])
    df_tile = df_tile.rename(columns={'P':tile_name}, inplace=False)
    df_in = pd.merge(df_in,df_tile, how='outer', on=['Date'])
    dictw = {}
    dictw[tag] = df_in
  return dictw

def make_matrix_item(i):
  for j in name_tile:
    out_tile =['Date'] + [j]
    in_tile = ['Date'] + name_tile.copy()
    in_tile.remove(j)
    dfw_in  = dict_dftime[time_list[i]][in_tile]
    dfw_out = dict_dftime[time_list[i+k]][out_tile]
    df_matrix = pd.merge(dfw_in, dfw_out, how='outer', on=['Date'])
    df_matrix = df_matrix.dropna().drop(['Date'], axis=1)
    new_recarray = df_matrix.to_records()
    np.save('{}{:>04}_{:>04}_{}.npy'.format(wdir, time_list[i], time_list[i+k], j), new_recarray)


if __name__ ==  '__main__':
  # parse command line
  if len(sys.argv) < 2:
    print("Usage :", sys.argv[0], "path/to/csv/input")
    exit(1)
  input_file = sys.argv[1]
  print('[PL-2-prepro] Pre-processing : {}'.format(input_file))

  # setup folder
  wdir = '../output/'
  try:
    os.makedirs(wdir)
  except FileExistsError:
    pass


  # preprocess input data
  df_input  = pd.read_csv(input_file, sep=';')
  df_datehour = pd.DataFrame(df_input.Timestamp.str.split(' ',1).tolist(),
                                     columns = ['Date','Hour'])
  df_input = pd.concat([df_input,df_datehour], axis=1)
  df_hour = pd.DataFrame(df_input.Hour.str.split(':',2).tolist(), columns=['Hour', 'Min', 'Sec'])
  df_input['IdHour']=df_hour['Hour']+df_hour['Min']
  time_group = df_input.groupby(['IdHour'])
  time_list = list(time_group.groups.keys())
  dict_dftime = {}

  tic = timer()
  ##### Parallel #############
  pool = mp.Pool(mp.cpu_count())
  dict_dftime = {k: v for d in pool.map(make_input, [time_group.get_group(j) for j in time_list]) for k, v in d.items()}
  pool.close()

  ##### Serial #############

  #for j in time_list:
  #  print('[PL-2-preprocessing] Processing data : {}'.format(j), flush=True)
  #  df_time = time_group.get_group(j)
  #  dict_dftime.update(make_input(df_time))

  tac = timer()
  print('[PL-2-preprocessing] Read each input: {}'.format(tac - tic), flush=True)

  name_tile = []
  group_df = time_group.get_group(time_list[0]).groupby(['TileX','TileY'])
  tile_list = list(group_df.groups.keys())
  for x in tile_list:
    tile_name = str(x[0])+'-'+str(x[1])
    name_tile.append(tile_name)

  count = 0

  for k in np.arange(1,5):
    print("Prediction next: ", k)
    tic = timer()
    for i in np.arange(0,len(time_list)-k):
      make_matrix_item(i)

    tac = timer()
    print('[PL-2-preprocessing] Make item of matrix: {}'.format(tac - tic), flush=True)




