#! /usr/bin/env python3

#################################
#### RUN FROM 'script' FOLDER ###
#################################

import pandas as pd
import numpy as np
import sys
import os
import random
from datetime import datetime, timedelta
import multiprocessing as mp
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)

#sys.stdout = open("shell.txt", "w")

def train_model(dict_matrix):
  name_model = list(dict_matrix.keys())[0]
  df = dict_matrix[name_model]
  df = df.sample(frac=1).reset_index(drop=True)

  item = name_model.split('/')[-1]
  item = item.split('_')
  time_in = item[0]
  time_out = item[1]

  column_name =df.columns.tolist()
  y_name = column_name[-1]
  X_name = column_name[:-1]

  X = df[X_name]
  y = df[y_name]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  test = SelectKBest(f_regression, k=7)
  fit_model = test.fit(X_train, y_train)
  X_train_new = fit_model.transform(X_train)
  X_test_new = fit_model.transform(X_test)
  X_name = df.columns[fit_model.get_support(indices=True)].tolist()
  ####################### LINEAR REGRESSION ############################
  model = LinearRegression()
  model.fit(X_train_new,y_train)
  r_sq = model.score(X_test_new,y_test)
  df_work = pd.DataFrame([[time_in, time_out, y_name, X_name, model.intercept_, model.coef_, r_sq]],
                    columns=['Hour_in','Hour_out', 'Tile_out', 'Tile_in', 'Intercept', 'Slope', 'Score'])
  df_work.to_json(name_model +'.json',orient='records')


def read_input(file_name):
  df_input  = pd.read_csv(file_name, sep=';')
  df_datehour = pd.DataFrame(df_input.Timestamp.str.split(' ',1).tolist(),
                                     columns = ['Date','Hour'])
  df_input = pd.concat([df_input,df_datehour], axis=1)
  df_hour = pd.DataFrame(df_input.Hour.str.split(':',2).tolist(), columns=['Hour', 'Min', 'Sec'])
  df_input['IdHour']=df_hour['Hour']+df_hour['Min']
  return df_input


def rearrange_input(df_time):
  tag = df_time.IdHour.iloc[0]
  group_df  =  df_time.groupby(['TileX','TileY'])
  tile_list = list(group_df.groups.keys())
  df_in =pd.DataFrame(columns=['Date'])

  for i in tile_list:
    df_tile = group_df.get_group(i)
    df_tile = df_tile.drop(['TileX', 'TileY', 'IdHour','Timestamp','Hour'], axis=1)
    tile_name = str(i[0])+'-'+str(i[1])
    df_tile = df_tile.rename(columns={'P':tile_name}, inplace=False)
    df_in = pd.merge(df_in,df_tile, how='outer', on=['Date'])
  dictw = {}
  dictw[tag] = df_in
  return dictw

def make_matrix_item(dict_dftime):
  time_list = list(dict_dftime.keys())
  input_key = time_list[0]
  output_key = time_list[1:]
  name_tile = dict_dftime[input_key].columns.tolist()
  name_tile.remove('Date')
  list_dict =[]
  for out_k in output_key:
    for j in name_tile:
      out_tile =['Date'] + [j]
      in_tile = ['Date'] + name_tile.copy()
      in_tile.remove(j)
      dfw_in  = dict_dftime[input_key][in_tile]
      dfw_out = dict_dftime[out_k][out_tile]
      df_matrix = pd.merge(dfw_in, dfw_out, how='outer', on=['Date'])
      df_matrix = df_matrix.dropna().drop(['Date'], axis=1)
      dict_matrix={}
      dict_matrix['{:>04}_{:>04}_{}'.format(input_key, out_k, j)] = df_matrix
      list_dict.append(dict_matrix)
  return list_dict

def normalize_before_date(df, x_date, norm):
    if norm==1:
        return df
    else:
        x_dt = datetime.strptime(x_date, '%Y-%m-%d %H:%M:%S')
        dates = pd.to_datetime(df.Timestamp, format = '%Y-%m-%d %H:%M:%S')
        df.P = df.P.astype('float32')
        df.loc[dates.loc[dates<x_dt].index, 'P'] *= norm
        df.P = df.P.astype('int64')
        return df

if __name__ ==  '__main__':
  # parse command line
  if len(sys.argv) < 4:
    print("Usage :", sys.argv[0], "path/to/csv/input", "output/dir", "number of prediction dts")
    sys.exit(1)
  input_file = sys.argv[1]
  wdir = sys.argv[2]
  number_prediction = int(sys.argv[3])
  min_dt = 15 # minutes between timestamps
  print('[PL-2-netle] Pre-processing : {}'.format(input_file))

  # setup folder
  try:
    os.makedirs(wdir)
  except FileExistsError:
    pass

  tic = timer()
  first_tic = tic
  df_input = read_input(input_file)

  x_date = '2020-02-19 10:00:00' # if any other format change the normalize function to match
  norm_factor = 3.125 # need exact value
  df_input = normalize_before_date(df_input, x_date, norm_factor)
  df_input = df_input.drop_duplicates()
  df_input = df_input.sort_values(by='Timestamp',ascending=True).reset_index(drop=True)

  time_group = df_input.groupby(['IdHour'])
  time_list = list(time_group.groups.keys())

  tac = timer()
  print('[PL-2-netle] Pre-processing time: {}'.format(tac-tic), flush=True)

  # """

  tic = timer()
  time_list = list(np.unique(df_input['IdHour']))
  df_input['XY'] = df_input['TileX'].astype(str) + '-' + df_input['TileY'].astype(str)
  df_input = df_input.drop(['TileX','TileY','Hour','Date'], axis=1)
  df_input = df_input[['Timestamp','P','IdHour','XY']]
  df_input['Timestamp'] = pd.to_datetime(df_input['Timestamp'], format = '%Y-%m-%d %H:%M:%S')

  dict_ts = {}
  for ts_in in time_list:
      dict_dftime = {}
      ts_slice_in = df_input.loc[df_input['IdHour'] == ts_in]

      group_tiles_in  =  ts_slice_in.groupby(['XY'])
      df_in = pd.DataFrame(columns=['Timestamp'])

      for t_in in list(group_tiles_in.groups.keys()):
        df_tile_in = group_tiles_in.get_group(t_in)
        df_tile_in = df_tile_in.drop(['XY', 'IdHour'], axis=1).rename(columns={'P': t_in}, inplace=False)
        df_in = pd.merge(df_in,df_tile_in, how='outer', on=['Timestamp'])

      dict_ts[ts_in] = df_in


  for ts_in in time_list:
    list_dict_matrix = []
    df_in = dict_ts[ts_in]

    for i in range(number_prediction):
      dt = timedelta(minutes=min_dt * (i+1))
      ts_out = (datetime.strptime(ts_in,'%H%M') + dt).strftime('%H%M')
      df_out = dict_ts[ts_out].copy()
      df_out['Timestamp'] = df_out['Timestamp'] - dt

      tile_list = df_out.columns.tolist()
      tile_list.remove('Timestamp')
      for t_out in tile_list:
        df1 = df_in.drop(t_out,axis=1)
        df2 = df_out[['Timestamp',t_out]]
        df = pd.merge(df1,df2,how='outer',on=['Timestamp'])
        df = df.dropna().drop(['Timestamp'], axis=1)
        list_dict_matrix.append({f'{ts_in}_{ts_out}_{t_out}' : df})

    for k in list_dict_matrix:
        old_name = list(k.keys())[0]
        new_name = wdir + '/' + old_name
        k[new_name] = k.pop(old_name)
        train_model(k)
  """

  parallel = False

  ##### Parallel #############
  if parallel:
    print('[PL-2-netle] MODE Parallel @ {} cpus'.format(mp.cpu_count()), flush=True)
    tic = timer()
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(make_matrix_item, [{k: v for d in pool.map(rearrange_input, [time_group.get_group(i) for i in time_list[j:j+min(number_prediction,len(time_list)-j)+1]]) for k, v in d.items()} for j in np.arange(0,len(time_list))])
    tac = timer()
    print('[PL-2-netle] Rearrange matrix: {}'.format(tac - tic), flush=True)
    tic = timer()
    for m in result:
      for i in m:
        old_name = list(i.keys())[0]
        new_name = wdir + '/' + old_name
        i[ new_name ] = i.pop(old_name)
    for m in result:
      pool.map(train_model, m)
    pool.close()

  ##### Serial #############
  if not parallel:
    tic = timer()
    for j in np.arange(0,len(time_list[:-number_prediction])):
      timew = time_list[j:(j+min(number_prediction, len(time_list)-j)+1)]
      dict_dftime = {}
      for i in timew:
        df_time = time_group.get_group(i)
        dict_dftime.update(rearrange_input(df_time))
      list_dict_matrix = make_matrix_item(dict_dftime)
      for i in list_dict_matrix:
        old_name = list(i.keys())[0]
        new_name = wdir + '/' + old_name
        i[ new_name ] = i.pop(old_name)
        train_model(i)
# """
  tac = timer()
  print('[PL-2-networklearning] Make item of matrix: {}'.format(tac - tic), flush=True)
  print('Total time: {}'.format(tac-first_tic), flush=True)

