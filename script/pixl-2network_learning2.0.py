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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


def train_model(dict_matrix):
    name_model = list(dict_matrix.keys())[0]
    df = dict_matrix[name_model]
    df = df.sample(frac=1).reset_index(drop=True)

    item = name_model.split('_')
    time_in = item[0]
    time_out = item[1]

    column_name =df.columns.to_list()
    y_name = column_name[-1]
    X_name = column_name[:-1]

    X = df[X_name]
    y = df[y_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    test = SelectKBest(f_regression, k=15)
    fit_model = test.fit(X_train, y_train)
    X_train_new = fit_model.transform(X_train)
    X_test_new = fit_model.transform(X_test)
    X_name = df.columns[fit_model.get_support(indices=True)].to_list()
    ####################### LINEAR REGRESSION ############################
    model = LinearRegression()
    model.fit(X_train_new,y_train)
    r_sq = model.score(X_test_new,y_test)
    df_work = pd.DataFrame([[time_in, time_out, y_name, X_name, model.intercept_, model.coef_, r_sq]],
                    columns=['Hour_in','Hour_out', 'Tile_out', 'Tile_in', 'Intercept', 'Slope', 'Score'])
    df_work.to_json('../output/'+ name_model +'.json',orient='records')


def read_input(file_name):
  df_input  = pd.read_csv(input_file, sep=';')
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
  name_tile = dict_dftime[input_key].columns.to_list()
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
      #new_recarray = df_matrix.to_records()
      #np.save('{}{:>04}_{:>04}_{}.npy'.format('../output/', input_key, out_k, j), new_recarray)


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

  tic = timer()
  df_input = read_input(input_file)

  time_group = df_input.groupby(['IdHour'])
  time_list = list(time_group.groups.keys())

  tac = timer()
  print('[PL-2-prepro] Pre-processing time: {}'.format(tac-tic), flush=True)
  number_prediction = 4

  ##### Parallel #############
  tic = timer()
  pool = mp.Pool(mp.cpu_count())
  result = pool.map(make_matrix_item, [{k: v for d in pool.map(rearrange_input, [time_group.get_group(i) for i in time_list[j:j+min(number_prediction,len(time_list)-j)+1]]) for k, v in d.items()} for j in np.arange(0,len(time_list))])
  tac = timer()
  print('[PL-2-preprocessing] Rearrange matrix: {}'.format(tac - tic), flush=True)
  tic = timer()
  for m in result:
    pool.map(train_model, m)
  pool.close()

  ##### Serial #############
  #tic = timer()
  #for j in np.arange(0,len(time_list[:-number_prediction])):
  #  timew = time_list[j:(j+min(number_prediction, len(time_list)-j)+1)]
  #  dict_dftime = {}
  #  for i in timew:
  #    df_time = time_group.get_group(i)
  #    dict_dftime.update(rearrange_input(df_time))
  #  list_dict_matrix = make_matrix_item(dict_dftime)
  #  for i in list_dict_matrix:
  #    train_model(i)

  tac = timer()
  print('[PL-2-networklearning] Make item of matrix: {}'.format(tac - tic), flush=True)




