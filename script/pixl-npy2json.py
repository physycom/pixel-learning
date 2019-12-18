#! /usr/bin/env python3

#################################
#### RUN FROM 'script' FOLDER ###
#################################

import pandas as pd
import numpy as np
import random
import sys
import os
from datetime import datetime
import multiprocessing as mp
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


def train_model(name_file):

  records = np.load(name_file)
  df = pd.DataFrame(records).drop('index', axis=1)
  df = df.sample(frac=1).reset_index(drop=True)

  item = name_file.split('\\')[-1]
  name_model = item.split('.')[0]
  item = item.split('.')[0].split('_')
  time_in = item[0]
  time_out = item[1]

  column_name =df.columns.to_list()
  y_name = column_name[-1]
  X_name = column_name[:-1]

  X = df[X_name]
  y = df[y_name]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  test = SelectKBest(f_regression, k=7)
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
  df_work.to_json('../output_json/'+ name_model +'.json',orient='records')


if __name__ ==  '__main__':
  # parse command line
  if len(sys.argv) < 2:
    print("Usage :", sys.argv[0], "path/to/npy/dir")
    exit(1)
  dir_path  = sys.argv[1]

  # setup folder
  wdir = '../output_json/'
  try:
    os.makedirs(wdir)
  except FileExistsError:
    pass

  # Fix random seed for reproducibility
  my_seed = 42
  random.seed(my_seed)
  np.random.seed(my_seed)

  tic = timer()

  file_list = [f for f in listdir(dir_path) if (isfile(join(dir_path, f)) and f.endswith('.npy'))]
  file_list = [dir_path+i for i in file_list]

  ##### Serial ########
  #for i in file_list:
  #  train_model(i)


  ##### Parallel #############
  pool = mp.Pool(mp.cpu_count())
  pool.map(train_model, file_list)
  pool.close()


  tac = timer()
  print('[PL-npy2json] : {}'.format(tac - tic), flush=True)




