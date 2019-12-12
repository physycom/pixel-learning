#! /usr/bin/env python3

import csv
import random
import sys
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from timeit import default_timer as timer
import multiprocessing as mp

def train_model(name_file):
    records = np.load(name_file)
    df = pd.DataFrame(records).drop('index', axis=1)
    df = df.sample(frac=1).reset_index(drop=True)

    item = name_file.split('\\')[-1]
    item = item.split('.')[0].split('_')
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
    return df_work

if __name__ ==  '__main__':
    # parse command line
    if len(sys.argv) < 2:
    	print("Usage :", sys.argv[0], "path/to/npy/dir")
    	exit(1)
    dir_path  = sys.argv[1]

    # Fix random seed for reproducibility
    my_seed = 42
    random.seed(my_seed)
    np.random.seed(my_seed)

    # train models
    tic = timer()

    file_list = [f for f in listdir(dir_path) if (isfile(join(dir_path, f)) and f.endswith('.npy'))]
    file_list = [dir_path+i for i in file_list]

    weight_matrix = pd.DataFrame(columns=['Hour_in','Hour_out', 'Tile_out', 'Tile_in', 'Intercept', 'Slope', 'Score'])

    ###### Parallel ######
    pool = mp.Pool(mp.cpu_count())
    weight_matrix = weight_matrix.append(pool.map(train_model, [i for i in file_list]))
    pool.close()

    ###### Serial ######
    #cnt_tot = len(file_list)
    #cnt = 0
    #for i in file_list:
    #    cnt+=1
    #    if(cnt%250==0):
    #        print('Model: {}/{}'.format(cnt, cnt_tot))
    #   weight_matrix = weight_matrix.append(train_model(i))

    weight_matrix = weight_matrix.sort_values(by=['Hour_in', 'Hour_out'])
    weight_matrix.to_json('weight_matrix.json',orient='records')

    tac = timer()
    print('[PL-3-netlearn]: {}'.format(tac - tic), flush=True)