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
from datetime import datetime

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
file_list = [f for f in listdir(dir_path) if (isfile(join(dir_path, f)) and f.endswith('.npy'))]
weight_matrix = pd.DataFrame(columns=['Hour', 'Tile_out', 'Tile_in', 'Intercept', 'Slope'])
with open('metadata.csv', 'w') as out:
  for i in file_list:
    #print('[PL-3-netlearn] Training model : {}'.format(i), flush=True)

    tnow = datetime.now()

    records = np.load(dir_path+'/'+i)
    df = pd.DataFrame(records).drop('index', axis=1)
    df = df.sample(frac=1).reset_index(drop=True)

    item = i.split('.')[0].split('_')
    time = item[0]

    column_name =df.columns.to_list()
    y_name = column_name[-1]
    X_name = column_name[:-1]

    X = df[X_name]
    y = df[y_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
    test = SelectKBest(f_regression, k=10)
    fit = test.fit(X_train, y_train)
    X_train_new = fit.transform(X_train)
    X_test_new = fit.transform(X_test)
    X_name = df.columns[fit.get_support(indices=True)].to_list()
    ####################### LINEAR REGRESSION ############################
    model = LinearRegression()
    model.fit(X_train_new,y_train)
    r_sq = model.score(X_test_new,y_test)
    out.write('{},{},{:.3f}\n'.format(time, y_name, r_sq))
    weight_matrix = weight_matrix.append(pd.DataFrame([[time, y_name, X_name, model.intercept_, model.coef_]],
                    columns=['Hour', 'Tile_out', 'Tile_in', 'Intercept', 'Slope']))

    #print('[PL-3-netlearn] Model perf {} trained in {}'.format(r_sq, datetime.now() - tnow), flush=True)

out.close

weight_matrix = weight_matrix.sort_values(by='Hour')
weight_matrix.to_json('weight_matrix.json',orient='records')
