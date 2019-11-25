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


# Fix random seed for reproducibility
my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)

np.set_printoptions(precision=3)
pd.set_option('precision', 3)

if len(sys.argv) < 2:
	print("Usage :", sys.argv[0], "path/to/npy/dir")
	exit(1)

dir_path  = sys.argv[1]
file_list = [f for f in listdir(dir_path) if (isfile(join(dir_path, f)) and f.endswith('.npy'))]

weight_matrix = pd.DataFrame(columns=['Hour', 'Tile_out', 'Tile_in', 'Intercept', 'Slope'])

with open('metadata.csv', 'w') as out:
  for i in file_list:
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


    ####################### LINEAR REGRESSION ############################
    model = LinearRegression()
    model.fit(X_train,y_train)
    r_sq = model.score(X_test,y_test)
    out.write(time +','+ y_name +','+ '{:.3f}'.format(r_sq)+'\n')
    weight_matrix = weight_matrix.append(pd.DataFrame([[int(time), y_name, X_name, model.intercept_, model.coef_]],
                    columns=['Hour', 'Tile_out', 'Tile_in', 'Intercept', 'Slope']))
out.close

weight_matrix = weight_matrix.sort_values(by='Hour')
weight_matrix.to_json('weight_matrix.json',orient='records')


