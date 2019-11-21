import csv
import random
import sys
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas.plotting import scatter_matrix

# Fix random seed for reproducibility
my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)

sns.set()
np.set_printoptions(precision=3)

if len(sys.argv) < 3:
	print("Usage :", sys.argv[0], "path/to/csv/input", "path/to/csv/output")
	exit(1)

file_input  = sys.argv[1]
file_output = sys.argv[2]

df_in  = pd.read_csv(file_input, sep='\t')
df_out = pd.read_csv(file_output, sep='\t')

df = pd.merge(df_in, df_out, how='inner', on=['month', 'day'])
X = df[['140043_93843_x', '140053_93842_x']]
y = df['140058_93850_y']

### Rescale data between 0 and 1 
pd.set_option('precision', 3)             
#scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)
#y = scaler.fit_transform(y)

####################### LINEAR REGRESSION ############################
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
r_sq = model.score(X,y)
print('coefficient of determination: ', r_sq)
print('intercept: ', model.intercept_)
print('slope: ', model.coef_)
y_pred = model.predict(X)

df_y = pd.DataFrame(y)
df_y['y_pred'] =y_pred
print(df_y)

