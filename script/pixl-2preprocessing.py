import pandas as pd 
import numpy as np 
import seaborn as sns 
import sys

sns.set()

if len(sys.argv) < 3:
	print("Usage :", sys.argv[0], "path/to/csv/input", "path/to/csv/output")
	exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

df_input  = pd.read_csv(input_file, sep=';')
df_output = pd.read_csv(output_file, sep=';')

group_input  =  df_input.groupby(['tileX','tileY'])
group_output = df_output.groupby(['tileX','tileY'])
tile_list_in = list(group_input.groups.keys())
tile_list_out = list(group_output.groups.keys())

df_in =pd.DataFrame(columns=['month','day'])
df_out =pd.DataFrame(columns=['month','day'])

for i in tile_list_in:
	df_tile = group_input.get_group(i)
	df_tile = df_tile.drop(['tileX', 'tileY'], axis=1)
	tile_name = str(i[0])+'_'+str(i[1])
	df_tile = df_tile.rename(columns={'P':tile_name}, inplace=False)
	df_in = pd.merge(df_in,df_tile, how='outer', on=['month','day'])

for i in tile_list_out:
	df_tile = group_output.get_group(i)
	df_tile = df_tile.drop(['tileX', 'tileY'], axis=1)
	tile_name = str(i[0])+'_'+str(i[1])
	df_tile = df_tile.rename(columns={'P':tile_name}, inplace=False)
	df_out = pd.merge(df_out,df_tile, how='outer', on=['month','day'])

df_in.to_csv('data/input_matrix.csv', sep='\t', index=False)
df_out.to_csv('data/output_matrix.csv', sep='\t', index=False)
