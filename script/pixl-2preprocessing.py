import pandas as pd
import numpy as np
import seaborn as sns
import sys

sns.set()


if len(sys.argv) < 2:
	print("Usage :", sys.argv[0], "path/to/csv/input")
	exit(1)

input_file = sys.argv[1]

df_input  = pd.read_csv(input_file, sep=';')
time_group = df_input.groupby(['time'])
time_list = list(time_group.groups.keys())
dict_dftime = {}
for j in time_list:
	df_time = time_group.get_group(j)
	group_df  =  df_time.groupby(['tileX','tileY'])
	tile_list = list(group_df.groups.keys())
	df_in =pd.DataFrame(columns=['month','day'])

	cnt = 0
	for i in tile_list:
		df_tile = group_df.get_group(i)
		df_tile = df_tile[df_tile.holiday == 0]
		df_tile = df_tile.drop(['tileX', 'tileY', 'time','holiday'], axis=1)
		tile_name = str(i[0])+'_'+str(i[1])
		df_tile = df_tile.rename(columns={'P':tile_name}, inplace=False)
		df_in = pd.merge(df_in,df_tile, how='outer', on=['month','day'])

	dict_dftime[j] = df_in

name_tile = []
for x in tile_list:
	tile_name = str(x[0])+'_'+str(x[1])
	name_tile.append(tile_name)

for i in np.arange(0,len(time_list)-1):
		for j in name_tile:
			out_tile =['month','day'] + [j]
			in_tile = ['month','day'] + name_tile.copy()
			in_tile.remove(j)
			dfw_in  = dict_dftime[time_list[i]][in_tile]
			dfw_out = dict_dftime[time_list[i+1]][out_tile]
			df_matrix = pd.merge(dfw_in, dfw_out, how='outer', on=['month','day'])
			df_matrix = df_matrix.dropna().drop(['month', 'day'], axis=1)
			new_recarray = df_matrix.to_records()
			new_recarray.tofile('../output/'+str(time_list[i])+'_'+str(j)+'.npy')
