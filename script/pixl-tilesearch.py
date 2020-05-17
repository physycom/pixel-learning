#! /usr/bin/env python3

import json
import sys
import mysql.connector
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dbc(config):
    return mysql.connector.connect(
            host=config['host'],
            database=config['database'],
            user=config['user'],
            passwd=config['passwd'],
            port = config['port']
           )

if len(sys.argv) < 2:
  print("Usage :", sys.argv[0], "path/to/json/config.json")
  exit(1)
conf_path = sys.argv[1]
# conf_path = "../cfg.json"

print("[Pixel-0-Tile Search] Starting")

# load config
with open(conf_path) as f:
  cfg = json.load(f)

time_query = 'SELECT MIN(Timestamp), MAX(Timestamp) FROM cells'

tquery = datetime.now()
try:
  time_limits = pd.read_sql(time_query, con=dbc(cfg))
  start_date = time_limits['MIN(Timestamp)'][0]
  stop_date = time_limits['MAX(Timestamp)'][0]
except Exception as e:
  print('[Pixel-0-Tile Search] Error : {}'.format(e))

print("[Pixel-0-Tile Search] Time interval obtained: {} // {}".format(start_date, stop_date))

query = '\n'.join([
        'SELECT',
        'AVG(c.P),',
        'c.TileX,',
        'c.TileY',
        'FROM',
        'cells c ',
        'WHERE',
        'c.Timestamp BETWEEN \'{}\' AND \'{}\''.format(start_date, stop_date),
        'GROUP BY ',
        'c.TileX, c.TileY',
        ';'
        ])

try:
    df = pd.read_sql(query, con=dbc(cfg))
except Exception as e:
  print('[Pixel-0-Tile Search] Error : {}'.format(e))

df = df.sort_values(by=['AVG(c.P)'],ascending=False).reset_index(drop=True)
tquery = datetime.now() - tquery
print('[Pixel-0-Tile Search] Query took {}'.format(tquery))

#%%
N = 1000
print("[Pixel-0-Tile Search] First {} tiles mean plot".format(N))
u = 50 # y ticks
w = 100 # x ticks
###
N = min(N,len(df))
plt.plot(df['AVG(c.P)'][:N],'.r')
plt.xticks(range(0,int((N//w+2)*w),w))
plt.yticks(range(0,int((df['AVG(c.P)'][0]//u+2)*u),u))
plt.grid()
plt.xlabel('Tiles')
plt.ylabel('Average people')
plt.title('First {} tiles mean plot'.format(N))
plt.show()

#%%
print("[Pixel-0-Tile Search] Thresholds % - number of tiles plot")
thresholds = [10,25,50,75]
###
cap = df['AVG(c.P)'][0]
tiles = [(len(df.loc[df["AVG(c.P)"]>(cap*(100-t)/100)])) for t in thresholds]
plt.plot(thresholds, tiles, '-r')
plt.xticks(thresholds)
plt.yticks(tiles)
plt.xlabel('Threshold on max of average people (%)')
plt.ylabel('Number of tiles')
plt.title('Tiles with average people bigger than threshold')
plt.show()

### show on map too TODO

#%%
N = 20
print("[Pixel-0-Tile Search] Weekly mean of top {} tiles plot".format(N))
###
N = min(N,len(df))
tl = df[:N][["TileX","TileY"]].values
week_query = '\n'.join([
            'SELECT',
            'WEEKDAY(c.Timestamp),',
            'AVG(c.P)',
            'FROM',
            'cells c ',
            'WHERE',
            *['(c.TileX = {} AND c.TileY = {}) {}'.format(tile[0], tile[1], 'OR' if i != len(tl) - 1 else '') for i,tile in enumerate(tl)],
            'GROUP BY ',
            'WEEKDAY(c.Timestamp)',
            ';'
            ])

print("[Pixel-0-Tile Search] Starting weekday mean query")
tquery = datetime.now()
try:
    dfw = pd.read_sql(week_query, con=dbc(cfg))
except Exception as e:
  print('[Pixel-0-Tile Search] Error : {}'.format(e))

dfw = dfw.sort_values(by=['WEEKDAY(c.Timestamp)'],ascending=True).reset_index(drop=True)
tquery = datetime.now() - tquery
print('[Pixel-0-Tile Search] Query took {}'.format(tquery))

days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
plt.plot(days, dfw['AVG(c.P)'], '-r')
plt.ylabel('Average people')
plt.title('Weekly mean of top {} tiles'.format(N))
plt.show()

#%%
N = 20
print("[Pixel-0-Tile Search] Daily mean of top {} tiles plot".format(N))
###
N = min(N,len(df))
tl = df[:N][["TileX","TileY"]].values
day_query = '\n'.join([
            'SELECT',
            'HOUR(c.Timestamp),',
            'AVG(c.P)',
            'FROM',
            'cells c ',
            'WHERE',
            *['(c.TileX = {} AND c.TileY = {}) {}'.format(tile[0], tile[1], 'OR' if i != len(tl) - 1 else '') for i,tile in enumerate(tl)],
            'GROUP BY ',
            'HOUR(c.Timestamp)',
            ';'
            ])

print("[Pixel-0-Tile Search] Starting daily mean query")
tquery = datetime.now()
try:
    dfd = pd.read_sql(day_query, con=dbc(cfg))
except Exception as e:
  print('[Pixel-0-Tile Search] Error : {}'.format(e))

dfd = dfd.sort_values(by=['HOUR(c.Timestamp)'],ascending=True).reset_index(drop=True)
tquery = datetime.now() - tquery
print('[Pixel-0-Tile Search] Query took {}'.format(tquery))

plt.plot(dfd['HOUR(c.Timestamp)'], dfd['AVG(c.P)'], '-r')
plt.ylabel('Average people')
plt.xlabel('Hours')
plt.title('Daily mean of top {} tiles'.format(N))
plt.show()