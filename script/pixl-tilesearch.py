#! /usr/bin/env python3

import json
import sys
import mysql.connector
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mercantile
import io
import random
import urllib.request
import PIL.Image

# transparent colormap
from matplotlib.colors import LinearSegmentedColormap
ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
my_cmap = LinearSegmentedColormap.from_list(name='jet_alpha',colors=color_array)
plt.register_cmap(cmap=my_cmap)


def _download_tile(tile: mercantile.Tile):
  server = random.choice(['a', 'b', 'c'])
  url = 'http://{server}.tile.openstreetmap.org/{zoom}/{x}/{y}.png'.format(
      server=server,
      zoom=tile.z,
      x=tile.x,
      y=tile.y
  )
  opener = urllib.request.build_opener()
  opener.addheaders = [('User-Agent', 'Googlebot/2.1'), ('Accept', '*/*')]
  response = opener.open(url)
  img = PIL.Image.open(io.BytesIO(response.read()))
  return img, tile

def _get_image(west, south, east, north, zoom):
  tiles = list(mercantile.tiles(west, south, east, north, zoom))
  tile_size = 256
  min_x = min_y = max_x = max_y = None

  for tile in tiles:
    min_x = min(min_x, tile.x) if min_x is not None else tile.x
    min_y = min(min_y, tile.y) if min_y is not None else tile.y
    max_x = max(max_x, tile.x) if max_x is not None else tile.x
    max_y = max(max_y, tile.y) if max_y is not None else tile.y

  out_img = PIL.Image.new(
    'RGB',
    ((max_x - min_x + 1) * tile_size, (max_y - min_y + 1) * tile_size)
  )

  lat_max = mercantile.bounds(max_x, min_y, zoom).north
  lat_min = mercantile.bounds(min_x, max_y, zoom).south
  lon_max = mercantile.bounds(max_x, max_y, zoom).east
  lon_min = mercantile.bounds(min_x, min_y, zoom).west

  results = []
  cnt = 0
  for tile in tiles:
    # print('Tile {}/{}'.format(cnt+1, len(tiles)))
    result = _download_tile(tile)
    results.append(result)
    cnt += 1

  for img, tile in results:
    left = tile.x - min_x
    top = tile.y - min_y
    bounds = (left * tile_size, top * tile_size, (left + 1) * tile_size, (top + 1) * tile_size)
    out_img.paste(img, bounds)

  return out_img, lat_min, lat_max, lon_min, lon_max

def get_image_roi(west, south, east, north, zoom):
  img, lat_min, lat_max, lon_min, lon_max = _get_image(west=west, south=south, east=east, north=north, zoom=zoom)
  left = img.width * (west - lon_min) / (lon_max - lon_min)
  right = img.width * (east - lon_min) / (lon_max - lon_min)
  top = img.height * (lat_max - north) / (lat_max - lat_min)
  bottom = img.height * (lat_max - south) / (lat_max - lat_min)
  return img.crop((left, top, right, bottom))

def dbc(config):
    return mysql.connector.connect(
            host=config['host'],
            database=config['database'],
            user=config['user'],
            passwd=config['passwd'],
            port = config['port']
           )
#%%
if len(sys.argv) < 2:
  print("Usage :", sys.argv[0], "path/to/json/config.json")
  exit(1)
conf_path = sys.argv[1]
# conf_path = "../cfg.json"

#%%
print("[Pixel-0-Tile Search] Starting")

# load config
with open(conf_path) as f:
  cfg = json.load(f)

time_query = 'SELECT MIN(Timestamp), MAX(Timestamp) FROM cells'
coords_query = 'SELECT MIN(TileX), MAX(TileX),  MIN(TileY), MAX(TileY) FROM cells'

tquery = datetime.now()
try:
  time_limits = pd.read_sql(time_query, con=dbc(cfg))
  coords_limits = pd.read_sql(coords_query, con=dbc(cfg))
  minx = coords_limits['MIN(TileX)'][0]
  maxx = coords_limits['MAX(TileX)'][0]
  miny = coords_limits['MIN(TileY)'][0]
  maxy = coords_limits['MAX(TileY)'][0]
  start_date = time_limits['MIN(Timestamp)'][0]
  stop_date = time_limits['MAX(Timestamp)'][0]
except Exception as e:
  print('[Pixel-0-Tile Search] Error : {}'.format(e))

print("[Pixel-0-Tile Search] Time interval obtained: {} // {}".format(start_date, stop_date))
print("[Pixel-0-Tile Search] Tiles interval obtained: X {}-{} / Y {}-{}".format(minx,maxx,miny,maxy))

query = '\n'.join([
        'SELECT',
        'AVG(c.P) AS Average,',
        'MAX(c.P) AS Maximum,',
        'c.TileX,',
        'c.TileY',
        'FROM',
        'cells c ',
        # 'INNER JOIN',
        # 'tiles_clusters_d t',
        # 'ON',
        # 'c.TileX = t.TileX AND c.TileY = t.TileY',
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

tquery = datetime.now() - tquery
print('[Pixel-0-Tile Search] Query took {}'.format(tquery))
#%%
### SELECT WHAT TO ANALYZE
# selector = "Maximum"
selector = "Average"
###
df = df.sort_values(by=[selector],ascending=False).reset_index(drop=True)

#%%
N = 1000
print("[Pixel-0-Tile Search] First {} tiles plot".format(N))
u = 100 # y ticks
w = 100 # x ticks
###
N = min(N,len(df))
plt.plot(df[selector][:N],'.r')
plt.xticks(range(0,int((N//w+2)*w),w))
plt.yticks(range(0,int((df[selector][0]//u+2)*u),u))
plt.grid()
plt.xlabel('Tiles')
plt.ylabel('{} people'.format(selector))
plt.title('First {} tiles plot'.format(N))
plt.show()

#%%
print("[Pixel-0-Tile Search] Thresholds - number of tiles plot")
thresholds = np.arange(0.1,100.1,0.1)
###
cap = df[selector][0]
means = [(cap*(100-t)/100) for t in thresholds]
tiles_list = [(df.loc[df[selector]>=m])[['TileX','TileY']] for m in means]
tiles = [len(t) for t in tiles_list]
plt.plot(means, tiles, '.r')
# plt.yticks(tiles)
plt.xlabel('Threshold on {} people'.format(selector))
plt.ylabel('Number of tiles')
plt.title('Tiles with {} people bigger than threshold'.format(selector))
plt.show()

# plt.hist(df['People'],bins=100)
# plt.hist(df['People'].loc[df['People']>100],bins=100)

#%%
print("[Pixel-0-Tile Search] Thresholds/tiles shown on map")
thresholds = np.arange(1,101,1)
zoom = 18
map_quality = 13 # use 12 or 13
###

latlon_min = mercantile.bounds(mercantile.Tile(minx, miny, zoom))
latlon_max = mercantile.bounds(mercantile.Tile(maxx, maxy, zoom))
print('[Pixel-0-Tile Search] Downloading map image ... ',end='')
im = get_image_roi(latlon_min.west, latlon_max.south, latlon_max.east, latlon_min.north, map_quality)
print('Done')

cap = df[selector][0]
listX = np.unique(df.TileX)
listY = np.unique(df.TileY)
dimx, dimy = len(listX),len(listY)
tileX2idx = { tile : idx for idx, tile in enumerate(listX)}
tileY2idy = { tile : idx for idx, tile in enumerate(listY)}

fig, ax = plt.subplots(1,1)
fig.suptitle("Selected tiles based on top % threshold")
ax.imshow(im, extent=[0,dimx,0,dimy])

cnt = np.zeros((dimy,dimx))
for t in thresholds[::-1]:
    mean = cap*(100-t)/100
    tiles_list = (df.loc[df[selector]>=mean])[['TileX','TileY']]
    tempx = [ tileX2idx[tile] for tile in tiles_list.TileX ]
    tempy = [ tileY2idy[tile] for tile in tiles_list.TileY ]
    for x,y in zip(tempx,tempy):
        cnt[y][x] = 100-t
hm = ax.imshow(cnt, extent=[0,dimx,0,dimy], cmap=my_cmap, alpha=1)
fig.colorbar(hm, ax=ax)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()

#%%
N = 20
print("[Pixel-0-Tile Search] Weekly {} of top {} tiles plot".format(selector, N))
###
N = min(N,len(df))
tl = df[:N][["TileX","TileY"]].values
week_query = '\n'.join([
            'SELECT',
            'WEEKDAY(c.Timestamp),',
            'AVG(c.P) AS Average,',
            'MAX(c.P) AS Maximum',
            'FROM',
            'cells c ',
            'WHERE',
            *['(c.TileX = {} AND c.TileY = {}) {}'.format(tile[0], tile[1], 'OR' if i != len(tl) - 1 else '') for i,tile in enumerate(tl)],
            'GROUP BY ',
            'WEEKDAY(c.Timestamp)',
            ';'
            ])

print("[Pixel-0-Tile Search] Starting weekday analysis query")
tquery = datetime.now()
try:
    dfw = pd.read_sql(week_query, con=dbc(cfg))
except Exception as e:
  print('[Pixel-0-Tile Search] Error : {}'.format(e))

dfw = dfw.sort_values(by=['WEEKDAY(c.Timestamp)'],ascending=True).reset_index(drop=True)
tquery = datetime.now() - tquery
print('[Pixel-0-Tile Search] Query took {}'.format(tquery))

days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
plt.plot(days, dfw[selector], '-r')
plt.ylabel('{} people'.format(selector))
plt.title('Weekly {} of top {} tiles plot'.format(selector, N))
plt.show()

#%%
N = 20
print("[Pixel-0-Tile Search] Daily {} of top {} tiles plot".format(selector, N))
###
N = min(N,len(df))
tl = df[:N][["TileX","TileY"]].values
day_query = '\n'.join([
            'SELECT',
            'HOUR(c.Timestamp),',
            'AVG(c.P) AS Average,',
            'MAX(c.P) AS Maximum',
            'FROM',
            'cells c ',
            'WHERE',
            *['(c.TileX = {} AND c.TileY = {}) {}'.format(tile[0], tile[1], 'OR' if i != len(tl) - 1 else '') for i,tile in enumerate(tl)],
            'GROUP BY ',
            'HOUR(c.Timestamp)',
            ';'
            ])

print("[Pixel-0-Tile Search] Starting daily analysis query")
tquery = datetime.now()
try:
    dfd = pd.read_sql(day_query, con=dbc(cfg))
except Exception as e:
  print('[Pixel-0-Tile Search] Error : {}'.format(e))

dfd = dfd.sort_values(by=['HOUR(c.Timestamp)'],ascending=True).reset_index(drop=True)
tquery = datetime.now() - tquery
print('[Pixel-0-Tile Search] Query took {}'.format(tquery))

plt.plot(dfd['HOUR(c.Timestamp)'], dfd[selector], '-r')
plt.ylabel('{} people'.format(selector))
plt.xlabel('Hours')
plt.title('Daily {} of top {} tiles plot'.format(selector, N))
plt.show()

#%%
# query e grafico per il totale sull'isola e media di giorno / notte
day_query = '\n'.join([
            'SELECT',
            'c.Timestamp AS time,',
            'SUM(c.P) AS tot',
            'FROM',
            'cells c ',
            'INNER JOIN',
            'tiles_clusters_d t',
            'ON',
            'c.TileX = t.TileX AND c.TileY = t.TileY',
            'WHERE',
            't.IDcluster = 1 ',#'IN (1,2,4)',# per fare il centro, 1 solo per san marco
            'AND',
            'c.Timestamp >= "2020-01-01 00:00:00"',
            'AND',
            'c.Timestamp <= "2020-07-01 00:00:00"',
            'GROUP BY ',
            'c.Timestamp',
            ';'
            ])

try:
    dfd = pd.read_sql(day_query, con=dbc(cfg))
except Exception as e:
  print('[Pixel-0-Tile Search] Error : {}'.format(e))

dfd = dfd.sort_values(by=['time'],ascending=True).reset_index(drop=True)

tsindex = pd.DatetimeIndex(dfd.time)
night = dfd.loc[(tsindex.hour < 7) | (tsindex.hour > 19)]
day = dfd.loc[(tsindex.hour < 19) & (tsindex.hour > 7)]
print(night.tot.mean())
print(day.tot.mean())

plt.plot(dfd['time'], dfd['tot'], '-or')
plt.ylabel('Tot people')
plt.xlabel('Ts')
plt.show()
