#! /usr/bin/env python3

import requests
import json
import os
import csv
import numpy as np
import pandas as pd
import argparse
from time import time
from datetime import datetime, timedelta
from dateutil import tz


##########################
#### request wrapper #####
##########################
def make_request(tag, type, url, headers, timeout, data, proxies):
  try:
    tnow = time()
    if type == 'POST':
      r = requests.post(
        url,
        data=data,
        headers=headers,
        proxies=proxies,
        timeout=timeout
      )
    else:
      r = requests.get(
        url,
        data=data,
        headers=headers,
        proxies=proxies,
        timeout=timeout
      )
    tela = time() - tnow
  except requests.exceptions.Timeout:
    print('[pixle-pred] {} timed out'.format(tag), flush=True)
    os.remove(lastrecname)
    exit(1)
  except Exception as e:
    print('[pixle-pred] {} generic error : {}'.format(tag, e), flush=True)
    os.remove(lastrecname)
    exit(1)
  return tela, r


##########################
#### ML ##################
##########################
def get_data(schedule):
  head={
    'Content-Type' : 'application/json'
  }

  tauth, rauth = make_request(
    tag = 'Authentication',
    type = 'POST',
    url=wapi_login,
    data=json.dumps(cred),
    headers=head,
    proxies=proxies,
    timeout=60
  )

  xauth = ''
  try:
    print('[pixle-pred] Authentication : %s'%(rauth.content), flush=True)
    xauth = rauth.headers['X-Auth']
  except:
    print('[pixle-pred] Authentication unexpected answer')
    os.remove(lastrecname)
    exit(1)

  print('[pixle-pred] Authentication took {}'.format(tauth), flush=True)

  head = {
    'Content-Type' : 'application/json',
    'X-Auth'       : xauth
  }

  timetag = schedule['date'] + '_' + schedule['time']
  data = {
    'data'        : schedule['utcdate'],
    'ora'         : schedule['utctime'],
    'granularita' : '15',
    'colors'      : ['P','Ni','Ns','Vr','Vp','Vi','Ve'],
    'ace'         : '05|027|042'
  }

  tdata, rdata = make_request(
    tag = 'Data',
    type = 'POST',
    url=wapi_data,
    headers=head,
    data=json.dumps(data),
    proxies=proxies,
    timeout=120
  )

  print('[pixle-pred] Data request for {}_{} (UTC {}_{}) took {}'.format(schedule['date'], schedule['time'], s['utcdate'], s['utctime'], tdata), flush=True)
  tiledata = rdata.json()

  df = pd.DataFrame(columns=['X', 'Y', schedule['time']])

  for jtile in tiledata["Data"]:
    for time, timeval in jtile["valuePresence"].items():
      for tag, cnt in timeval.items():
        if tag == "P":
          df.loc[df.shape[0]] = [jtile["tileX"], jtile["tileY"], cnt]

  print('[pixle-pred] Tile data {}/{} ({} %)'.format(df.shape[0], len(tiledata['Data']), (100 * df.shape[0]) // len(tiledata['Data'])))
  if df.shape[0] == 0:
    print('[pixle-pred] EMPTY tile data')
    os.remove(lastrecname)
    exit(1)

  return df

def make_prediction(df_input, weight_file):
  df_input['Tile'] = ['{:.0f}-{:.0f}'.format(i, j) for i,j in zip(df_input.X, df_input.Y)]
  tag_hour_now = df_input.columns[2]

  df_weights = pd.read_json(weight_file, orient='records')
  df_weights.Hour_in = ['{:>04}'.format(i) for i in df_weights.Hour_in]
  df_weights.Hour_out = ['{:>04}'.format(i) for i in df_weights.Hour_out]
  df_hour_now = df_weights[df_weights.Hour_in == tag_hour_now]

  column_list = ['X','Y','Hour_in','Hour_out','P']
  df_output = pd.DataFrame( columns = column_list)
  for i in np.arange(0, df_hour_now.shape[0]):
    df_model = df_hour_now.iloc[i]
    time_in = df_model.Hour_in
    time_out = df_model.Hour_out
    intercept = df_model.Intercept
    X_out = df_model.Tile_out.split('-')[0]
    Y_out = df_model.Tile_out.split('-')[1]

    tile_out = df_model.Tile_out
    df_mod_in = df_input[df_input.Tile.isin(df_model.Tile_in)]
    df_mod_in = df_mod_in.sort_values(by=['X','Y'])

    if df_mod_in.shape[0] != len(df_model.Tile_in):
      print('[pixle-pred] Missing prediction tile : {}'.format(tile_out))
      continue

    prediction = round((np.array(df_mod_in[tag_hour_now])*np.array(df_model.Slope)).sum()+intercept)

    if prediction < 0:
      print('[pixle-pred] WARNING: Negative prediction')
      #exit(1)

    df_output = df_output.append(pd.DataFrame(
      [[X_out, Y_out, time_in, time_out, prediction]],
      columns = column_list
    ))
    df_output = df_output.astype({ 'X' : 'int', 'Y' : 'int'})

  return df_output

def dump_data(df_tosend, schedule, geo_tile_file):
  df_maptile = pd.read_csv(geo_tile_file, sep=',')
  df_maptile = df_maptile.astype({ 'X' : 'int', 'Y' : 'int'})
  df_tosend = df_tosend.astype({ 'P' : 'int' })
  df_tosend = df_tosend.merge(df_maptile, how='left', on=['X', 'Y'])
  df_tosend['Datetime'] = [ datetime.strptime(s['date']+'-'+hour_out, '%y%m%d-%H%M').strftime('%Y-%m-%d %H:%M:%S') for hour_out in df_tosend.Hour_out ]
  df_tosend.to_csv(
    wdir + '/' + schedule['date']+'_'+schedule['time']+'.csv',
    columns=['Datetime','X','Y','LatMin','LatMax','LonMin','LonMax','P'],
    index=False,
    float_format='%.6f'
  )


##########################
#### global scope ########
##########################
lastrecname = 'lastrec.txt'


##########################
#### parse config ########
##########################
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='configuration JSON')
args = parser.parse_args()

if args.config != '':
  with open(args.config) as f:
    conf = json.load(f)

# credentials
if 'credentials' in conf:
  cred = conf['credentials']
else:
  print('[pixle-pred] Missing credentials in config JSON', flush=True)
  exit(1)

# proxies
if 'proxies' in conf:
  proxies = conf['proxies']
else:
  proxies = {}

# work dir
if 'workdir' in conf:
  wdir = conf['workdir']
else:
  wdir = 'data/'
try:
  os.makedirs(wdir)
except FileExistsError:
  pass

# hour delay
if 'hour_delay' in conf:
  hour_delay = conf['hour_delay']
else:
  hour_delay = 0

# prediction dt
if 'pred_dt_min' in conf:
  pred_dt = conf['pred_dt_min']
else:
  pred_dt = 0

# prediction dt
if 'weight_file' in conf:
  weight_file = conf['weight_file']
else:
  print('[pixle-pred] Weight file not found in JSON config')
  exit(1)

# prediction dt
if 'geo_tile_file' in conf:
  geo_tile_file = conf['geo_tile_file']
else:
  print('[pixle-pred] GEOtile file not found in JSON config')
  exit(1)

# web api url
try:
  wapi_login = conf['webapi_login']
  wapi_data = conf['webapi_data']
except Exception as e:
  print('[pixle-pred] Problems when loading JSON config file : {}'.format(e))
  exit(1)

# time conversion
HERE = tz.tzlocal()
UTC = tz.gettz('UTC')
schedule = []

# realtime mode
utcnow = datetime.utcnow() - timedelta(hours=hour_delay)
utcmin = (int(utcnow.strftime('%M')) // 15) * 15
utctt = utcnow.strftime('%y%m%d_%H') + '{:02}'.format(utcmin)

start_utc = datetime.strptime(utctt, '%y%m%d_%H%M')
stop_utc = start_utc + timedelta(minutes=15)

start = start_utc.replace(tzinfo=UTC).astimezone(HERE)
stop = stop_utc.replace(tzinfo=UTC).astimezone(HERE)

schedule.append({
  'date'    : start.strftime('%y%m%d'),
  'time'    : start.strftime('%H%M'),
  'utcdate' : start_utc.strftime('%y%m%d'),
  'utctime' : start_utc.strftime('%H%M')
})
timetag = '{}_{}'.format(schedule[0]['date'], schedule[0]['time'])

# check if work is needed and setup folders
try:
  os.stat(lastrecname)
except:
  lastrec = open(lastrecname, 'w')
  lastrec.write('YYMMDD_HHMM')
  lastrec.close()

with open(lastrecname) as lr:
  lasttag = lr.readlines()
lasttag = [x.strip() for x in lasttag] # trim spaces and \n

if lasttag[0] == timetag:
  print('[pixle-pred] No job for {}'.format(timetag))
  exit(1)
else:
  print('[pixle-pred] Requesting data for {}'.format(timetag), flush=True)
  with open(lastrecname, "w") as lastrec:
    lastrec.write(timetag)

for s in schedule:
  df_in = get_data(s)
  df_out = make_prediction(df_in, weight_file)
  dump_data(df_out, s, geo_tile_file)
