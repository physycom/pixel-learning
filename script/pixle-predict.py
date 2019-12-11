#! /usr/bin/env python3

import requests
import json
import os
import csv
import pandas as pd
import argparse
from time import time
from datetime import datetime, timedelta
from dateutil import tz

##########################
#### global scope ########
##########################
lastrecname = 'lastrec.txt'


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
        headers=head,
        proxies=proxies,
        timeout=timeout
      )
    else:
      r = requests.get(
        url,
        data=data,
        headers=head,
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

timetag = '{}_{}'.format(schedule[0]['date'], schedule[0]['time'])

for s in schedule:
  s["prediction_dt_sec"] = pred_dt * 60
  with open(wdir + 'px-prediction-{}.json'.format(timetag), 'w') as out:
    json.dump(s, out, sort_keys=True)

