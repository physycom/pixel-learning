#! /usr/bin/env python3

import os
import json
import mysql.connector
from datetime import datetime
import pandas as pd
import argparse
import matplotlib.pyplot as plt

version='1.0.0'
print("[Query-MariaDB] QueryMariadb v" + version)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', required=True)
args = parser.parse_args()

# load toi
toi_dict = {'station'    :[140043, 93843],
            'square'     :[140058, 93850],
            'stnuova'    :[140053, 93842],
            'center'     :[140057, 93849],
            'rialto'     :[140052, 93846],
            'port'       :[140037, 93852],
            'academy'    :[140049, 93852],
            'university' :[140046, 93850]
            }

# load config
with open(args.cfg) as f:
  config = json.load(f)

query_string = '\n'.join([
  'SELECT tileX, tileY, month, day, time, holiday, P FROM test_year',
  'WHERE(',
  *['(tileX = {} AND tileY = {}) {}'.format(v[0], v[1], 'OR' if i != len(toi_dict) - 1 else '') for i,v in enumerate(toi_dict.values())],
  ')',
  'AND',
  'year = 17',
  ';'
])

query = query_string

try:
  db = mysql.connector.connect(
    host=config['host'],
    database=config['database'],
    user=config['user'],
    passwd=config['passwd']
  )
  cursor = db.cursor()

  tquery = datetime.now()
  #cursor.execute(query)
  #result = cursor.fetchall()

  df = pd.read_sql(query, con=db)
  df = df.sort_values(by =['tileX','month','day', 'time'] , ascending=True)
  df.to_csv('toi_P.csv', index=None, sep=';')
  tquery = datetime.now() - tquery
  print('[Query-MariaDB] Query for "{}" took {}'.format(query, tquery))

except Exception as e:
  print('[Query-MariaDB] Error : {}'.format(e))
