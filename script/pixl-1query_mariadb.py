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

# load config
with open(args.cfg) as f:
  config = json.load(f)

#query = 'SHOW DATABASES;'
#query = 'SHOW TABLES;'
#query = 'SELECT * FROM cuore;'
#query = 'DESCRIBE test_year;'
#query = """SELECT time, P FROM test_year
#WHERE
#  tileX = 140057
#  AND
#  tileY = 93850
#  AND
#  year = 17 
#  AND
#  month = 9
#  AND
#  day = 3
#;
#""" ##15 min

query = """SELECT tileX, tileY, month, day,P FROM test_year   
WHERE ((tileX = 140058 AND tileY = 93850) #square
  OR (tileX = 140043 AND tileY = 93843)   #station
  OR (tileX = 140053 AND tileY = 93842)   #stnuova
  OR (tileX = 140057 AND tileY = 93849)   #center
  OR (tileX = 140052 AND tileY = 93846)   #rialto
  OR (tileX = 140037 AND tileY = 93852)   #port
  OR (tileX = 140048 AND tileY = 93854))  #academy
  AND
  year = 17 
  AND
  time = 900
;
""" ## 33 min


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
  df = df.sort_values(by =['tileX','month','day'] , ascending=True)
  df.plot('P',color='red',style='.-')
  df.to_csv(args.cfg + '.csv', index=None, sep=';')
  tquery = datetime.now() - tquery
  print('[Query-MariaDB] Query for "{}" took {}'.format(query, tquery))
  plt.show()

  #for x in result:
  #  print(x)


except Exception as e:
  print('[Query-MariaDB] Error : {}'.format(e))
