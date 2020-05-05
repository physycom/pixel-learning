#! /usr/bin/env python3
# Usage: python pixel_train_1_query.py config.json
# config.json must contain: host, port, database, user, passwd.

import json
import sys
import mysql.connector
from datetime import datetime
import pandas as pd

def count_db_tiles(df_input):
    dti = pd.date_range(start=df_input.Timestamp.min(), end = df_input.Timestamp.max(),freq='15min')
    tile_list = []
    time_list = []
    for i in dti:
        l=len(df_input.loc[df_input.Timestamp==i])
        tile_list.append(l)
        time_list.append(i)
    return pd.DataFrame({"Tiles" : tile_list, "Timestamp" : time_list}) # df_tiles

def fix_db_tiles(df_input, df_tiles):
    df = df_input.copy()
    tiles = df[["TileX","TileY"]].drop_duplicates()
    problems = df_tiles.loc[df_tiles.Tiles > len(tiles)].Timestamp
    for p in problems:
        df_p = df.loc[df.Timestamp==p]
        for t in tiles.values:
            people = df_p.loc[(df_p.TileX==t[0]) & (df_p.TileY==t[1])].P.values
            if len(people) != 1:
                count = people[people!=0].mean()
                df.loc[(df_input["Timestamp"]==p) & (df_input.TileX==t[0]) & (df_input.TileY==t[1]), "P"] = count
    return df.drop_duplicates()

if len(sys.argv) < 2:
  print("Usage :", sys.argv[0], "path/to/json/config.json")
  exit(1)
conf_path = sys.argv[1]

print("[Pixel-1-Query] Starting")

########## Island venice #############

# load toi
# toi_dict = {'station'    :[140043, 93843],
#             'square'     :[140058, 93850],
#             'stnuova'    :[140053, 93842],
#             'center'     :[140057, 93849],
#             'rialto'     :[140052, 93846],
#             'port'       :[140035, 93845],
#             'academy'    :[140049, 93852],
#             'university' :[140046, 93850]
#            }

# toi_dict = {'0': [140043, 93843], '1': [140044, 93845], '2': [140049, 93846], '3': [140051, 93846], '4': [140052, 93846], '5': [140052, 93850], '6': [140053, 93842], '7': [140053, 93846], '8': [140053, 93847], '9': [140053, 93849], '10': [140053, 93850], '11': [140054, 93845], '12': [140054, 93846], '13': [140054, 93847], '14': [140054, 93849], '15': [140054, 93850], '16': [140054, 93851], '17': [140054, 93852], '18': [140055, 93845], '19': [140055, 93846], '20': [140055, 93847], '21': [140055, 93849], '22': [140055, 93850], '23': [140057, 93846], '24': [140057, 93847], '25': [140057, 93849], '26': [140057, 93850], '27': [140058, 93846], '28': [140058, 93847], '29': [140058, 93849], '30': [140058, 93850], '31': [140059, 93849], '32': [140059, 93850], '33': [140059, 93851]}
# toi_dict = {'0': [140043, 93843], '1': [140044, 93843], '2': [140044, 93845], '3': [140044, 93846], '4': [140046, 93843], '5': [140046, 93845], '6': [140046, 93846], '7': [140047, 93839], '8': [140047, 93841], '9': [140047, 93843], '10': [140047, 93846], '11': [140047, 93847], '12': [140047, 93849], '13': [140048, 93841], '14': [140048, 93843], '15': [140048, 93846], '16': [140048, 93847], '17': [140048, 93849], '18': [140049, 93839], '19': [140049, 93841], '20': [140049, 93843], '21': [140049, 93845], '22': [140049, 93846], '23': [140049, 93847], '24': [140051, 93841], '25': [140051, 93843], '26': [140051, 93845], '27': [140051, 93846], '28': [140051, 93847], '29': [140052, 93841], '30': [140052, 93842], '31': [140052, 93845], '32': [140052, 93846], '33': [140052, 93847], '34': [140052, 93849], '35': [140052, 93850], '36': [140053, 93842], '37': [140053, 93843], '38': [140053, 93846], '39': [140053, 93847], '40': [140053, 93849], '41': [140053, 93850], '42': [140053, 93851], '43': [140054, 93842], '44': [140054, 93843], '45': [140054, 93845], '46': [140054, 93846], '47': [140054, 93847], '48': [140054, 93849], '49': [140054, 93850], '50': [140054, 93851], '51': [140054, 93852], '52': [140055, 93842], '53': [140055, 93843], '54': [140055, 93845], '55': [140055, 93846], '56': [140055, 93847], '57': [140055, 93849], '58': [140055, 93850], '59': [140055, 93851], '60': [140057, 93843], '61': [140057, 93845], '62': [140057, 93846], '63': [140057, 93847], '64': [140057, 93849], '65': [140057, 93850], '66': [140057, 93851], '67': [140058, 93843], '68': [140058, 93845], '69': [140058, 93846], '70': [140058, 93847], '71': [140058, 93849], '72': [140058, 93850], '73': [140058, 93851], '74': [140059, 93846], '75': [140059, 93847], '76': [140059, 93849], '77': [140059, 93850], '78': [140059, 93851], '79': [140060, 93855], '80': [140063, 93850]}
# toi_dict = {'0': [140041, 93852], '1': [140042, 93852], '2': [140043, 93842], '3': [140043, 93843], '4': [140043, 93845], '5': [140043, 93847], '6': [140044, 93842], '7': [140044, 93843], '8': [140044, 93845], '9': [140044, 93846], '10': [140044, 93847], '11': [140044, 93850], '12': [140044, 93851], '13': [140044, 93852], '14': [140046, 93839], '15': [140046, 93842], '16': [140046, 93843], '17': [140046, 93845], '18': [140046, 93846], '19': [140046, 93847], '20': [140046, 93849], '21': [140046, 93850], '22': [140046, 93851], '23': [140046, 93852], '24': [140047, 93839], '25': [140047, 93841], '26': [140047, 93842], '27': [140047, 93843], '28': [140047, 93845], '29': [140047, 93846], '30': [140047, 93847], '31': [140047, 93849], '32': [140047, 93850], '33': [140047, 93851], '34': [140047, 93852], '35': [140048, 93839], '36': [140048, 93841], '37': [140048, 93842], '38': [140048, 93843], '39': [140048, 93845], '40': [140048, 93846], '41': [140048, 93847], '42': [140048, 93849], '43': [140048, 93854], '44': [140048, 93855], '45': [140049, 93838], '46': [140049, 93839], '47': [140049, 93841], '48': [140049, 93842], '49': [140049, 93843], '50': [140049, 93845], '51': [140049, 93846], '52': [140049, 93847], '53': [140049, 93849], '54': [140049, 93850], '55': [140049, 93851], '56': [140051, 93838], '57': [140051, 93839], '58': [140051, 93841], '59': [140051, 93842], '60': [140051, 93843], '61': [140051, 93845], '62': [140051, 93846], '63': [140051, 93847], '64': [140051, 93849], '65': [140051, 93850], '66': [140052, 93839], '67': [140052, 93841], '68': [140052, 93842], '69': [140052, 93843], '70': [140052, 93845], '71': [140052, 93846], '72': [140052, 93847], '73': [140052, 93849], '74': [140052, 93850], '75': [140052, 93851], '76': [140053, 93841], '77': [140053, 93842], '78': [140053, 93843], '79': [140053, 93845], '80': [140053, 93846], '81': [140053, 93847], '82': [140053, 93849], '83': [140053, 93850], '84': [140053, 93851], '85': [140053, 93852], '86': [140054, 93841], '87': [140054, 93842], '88': [140054, 93843], '89': [140054, 93845], '90': [140054, 93846], '91': [140054, 93847], '92': [140054, 93849], '93': [140054, 93850], '94': [140054, 93851], '95': [140054, 93852], '96': [140055, 93841], '97': [140055, 93842], '98': [140055, 93843], '99': [140055, 93845], '100': [140055, 93846], '101': [140055, 93847], '102': [140055, 93849], '103': [140055, 93850], '104': [140055, 93851], '105': [140055, 93852], '106': [140057, 93841], '107': [140057, 93842], '108': [140057, 93843], '109': [140057, 93845], '110': [140057, 93846], '111': [140057, 93847], '112': [140057, 93849], '113': [140057, 93850], '114': [140057, 93851], '115': [140058, 93842], '116': [140058, 93843], '117': [140058, 93845], '118': [140058, 93846], '119': [140058, 93847], '120': [140058, 93849], '121': [140058, 93850], '122': [140058, 93851], '123': [140059, 93843], '124': [140059, 93845], '125': [140059, 93846], '126': [140059, 93847], '127': [140059, 93849], '128': [140059, 93850], '129': [140059, 93851], '130': [140060, 93846], '131': [140060, 93847], '132': [140060, 93849], '133': [140060, 93850], '134': [140060, 93851], '135': [140060, 93855], '136': [140061, 93847], '137': [140061, 93849], '138': [140061, 93850], '139': [140063, 93849], '140': [140063, 93850], '141': [140064, 93849], '142': [140064, 93850], '143': [140064, 93851], '144': [140065, 93852], '145': [140066, 93852]}

########## Mestre venice #############
toi_dict = {'0': [139956, 93800], '1': [139962, 93798], '2': [139964, 93808], '3': [139966, 93800], '4': [139966, 93802], '5': [139966, 93807], '6': [139967, 93804], '7': [139967, 93806], '8': [139968, 93802], '9': [139968, 93803], '10': [139969, 93797], '11': [139970, 93793], '12': [139970, 93794], '13': [139972, 93791], '14': [139972, 93795], '15': [139972, 93797], '16': [139972, 93798], '17': [139972, 93806], '18': [139973, 93797], '19': [139973, 93798], '20': [139973, 93802], '21': [139973, 93803], '22': [139973, 93807], '23': [139973, 93808], '24': [139974, 93771], '25': [139974, 93794], '26': [139974, 93795], '27': [139974, 93797], '28': [139974, 93798], '29': [139974, 93800], '30': [139974, 93802], '31': [139974, 93804], '32': [139974, 93807], '33': [139975, 93771], '34': [139975, 93791], '35': [139975, 93793], '36': [139975, 93794], '37': [139975, 93795], '38': [139975, 93797], '39': [139975, 93798], '40': [139975, 93799], '41': [139975, 93800], '42': [139977, 93785], '43': [139977, 93790], '44': [139977, 93791], '45': [139977, 93793], '46': [139977, 93794], '47': [139977, 93795], '48': [139977, 93797], '49': [139977, 93798], '50': [139977, 93799], '51': [139977, 93800], '52': [139977, 93802], '53': [139977, 93803], '54': [139978, 93771], '55': [139978, 93782], '56': [139978, 93784], '57': [139978, 93789], '58': [139978, 93790], '59': [139978, 93791], '60': [139978, 93793], '61': [139978, 93794], '62': [139978, 93795], '63': [139978, 93797], '64': [139978, 93798], '65': [139978, 93799], '66': [139978, 93800], '67': [139978, 93802], '68': [139978, 93803], '69': [139979, 93769], '70': [139979, 93778], '71': [139979, 93780], '72': [139979, 93789], '73': [139979, 93790], '74': [139979, 93791], '75': [139979, 93793], '76': [139979, 93794], '77': [139979, 93795], '78': [139979, 93797], '79': [139979, 93798], '80': [139979, 93799], '81': [139979, 93800], '82': [139979, 93802], '83': [139979, 93807], '84': [139979, 93808], '85': [139980, 93769], '86': [139980, 93776], '87': [139980, 93789], '88': [139980, 93790], '89': [139980, 93791], '90': [139980, 93793], '91': [139980, 93794], '92': [139980, 93795], '93': [139980, 93797], '94': [139980, 93798], '95': [139980, 93799], '96': [139980, 93800], '97': [139980, 93802], '98': [139980, 93803], '99': [139980, 93804], '100': [139980, 93806], '101': [139981, 93785], '102': [139981, 93786], '103': [139981, 93788], '104': [139981, 93789], '105': [139981, 93790], '106': [139981, 93791], '107': [139981, 93793], '108': [139981, 93794], '109': [139981, 93795], '110': [139981, 93797], '111': [139981, 93798], '112': [139981, 93799], '113': [139981, 93800], '114': [139981, 93803], '115': [139983, 93780], '116': [139983, 93785], '117': [139983, 93786], '118': [139983, 93788], '119': [139983, 93789], '120': [139983, 93790], '121': [139983, 93791], '122': [139983, 93793], '123': [139983, 93794], '124': [139983, 93795], '125': [139983, 93797], '126': [139983, 93798], '127': [139983, 93799], '128': [139983, 93800], '129': [139983, 93803], '130': [139983, 93806], '131': [139984, 93780], '132': [139984, 93785], '133': [139984, 93786], '134': [139984, 93788], '135': [139984, 93789], '136': [139984, 93790], '137': [139984, 93791], '138': [139984, 93793], '139': [139984, 93794], '140': [139984, 93795], '141': [139984, 93797], '142': [139984, 93798], '143': [139984, 93799], '144': [139984, 93800], '145': [139985, 93777], '146': [139985, 93784], '147': [139985, 93785], '148': [139985, 93786], '149': [139985, 93788], '150': [139985, 93789], '151': [139985, 93790], '152': [139985, 93791], '153': [139985, 93793], '154': [139985, 93794], '155': [139985, 93795], '156': [139985, 93797], '157': [139985, 93798], '158': [139985, 93799], '159': [139985, 93800], '160': [139986, 93781], '161': [139986, 93782], '162': [139986, 93785], '163': [139986, 93786], '164': [139986, 93788], '165': [139986, 93789], '166': [139986, 93790], '167': [139986, 93791], '168': [139986, 93793], '169': [139986, 93794], '170': [139986, 93795], '171': [139986, 93797], '172': [139986, 93798], '173': [139986, 93799], '174': [139986, 93800], '175': [139987, 93782], '176': [139987, 93784], '177': [139987, 93785], '178': [139987, 93786], '179': [139987, 93788], '180': [139987, 93789], '181': [139987, 93790], '182': [139987, 93791], '183': [139987, 93793], '184': [139987, 93794], '185': [139987, 93795], '186': [139987, 93797], '187': [139987, 93798], '188': [139987, 93799], '189': [139987, 93802], '190': [139987, 93806], '191': [139987, 93807], '192': [139989, 93781], '193': [139989, 93782], '194': [139989, 93784], '195': [139989, 93785], '196': [139989, 93786], '197': [139989, 93788], '198': [139989, 93789], '199': [139989, 93790], '200': [139989, 93791], '201': [139989, 93793], '202': [139989, 93794], '203': [139989, 93795], '204': [139989, 93797], '205': [139990, 93780], '206': [139990, 93781], '207': [139990, 93782], '208': [139990, 93784], '209': [139990, 93785], '210': [139990, 93786], '211': [139990, 93788], '212': [139990, 93789], '213': [139990, 93790], '214': [139990, 93791], '215': [139990, 93793], '216': [139990, 93794], '217': [139990, 93795], '218': [139990, 93797], '219': [139991, 93775], '220': [139991, 93776], '221': [139991, 93780], '222': [139991, 93781], '223': [139991, 93782], '224': [139991, 93784], '225': [139991, 93785], '226': [139991, 93786], '227': [139991, 93788], '228': [139991, 93789], '229': [139991, 93790], '230': [139991, 93791], '231': [139991, 93793], '232': [139991, 93794], '233': [139991, 93795], '234': [139991, 93798], '235': [139991, 93800], '236': [139992, 93775], '237': [139992, 93776], '238': [139992, 93777], '239': [139992, 93778], '240': [139992, 93780], '241': [139992, 93781], '242': [139992, 93782], '243': [139992, 93784], '244': [139992, 93785], '245': [139992, 93786], '246': [139992, 93788], '247': [139992, 93789], '248': [139992, 93790], '249': [139992, 93791], '250': [139992, 93793], '251': [139992, 93794], '252': [139992, 93795], '253': [139992, 93798], '254': [139992, 93799], '255': [139992, 93800], '256': [139994, 93777], '257': [139994, 93780], '258': [139994, 93781], '259': [139994, 93782], '260': [139994, 93784], '261': [139994, 93785], '262': [139994, 93786], '263': [139994, 93788], '264': [139994, 93789], '265': [139994, 93790], '266': [139994, 93791], '267': [139994, 93793], '268': [139994, 93794], '269': [139994, 93795], '270': [139994, 93797], '271': [139994, 93799], '272': [139995, 93775], '273': [139995, 93776], '274': [139995, 93777], '275': [139995, 93778], '276': [139995, 93780], '277': [139995, 93781], '278': [139995, 93782], '279': [139995, 93784], '280': [139995, 93785], '281': [139995, 93786], '282': [139995, 93788], '283': [139995, 93789], '284': [139995, 93790], '285': [139995, 93791], '286': [139995, 93793], '287': [139995, 93797], '288': [139995, 93798], '289': [139996, 93778], '290': [139996, 93780], '291': [139996, 93781], '292': [139996, 93782], '293': [139996, 93784], '294': [139996, 93786], '295': [139996, 93788], '296': [139996, 93789], '297': [139996, 93797], '298': [139996, 93798], '299': [139997, 93780], '300': [139997, 93781], '301': [139997, 93786], '302': [139997, 93788], '303': [139997, 93789], '304': [139998, 93777], '305': [140000, 93776], '306': [140000, 93777], '307': [140000, 93778], '308': [140001, 93776], '309': [140002, 93776], '310': [140002, 93777], '311': [140002, 93778], '312': [140002, 93786], '313': [140002, 93788], '314': [140002, 93789], '315': [140003, 93785], '316': [140003, 93786], '317': [140004, 93784], '318': [140004, 93797], '319': [140006, 93793], '320': [140007, 93790], '321': [140034, 93841], '322': [140034, 93849], '323': [140037, 93846], '324': [140037, 93852], '325': [140038, 93851], '326': [140038, 93852], '327': [140040, 93847], '328': [140040, 93849], '329': [140040, 93851], '330': [140040, 93852], '331': [140041, 93838], '332': [140041, 93839], '333': [140041, 93846], '334': [140041, 93847], '335': [140041, 93849], '336': [140041, 93850], '337': [140041, 93851], '338': [140041, 93852], '339': [140042, 93837], '340': [140042, 93838], '341': [140042, 93839], '342': [140042, 93841], '343': [140042, 93842], '344': [140042, 93843], '345': [140042, 93845], '346': [140042, 93846], '347': [140042, 93847], '348': [140042, 93850], '349': [140042, 93851], '350': [140042, 93852], '351': [140042, 93854], '352': [140043, 93837], '353': [140043, 93838], '354': [140043, 93839], '355': [140043, 93841], '356': [140043, 93842], '357': [140043, 93843], '358': [140043, 93845], '359': [140043, 93846], '360': [140043, 93847], '361': [140043, 93849], '362': [140043, 93850], '363': [140043, 93851], '364': [140043, 93852], '365': [140043, 93854], '366': [140043, 93858], '367': [140043, 93859], '368': [140044, 93837], '369': [140044, 93838], '370': [140044, 93839], '371': [140044, 93841], '372': [140044, 93842], '373': [140044, 93843], '374': [140044, 93845], '375': [140044, 93846], '376': [140044, 93847], '377': [140044, 93849], '378': [140044, 93850], '379': [140044, 93851], '380': [140044, 93852], '381': [140044, 93854], '382': [140044, 93859], '383': [140046, 93837], '384': [140046, 93838], '385': [140046, 93839], '386': [140046, 93841], '387': [140046, 93842], '388': [140046, 93843], '389': [140046, 93845], '390': [140046, 93846], '391': [140046, 93847], '392': [140046, 93849], '393': [140046, 93850], '394': [140046, 93851], '395': [140046, 93852], '396': [140046, 93854], '397': [140046, 93859], '398': [140047, 93837], '399': [140047, 93838], '400': [140047, 93839], '401': [140047, 93841], '402': [140047, 93842], '403': [140047, 93843], '404': [140047, 93845], '405': [140047, 93846], '406': [140047, 93847], '407': [140047, 93849], '408': [140047, 93850], '409': [140047, 93851], '410': [140047, 93852], '411': [140047, 93854], '412': [140047, 93855], '413': [140047, 93859], '414': [140047, 93860], '415': [140048, 93836], '416': [140048, 93837], '417': [140048, 93838], '418': [140048, 93839], '419': [140048, 93841], '420': [140048, 93842], '421': [140048, 93843], '422': [140048, 93845], '423': [140048, 93846], '424': [140048, 93847], '425': [140048, 93849], '426': [140048, 93850], '427': [140048, 93851], '428': [140048, 93852], '429': [140048, 93854], '430': [140048, 93855], '431': [140048, 93859], '432': [140048, 93860], '433': [140049, 93837], '434': [140049, 93838], '435': [140049, 93839], '436': [140049, 93841], '437': [140049, 93842], '438': [140049, 93843], '439': [140049, 93845], '440': [140049, 93846], '441': [140049, 93847], '442': [140049, 93849], '443': [140049, 93850], '444': [140049, 93851], '445': [140049, 93852], '446': [140049, 93854], '447': [140049, 93855], '448': [140049, 93860], '449': [140051, 93837], '450': [140051, 93838], '451': [140051, 93839], '452': [140051, 93841], '453': [140051, 93842], '454': [140051, 93843], '455': [140051, 93845], '456': [140051, 93846], '457': [140051, 93847], '458': [140051, 93849], '459': [140051, 93850], '460': [140051, 93851], '461': [140051, 93852], '462': [140051, 93855], '463': [140051, 93860], '464': [140052, 93837], '465': [140052, 93838], '466': [140052, 93839], '467': [140052, 93841], '468': [140052, 93842], '469': [140052, 93843], '470': [140052, 93845], '471': [140052, 93846], '472': [140052, 93847], '473': [140052, 93849], '474': [140052, 93850], '475': [140052, 93851], '476': [140052, 93852], '477': [140052, 93855], '478': [140052, 93860], '479': [140053, 93838], '480': [140053, 93839], '481': [140053, 93841], '482': [140053, 93842], '483': [140053, 93843], '484': [140053, 93845], '485': [140053, 93846], '486': [140053, 93847], '487': [140053, 93849], '488': [140053, 93850], '489': [140053, 93851], '490': [140053, 93852], '491': [140053, 93854], '492': [140053, 93855], '493': [140054, 93841], '494': [140054, 93842], '495': [140054, 93843], '496': [140054, 93845], '497': [140054, 93846], '498': [140054, 93847], '499': [140054, 93849], '500': [140054, 93850], '501': [140054, 93851], '502': [140054, 93852], '503': [140054, 93854], '504': [140055, 93841], '505': [140055, 93842], '506': [140055, 93843], '507': [140055, 93845], '508': [140055, 93846], '509': [140055, 93847], '510': [140055, 93849], '511': [140055, 93850], '512': [140055, 93851], '513': [140055, 93852], '514': [140057, 93769], '515': [140057, 93777], '516': [140057, 93780], '517': [140057, 93841], '518': [140057, 93842], '519': [140057, 93843], '520': [140057, 93845], '521': [140057, 93846], '522': [140057, 93847], '523': [140057, 93849], '524': [140057, 93850], '525': [140057, 93851], '526': [140058, 93776], '527': [140058, 93777], '528': [140058, 93778], '529': [140058, 93842], '530': [140058, 93843], '531': [140058, 93845], '532': [140058, 93846], '533': [140058, 93847], '534': [140058, 93849], '535': [140058, 93850], '536': [140058, 93851], '537': [140058, 93858], '538': [140059, 93843], '539': [140059, 93845], '540': [140059, 93846], '541': [140059, 93847], '542': [140059, 93849], '543': [140059, 93850], '544': [140059, 93851], '545': [140059, 93855], '546': [140059, 93856], '547': [140060, 93843], '548': [140060, 93845], '549': [140060, 93846], '550': [140060, 93847], '551': [140060, 93849], '552': [140060, 93850], '553': [140060, 93851], '554': [140060, 93855], '555': [140060, 93856], '556': [140061, 93845], '557': [140061, 93846], '558': [140061, 93847], '559': [140061, 93849], '560': [140061, 93850], '561': [140061, 93851], '562': [140061, 93855], '563': [140061, 93856], '564': [140063, 93836], '565': [140063, 93846], '566': [140063, 93847], '567': [140063, 93849], '568': [140063, 93850], '569': [140063, 93851], '570': [140064, 93830], '571': [140064, 93832], '572': [140064, 93847], '573': [140064, 93849], '574': [140064, 93850], '575': [140064, 93851], '576': [140065, 93830], '577': [140065, 93832], '578': [140065, 93851], '579': [140065, 93852], '580': [140066, 93830], '581': [140066, 93832], '582': [140066, 93851], '583': [140066, 93852], '584': [140068, 93768], '585': [140068, 93851], '586': [140068, 93852], '587': [140068, 93854], '588': [140069, 93851], '589': [140069, 93852], '590': [140069, 93854], '591': [140070, 93851], '592': [140070, 93852], '593': [140070, 93854]}
starting_date = '2019-12-15' # Starting date set

# load config
with open(conf_path) as f:
  config = json.load(f)

query = '\n'.join([
  'SELECT TileX, TileY, P, Timestamp FROM cells',
  'WHERE(',
  *['(TileX = {} AND TileY = {}) {}'.format(v[0], v[1], 'OR' if i != len(toi_dict) - 1 else '') for i,v in enumerate(toi_dict.values())],
  ') AND Timestamp >= \'{}\''.format(starting_date),
  ';'
])


try:
  db = mysql.connector.connect(
    host=config['host'],
    database=config['database'],
    user=config['user'],
    passwd=config['passwd'],
    port = config['port']
  )
  cursor = db.cursor()

  tquery = datetime.now()

  df = pd.read_sql(query, con=db)
  df = df.sort_values(by =['TileX', 'Timestamp'] , ascending=True)

  df = df.drop_duplicates()
  df = fix_db_tiles(df, count_db_tiles(df))

  df.to_csv('toi_P.csv', index=None, sep=';')
  tquery = datetime.now() - tquery
  print('[Pixel-1-Query] Query for "{}" took {}'.format(query, tquery))

except Exception as e:
  print('[Pixel-1-Query] Error : {}'.format(e))
