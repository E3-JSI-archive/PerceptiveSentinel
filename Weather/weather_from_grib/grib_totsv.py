#python 2.7
#from datetime import date
from weather.weather import WeatherExtractor

import pandas as pd
#import numpy as np
import json
from tqdm import tqdm

#months to extract
start = (1, 2016)
end = (1, 2016)

list_of_months = []
br = False
for i in range(2010, 2100, 1):
    for j in range(1, 13, 1):
        if start[1] > i or (start[1] == i and start[0] > j):
            continue
        if end[1] < i or (end[1] == i and end[0] < j):
            br = True
            break
        list_of_months.append((j, i))
    if br:
        break
    
# query the downloaded data
we = WeatherExtractor()

# load actual and forecasted weather data
#we.load(['.\\grib_files\\weather-slovenia_m1_y2017.grib'])
exrtractors = []
pbar = tqdm(total=len(list_of_months))
for ind, i in enumerate(list_of_months):
    exrtractors.append(WeatherExtractor())
    exrtractors[ind].load(['./grib_files/weather-slovenia_m' + str(i[0]) + '_y' + str(i[1]) + '.grib'])
    pbar.update(1)

with open('coordinate.json') as json_file:  
    dic_list = json.load(json_file)
pbar = tqdm(total=len(list_of_months))
for ind, i in enumerate(list_of_months):
    exrtractors[ind].export_qminer('./tsv_help/weather-slovenia_exported_m' + str(i[0]) + '_y' + str(i[1]) + '.tsv', dic_list)
    df = pd.read_csv('./tsv_help/weather-slovenia_exported_m' + str(i[0]) + '_y' + str(i[1]) + '.tsv', sep = '\t')
    df = df[df['dayOffset'] == 0]
    features = list(set(df['param'].values))
    columns = ['timestamp', 'region', 'dayOffset', '10u',
               '10v', '2d', '2t', 'rh', 'sd', 'sf', 'sp',
               'ssr', 'sund', 'tcc', 'tp', 'vis', 'ws']
    data_list = ['10u','10v', '2d', '2t', 'rh', 'sd', 'sf', 'sp',
                 'ssr', 'sund', 'tcc', 'tp', 'vis', 'ws']

    df_new = df.reindex( columns = columns).drop_duplicates().reset_index().drop('index', 1).set_index(['timestamp', 'region'])
    df = df.reset_index(drop = True)
    df = df.set_index(['timestamp', 'param', 'region'])
    for j in data_list:
        df_new[j] = df.xs(j, level = 1)['value']
    df_new = df_new.reset_index()
    df_new.to_csv('./tsv_files/weather-slovenia_m' + str(i[0]) + '_y' + str(i[1]) + '.tsv', sep='\t', index=False)
    pbar.update(1)