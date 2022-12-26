from collections import defaultdict
from this import d
import numpy as np
import pickle
import argparse
import time
from copyreg import constructor
from datetime import timedelta
from datetime import datetime
import os.path 
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from mplfinance.original_flavor import candlestick2_ochl, volume_overlay
from pandas_datareader import data
import time
import math
from tqdm import trange, tqdm

def scoring(base_df):
    pred = base_df['Predicted']
    label = base_df['Label']
    
    if np.isnan(label):
        return 1
    elif label == 1:
        #print(1)
        return 1
    else:
        #print(0)
        return 0

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--csv',
                        help='an input csv file', required=True)
parser.add_argument('-s', '--start',
                        help='an input start date', default='2019-01-01')
parser.add_argument('-e', '--end',
                        help='an input end date', default='2019-12-31')
parser.add_argument('-d', '--dir',
                        help='a path of output directory of PNG file', required=True)
parser.add_argument('-l', '--labeling',
                        help='labeling file path', required=True)
args = parser.parse_args()
path_csv = args.csv
start = args.start
end = args.end
output_dir = args.dir
label_csv = args.labeling

df = pd.read_csv(path_csv)
s_date = datetime.strptime(start, "%Y-%m-%d")
e_date = datetime.strptime(end, "%Y-%m-%d")
start_date = s_date.strftime('%Y%m%d')
end_date = e_date.strftime('%Y%m%d')
KOSPI = data.get_data_yahoo("^KS11", start_date, end_date)
# x축 : 날짜
# y축 : 확률값
# 1. 새롭게 만들 dataframe에서 index로 사용할 날짜를 뽑는다.
grouped = df.groupby('Ticker')
Date = None
for group_name, group_data in grouped:
    Date = group_data['Date']
    break

#Date = KOSPI.index.strftime('%Y-%m-%d')
Date = pd.DataFrame(KOSPI.index.format())
Date.columns = ['Date']
#print(Date)
#exit()
# 2. Prediction이 1인 행만 골라낸다.
df = df.loc[df['Predicted'] == 1]

name = path_csv.split('/')[-1]
name = name.split('.')[0]
plt.figure(figsize=(20,10))
plt.grid(True)

Correct = []

# 1. 날짜별로 묶기
# 2. 날짜별로 probability 기준으로 정렬 후, 상위 N개만 남기기
groupby_date = df.groupby('Date')
new_df = pd.DataFrame(columns=df.columns)
for date, group_data in tqdm(groupby_date):
    group_data = group_data.sort_values('Probability', ascending=False)
    # 상위 N개만 남기기
    if len(group_data) > 20:
        group_data = group_data.iloc[:20]
    new_df = pd.concat([new_df, group_data])

new_df = new_df[['Ticker', 'Date', 'Predicted', 'Probability']]
# 3. True/False 판별하여 리스트에 담기
# correct_bool = 0 / 1 / 2
# KOSPI : /home/ubuntu/2022_VAIV_Dataset/Labeling/Kospi/4%_01_2_20_5.csv ???
# KOSDAQ : /home/ubuntu/2022_VAIV_Dataset/Labeling/1/Kosdaq/4%_01_2_20_5.csv
label_df = pd.read_csv(label_csv)

total_df = pd.merge(new_df, label_df, on=['Date', 'Ticker'], how='outer')
total_df = total_df[['Ticker', 'Date', 'Probability', 'Predicted', 'Label']]
total_df = total_df[total_df['Predicted'].notna()]
total_df = total_df[total_df['Probability'].notna()]
total_df = total_df[total_df['Ticker'].notna()]
total_df = total_df[total_df['Date'].notna()]

print(total_df)
total_df['Correct'] = total_df.apply(scoring, axis=1)
total_df.to_csv('total_df.csv')

# 5. 'Correct' (True/False) 그룹화
grouped_correct = total_df.groupby('Correct')
tickers = []
nb_date = 0
count = 0
X = Date['Date']

for name_correct, data_correct in grouped_correct:
    # 5-1. Ticker 그룹화
    grouped_ticker = data_correct.groupby('Ticker')
    for name_ticker, data_ticker in grouped_ticker:
        merged_data = pd.merge(Date, data_ticker, on='Date', how='outer')
        merged_data = merged_data.drop_duplicates(['Date'])
        tickers.append(name_ticker)
        #NaN
        Y = merged_data['Probability'].values
        nb_date = len(Y)
        try:
            if name_correct == 1:
                plt.scatter(X, Y, color='green', alpha=0.3)
            elif name_correct == 0:
                plt.scatter(X, Y, color='red', alpha=0.3)
            elif name_correct == -1:
                #print(X)
                #print(len(Y))
                #exit()
                plt.scatter(X, Y, color='orange', alpha=0.3)
            else:
                continue
        except Exception as e:
            print(e)
            print(merged_data)
            merged_data.to_csv('merged_data.csv')
            print(len(merged_data))
            print(len(X))
            print(name_ticker)

print(nb_date)

plt.xticks(np.arange(0, nb_date, int(nb_date/15)))
#plt.yticks(np.arange(0.99995, 1.0, 0.00005))
plt.yticks(np.arange(0.5, 1.0, 0.05))
plt.xlabel('Date')
plt.ylabel('Probability')
plt.title(name)
plt.savefig(f'{output_dir}/buy_scatter_{name}_without_orange.png')