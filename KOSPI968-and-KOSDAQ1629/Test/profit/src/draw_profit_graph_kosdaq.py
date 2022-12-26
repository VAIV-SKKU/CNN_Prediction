from keras.models import load_model
import sys, traceback
import logging

logging.basicConfig(level=logging.ERROR)
import os
from collections import defaultdict
import numpy as np
import scipy.misc
import imageio
import cv2
import pickle
import argparse
import time
from copyreg import constructor
import tensorflow as tf
from datetime import timedelta
from datetime import datetime
import os.path 
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from mplfinance.original_flavor import candlestick2_ochl, volume_overlay
from pandas_datareader import data
import time
from sys import exit
from tqdm import trange, tqdm
from profit_VAIV import *

def predict_dataset(start, end, output, title, topN, isTopN):
    # 1. KOSDAQ 데이터로부터 2019 ~ 2021 전체 날짜 (Date) 구해서 -> X 축으로 활용
    # 2. (KOSDAQ 지수 수익률을 Plot)
    # 3. model_list for문으로 돌리기
    # 3-1. 2019, 2020, 2021 각각 누적 수익률 y-list 구하기 (여기서 시간 걸리는 연산은 끝)
    # 3-2. 2019~2021 전체 누적 수익률 y-list 마련 (위의 세 list 더해서 하나의 list로)
    # 3-3. 2019~2021 연도마다 초기화된 누적 수익률 y-list 마련 (위의 세 list 단순 merge)
    # 4. y-list Plot하기
    # 4-1. KOSDAQ 지수 수익률을 Plot
    # 4-2. (3-2) Plot
    # 4-3. (3-1) 각각 색 다르게 해서 위에 입히기
    # 4-4. 그래프 이미지로 저장 후 plot 초기화, (3-3)에 대해서도 진행
    probability = [0.675, 0.75, 0.55, 0.5]
    text = [-20, -30, -40, -50, -60, -70]
    color = ['red', 'black', 'blue', 'purple']
    label = ['VGG16 (Train Data : KOSDAQ)', 'VGG16 (Train Data : KOSPI, KOSDAQ)', 'EfficientNet B7 (Train Data : KOSDAQ)', 'EfficientNet B7 (Train Data : KOSPI, KOSDAQ)']
    trans = [0, 0]

    # 1. KOSDAQ 데이터로부터 2019 ~ 2021 전체 날짜 (Date) 구해서 -> X 축으로 활용
    start_date_ = datetime.strptime(start, "%Y-%m-%d")
    end_date_ = datetime.strptime(end,"%Y-%m-%d" )
    start_date = start_date_.strftime("%Y%m%d")
    end_date = end_date_.strftime("%Y%m%d")

    df = data.get_data_yahoo("^KQ11", start_date, end_date)
    KOSPI = data.get_data_yahoo("^KS11", start_date, end_date)
    #print(df)


    buy_price = df['Close'].iloc[0]
    diff = df['Close'] - buy_price

    rate = ( diff / buy_price ) * 100

    df['Rate'] = rate

    cumulative_rate = rate.copy()
    for i in range(1, len(rate)):
        cumulative_rate[i] = rate[i] + cumulative_rate[i-1]

    df['Cumulative rate'] = cumulative_rate
    df.to_csv(f'../Stock_Data/KOSDAQ_INDEX_{start_date}_{end_date}.csv', mode="w")
    KOSDAQ = df.copy()
    #exit()
    #print(KOSDAQ.index)
    Date_all = KOSDAQ.index  # x-label
    Date_all_KOSPI = KOSPI.index
    
    #KOSPI INDEX
    KOSPI_2019 = data.get_data_yahoo("^KS11", "20190101", "20191231")
    KOSPI_2020 = data.get_data_yahoo("^KS11", "20200101", "20201231")
    KOSPI_2021 = data.get_data_yahoo("^KS11", "20210101", "20211231")
    
    buy_price_2019_kospi = KOSPI_2019['Close'].iloc[0]
    diff_2019_kospi = KOSPI_2019['Close'] - buy_price_2019_kospi
    kospi_2019 = ( diff_2019_kospi / buy_price_2019_kospi ) * 100
    
    buy_price_2020_kospi = KOSPI_2020['Close'].iloc[0]
    diff_2020_kospi = KOSPI_2020['Close'] - buy_price_2020_kospi
    kospi_2020 = ( diff_2020_kospi / buy_price_2020_kospi ) * 100
    
    buy_price_2021_kospi = KOSPI_2021['Close'].iloc[0]
    diff_2021_kospi = KOSPI_2021['Close'] - buy_price_2021_kospi
    kospi_2021 = ( diff_2021_kospi / buy_price_2021_kospi ) * 100
    
    # KOSDAQ INDEX
    KOSDAQ_2019 = data.get_data_yahoo("^KQ11", "20190101", "20191231")
    KOSDAQ_2020 = data.get_data_yahoo("^KQ11", "20200101", "20201231")
    KOSDAQ_2021 = data.get_data_yahoo("^KQ11", "20210101", "20211231")

    df_2019 = data.get_data_yahoo("^KQ11", "20190101", "20191231")
    #print(df)
    buy_price_2019 = df_2019['Close'].iloc[0]
    diff_2019 = df_2019['Close'] - buy_price_2019
    kosdaq_2019 = ( diff_2019 / buy_price_2019 ) * 100
    
    df_2020 = data.get_data_yahoo("^KQ11", "20200101", "20201231")
    #print(df)
    buy_price_2020 = df_2020['Close'].iloc[0]
    diff_2020 = df_2020['Close'] - buy_price_2020
    kosdaq_2020 = ( diff_2020 / buy_price_2020 ) * 100

    df_2021 = data.get_data_yahoo("^KQ11", "20210101", "20211231")
    #print(df)
    buy_price_2021 = df_2021['Close'].iloc[0]
    diff_2021 = df_2021['Close'] - buy_price_2021
    kosdaq_2021 = ( diff_2021 / buy_price_2021 ) * 100

    kosdaq_2019_real = kosdaq_2019.copy()
    kosdaq_2019 = kosdaq_2019.append(kosdaq_2020)
    kosdaq_2019 = kosdaq_2019.append(kosdaq_2021)
    #exit()
    Date_list = [KOSDAQ_2019.index, KOSDAQ_2020.index, KOSDAQ_2021.index]

    plt.figure(figsize=(20,10))
    
    models = []
    
    # VGG16 - KOSDAQ
    profit_2019_kosdaq_2 = pd.read_csv('../profit_csv/test_kosdaq/test_2019_kosdaq_vgg16_train_kosdaq_Batch128_Epochs8_Dropout20_profit.csv')
    profit_2020_kosdaq_2 = pd.read_csv('../profit_csv/test_kosdaq/test_2020_kosdaq_vgg16_train_kosdaq_Batch128_Epochs8_Dropout20_profit.csv')
    profit_2021_kosdaq_2 = pd.read_csv('../profit_csv/test_kosdaq/test_2021_kosdaq_vgg16_train_kosdaq_Batch128_Epochs8_Dropout20_profit.csv')
    vgg16_kosdaq_2 = [profit_2019_kosdaq_2, profit_2020_kosdaq_2, profit_2021_kosdaq_2]
    models.append(vgg16_kosdaq_2)

    # VGG16 - KOSPI, KOSDAQ
    profit_2019_kosdaq = pd.read_csv('../profit_csv/test_kosdaq/test_2019_kosdaq_vgg16_train_kospi_and_kosdaq_Batch64_Epochs8_Dropout20_profit.csv')
    profit_2020_kosdaq = pd.read_csv('../profit_csv/test_kosdaq/test_2020_kosdaq_vgg16_train_kospi_and_kosdaq_Batch64_Epochs8_Dropout20_profit.csv')
    profit_2021_kosdaq = pd.read_csv('../profit_csv/test_kosdaq/test_2021_kosdaq_vgg16_train_kospi_and_kosdaq_Batch64_Epochs8_Dropout20_profit.csv')
    vgg16_kosdaq = [profit_2019_kosdaq, profit_2020_kosdaq, profit_2021_kosdaq]
    models.append(vgg16_kosdaq)
    
    # EfficientNet B7 - KOSDAQ
    profit_2019_kosdaq_effi = pd.read_csv('../profit_csv/test_kosdaq/test_2019_kosdaq_efficientNetB7_train_kosdaq_Batch64_Dropout30_profit.csv')
    profit_2020_kosdaq_effi = pd.read_csv('../profit_csv/test_kosdaq/test_2020_kosdaq_efficientNetB7_train_kosdaq_Batch64_Dropout30_profit.csv')
    profit_2021_kosdaq_effi = pd.read_csv('../profit_csv/test_kosdaq/test_2021_kosdaq_efficientNetB7_train_kosdaq_Batch64_Dropout30_profit.csv')
    effi_kosdaq = [profit_2019_kosdaq_effi, profit_2020_kosdaq_effi, profit_2021_kosdaq_effi]
    models.append(effi_kosdaq)

    # EfficientNet B7 - KOSPI, KOSDAQ
    profit_2019_kospi_effi = pd.read_csv('../profit_csv/test_kosdaq/test_2019_kosdaq_efficientNetB7_train_kospi_and_kosdaq_Batch32_Dropout35_profit.csv')
    profit_2020_kospi_effi = pd.read_csv('../profit_csv/test_kosdaq/test_2020_kosdaq_efficientNetB7_train_kospi_and_kosdaq_Batch32_Dropout35_profit.csv')
    profit_2021_kospi_effi = pd.read_csv('../profit_csv/test_kosdaq/test_2021_kosdaq_efficientNetB7_train_kospi_and_kosdaq_Batch32_Dropout35_profit.csv')
    effi_kospi = [profit_2019_kospi_effi, profit_2020_kospi_effi, profit_2021_kospi_effi]
    models.append(effi_kospi)
    
    for i in range(len(models)):
      cumul_profit_2019 = np.array(calculate_cumul_profit(models[i][0], probability[i], '20190101', '20191231', topN, isTopN))
      cumul_profit_2020 = np.array(calculate_cumul_profit(models[i][1], probability[i], '20200101', '20201231', topN, isTopN))
      cumul_profit_2021 = np.array(calculate_cumul_profit(models[i][2], probability[i], '20210101', '20211231', topN, isTopN))

      last_2019 = cumul_profit_2019[-1]
      last_2020 = cumul_profit_2020[-1]
      last_2021 = cumul_profit_2021[-1]
      
      cumul_profit_topN_list = list(cumul_profit_2019) + list(cumul_profit_2020) + list(cumul_profit_2021)
      plt.plot(Date_all, cumul_profit_topN_list, color[i], label=f'{label[i]}', linewidth=2)
      plt.text(Date_all[-1], text[i], '%.2f%%' %last_2021, color=color[i], ha='center', va='bottom', size = 15)
      plt.text(Date_all[len(cumul_profit_2019) - 1], text[i], '%.2f%%' %last_2019, color=color[i], ha='center', va='bottom', size = 15)
      plt.text(Date_all[len(cumul_profit_2019) + len(cumul_profit_2020) - 1], text[i], '%.2f%%' %last_2020, color=color[i], ha='center', va='bottom', size = 15)

    
    
    # # Plotting KOSPI Index
    # kospi_all = kospi_2019.copy()
    # kospi_all = kospi_all.append(kospi_2020)
    # kospi_all = kospi_all.append(kospi_2021)
    # plt.plot(Date_all_KOSPI, kospi_all, 'green', label='KOSPI INDEX')
    # plt.text(Date_all[-1], -80, '%.1f%%' %kospi_2021[-1], color='green', ha='center', va='bottom', size = 15)
    # plt.text(Date_all[len(kospi_2019) - 1],-80, '%.1f%%' %kospi_2019[-1], color='green', ha='center', va='bottom', size = 15)
    # plt.text(Date_all[len(kospi_2019) + len(kospi_2020) - 1], -80, '%.1f%%' %kospi_2020[-1], color='green', ha='center', va='bottom', size = 15)

    # Plotting KOSDAQ Index
    plt.plot(Date_all, kosdaq_2019, 'green', label='KOSDAQ INDEX')
    plt.text(Date_all[-1], -60, '%.1f%%' %kosdaq_2019[-1], color='green', ha='center', va='bottom', size = 15)
    plt.text(Date_all[len(kosdaq_2019_real) - 1],-60, '%.1f%%' %kosdaq_2019_real[-1], color='green', ha='center', va='bottom', size = 15)
    plt.text(Date_all[len(kosdaq_2019_real) + len(kosdaq_2020) - 1], -60, '%.1f%%' %kosdaq_2020[-1], color='green', ha='center', va='bottom', size = 15)
    
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.legend(loc='upper left')
    #plt.xticks(np.arange(0, len(Date_all), 10))
    plt.yticks(np.arange(-80, 85, 5))
    #plt.xticks(rotation=45)
    plt.grid(True)
    plt.title(f'{title}')
    plt.savefig(f'{output}')


      # 4-4. 그래프 이미지로 저장 후 plot 초기화, (3-3)에 대해서도 진행
      #plt.cla()
    
    return 0


def main():
  parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-s', '--start',
                          help='an input start date', default='2019-01-01')
  parser.add_argument('-e', '--end',
                          help='an input end date', default='2021-12-31')
  parser.add_argument('-f', '--forecast',
                        help='forecast', default=5)
  parser.add_argument('-o', '--output',
                        help='output png file path', required=True)
  parser.add_argument('-t', '--title',
                        help='graph title', required=True)
  parser.add_argument('-n', '--topN',
                        help='topN', type=int, default=20)
  parser.add_argument('-u', '--isTopN',
                        help='use topN or not', type=bool, default=True)
  args = parser.parse_args()
  start = args.start
  end = args.end
  fore = args.forecast
  output = args.output
  title = args.title
  isTopN = args.isTopN
  topN = args.topN
  
  predict_dataset(start, end, output, title, topN, isTopN)
      
  return 0


if __name__ == "__main__":
    main()
