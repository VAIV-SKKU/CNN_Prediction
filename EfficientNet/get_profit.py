from keras.models import load_model
import sys, traceback
import logging
logging.basicConfig(level=logging.ERROR)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


def predict_dataset(base_dir,   start, end, model_list, fore, pr, year):
    start_time = time.time()
    model1 = pd.read_csv(model_list[0])
    st_date = datetime.strptime(start, "%Y-%m-%d")
    e_date = datetime.strptime(end,"%Y-%m-%d" )
    end_date = e_date.strftime("%Y%m%d")
    ticker_data = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kospi.csv')
    ticker_list = ticker_data['Symbol'].values
    start_date = st_date.strftime("%Y%m%d")
    
    
    count=0
    df = data.get_data_yahoo("^KS11", start_date, end_date)



    buy_price = df['Close'].iloc[0]
    diff = df['Close'] - buy_price

    rate = ( diff / buy_price ) * 100

    df['Rate'] = rate

    cumulative_rate = rate.copy()
    for i in range(1, len(rate)):
        cumulative_rate[i] = rate[i] + cumulative_rate[i-1]

    df['Cumulative rate'] = cumulative_rate

    df.to_csv(f'/home/ubuntu/2022_VAIV_SeoHwan/make_graph/graph/KOSPI_INDEX_{start_date}_{end_date}.csv', mode="w")
    kospi_csv_data = pd.read_csv(f'/home/ubuntu/2022_VAIV_SeoHwan/make_graph/graph/KOSPI_INDEX_{start_date}_{end_date}.csv')
    color = []
    color_for_com = []
    plt.figure(figsize=(20,10))
    trans_count = 0
    trans_list = []
    color_for_com = ['blue']
    label_list = ['efficientB0']
    kospi_check = False
    date_check= False
    kospi_date_list = []
    kospi_list = []
    d_list = []
    probability = float(pr)
    prob_str = str(int(probability*100))
    print(type(year))
    total_date = 365
    if year =="2020":
      total_date=366
    for i in range(total_date):
      t_date = datetime.strptime(start, "%Y-%m-%d") + timedelta(days = i)
      d_list.append(t_date.strftime("%Y-%m-%d"))
    for mod in range(len(model_list)):
      check_profit = []
      year_buy_list = []
      date_list = []
      print(model_list[mod])
      s_date = datetime.strptime(start, "%Y-%m-%d")
      e_date = datetime.strptime(end,"%Y-%m-%d" )
      d = []
      real_start = None
      start_data =0
      start_check=False
      
      no_list=[]
      cu_kospi = 0
      end_date = e_date.strftime("%Y%m%d")
      profit_list = []
      ac_profit =0
      ac_profit_no =0
      transaction=0
      forecast_list = []
      while s_date<=e_date:
        #/////////////////
        try : 
          if kospi_check == False:
            print(kospi_csv_data[kospi_csv_data['Date']==s_date.strftime('%Y-%m-%d')].values)
            rate = kospi_csv_data[kospi_csv_data['Date']==s_date.strftime('%Y-%m-%d')].values[0][7]
            # cu_kospi  = cu_kospi + rate
            print(rate)
            kospi_list.append(rate)
            kospi_date_list.append(s_date.strftime('%Y-%m-%d'))
        except :
          ko = None
          if len(kospi_list)==0:
            ko = 0
          else : 
            ko=kospi_list[-1]
          kospi_list.append(ko)
          kospi_date_list.append(s_date.strftime('%Y-%m-%d'))

      #//////////////////////////
        count = count+1
        print(count)
        try : 
          a=time.time()
          d=[]
          pred_csv = pd.read_csv(model_list[mod])
          temp_date = s_date.strftime('%Y-%m-%d')
          pred_csv_data = pred_csv[pred_csv['Date']==temp_date].values
          pred = []
          probL = []
          for i in pred_csv_data:
            d.append(i[2])
            pred.append(i[4])
            probL.append(float(i[5]))
          X=[]
          b= time.time()
          
          print('time1 : ', (b-a))

          buy_list = []
          
          a = time.time()
          for i in range(len(pred)):
            if pred[i]== 1:
              if probL[i] >= probability:
                transaction = transaction+1
                p = d[i].split('/')[-1]
                ticker = p.split('_')[0]
                buy_list.append(ticker)
          csv_date = s_date.strftime('%Y%m%d')
          trans_list.append(len(buy_list))
          trans_count= trans_count+len(buy_list)
          buy_profit_list = []
          buy_profit_no_list = []
          sum =0
          pred_date = None

          b = time.time()
          print('time2 : ', b-a)

          a = time.time()
          print('day tran : ', len(buy_list))
          for t in buy_list:

            try:
            
              stock_data = pd.read_csv(f'/home/ubuntu/2022_VAIV_SeoHwan/make_graph/{year}_stcok_data/{t}.csv',encoding='CP949')
              now_row = stock_data[stock_data['Date']==int(csv_date)]
              index = int(now_row.values[0][0])
              pred_row = stock_data[stock_data['Unnamed: 0']==index+int(fore)]
              now_close = float(now_row.values[0][3])
              pred_close = float(pred_row.values[0][3])
              pred_date = str(int(pred_row.values[0][2]))
              pro = (pred_close-now_close)/now_close *100
              pro_no = (pred_close*0.9975-now_close)/now_close *100
              buy_profit_list.append(pro)
              buy_profit_no_list.append(pro_no)
              sum = sum + pred_close
            
            except Exception as e:
              print(e)
          b = time.time()

          print('time3 : ', b-a)
          kospi_avr = sum / len(buy_profit_list)

          sum_profit =0
          for i in range(len(buy_profit_list)):
            sum_profit = sum_profit + buy_profit_list[i]

          sum_profit_no=0
          for i in range(len(buy_profit_no_list)):
            sum_profit_no = sum_profit_no + buy_profit_no_list[i]

          avr = (sum_profit/len(buy_profit_list))/float(fore)
          avr_no = (sum_profit_no/len(buy_profit_no_list))/float(fore)
          check_profit.append(avr_no)
          ac_profit = ac_profit+avr
          ac_profit_no = ac_profit_no+avr_no
          profit_list.append(ac_profit)
          no_list.append(ac_profit_no)
          date_list.append(datetime.strptime(pred_date, '%Y%m%d').strftime('%Y-%m-%d'))
          forecast_date = s_date+timedelta(days=int(fore))
          forecast_list.append(forecast_date)
          year_buy_list.append(buy_list)
          if start_check==False:
            start_check = True
            start_data= kospi_avr

         
          
        except:

          check_profit.append(0)
          profit_list.append(ac_profit)
          no_list.append(ac_profit_no)
          year_buy_list.append(['nothing'])
        s_date = s_date + timedelta(days=1)



      print("date_list : ",date_list)
      print("profit_list : ",profit_list)
      print("no_list : ",no_list)
      

      

      plt.plot(d_list, no_list, color_for_com[mod], label= label_list[mod]+'(commission)')
      kospi_check  = True
      date_check = True
      f_output = open('/home/ubuntu/2022_VAIV_SeoHwan/make_graph/check_profit.txt', 'a')
      f_output.write(f'{model_list[mod]}(commission)(prob{prob_str})\n')
      for e in range(len(d_list)):
        f_output.write(f'[{d_list[e]}] profit : {check_profit[e]}  && cu_profit : {no_list[e]}  <{year_buy_list[e]}>\n')
      f_output.write('================================================================\n')
      f_output.close()
      plt.text(d_list[-1],  no_list[-1]+ 0.25, '%.2f' %no_list[-1], ha='center', va='bottom', size = 12)
      f_output = open('/home/ubuntu/2022_VAIV_SeoHwan/make_graph/cumulative_profit3.txt', 'a')
      f_output.write(f'{model_list[mod]}(commission)(prob{prob_str}) : {no_list[-1]}\n')
      f_output.write(f'trans count : {transaction}\n')
      f_output.write(f'trans day count : {transaction/246}\n')
      f_output.write('================================================================\n')
      f_output.close()

 
    
    f_output = open('/home/ubuntu/2022_VAIV_SeoHwan/make_graph/cumulative_profit3.txt', 'a')
    f_output.write(f'kospi(commission) : {kospi_list[-1]}\n')
    f_output.write(f'last transaction date : {forecast_list[-1].strftime("%Y%m%d")}\n')
    f_output.write('================================================================\n')
    
    f_output.close()

    end_time = time.time()

    print("time : ", end_time-start_time)
    
    return 0


def main():
  parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-s', '--start',
                          help='an input start date', required=True)
  parser.add_argument('-e', '--end',
                          help='an input end date', required=True)
  parser.add_argument('-f', '--forecast',
                        help='forecast', required=True)
  parser.add_argument('-m', '--model',
                        help='model path', required=True)
  parser.add_argument('-t', '--threshold',
                        help='threshold', required=True)   
  parser.add_argument('-d', '--dimension',
                        help='image size', required=True) 
  parser.add_argument('-y', '--year',
                        help='test year', required=True)               
  args = parser.parse_args()
  start = args.start
  end = args.end
  fore = args.forecast
  path = args.model
  pr = args.threshold
  size = args.dimension
  year = args.year

  rise={}
  model = None
  vgg_rise = []
  effi_rise = []
  ##########################################################################################
  model1_path = path
  ##########################################################################################
  model_list = []
  
  model_list.append(model1_path)

  predict_dataset(f'/home/ubuntu/2022_VAIV_Dataset/Image/1/{size}x{size}/Kosdaq',start, end ,model_list, fore, pr, year)
      
  return 0


if __name__ == "__main__":
    main()