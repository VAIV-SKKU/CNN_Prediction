import os
import sys, traceback
import logging
logging.basicConfig(level=logging.ERROR)
import math
import json
import sys
import imageio
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data
import argparse
import time
from datetime import timedelta
from datetime import datetime
import pandas as pd
from PIL import Image

def regenerate( start, end, predict_file,    kos, pr, output_path):
  pred_csv = pd.read_csv(predict_file)
  tickers = []
  date=[]
  day_close=[]
  first_day=[]
  second_day=[]
  third_day=[]
  fourth_day=[]
  fifth_day=[]
  s_date = datetime.strptime(start, "%Y-%m-%d")
  e_date = datetime.strptime(end,"%Y-%m-%d" )


  stock_data_label = None
  count=0
  stock_data_label = 'stcok'
  if(kos=='Kosdaq'):
    stock_data_label = 'kosdaq'
  
  while s_date<=e_date:
    count = count+1
    check_start = time.time()
    print(count)
    try : 
      a=time.time()
      temp_date = s_date.strftime('%Y-%m-%d')
      temp_year = temp_date.split('-')[0]
      pred_csv_data = pred_csv[pred_csv['Date']==temp_date].values
    

      recommendList = []
      for i in pred_csv_data:
        tempList = []
        if(int(i[4])==1):
          if(float(i[5])>float(pr)):
          
            tempList.append(float(i[5]))
            tempList.append(i[2])
            tempList.append(i[4])
        if(len(tempList)!=0):
          recommendList.append(tempList)
      recommendList.sort(reverse=True)
      d=[]
      pred = []
      probL = []
      if(len(recommendList)<20):
        for i in range(len(recommendList)):
          probL.append(recommendList[i][0])
          d.append(recommendList[i][1])
          pred.append(recommendList[i][2])
      else:
        for i in range(20):
          probL.append(recommendList[i][0])
          d.append(recommendList[i][1])
          pred.append(recommendList[i][2])

      print("len : ", len(recommendList))
        

      buy_list = []
      
      for i in range(len(pred)):  
        buy_list.append(d[i])
      csv_date = s_date.strftime('%Y%m%d')
      sum =0
      pred_date = None
      for t in buy_list:
        
        try:
          stock_data = pd.read_csv(f'/home/ubuntu/2022_VAIV_SeoHwan/make_graph/{temp_year}_{stock_data_label}_data/{t}.csv',encoding='CP949')
          now_row = stock_data[stock_data['Date']==int(csv_date)]
          
          index = int(now_row.values[0][0])
          now_close = float(now_row.values[0][3])
          day_close.append(now_close)
          print(temp_date)
          try:
            first_row = stock_data[stock_data['Unnamed: 0']==index+int(1)]
            first_close = float(first_row.values[0][3])
            first_profit = ((first_close*0.9975-now_close)/now_close *100)/1
            first_day.append(first_profit)
          except:
            first_day.append(None)
          try:
            second_row = stock_data[stock_data['Unnamed: 0']==index+int(2)]
            second_close = float(second_row.values[0][3])
            second_profit = ((second_close*0.9975-now_close)/now_close *100)/2
            second_day.append(second_profit)
          except:
            second_day.append(None)
          try:
            third_row = stock_data[stock_data['Unnamed: 0']==index+int(3)]
            third_close = float(third_row.values[0][3])
            third_profit = ((third_close*0.9975-now_close)/now_close *100)/3
            third_day.append(third_profit)
          except:
            third_day.append(None)
          try:
            fourth_row = stock_data[stock_data['Unnamed: 0']==index+int(4)]
            fourth_close = float(fourth_row.values[0][3])
            fourth_profit = ((fourth_close*0.9975-now_close)/now_close *100)/4
            fourth_day.append(fourth_profit)
          except:
            fourth_day.append(None)
          try:
            fifth_row = stock_data[stock_data['Unnamed: 0']==index+int(5)]
            fifth_close = float(fifth_row.values[0][3])
            fifth_profit = ((fifth_close*0.9975-now_close)/now_close *100)/5
            fifth_day.append(fifth_profit)
          except:
            fifth_day.append(None)
          tickers.append(t)
          date.append(temp_date)
        
        except:
          tickers.append(t)
          day_close.append(None)
          first_day.append(None)
          second_day.append(None)
          third_day.append(None)
          fourth_day.append(None)
          fifth_day.append(None)
          date.append(temp_date)
          logging.error(traceback.format_exc())
          #print(e)
          
        

      
      
    

      
      
    except Exception as e:
      print(e)
    s_date = s_date + timedelta(days=1)


        



      
  df = pd.DataFrame({'Ticker':tickers, 'Date':date, 'day close':day_close, '1day':first_day, '2day':second_day, '3day':third_day, '4day':fourth_day, '5day': fifth_day})
  df.reset_index(inplace=True)
  df.to_csv(output_path, mode='w', index=False)











def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
   
    parser.add_argument('-s', '--start',
                        help='start date', type=str, required=True)
    parser.add_argument('-e', '--end',
                        help='end date', type=str, required=True)
    parser.add_argument('-k', '--kos',
                        help='Kospi or Kosdaq', type=str, required=True)
    parser.add_argument('-m', '--model',
                    help='effi / vgg',nargs='+', required=True)

    args = parser.parse_args()

    # kospi / kosdaq
    kos = args.kos
    # model list
    m = args.model
    start_date = args.start
    end_date = args.end

    output_path = '/home/ubuntu/2022_VAIV_Dataset/try/predict_csv/'
    
    s_date = datetime.strptime(start_date, "%Y-%m-%d")
    e_date = datetime.strptime(end_date, "%Y-%m-%d")
    ticker_data = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kosdaq.csv')
    if(kos == 'Kospi'):
      ticker_data = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kospi.csv')
    
    
    ticker_list = ticker_data['Symbol'].values
    #for i in range(len(ticker_list)):
    # csv path of model
    csv_file= None
    pr_list=[]
    label_list = []
    pr=0
    for i in range(len(m)):
      if(m[i]=='effi'):
        if(kos == 'Kospi'):
          csv_file= '/home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSPI/efficient_4.csv'
          output_path = output_path+'KOSPI/'
          pr=0.625
        elif(kos == 'Kosdaq'):
          csv_file='/home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSDAQ/efficient_kosdaq.csv'
          output_path = output_path+'KOSDAQ/'
          pr=0.55
      elif(m[i]=='vgg'):
        if(kos == 'Kospi'):
          csv_file='/home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSPI/vgg16_4.csv'
          output_path = output_path+'KOSPI/'
          pr=0.625
        elif(kos == 'Kosdaq'):
          csv_file='/home/ubuntu/2022_VAIV_Dataset/try/predict_csv/KOSDAQ/vgg16_kosdaq.csv'
          output_path = output_path+'KOSDAQ/'
          pr=0.675
    output_path = output_path+"new_"+csv_file.split("/")[-1]
    size = 224
    fore = 5
    print("pr: ", pr)
    regenerate(start_date, end_date ,csv_file,  kos, pr, output_path)
    

if __name__ == "__main__":
    main()