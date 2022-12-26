from cmath import nan
import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import trange, tqdm
from pandas_datareader import data

def calculate_profit(input_csv, year, forecast):
      print("== Process 1 : Calculating Profit Per Transaction ==")
      
      input_df = pd.read_csv(input_csv)
      input_df = input_df.loc[input_df['Predicted'] == 1]
      print(input_df)
      
      profit_list = []
      for row in tqdm(input_df.values):
            date = row[3]
            now_date_bar = datetime.strptime(date, "%Y-%m-%d")
            now_date = now_date_bar.strftime("%Y%m%d")
            ticker = row[2]
            
            stock_data = pd.read_csv(f'..Stock_Data/{year}_kosdaq_data/{ticker}.csv',encoding='CP949')
            try:
                  now_index = stock_data[stock_data['Date'] == int(now_date)].index
                  pred_index = now_index + forecast
                  now_close = stock_data.iloc[now_index]['Close'].values[0]
                  pred_close = stock_data.iloc[pred_index]['Close'].values[0]
                  profit = (pred_close*0.9975 - now_close)/(now_close*int(forecast))*100
                  profit_list.append(float(profit))
            except:
                  profit_list.append(None)
                  print(date)
                  print(ticker)
                  continue
            
      input_df['Profit'] = profit_list
      return input_df

def make_report(profit_df, start, end, forecast, year, topN):
      KOSDAQ = data.get_data_yahoo("^KQ11", start, end)
      nb_date = len(KOSDAQ.index)
      print(f'KOSDAQ Date : {nb_date}')
      
      prob_list = np.array(list(range(50, 105, 5)))
      prob_list = prob_list/100
      #print(len(thres_list))
      
      l_pred_conds = []   # 예측 확률값 범위
      l_stock_type = []   # 산업
      l_topN = []         # 하루에 (확률값 기준) 상위 몇 종목을 거래할 것인지 (VAIV : 20종목)
      l_tr_date = []      # 총 거래일
      l_mean_tr_per_day = []   # 일별 평균 구매수
      l_rate_tr_date = [] # 총 거래일 비중 = 총 거래일 / 전체 날짜 수 (2019년도 : 246일, 2020년도 : 247일, 2021년도 : 247일)
      l_r_5 = [] # 총 거래일 중에 5개 이상 거래된 날의 비율
      l_r_10 = [] # 총 거래일 중에 10개 이상 거래된 날의 비율
      l_r_20 = [] # 총 거래일 중에 20개 이상 거래된 날의 비율
      l_avr_profit = []
      l_rate_profit = []
      
      l_stock_rate_mp1 = []
      l_stock_rate_p1p5 = []
      l_stock_rate_p5p10 = []
      l_stock_rate_p10 = []
      l_stock_rate_m1m5 = []
      l_stock_rate_m5m10 = []
      l_stock_rate_m10 = []
      print("== Process 2 : per probability ==")
      for i in range(len(prob_list) - 1):
            day_5 = 0
            day_10 = 0
            day_20 = 0
            mp1 = 0
            p1p5 = 0
            p5p10 = 0
            p10 = 0
            m1m5 = 0
            m5m10 = 0
            m10 = 0
            start_prob = prob_list[i]   # 범위에 포함
            end_prob = prob_list[i+1]   # 범위에 포함x
            
            pred_conds = f'{start_prob : .2f}~{end_prob : .2f}'
            print()
            l_pred_conds.append(pred_conds)
            l_stock_type.append('KOSDAQ')
            l_topN.append(topN)
            
            profit_df_range = profit_df.copy()
            profit_df_range = profit_df_range.loc[profit_df_range['Probability'] >= start_prob]
            profit_df_range = profit_df_range.loc[profit_df_range['Probability'] < end_prob]
            
            date_list = list(set(profit_df_range['Date'].values))
            tr_date = len(date_list)
            l_tr_date.append(tr_date)
            
            # 일별 평균 구매수 구하기
            # 1. 날짜별로 묶기
            # 2. 날짜별로 probability 기준으로 정렬 후, 상위 N개만 남기기
            print(f"pred_conds : {pred_conds}")
            count_tr_per_day = 0
            profit_per_day = 0
            groupby_date = profit_df_range.groupby('Date')
            total_tr = 0
            new_profit_df_range = pd.DataFrame(columns=profit_df_range.columns)
            for date, group_data in tqdm(groupby_date):
                  group_data = group_data.sort_values('Probability', ascending=False)
                  
                  # 상위 N개만 남기기
                  if len(group_data) > topN:
                        group_data = group_data.iloc[:topN]
                  count_tr_per_day += len(group_data)
                  profit_per_day += group_data['Profit'].sum()
                  #print(profit_per_day)
                  
                  new_profit_df_range += group_data
                  total_tr += len(group_data)
                  
                  if len(group_data) >= 5:
                        day_5 += 1
                  if len(group_data) >= 10:
                        day_10 += 1
                  if len(group_data) >= 20:
                        day_20 += 1
                  #print(len(data))
                  
                  mp1_data = group_data.loc[group_data['Profit'] > -1]
                  mp1_data = mp1_data.loc[mp1_data['Profit'] < 1]
                  mp1 += len(mp1_data)
                  
                  p1p5_data = group_data.loc[group_data['Profit'] >= 1]
                  p1p5_data = p1p5_data.loc[p1p5_data['Profit'] < 5]
                  p1p5 += len(p1p5_data)
                  
                  p5p10_data = group_data.loc[group_data['Profit'] >= 5]
                  p5p10_data = p5p10_data.loc[p5p10_data['Profit'] < 10]
                  p5p10 += len(p5p10_data)
                  
                  p10_data = group_data.loc[group_data['Profit'] >= 10]
                  p10 += len(p10_data)
                  
                  m1m5_data = group_data.loc[group_data['Profit'] > -5]
                  m1m5_data = m1m5_data.loc[m1m5_data['Profit'] <= -1]
                  m1m5 += len(m1m5_data)
                  
                  m5m10_data = group_data.loc[group_data['Profit'] > -10]
                  m5m10_data = m5m10_data.loc[m5m10_data['Profit'] <= -5]
                  m5m10 += len(m5m10_data)
                  
                  m10_data = group_data.loc[group_data['Profit'] <= -10]
                  m10 += len(m10_data)
            
            #total_tr = len(new_profit_df_range)
            if tr_date == 0:
                  l_r_5.append(None)
                  l_r_10.append(None)
                  l_r_20.append(None)
                  l_mean_tr_per_day.append(None)
                  l_rate_tr_date.append(None)
                  l_avr_profit.append(None)
                  l_stock_rate_mp1.append(None)
                  l_stock_rate_p1p5.append(None)
                  l_stock_rate_p5p10.append(None)
                  l_stock_rate_p10.append(None)
                  l_stock_rate_m1m5.append(None)
                  l_stock_rate_m5m10.append(None)
                  l_stock_rate_m10.append(None)
                  
            else:
                  r_5 = round(day_5/tr_date, 2)
                  l_r_5.append(r_5)
            
                  r_10 = round(day_10/tr_date, 2)
                  l_r_10.append(r_10)
            
                  r_20 = round(day_20/tr_date, 2)
                  l_r_20.append(r_20)
            
                  mean_tr_per_day = count_tr_per_day/tr_date
                  l_mean_tr_per_day.append(round(mean_tr_per_day,2))
            
                  rate_tr_date = round(tr_date/nb_date, 2)
                  l_rate_tr_date.append(rate_tr_date)
            
                  print(profit_per_day)
                  avr_profit = round(profit_per_day/(total_tr),4)
                  print(avr_profit)
                  l_avr_profit.append(avr_profit)
            
                  l_stock_rate_mp1.append(round(mp1/total_tr, 2))
                  l_stock_rate_p1p5.append(round(p1p5/total_tr, 2))
                  l_stock_rate_p5p10.append(round(p5p10/total_tr, 2))
                  l_stock_rate_p10.append(round(p10/total_tr, 2))
                  l_stock_rate_m1m5.append(round(m1m5/total_tr, 2))
                  l_stock_rate_m5m10.append(round(m5m10/total_tr, 2))
                  l_stock_rate_m10.append(round(m10/total_tr, 2))
            
      
      output_df = pd.DataFrame({'pred_conds' : l_pred_conds, '산업' : l_stock_type, 'topN' : l_topN, '총 거래일' : l_tr_date,
                                '일별 평균 구매수' : l_mean_tr_per_day, '총 거래일 비중' : l_rate_tr_date, 
                                'R_5' : l_r_5, 'R_10' : l_r_10, 'R_20' : l_r_20,
                                '평균 수익률(%)' : l_avr_profit,
                                '+-1%' : l_stock_rate_mp1, '1~5%' : l_stock_rate_p1p5, '5~10%' : l_stock_rate_p5p10, '10%~' : l_stock_rate_p10,
                                '(M)1~5%' : l_stock_rate_m1m5, '(M)5~10%' : l_stock_rate_m5m10, '(M)10%~' : l_stock_rate_m10})
            
      return output_df         

def calculate_cumul_profit(profit_df, threshold, start, end, topN, isTopN):
      KOSDAQ = data.get_data_yahoo("^KQ11", start, end)
      Date_list = KOSDAQ.index
      
      prob_list = np.array(list(range(50, 105, 5)))
      prob_list = prob_list/100
          
      profit_df_range = profit_df.copy()
      profit_df_range = profit_df_range.loc[profit_df_range['Probability'] >= threshold]
            
      # 일별 평균 구매수 구하기
      # 1. 날짜별로 묶기
      # 2. 날짜별로 probability 기준으로 정렬 후, 상위 N개만 남기기
      groupby_date = profit_df_range.groupby('Date')

      daily_profit = {}
      count_total_date = 0
      count_tr_by_date = 0
      print("Slicing TopN")
      for date, group_data in tqdm(groupby_date):
            count_total_date += 1
            group_data = group_data.sort_values('Probability', ascending=False)
                  
            # 상위 N개만 남기기
            if (isTopN == True) and (len(group_data) > topN):
                  group_data = group_data.iloc[:topN]
                  
            tr_per_day = len(group_data)

            count_tr_by_date += tr_per_day
            
            profit_sum = group_data['Profit'].sum()
                  
            profit_today = profit_sum / tr_per_day
            #print(profit_today)
            daily_profit[date] = profit_today
      
      cumul_profit_list = []
      cumul_profit = 0
      for temp_date in tqdm(Date_list):
            #print(f'temp date : {temp_date}')
            today = temp_date.strftime("%Y-%m-%d")
            
            try:
                  profit_today = daily_profit[today]
                  #print(profit_today)
                  cumul_profit += profit_today
                  cumul_profit_list.append(cumul_profit)
                  
            except:
                  cumul_profit_list.append(cumul_profit)
                  continue
      
      #print(cumul_profit_list)
      #print(len(cumul_profit_list))
      
      print(f'total # of TR : {int(count_tr_by_date)}')
      print(f'TR per day : {int(count_tr_by_date/len(Date_list))}')

      return cumul_profit_list
            
            


# 새로 만들어질 csv 파일 이름 : profit_VAIV_(input_csv파일이름)
def main():
      parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser.add_argument('-c', '--csv',
                          help='an input csv file', required=True)
      parser.add_argument('-f', '--forecast',
                          help='forecast interval', type=int, default=5)
      parser.add_argument('-s', '--start',
                          help='an input start date', default='20190101')
      parser.add_argument('-e', '--end',
                          help='an input end date', default='20191231')
      parser.add_argument('-y', '--year',
                          help='year', required=True)
      parser.add_argument('-n', '--topN',
                          help='topN', type=int, default=20)
      parser.add_argument('-u', '--isTopN',
                          help='use topN or not', type=bool, default=True)
      parser.add_argument('-o', '--output',
                          help='output directory path', required=False)
  
      args = parser.parse_args()
      input_csv = args.csv
      start = args.start
      end = args.end
      year = args.year
      forecast = args.forecast
      topN = args.topN
      isTopN = args.isTopN
      output_dir = args.output
  
      name = input_csv.split('/')[-1]
  
      output_csv = f'{output_dir}/profit_VAIV_top{topN}_{name}'
      profit_csv = input_csv.split('.')[0] + '_profit.csv'
  
      profit_df = None
      
      if os.path.isfile(profit_csv):
            profit_df = pd.read_csv(profit_csv)
            print("Profits are already calculated.")
      else:
            profit_df = calculate_profit(input_csv, year, forecast)
            profit_df.to_csv(profit_csv)
      
      print(profit_df)
      #calculate_cumul_profit(profit_df, 0.67, '20210101', '20211231', 20)
      output_df = make_report(profit_df, start, end, forecast, year, topN)
      
      output_df.to_csv(output_csv, encoding='CP949')
if __name__ == "__main__":
      main()