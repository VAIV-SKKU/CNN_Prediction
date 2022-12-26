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
from profit_kosdaq import *

def predict_dataset(csv, topN, isTopN, prob, year):

    # 224x224 VGG16
    profit = pd.read_csv(csv)
    cumul_profit = None
    if year == 2019:
      cumul_profit = np.array(calculate_cumul_profit(profit, prob, '20190101', '20191231', topN, isTopN))
    elif year == 2020:
      cumul_profit = np.array(calculate_cumul_profit(profit, prob, '20200101', '20201231', topN, isTopN))
    else:
      cumul_profit = np.array(calculate_cumul_profit(profit, prob, '20210101', '20211231', topN, isTopN))
      
    print(cumul_profit[-1])
    
    return 0


def main():
  parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-c', '--csv',
                        help='csv file with profit', required=True)
  parser.add_argument('-n', '--topN',
                        help='topN', type=int, default=20)
  parser.add_argument('-u', '--isTopN',
                        help='use topN or not', type=bool, default=True)
  parser.add_argument('-p', '--prob',
                        help='probability', type=float, default=0.5)
  parser.add_argument('-y', '--year',
                        help='year', type=int, required=True)
  args = parser.parse_args()
  csv = args.csv
  prob = args.prob
  isTopN = args.isTopN
  topN = args.topN
  year = args.year
  
  predict_dataset(csv, topN, isTopN, prob, year)
      
  return 0


if __name__ == "__main__":
    main()
