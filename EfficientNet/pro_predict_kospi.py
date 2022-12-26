import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import sys, traceback
import logging
logging.basicConfig(level=logging.ERROR)
import math
import json
import sys
import imageio
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import *
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils import dataset_sampling as dataset
import argparse

import time
from datetime import timedelta
from datetime import datetime
import pandas as pd
from keras.models import load_model

def build_dataset(data_directory, img_width):
    X, y, tags, tickers, date = dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("test size : {}".format(train_size))
    feature = X
    print(y)
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes, tickers, date

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--dir',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-m', '--source',
                        help='a path of the model checkpoints', required=True)
    parser.add_argument('-o', '--output',
                        help='csv output file path', type=str, required=True)
    parser.add_argument('-s', '--start',
                        help='start date', type=str, required=True)
    parser.add_argument('-e', '--end',
                        help='end date', type=str, required=True)



    args = parser.parse_args()
    data_directory = args.dir
    #dataset_name = data_directory.split('/')[1]
    path_model = args.source
    output_csv = args.output
    start_date = args.start
    end_date = args.end
    
    s_date = datetime.strptime(start_date, "%Y-%m-%d")
    e_date = datetime.strptime(end_date, "%Y-%m-%d")
    ticker_data = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kospi.csv')
    ticker_list = ticker_data['Symbol'].values
    # Testing
    model = keras.models.load_model(path_model)
    tickers = []
    date = []
    pred_01 = []
    prob_list = []



    while s_date<=e_date:
      try : 
        print(s_date)
        d=[]
        temp_date = s_date.strftime('%Y-%m-%d')
        
        for t in ticker_list:
          try : 
            file_path = data_directory + f'/A{t}_{temp_date}_20_224x224.png'
            if os.path.isfile(file_path):
              d.append(file_path)
          except Exception as e2:
            print(e2)
        file_list =[]
        for i in range(len(d)):
            p = d[i].split('/')[-1]
            ticker = p.split('_')[0]
            file_list.append(ticker)
        filenames = d
        X=[]
        for filename in filenames:

          img = imageio.imread(filename)

          X.append(img)

        predictSet = np.array(X).astype(np.float32)
        try:
          predictSet = predictSet[:,:,:,:3]
        except Exception as e2:
          print(e2)
        

        predicted = model.predict(predictSet)

        pred = np.argmax(predicted, axis=1)
        probability = np.max(predicted, axis=1)

        for i in pred:
          pred_01.append(i)
        for i in probability:
          prob_list.append(i)
        for i in file_list:
          tickers.append(i)
        for i in range(len(file_list)):
          date.append(temp_date)

        

      except :
        logging.error(traceback.format_exc())
      s_date = s_date + timedelta(days=1)


    
    
    #print(y_pred)
    df = pd.DataFrame({'Ticker':tickers, 'Date':date, 'Predicted':pred_01, 'Probability':prob_list})
    model_name = path_model.split('.')[0]
    df.sort_values(by=['Date', 'Ticker'], axis=0, inplace=True)
    df.reset_index(inplace=True)
    df.to_csv(output_csv, mode='w')
    
    
    

if __name__ == "__main__":
    main()