import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import math
import json
import sys

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
from utils import dataset_testing as dataset
import argparse

import time
from datetime import timedelta
from datetime import datetime

import pandas as pd


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
                        help='an input directory of dataset', default=None)
    parser.add_argument('-m', '--source',
                        help='a path of the model checkpoints', default=None)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=224)
    parser.add_argument('-c', '--csv',
                        help='csv input/output file path', type=str, required=True)      
    parser.add_argument('-t', '--threshold',
                        help='probability threshold', type=float, required=True)            
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="output_test.txt")

    args = parser.parse_args()
    data_directory = args.dir
    path_model = args.source
    csv = args.csv
    threshold = args.threshold

    
    model_name = csv.split('/')[-1]
    model_name = model_name.split('.')[0]
    # Testing
    if not path_model == None and not data_directory == None:
        # prediction 결과 저장
        print("loading dataset")
        X_test, Y_test, nb_classes, tickers, date = build_dataset(
            "{}/test".format(data_directory), args.dimension)
        print("number of classes : {}".format(nb_classes))
        model = keras.models.load_model(path_model)       
        
        predicted = model.predict(X_test)
        
        probability = np.max(predicted, axis=1) #예측 확률
        y_pred = np.argmax(predicted, axis=1)   #예측 레이블
        Y_test = np.argmax(Y_test, axis=1)  #정답 레이블
        
        df = pd.DataFrame({'Ticker':tickers, 'Date':date, 'Predicted':y_pred, 'Label':Y_test, 'Probability':probability})
        
        df.sort_values(by=['Date', 'Ticker'], axis=0, inplace=True)
        df.reset_index(inplace=True)
        df.to_csv(csv, mode='w')
        
    prediction_result = pd.read_csv(csv)
    
    # 일정 확률 이상인 것만 걸러내기
    prediction_result = prediction_result.loc[prediction_result['Probability'] >= threshold]
    
    Y_test = prediction_result['Label'].values
    y_pred = prediction_result['Predicted'].values
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    if tp == 0:
        tp = 1
    if tn == 0:
        tn = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1
    TPR = float(tp)/(float(tp)+float(fn))
    FPR = float(fp)/(float(fp)+float(tn))
    accuracy = round((float(tp) + float(tn))/(float(tp) +
                                              float(fp) + float(fn) + float(tn)), 3)
    specitivity = round(float(tn)/(float(tn) + float(fp)), 3)
    sensitivity = round(float(tp)/(float(tp) + float(fn)), 3)
    precision = round(float(tp)/(float(tp) + float(fp)), 3)
    mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
        (float(tp)+float(fp))
        * (float(tp)+float(fn))
        * (float(tn)+float(fp))
        * (float(tn)+float(fn))
    ), 3)
    f1score = round(float(precision) * float(sensitivity) * 2 / (float(precision) + float(sensitivity)), 3)
    f_output = open(args.output, 'a')
    f_output.write('=======\n')
    f_output.write('only for testing model\n')
    f_output.write('from model path - {}\n'.format(path_model))
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write("precision : {}\n".format(precision))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("f1 score : {}\n".format(f1score))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write('=======\n')
    f_output.close()
    
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print('sensitivity: {}'.format(sensitivity))
    print('f1 score: {}'.format(f1score))

if __name__ == "__main__":
    main()