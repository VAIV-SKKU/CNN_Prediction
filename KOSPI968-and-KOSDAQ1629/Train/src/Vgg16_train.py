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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add, Dropout
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
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from utils import dataset as dataset
import argparse

import time
from datetime import timedelta

def build_dataset(data_directory, img_width):
    print("{ method - build_dataset }") 
    X, y, tags = dataset(data_directory, int(img_width))
    nb_classes = len(tags)
    #print(tags)
    #print(X)
    #print(y)
    #print(data_directory)
    #quit()
    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes

def build_model(SHAPE, nb_classes, dropout):
    model = VGG16(input_shape=SHAPE, include_top=False)
    
    x = model.output
    
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(dropout)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(dropout)(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    model = Model(model.input, x)

    return model

def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-p', '--tradingPeriod',
                        help='trading period', type=int, default=20)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=224)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=3)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    parser.add_argument('-v', '--phase',
                        help='?', type=str, default='01')
    parser.add_argument('-r', '--dropout',
                        help='dropout rate', type=float, default=0)
    # parser.add_argument('-o', '--optimizer',
    #                     help='choose the optimizer (rmsprop, adagrad, adadelta, adam, adamax, nadam)', default="adam")
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="output_large.txt")
    args = parser.parse_args()

    seq_len = args.tradingPeriod
    batch_size = args.batch_size
    phase = args.phase

    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    dropout = args.dropout

    print("* expected img_width :", img_width)   # 50
    print("* expected img_height :", img_height) # 50
    print("* expected channel :", channel)       # 3 -> 4

    SHAPE = (img_width, img_height, channel)
    bn_axis = 3 if K.image_data_format() == 'tf' else 1

    data_directory = args.input
    dataset_name = data_directory.split('/')[-1]
    
    
    print("loading dataset")
    X_train, Y_train, nb_classes = build_dataset(
        "{}/train".format(data_directory), args.dimension)
    X_test, Y_test, nb_classes = build_dataset(
        "{}/test".format(data_directory), args.dimension)
    print("number of classes : {}".format(nb_classes))
    X_val, Y_val, nb_classes = build_dataset(
        "{}/valid".format(data_directory), args.dimension)
    
    
    # Load model
    # It can be used to reconstruct the model identically.
    # reconstructed_model = keras.models.load_model("my_h5_model.h5")
    # make checkpoints
    model = build_model(SHAPE, 2, dropout)
    #print(model)
    #exit()
    #model = keras.models.load_model('checkpoints3/Vgg16_8_4%_01/phase03/model-016.h5')
    #print(model.summary())
    #exit()

    adam=keras.optimizers.Adam(learning_rate=1.0e-5)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    if not os.path.isdir(f'../checkpoints/{dataset_name}'):
        os.mkdir(f'../checkpoints/{dataset_name}')
    if not os.path.isdir(f'../checkpoints/{dataset_name}/{batch_size}batch'):
        os.mkdir(f'../checkpoints/{dataset_name}/{batch_size}batch')
    path_checkpoint = f'../checkpoints/{dataset_name}/{batch_size}batch/{phase}' + '/model-{epoch:03d}.h5'
    
    if not os.path.isdir(f'../logs/{dataset_name}'):
        os.mkdir(f'../logs/{dataset_name}')
    if not os.path.isdir(f'../logs/{dataset_name}/{batch_size}batch'):
        os.mkdir(f'../logs/{dataset_name}/{batch_size}batch')
    path_logs = f'../logs/{dataset_name}/{batch_size}batch/{phase}.csv'
    # Create a callback that saves the model's structure by five epochs
    callback_cp = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint, period=2)
    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, mode='min', verbose=1)
    callback_csv = tf.keras.callbacks.CSVLogger(path_logs)
    
    # Fit the model
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), callbacks=[callback_cp, callback_csv])
    #history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), callbacks=[callback_cp, callback_csv, callback_stop])
    #history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback_cp, callback_csv, callback_stop])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    path_model = f'../models/{dataset_name}/Epochs{len(acc)}_Batch{batch_size}_{phase}.h5'
    if not os.path.isdir(f'../models/{dataset_name}'):
        os.mkdir(f'../models/{dataset_name}')
    model.save(path_model, overwrite=True)
    
    epochs = range(1, len(acc) + 1)
 
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Accuracys')
    plt.legend()
    
    path_history = f'../history/{dataset_name}/Epochs{len(acc)}_Batch{batch_size}_{dataset_name}_{phase}'
    if not os.path.isdir(f'../history/{dataset_name}'):
        os.mkdir(f'../history/{dataset_name}')
    plt.savefig(f'{path_history}_accuracy.png')

    plt.cla()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()

    plt.savefig(f'{path_history}_loss.png')
    
    
    # del model  # deletes the existing model
    predicted = model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
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
    mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
        (float(tp)+float(fp))
        * (float(tp)+float(fn))
        * (float(tn)+float(fp))
        * (float(tn)+float(fn))
    ), 3)

    end_time = time.monotonic()
    f_output = open(args.output, 'a')
    f_output.write('=======\n')
    output_title = f'Vgg16_Epochs{len(acc)}_Batch{batch_size}_{dataset_name}_{phase}\n'
    f_output.write(output_title)
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write('specitivity: {}\n'.format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write("\nDuration : {}\n".format(timedelta(seconds=end_time - start_time)))
    f_output.write('=======\n')
    f_output.close()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))
    

if __name__ == "__main__":
    main()
