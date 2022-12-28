# CNN 모델 KOSPI 968종목, KOSDAQ 1,629종목 실험
Reference paper : "Using Deep Learning Neural Networks and Candlestick Chart Representation to Predict Stock Market" by Rosdyna. Mangir Irawan Kusuma, Tang-Thi Ho, Wei-Chun Kao, Yu-Yen Ou and Kai-Lung Hua

## 1. Getting Started
Install packages with:
```
$ pip install -r requirements.txt
```

## 2. Data
데이터셋 생성 : https://github.com/VAIV-SKKU/Data.git


### 2-1. 데이터셋 구조 예시

```
.
├── KOSPI968
│   └── 224x224
│       └── Kospi
│           ├── Dataset
│           │   ├── 4%_01_2_5
│           │   │   ├── test
│           │   │   │   ├── 0
│           │   │   │   │   ├── A900140_2019-10-26_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_0.png
│           │   │   │   │   └── A900140_2019-12-14_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_0.png
│           │   │   │   │   └── ...
│           │   │   │   └── 1
│           │   │   │       ├── A900140_2019-11-30_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_1.png
│           │   │   │       └── A900140_2019-12-29_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_1.png
│           │   │   │       └── ...
│           │   │   ├── train
│           │   │   │   ├── 0
│           │   │   │   │   ├── A900140_2006-10-26_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_0.png
│           │   │   │   │   └── A900140_2006-12-14_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_0.png
│           │   │   │   │   └── ...
│           │   │   │   └── 1
│           │   │   │       ├── A900140_2006-11-30_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_1.png
│           │   │   │       └── A900140_2006-12-29_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_1.png
│           │   │   │       └── ...
│           │   │   └── valid
│           │   │   │   ├── 0
│           │   │   │   │   ├── A900140_2018-10-26_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_0.png
│           │   │   │   │   └── A900140_2018-12-14_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_0.png
│           │   │   │   │   └── ...
│           │   │   │   └── 1
│           │   │   │       ├── A900140_2018-11-30_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_1.png
│           │   │   │       └── A900140_2018-12-29_20_224x224_5_vol(X)_ ma(5,10,30)_MACD(X),(4%_01_2)_1.png
│           │   │   │       └── ...
│           │   └── 4%_01_2_5_pickle
│           │       ├── test
│           │       │   ├── 0
│           │       │   └── 1
│           │       ├── train
│           │       │   ├── 0
│           │       │   └── 1
│           │       └── valid
│           │           ├── 0
│           │           └── 1
│           └── Label
│               └── 4%_01_2_5
│                   ├── test_label.csv
│                   ├── train_label.csv
│                   └── valid_label.csv

```

### 2-2. Data Preprocessing

png 파일을 pickle 형식으로 변환

[CNN_Prediction/KOSPI968-and-KOSDAQ1629/Train/src/utils/png2pickle.py](https://github.com/VAIV-SKKU/CNN_Prediction/blob/main/KOSPI968-and-KOSDAQ1629/Train/src/utils/png2pickle.py) 사용

```
$ python png2pickle.py KOSPI968/224x224/Kospi/Dataset/4%_01_2_5
```

## 3. Training
+ VGG16
  + [CNN_Prediction/KOSPI968-and-KOSDAQ1629/Train/src/Vgg16_train.py](https://github.com/VAIV-SKKU/CNN_Prediction/blob/main/KOSPI968-and-KOSDAQ1629/Train/src/Vgg16_train.py) 사용
  + Arguments 설명
    + -i : 사용할 데이터셋 경로
    + -p : trading period (default : 20)
    + -d : image dimension (default : 224x224)
    + -c : image channel (default : 3, RGB)
    + -e : number of max epochs (default : 20)
    + -b : batch size
    + -v : version name (학습된 모델명 구분을 위해 사용)
    + -r : drop-out rate
    + -o : accuracy, recall 등의 test 결과 저장할 txt file 경로
```
$ python Vgg16_train.py -i KOSPI968/224x224/Kospi/Dataset/4%_01_2_5 -p 20 -d 224 -c 3 -e 20 -b 128 -v model1 -r 0.2 -o outputresult.txt
```


## 4. Testing
### 4-1. General metrics

### 4-2. Profit

