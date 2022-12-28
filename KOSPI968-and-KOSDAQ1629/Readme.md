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
### 3-1. VGG16
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
    + -o : accuracy, recall 등의 test 결과 저장할 txt file 경로 (.txt)
```
$ python Vgg16_train.py -i KOSPI968/224x224/Kospi/Dataset/4%_01_2_5 -p 20 -d 224 -c 3 -e 20 -b 128 -v model1 -r 0.2 -o outputresult.txt
```

### 3-2. Train 결과 저장 구조
```
.
├── checkpoints
│   └── <Dataset name>
│       └── <batch size>
│           ├── <version name>
│           │   ├── model-{epoch:03d}.h5  // checkpoint 마다 model 저장 (.h5 형식)
├── logs
│   └── <Dataset name>
│       └── <batch size>
│           ├── <version name>.csv        // epoch 마다 train acc, train loss, validation acc, validation loss 저장
├── models
│   └── <Dataset name>
│       └── Epochs{# of epochs}_Batch{batch size}_{version name}.h5    // 학습이 끝난 뒤 마지막 epoch까지 학습된 모델 저장 (최종 모델)
├── history
│   └── <Dataset name>
│       └── Epochs{# of epochs}_Batch{batch size}_{version name}_accuracy.png   // epoch에 따른 train accuracy, validation accuracy 결과 기록한 그래프를 png file로 저장
│       └── Epochs{# of epochs}_Batch{batch size}_{version name}_loss.png       // epoch에 따른 train loss, validation loss 결과 기록한 그래프를 png file로 저장
├── outputresult.txt    // 최종 모델 test 결과 저장

```
* Dataset_name : 입력한 Dataset 경로에서 가장 마지막 부분 (ex. "KOSPI968/224x224/Kospi/Dataset/4%_01_2_5"의 경우 Dataset_name은 "4%_01_2_5")


## 4. Performance Evaluation
### 4-1. General metrics
+ Accuracy
+ Precision
+ Sensitivity
+ F1 score

[CNN_Prediction/KOSPI968-and-KOSDAQ1629/Test/general_metrics/src/prediction_result.py](https://github.com/VAIV-SKKU/CNN_Prediction/blob/main/KOSPI968-and-KOSDAQ1629/Test/general_metrics/src/prediction_result.py) 사용

  + Arguments 설명
    + -i : 사용할 데이터셋 경로
    + -m : 테스트할 모델 경로 (.h5)
    + -o : 결과 저장할 txt file 경로 (.txt)
    + -c : 상세한 prediction 결과 저장할 csv file 경로 (.csv)
      + 이미지 정보(날짜, 종목), 예측 결과, 예측 확률
    + -t : 예측 확률값에 대한 threshold
    + -d : image dimension (default : 224x224)
```
$ python prediction_result.py -i KOSPI968/224x224/Kospi/Dataset/4%_01_2_5 -m model1.h5 -o output_predict.txt -c model1_prediction.csv -t 0.5 -d 224
```

### 4-2. Profit
#### 4-2-1. Make prediction file
테스트하고자 하는 모델의 2019, 2020, 2021년도에 대한 prediction 결과를 구한다.

+ KOSPI 968 종목 테스트 : [CNN_Prediction/KOSPI968-and-KOSDAQ1629/Test/profit/src/make_prediction_kospi.py](https://github.com/VAIV-SKKU/CNN_Prediction/blob/main/KOSPI968-and-KOSDAQ1629/Test/profit/src/make_prediction_csv_kospi.py)
  + Arguments 설명
    + -i : 이미지(.png) 폴더 경로, KOSPI 968 종목 전체 이미지가 들어 있어야 한다.
    + -s : prediction 결과를 구할 시작 날짜
    + -e : prediction 결과를 구할 마지막 날짜
    + -d : image dimension (default : 224)
    + -o : prediction 결과를 저장할 csv file path
      + 이미지 정보(날짜, 종목), 예측 결과, 예측 확률
    + -m : model path
```
// 2019 prediction 결과 생성
$ python make_prediction_kospi.py -i <KOSPI 968 종목 전체 이미지가 들어있는 디렉토리> -s 2019-01-01 -e 2019-12-31 -d 224 -o model1_kospi968_2019_prediction.csv -m model1.h5

// 2020 prediction 결과 생성
$ python make_prediction_kospi.py -i <KOSPI 968 종목 전체 이미지가 들어있는 디렉토리> -s 2020-01-01 -e 2020-12-31 -d 224 -o model1_kospi968_2020_prediction.csv -m model1.h5

// 2021 prediction 결과 생성
$ python make_prediction_kospi.py -i <KOSPI 968 종목 전체 이미지가 들어있는 디렉토리> -s 2021-01-01 -e 2021-12-31 -d 224 -o model1_kospi968_2021_prediction.csv -m model1.h5
```
+ KOSDAQ 1,629 종목 테스트 : [CNN_Prediction/KOSPI968-and-KOSDAQ1629/Test/profit/src/make_prediction_kosdaq.py](https://github.com/VAIV-SKKU/CNN_Prediction/blob/main/KOSPI968-and-KOSDAQ1629/Test/profit/src/make_prediction_csv_kosdaq.py)


#### 4-2-2. Calculate profit line by line
<4-2-1>에서 생성한 csv file에서 각각의 prediction 결과를 기반으로 한 수익률을 구한다. 
기존의 csv file에서 "Profit" Column이 추가되는 방식으로 새로운 csv file을 생성한다.

+ KOSPI 968 종목의 prediction 결과에 대한 수익률 계산 : [CNN_Prediction/KOSPI968-and-KOSDAQ1629/Test/profit/src/profit_kospi.py](https://github.com/VAIV-SKKU/CNN_Prediction/blob/main/KOSPI968-and-KOSDAQ1629/Test/profit/src/profit_kospi.py)
  + Arguments 설명
    + -i : 이미지(.png) 폴더 경로, KOSPI 968 종목 전체 이미지가 들어 있어야 한다.
    + -s : prediction 결과를 구한 시작 날짜
    + -e : prediction 결과를 구한 마지막 날짜
    + -y : prediction 결과를 구한 연도
    + -c : <4.2.1>에서 저장한 prediction 결과 csv file path
   + Output file : 기존 파일(prediction 결과 csv file) 이름 끝에 "_profit"가 추가된 이름으로 저장된다. (ex. model1_kospi968_2019_prediction.csv 가 input으로 들어갈 경우 생성되는 파일 이름은 model1_kospi968_2019_prediction_profit.csv)
```
// 2019 prediction 결과에 대한 수익률 계산
$ python make_prediction_kospi.py -s 2019-01-01 -e 2019-12-31 -y 2019 -c model1_kospi968_2019_prediction.csv -o Report

// 2020 prediction 결과에 대한 수익률 계산
$ python make_prediction_kospi.py -s 2020-01-01 -e 2020-12-31 -y 2020 -c model1_kospi968_2020_prediction.csv -o Report

// 2021 prediction 결과에 대한 수익률 계산
$ python make_prediction_kospi.py -s 2021-01-01 -e 2021-12-31 -y 2021 -c model1_kospi968_2021_prediction.csv -o Report
```

+ KOSDAQ 1,629 종목의 prediction 결과에 대한 수익률 계산 : [CNN_Prediction/KOSPI968-and-KOSDAQ1629/Test/profit/src/profit_kosdaq.py](https://github.com/VAIV-SKKU/CNN_Prediction/blob/main/KOSPI968-and-KOSDAQ1629/Test/profit/src/profit_kosdaq.py)


#### 4-2-3. Draw scatter
<4-2-1> 또는 <4-2-2>에서 생성된 csv file에서 Class 1로 예측한 결과의 예측 확률값에 대한 scatter plot을 생성한다.
+ x축은 날짜, y축은 예측 확률값(0.5~1.0)이다.
+ Labeling 방식 : 4%_01_2_5 (binary)
  + Class 1 : 5일 뒤 4% 이상 상승
  + Class 0 : 5일 뒤 하락
+ 색깔 구분
  + Green : 정답 (Class 1로 예측했는데 정답은 Class 1인 경우)
  + Orange : 미정 (정답 Class가 없는 경우 - 0 이상 4% 미만 상승)
  + Red : 오답 (Class 1로 예측했는데 정답은 Class 0인 경우)
 
 [CNN_Prediction/KOSPI968-and-KOSDAQ1629/Test/profit/src/draw_scatter_kospi.py](https://github.com/VAIV-SKKU/CNN_Prediction/blob/main/KOSPI968-and-KOSDAQ1629/Test/profit/src/draw_scatter_kospi.py)
 ```
 
 ```

생성 예시
![buy_scatter_2019_Batch16_Epochs8_Dropout30_profit_without_orange](https://user-images.githubusercontent.com/100757275/209842356-e485dbfb-0b69-4e31-9555-3f1e5a74ba31.png)


#### 4-2-4.

#### 4-2-5.
