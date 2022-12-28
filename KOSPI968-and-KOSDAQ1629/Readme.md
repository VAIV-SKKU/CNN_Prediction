#CNN 모델 KOSPI 968종목, KOSDAQ 1,629종목 실험
Reference paper : "Using Deep Learning Neural Networks and Candlestick Chart Representation to Predict Stock Market" by Rosdyna Mangir Irawan Kusuma, Tang-Thi Ho, Wei-Chun Kao, Yu-Yen Ou and Kai-Lung Hua

## Data
### 데이터셋 구조 예시

'''
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

'''

### Data Preprocessing
png 파일을 pickle
