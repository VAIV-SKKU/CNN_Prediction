---

## 변경점


### 1) Data Sampling 
0/1 Class 간 Unbalance -> Random + Balance (0/1)
=> Label class 간 imbalance 문제 해결


### 2) Forecast Interval
1 거래일 후 -> 5 거래일 후
=> Transaction 수를 줄임으로써 수수료 절감으로 수익률 향상


### 3) Candlestick chart image size
50 x 50 -> 224 x 224
=> Pretrained CNN 모델의 input size와 동일하게 했을 때 좋은 성능 보임


### 4) Labeling
상승 4% 이상 '1' 나머지 '0' -> 상승 4% 이상 '1' 0% 미만 '0'(상승 0~4%는 무시)
=> 명확한 상승 구간의 데이터를 학습하여, Buy 에 대한 변별력 향상


### 5) Hyper-parameters
Batch size, Dropout  -> 모델마다 dropout, batch size 변경하며 학습
=> Bacth size, Dropout 이 가장 큰 영향을 끼침


### 6) Evaluation Matric
Accuracy, Precision, Recall, F1 score 값에 집중 -> Profit에 집중
=> 매수 예측 성공 여부를 수익률 기준으로 변경



