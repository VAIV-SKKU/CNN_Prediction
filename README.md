---

## 2. 변경점

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
    
---
## 3.데이터셋

### 3-1) KOSPI Dataset
![kospi dataset](https://user-images.githubusercontent.com/82187742/209559303-77f18df2-c866-420f-b72d-f9db4b92f3a8.png)

### 3-2) KOSDAQ Dataset
![kosdaq dataset](https://user-images.githubusercontent.com/82187742/209559304-aba211b3-088f-4a41-8f45-d5ad598646bb.png)

### 3-3) KOSPI + KOSDAQ Dataset
![kospikosdaq dataset](https://user-images.githubusercontent.com/82187742/209559301-dfe214b1-e491-46bf-ad95-26ab2ff7fb36.png)

## 4. 결과

### 4-1) KOSPI
![kospi test](https://user-images.githubusercontent.com/82187742/209559307-d6b1ee7b-607a-4c3d-80f1-fccc5a613048.png)

### 4-2) KOSDAQ
![kosdaq test](https://user-images.githubusercontent.com/82187742/209559245-19bd98b9-ec54-459b-b7a5-af97aee88507.png)
