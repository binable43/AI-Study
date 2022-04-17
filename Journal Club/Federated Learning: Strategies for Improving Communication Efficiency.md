# Federated Learning: Strategies for Improving Communication Efficiency

- FL의 목적 : 좋은 성능의 Centralized model 학습
- 학습 데이터 조건 : 불안정하고 상대적으로 느린 네트워크 연결을 가진 clients에 분포되어 있어야 함)
- Client가 각 round마다 독립적으로 local data를 이용하여 현재 모델에 대한 update 계산하고 이를 central server에 전송. 모이게 된 Update들은 새로운 global model을 계산하기 위해 합해짐

<br>

- 본 논문에서는 Uplink communication costs를 줄이기 위한 두 가지 방법 제시
    1) structured updates   
    적은 수의 변수를 사용하여 parameterized된 제한된 space로부터 직접 update 학습 ex) low-rank 혹은 random mask
    
    2) sketched updates   
    full model update 학습하여 서버 전송 전에 quantization, random rotations, subsampling 등의 기법들을 함께 사용해 압축
    
→ Convolutional, Recurrent networks에 대해 communication cost를 거의 100배 수준으로 감소


## Reference
- [Paper Link](https://arxiv.org/abs/1610.05492)
- [Google AI Blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
