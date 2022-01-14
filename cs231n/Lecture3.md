# Lecture 3. Loss Functions and Optimization

<img src="https://user-images.githubusercontent.com/81629116/149471690-aea055be-466b-48d5-929d-f880243c0475.png" width="50%" height="50%"> 
만일 이미지 분류기가 이미지를 잘 분류해서 Loss 값이 0이 되는 W의 값을 찾았다고 해보자. 그렇다면 이 W는 유일한 값일까 ? 

→ No. 위와 같이 2W 역시 Loss 값은 0이 된다.

- - - 

<img src="https://user-images.githubusercontent.com/81629116/149471833-50767cc4-84c3-4e01-84b7-a075fcfcecef.png" width="50%" height="50%"> 
많은 W의 값들 중에서 loss 값이 0이 되는 W를 고르는 것은 좋지 않다. 왜냐하면 loss 값이 0이 되는 W을 찾는 것은 특정 training data에만 fit한 Loss function만 찾는 것과 같기 때문이다. 우리는 training data에 fit한 classifier를 찾는 것보다 test data에 fit한 classifier를 찾는 것에 좀 더 집중해야하는데, 이렇게 되면 새로운 test data를 적용했을 때의 성능은 보장하지 못하게 된다. 

- - - 

<img src="https://user-images.githubusercontent.com/81629116/149472212-acab90d3-c74e-4e0d-a1d8-8639b4b342d1.png" width="50%" height="50%"> 
위의 예시를 보자면, classifier는 파란색 점들에 대해선 perfectly fit한 분류를 했지만 새로운 녹색 점들에 대해서는 완벽하게 분류하지 못하고, 녹색 선들에 대해 잘 설명해주지 못한다. 우리가 의도했던 것은 초록색 직선과 같이 분류하는 것이다. 

###  
이 문제를 해결하기 위해 Regularization항을 추가한다. 좀 더 단순한 W를 선택할 수 있도록 도와주고, classifier가 training data에 fit할 수 있도록 도와준다. 다시 말해 예측 모델이 복잡한 고차 다항식을 선택할 때 패널티를 줘서 보다 단순한 저차 다항식을 선택하도록 만든다.    

### 
data loss는 training data에 맞는 W를 찾으려고 하지만, Regularization은 test data를 잘 설명할 수 있는 일반화된 W를 찾으려고 한다. data loss와 Regularization이 서로 경쟁하며 최적화된 W를 찾는다.

- - - 

<img src="https://user-images.githubusercontent.com/81629116/149472696-3c2036f7-f83b-442e-9a28-881f551693fe.png" width="50%" height="50%">
Regularization에는 여러가지가 있지만, 이 중에서도 가장 common하게 쓰이는 방식은 L2 Regularization(weight decay)이다. 전체적으로 Regularization은 training data에 fit하게 만드는 것보다 모델의 복잡성에 대해 패널티를 주는 방식이라고 생각하기. 

- - - 

<img src="https://user-images.githubusercontent.com/81629116/149472869-0897b93e-e690-4654-86fc-3b71047a6cd5.png" width="50%" height="50%">   

* L2 Regularization은 복잡도를 어떻게 판단하는가?    
→ 위와 같이 w1와 w2 값이 있을 때, 상대적으로 어떤 값이 좀 더 매끄러운지를 판단한다. x의 특정 요소에 의존하기보다 x의 모든 요소가 골고루 영향을 미치길 원하기 때문에 동일한 점수를 갖게 된다면 좀 더 넓게 펼쳐져있는 것을 선호한다. 따라서 L2 방식에서는 w2를 선호하고, 이것을 coarse solution이라고 한다.    

  이에 반하여 L1 Regularization은 w1을 선호한다. L1에서는 w에서 0의 개수에 따라 모델의 복잡도(복잡도 = w에서 0이 아닌 수의 개수)가 달라지고, w의 대부분의 원소를 0이 되게 한다. 이것을 sparse solution이라고 한다. 

  숫자가 넓게 퍼져있을수록 모델 복잡도는 덜 복잡해진다. 이 문제는 데이터와 problem에 따라 달라지기 때문에 모델에 맞춰서 선택해야 한다.

- - - 

<img src="https://user-images.githubusercontent.com/81629116/149475995-623af87c-be20-4ef8-87fe-10361836d2c6.png" width="50%" height="50%">

딥러닝에서 좀 더 보편적으로 쓰는 classifier로는 Softmax(multinomial logistic regression)가 있다. 앞에서 본 multi-class SVM loss는 score 자체를 고려하기보다, correct 클래스의 점수와 incorrect 클래스 점수 사이의 격차에만 집중했다. 하지만 이에 반해 softmax는 클래스별 확률분포를 사용하여 예측 점수 자체에 추가적인 의미를 부여한다.

softmax classifier에서 사용하는 loss는 cross entropy loss라고 한다. 정답이 나올 확률의 최소값을 의미한다. 따라서 loss는 **-log(correct 클래스가 정답일 확률)** 이다. 

- - - 

<img src="https://user-images.githubusercontent.com/81629116/149477001-6cd147ca-703c-4f8a-9e85-27be5eb47a34.png" width="50%" height="50%">

softmax에서는 score들에 지수를 취해서 양수로 만들어주고, 그 지수들의 합으로 다시 정규화시킨다. 따라서 softmax 함수를 거치게 되면 확률분포가 얻어지고, 이 값들은 해당 클래스에 속하게 될 확률을 의미하게 된다. 따라서 모든 값을 합했을 때는 1이 되고, 우리는 correct 클래스에 해당하는 클래스의 확률값이 1에 가깝도록 계산되는 것을 목표로 한다. 이렇게 되면 loss는 `-log(correct 클래스의 확률)` 이 된다.     

위의 예제에서 correct 클래스가 정답일 확률은 0.13이며, 이 때의 cross entropy loss 값은 0.89이다.    

Softmax classifier는 correct 클래스가 정답일 확률은 1, loss 값은 0에 가깝도록 만들고자 한다.

- - - 

<img src="https://user-images.githubusercontent.com/81629116/149478193-c4d2f8ec-6eef-4d0c-aaa6-7309ddbd309f.png" width="50%" height="50%">
위는 2개의 loss function을 비교한 그림이다. 두 함수 모두 예측 score는 같지만 이 점수를 해석하는 방식이 다르다. SVM은 correct 클래스와 incorrect 클래스 간의 점수 차이를 중점으로 보고, softmax는 예측 점수 자체를 해석하려고 한다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149478766-6a422bcb-da83-498d-922f-a502073ae41a.png" width="50%" height="50%">

* 만일 score를 변화시킨다면 각각의 함수는 loss 값을 어떻게 가지게 될까?    
  → SVM은 score 자체보다는 correct 클래스와 incorrect 클래스 간의 점수 차이를 중점으로 보기 때문에 loss 값이 변하지 않지만, softmax는 정답 클래스의 예측 점수를 확률로 만들기 때문에 loss 값이 변한다. 하지만 예측 score가 많이 변하게 되면 hinge loss도 변화한다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149482017-66d82f13-a812-4d6c-8042-de000d15c53a.png" width="50%" height="50%">

앞의 내용 과정을 다시 정리하자면, 우리가 (x, y)라는 데이터셋을 가지고 있을 때, dot product를 통해 score를 구한다. 또한 SVM이나 Softmax 등과 같은 loss function으로 loss를 알아낸다. 또한 Regularization을 통해 최종 loss 값을 구할 수 있게 된다.    

근데 그러면 어떻게 loss 값을 최소화시키는 w의 값은 어떻게 찾는건가 ?

- - -

<img src="https://user-images.githubusercontent.com/81629116/149482539-519216d0-ae14-4dd3-80e5-27e0451507bf.png" width="50%" height="50%">
바로 optimization이라는 방법을 통해 찾을 수 있다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149482892-14d873ed-447e-44f3-9b3a-baa5df567f1d.png" width="50%" height="50%">
산에 비유를 해보자면, 만일 이 산을 내려가려고 한다면 반복적인 행동을 통해 산에서 내려갈 수 있을 것이다. 즉, 산의 높이를 loss, 사람의 위치를 w라고 한다면 loss가 적은 곳으로 w를 계속해서 변화시키며 내려가야한다. 이 과정이 바로 optimization이다.

- - -

<img src="https://user-images.githubusercontent.com/81629116/149483223-27ac072f-603a-4937-96fb-cc3d1ac32e2f.png" width="50%" height="50%">
우선 가장 멍청한 방법부터 살펴보자. random search. w의 값을 랜덤으로 바꿔가면서 최적의 loss를 찾는 방법이다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149483577-01181d37-a9e7-40e0-bd26-9bd00a49f865.png" width="50%" height="50%">
이걸 test set에 적용시켜보면 15.5%의 정확도가 나오긴 한다. 참고로 현대 최신 기술(state of the art)로는 95%가 나온다고 한다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149483889-5244e8f5-7fc8-47f2-b4f0-6e5e459d4c18.png" width="50%" height="50%">
두 번째 방식은 땅에 발을 가져다대서 조금 더 아래의 방향으로 내려갈 수 있는 경사면을 찾는 것이다. 그 방향으로 계속해서 발을 내딛어서 더 낮은 곳으로 내려가보는 전략이다. slope를 구하기 위해서 어떤 방식을 써야할까?

- - -

<img src="https://user-images.githubusercontent.com/81629116/149485080-565f14ce-9889-4cb2-8d8a-c9949724955f.png" width="50%" height="50%">
slope를 구할 때 '미분'을 사용한다. slope는 1차원 공간에서 어떤 함수의 미분 값이라고 할 수 있다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149485394-00fa7766-bda1-4acf-9dd1-e9f197f71a45.png" width="50%" height="50%">
현재의 w 값을 가지고 있을 때 loss값이 1.25347이 나온다고 해보자. 그러면 아래 방향으로 내려가게 될 것이다. 이 때 w 값에 0.0001이라는 아주 작은 숫자를 더해보니 loss 값이 1.25322으로 조금 줄었다. 그리고 이 때 미분을 해보니 기울기는 -2.5가 나왔다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149485991-3d91e77b-5a9d-4157-9132-b874b4c68e40.png" width="50%" height="50%">

이번에 또 w 값에 0.0001을 더해봤다. 그랬더니 이번에는 loss 값이 1.25353으로 조금 커졌다. 그리고 미분을 해보면 기울기값이 양수로 나온 것을 확인할 수 있었다. 

이 과정을 무한으로 반복해야하는데... 근데 w은 아주 크고, 0.0001이라는 아주 작은 값을 더해보기 때문에 super slow하고 좋지 않은 방식이다.

- - -

<img src="https://user-images.githubusercontent.com/81629116/149486265-dc89aa56-b65a-4dfc-b300-7b7a2ac15be0.png" width="50%" height="50%">
다행히 뉴턴과 라이프니치의 도움으로 미분을 이용하면 간단하게 해결 가능하다.

- - -

<img src="https://user-images.githubusercontent.com/81629116/149486478-4276ffce-4797-454d-b2bb-d073518e20cd.png" width="50%" height="50%">
짜잔. 이렇게 보다 간편하게 값을 구할 수 있다.

- - -

<img src="https://user-images.githubusercontent.com/81629116/149486559-49ae1a59-4679-4993-9a3a-7b4d39e8b327.png" width="50%" height="50%">

- - -

<img src="https://user-images.githubusercontent.com/81629116/149486646-0dac9675-c1d8-458e-937f-66cf3369b02f.png" width="50%" height="50%">
우리는 Gradient Descent, 즉 경사하강법을 통해 가장 적절한 w의 값을 찾아낼 것이다. 이 때 step_size는 한번 움직일 때의 보폭이라고 생각하면 됨. step_size는 우리가 잘 정해줘야하는 hyperparameter이다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149486769-63add227-3ea2-47cd-8512-eb45bdd1b332.png" width="50%" height="50%">
빨간색 부분이 가장 낮은 부분이라고 생각했을 때, 우리가 가야하는 방향(미분해서 나온 기울기의 방향)은 흰 화살표 방향과 유사할 것이다. 즉, 중앙으로 가길 원함. 이 때 step_size를 너무 크게 한다면 빨간색에서 멀어지게 될 것이고, 그러면 기울기가 더욱 커지게 됨. 그러면 더욱 더 멀어지게 되는 악순환이 반복. 반면 step_size를 너무 작게 한다면 내려가는데 시간이 너무 오래 걸리게 됨. 따라서 적절한 크기의 step_size가 필요.

- - -

<img src="https://user-images.githubusercontent.com/81629116/149486859-b0ab6f68-d045-4cde-aede-1f8a93e78d14.png" width="50%" height="50%">
이 때, 모든 것을 이런 식으로 계산하기에는 시간이 너무 오래 소요. 그래서 생각해낸 방식이 SGD(Stochastic Gradient Descent)이다. 전체 training set에 대한 loss와 gradient를 계산하는 것이 아니라, mini-batch라고 불리는 small random set을 training set으로부터 가지고 온다. (32, 64, 128 등) 랜덤으로 가져와서 그 사진들에 대한 Loss를 계산하고 weight을 계산해서 전체의 gradient와 loss를 추정하는 방식. 이 방식으로 좀 더 빠른 계산이 가능하다. 

- - -

<img src="https://user-images.githubusercontent.com/81629116/149486949-51354b38-60b9-41bf-b53f-9bd0c0e25647.png" width="50%" height="50%">

- - -

<img src="https://user-images.githubusercontent.com/81629116/149487035-f4c64e67-5aee-4e37-98bd-4ff8583dc54d.png" width="50%" height="50%">

- - -

<img src="https://user-images.githubusercontent.com/81629116/149487124-644c25da-208a-4a3e-a78f-bf8568ab5179.png" width="50%" height="50%">

- - -

<img src="https://user-images.githubusercontent.com/81629116/149487195-ca5d5b1c-f6f5-4029-baf4-c6044d3cc883.png" width="50%" height="50%">

- - -

<img src="https://user-images.githubusercontent.com/81629116/149487240-ef624fb8-e0c4-4176-ad88-aa4c4f242703.png" width="50%" height="50%">

