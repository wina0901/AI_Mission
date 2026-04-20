# FashionMNIST 조건부 이미지 생성 프로젝트

## 프로젝트 개요

이 프로젝트는 FashionMNIST 데이터셋의 각 패션 아이템 클래스를 조건으로 받아, 해당 클래스에 맞는 이미지를 생성하는 Conditional GAN 기반 이미지 생성 프로젝트입니다.  
단순히 랜덤 이미지를 생성하는 것이 아니라, 티셔츠, 바지, 스니커즈, 가방처럼 사용자가 지정한 클래스 조건에 맞는 이미지를 생성할 수 있도록 모델을 설계하고 학습했습니다.

이번 프로젝트에서는 cGAN 구조를 직접 구현하고, Generator와 Discriminator의 학습 균형을 분석했으며, 최종적으로 FID와 Inception Score를 통해 생성 이미지의 품질과 다양성을 정량적으로 평가했습니다.  
즉, 단순 구현에서 끝나는 것이 아니라, 생성 모델을 설계하고 안정적으로 학습시키며 결과를 수치로 검증하는 전체 GAN 실험 과정을 직접 수행한 프로젝트입니다.

---

## 이 프로젝트를 왜 했는가

생성형 AI는 최근 가장 중요한 인공지능 분야 중 하나이며, 단순 분류나 예측과는 달리 새로운 데이터를 직접 만들어낸다는 점에서 매우 큰 의미가 있습니다.  
특히 조건부 생성은 원하는 클래스나 속성에 맞는 출력을 제어할 수 있기 때문에, 이미지 생성, 데이터 증강, 콘텐츠 제작, 맞춤형 디자인 같은 다양한 분야에 활용될 수 있습니다.

이번 프로젝트는 다음과 같은 이유로 진행했습니다.

- 생성형 모델의 기본 원리인 GAN을 직접 구현해보고 싶었습니다.
- 단순 생성이 아니라 클래스 조건을 반영하는 Conditional GAN 구조를 설계해보고 싶었습니다.
- Generator와 Discriminator가 경쟁적으로 학습하는 구조가 실제로 어떻게 안정화되는지 경험해보고 싶었습니다.
- 생성 결과를 눈으로만 확인하는 것이 아니라, FID와 Inception Score 같은 정량 지표로 평가하는 경험을 쌓고 싶었습니다.
- 이미지 생성 프로젝트를 통해 분류 중심의 비전 프로젝트와는 다른 문제 설정과 모델링 방식을 경험해보고 싶었습니다.

즉, 이번 프로젝트는 단순한 이미지 생성 실습이 아니라, 조건부 생성 모델을 직접 설계하고 학습하며 생성형 AI의 핵심 구조와 평가 방식을 실전적으로 익히기 위해 진행했습니다.

---

## 문제 정의

- 주제: FashionMNIST 클래스 조건부 이미지 생성
- 문제 유형: Conditional Image Generation
- 입력:
  - 랜덤 노이즈 벡터
  - 클래스 레이블
- 출력:
  - 주어진 클래스 조건에 맞는 FashionMNIST 이미지
- 클래스 수: 10개
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

이번 프로젝트의 핵심은 무작위 이미지를 만드는 것이 아니라, 조건으로 넣은 클래스에 맞는 이미지를 생성할 수 있도록 모델을 학습시키는 것이었습니다.

---

## 데이터 설명

사용한 데이터는 torchvision에서 제공하는 FashionMNIST 데이터셋입니다.

데이터 특성은 다음과 같습니다.

- 학습 데이터: 60,000장
- 테스트 데이터: 10,000장
- 이미지 크기: 28x28
- 채널 수: 1채널 grayscale
- 클래스 수: 10개

FashionMNIST는 비교적 단순한 구조의 패션 아이템 이미지로 구성되어 있기 때문에, GAN의 기본 동작 원리와 조건부 생성 구조를 실험하기에 적절한 데이터셋이라고 판단했습니다.

---

## 프로젝트 진행 과정

### 1. 데이터 전처리 및 DataLoader 구성

먼저 FashionMNIST 데이터를 불러오고, 학습용과 테스트용 transform을 구분해 구성했습니다.

학습 데이터에는 다음과 같은 증강을 적용했습니다.

- RandomHorizontalFlip
- RandomAffine
- ColorJitter
- 정규화

테스트 데이터에는 기본적인 tensor 변환과 정규화만 적용했습니다.

이 과정은 단순 전처리가 아니라, 생성 모델이 특정 샘플 패턴에 과도하게 의존하지 않도록 약간의 변형을 허용해 학습 다양성을 높이기 위한 설계였습니다.

---

### 2. Conditional GAN 구조 설계

이번 프로젝트에서는 Generator와 Discriminator 모두 클래스 레이블을 입력으로 받는 Conditional GAN 구조를 직접 구현했습니다.

#### Generator
Generator는 다음과 같은 흐름으로 구성했습니다.

- 입력:
  - latent noise vector
  - label embedding
- noise와 label embedding을 결합
- 선형층으로 초기 feature map 생성
- upsampling과 convolution block을 거쳐 28x28 이미지 생성
- 최종 출력은 Tanh를 사용해 [-1, 1] 범위로 정규화

이 구조를 통해 클래스 조건이 단순 부가 정보가 아니라, 이미지 생성 과정 전반에 반영되도록 설계했습니다.

#### Discriminator
Discriminator는 다음과 같이 구성했습니다.

- 입력 이미지와 label embedding을 spatial map 형태로 결합
- 이미지 채널과 라벨 맵을 concat
- convolution 기반 판별 네트워크 통과
- 최종적으로 real / fake 판별 수행

또한 Discriminator에는 spectral normalization을 적용해 학습 안정성을 높이도록 구성했습니다.

즉, 이번 프로젝트는 단순 cGAN 개념 사용이 아니라, label embedding과 spectral norm을 포함한 구조적 설계를 직접 수행한 프로젝트였습니다.

---

## 학습 전략

학습은 Generator와 Discriminator를 번갈아 업데이트하는 전형적인 GAN 방식으로 진행했습니다.

주요 설정은 다음과 같습니다.

- epoch: 30
- latent dimension: 128
- label embedding dimension: 32
- batch size: 64
- loss: BCEWithLogitsLoss
- optimizer:
  - Generator: Adam
  - Discriminator: Adam
- learning rate:
  - Generator: 2e-4
  - Discriminator: 1e-4
- Adam betas: (0.5, 0.999)

학습 안정화를 위해 다음과 같은 요소도 반영했습니다.

- real label smoothing 적용
- Generator와 Discriminator의 learning rate를 다르게 설정
- Discriminator에 spectral normalization 적용

이러한 설정은 GAN 학습에서 자주 발생하는 불안정성, 한쪽 네트워크의 과도한 우세, mode collapse 가능성을 줄이기 위한 목적이었습니다.

---

## 학습 과정 분석

이번 프로젝트에서는 단순히 최종 생성 결과만 보는 것이 아니라, 학습 중 Generator와 Discriminator의 균형이 어떻게 변하는지도 함께 분석했습니다.

기록한 주요 지표는 다음과 같습니다.

- D loss
- G loss
- D(x)
- D(G(z)) before G update
- D(G(z)) after G update

학습 결과를 보면,

- Discriminator loss는 약 1.34에서 1.22 수준으로 완만하게 감소
- Generator loss는 약 0.86에서 1.03 수준으로 증가
- D(x)는 약 0.47~0.51 수준 유지
- D(G(z))는 약 0.43~0.38 수준 유지

이 결과는 Discriminator가 실제 이미지와 생성 이미지를 완벽하게 구분하지 못할 정도로 Generator가 점차 품질을 높여갔고, 두 네트워크가 어느 한쪽으로 완전히 무너지지 않고 비교적 균형 있게 경쟁하며 학습했다는 의미로 해석할 수 있습니다.

즉, 이번 프로젝트는 단순히 생성 이미지를 보는 데 그치지 않고, GAN 특유의 학습 균형을 수치적으로 모니터링하며 분석했다는 점에 의미가 있습니다.

---

## 생성 결과 확인

학습 중간마다 고정된 noise와 클래스 레이블을 사용해 이미지를 생성하고 시각화했습니다.

이 방식을 통해 다음을 확인할 수 있었습니다.

- 같은 클래스 조건에 대해 epoch가 진행될수록 형태가 더 뚜렷해지는지
- 클래스별 특징이 유지되는지
- 생성 결과가 점점 다양해지는지
- 특정 클래스에 mode collapse가 발생하지 않는지

즉, 생성 결과를 정성적으로도 계속 확인하면서 모델이 단순 노이즈 출력에서 점차 클래스 의미를 가진 이미지로 발전하는 과정을 추적했습니다.

---

## 정량 평가

이번 프로젝트에서는 생성 모델의 결과를 정량적으로 평가하기 위해 FID와 Inception Score를 계산했습니다.

평가를 위해 수행한 작업은 다음과 같습니다.

- grayscale 생성 이미지를 3채널로 확장
- Inception 입력 크기인 299x299로 resize
- uint8 형태로 변환
- 실제 이미지와 생성 이미지를 각각 5,000장씩 사용해 FID 계산
- 생성 이미지 5,000장으로 Inception Score 계산

최종 결과는 다음과 같습니다.

- FID: 34.05
- Inception Score: 3.99 ± 0.10

이 결과는 FashionMNIST 기준으로 비교적 안정적인 생성 품질을 보여주는 수치로 해석할 수 있으며, 클래스 특징을 어느 정도 구분 가능하게 생성하면서도 다양한 샘플을 만들 수 있었다는 점을 의미합니다.

---

## 최종 결과 해석

이번 프로젝트의 핵심 결과는 다음과 같습니다.

- FashionMNIST의 10개 클래스를 조건으로 받아 이미지 생성 가능
- Generator와 Discriminator가 비교적 균형 있게 학습
- FID 34.05로 실제 데이터 분포와의 차이를 일정 수준까지 줄임
- Inception Score 3.99 ± 0.10으로 품질과 다양성 모두 안정적인 결과 확보

즉, 이번 실험을 통해 조건부 생성 모델이 클래스 조건을 반영한 패션 아이템 이미지를 비교적 안정적으로 생성할 수 있음을 확인했습니다.

또한 단순히 이미지를 생성하는 것에 그치지 않고, GAN 학습 안정화 기법과 정량 평가 방식을 함께 적용했다는 점이 프로젝트의 핵심 가치였습니다.

---

## 이 프로젝트를 통해 갖추게 된 능력

### 1. 생성형 모델을 직접 설계하고 구현하는 능력

분류나 회귀가 아니라, 새로운 이미지를 생성하는 모델을 직접 설계하고 학습시켰습니다.  
이를 통해 생성형 AI의 기본 구조와 학습 방식을 실제 코드 수준에서 이해할 수 있게 되었습니다.

### 2. Conditional GAN 구조 이해 및 구현 능력

noise만 입력받는 일반 GAN이 아니라, 클래스 레이블을 함께 입력받는 cGAN 구조를 구현했습니다.  
이를 통해 조건 정보를 생성 과정에 반영하는 방법과 label embedding 활용 방식을 익힐 수 있었습니다.

### 3. GAN 학습 안정화 기법 적용 능력

GAN은 일반적인 지도학습보다 훨씬 불안정한 학습 구조를 가지기 때문에, spectral normalization, real label smoothing, learning rate 분리 같은 안정화 기법을 적용했습니다.  
이를 통해 생성 모델 학습에서 발생할 수 있는 불안정성을 줄이는 실전적 접근을 경험할 수 있었습니다.

### 4. Generator와 Discriminator의 균형을 해석하는 능력

loss 값만 보는 것이 아니라 D(x), D(G(z)) 같은 GAN 전용 지표를 함께 추적하며, 두 네트워크의 경쟁 상태를 해석했습니다.  
이를 통해 생성 모델은 단순 성능 수치가 아니라, 학습 균형 자체를 읽는 것이 중요하다는 점을 배울 수 있었습니다.

### 5. 생성 모델 정량 평가 능력

FID와 Inception Score를 직접 계산하고 해석하면서, 생성 결과를 단순 시각화에만 의존하지 않고 정량적으로 검증하는 방법을 익힐 수 있었습니다.

### 6. PyTorch 기반 생성 모델 프로젝트 구현 능력

Dataset, DataLoader, Generator, Discriminator, 학습 루프, 시각화, 평가 지표 계산까지 전체 생성 프로젝트 파이프라인을 직접 구성했습니다.  
이를 통해 PyTorch 기반 생성 모델 실험을 end-to-end로 구현하는 능력을 강화할 수 있었습니다.

---

## 이 프로젝트에서 중요하게 본 점

이번 프로젝트에서 가장 중요하게 생각한 점은 이미지를 얼마나 그럴듯하게 생성하는가만이 아니라, 조건부 생성이 제대로 이루어지고 있는지, 그리고 학습이 얼마나 안정적으로 진행되는지를 함께 확인하는 것이었습니다.

그래서 다음과 같은 원칙으로 프로젝트를 진행했습니다.

- 클래스 조건이 생성 결과에 실제로 반영되도록 구조 설계하기
- Generator와 Discriminator를 균형 있게 학습시키기
- 생성 결과를 중간중간 시각화해 변화 추적하기
- GAN 전용 학습 지표를 함께 기록하기
- FID와 Inception Score로 정량 평가하기
- 단순 구현이 아니라 안정화 기법까지 포함해 실험하기

이러한 흐름은 향후 이미지 생성, 데이터 증강, 스타일 생성, 맞춤형 콘텐츠 생성 같은 생성형 AI 응용 프로젝트를 수행할 때 중요한 기반이 될 수 있다고 생각합니다.

---

## 개선 방향

이번 결과를 바탕으로 향후 다음과 같은 방향으로 성능을 더 높일 수 있다고 판단했습니다.

- 더 깊은 Generator / Discriminator 구조 실험
- class embedding 방식 다양화
- Wasserstein GAN 계열과 비교
- self-attention 기반 생성 구조 도입
- training epoch 확대 및 scheduler 적용
- 클래스별 생성 품질을 따로 측정하는 추가 분석 수행

즉, 현재 모델은 FashionMNIST 수준에서는 안정적인 생성 성능을 보였지만, 더 복잡한 데이터셋으로 확장하거나 더 정교한 생성 구조를 적용하면 추가 개선 가능성이 충분하다고 볼 수 있습니다.

---

## 사용 기술

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- torchmetrics
- torch-fidelity

---

## 한 줄 정리

이 프로젝트는 FashionMNIST의 클래스 조건을 반영해 이미지를 생성하기 위해 Conditional GAN을 직접 구현하고, 학습 안정화 기법 적용, 생성 결과 시각화, FID 및 Inception Score 평가까지 전 과정을 수행한 생성형 AI 프로젝트입니다.