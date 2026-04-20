# 흉부 X-Ray 기반 폐렴 분류 프로젝트

## 프로젝트 개요

이 프로젝트는 흉부 X-Ray 이미지를 바탕으로 폐렴 여부를 판별하는 의료영상 이진 분류 프로젝트입니다.  
단순히 하나의 CNN 모델을 학습하는 데서 끝나지 않고, 데이터 불균형 문제를 해결하고, 다양한 Transfer Learning 전략을 비교하며, 최종적으로 의료 도메인에서 더 중요한 오분류 기준까지 함께 고려해 모델을 선택하는 방식으로 진행했습니다.

이번 프로젝트에서는 Custom CNN을 기준 모델로 두고, ResNet18, ResNet50, DenseNet121을 활용해 Feature Extraction, Partial Fine-Tuning, Full Fine-Tuning을 각각 비교했습니다. 또한 Grad-CAM을 통해 모델이 실제로 어떤 영역을 보고 판단하는지도 함께 확인했습니다.

---

## 이 프로젝트를 왜 했는가

의료영상 분류 문제는 일반 이미지 분류와 달리, 단순 정확도만 높은 모델보다 실제 위험 상황을 얼마나 잘 탐지하느냐가 더 중요합니다.  
특히 폐렴 분류에서는 실제 환자를 정상으로 잘못 판단하는 False Negative가 매우 치명적일 수 있기 때문에, 성능 수치만이 아니라 오분류의 의미까지 함께 해석할 수 있어야 한다고 생각했습니다.

이번 프로젝트는 이러한 이유로 진행했습니다.

- 컴퓨터 비전 분류 문제를 의료 데이터에 적용해보고 싶었습니다.
- 작은 데이터셋 환경에서 Transfer Learning이 얼마나 효과적인지 직접 검증해보고 싶었습니다.
- 단순 Accuracy 비교가 아니라 Precision, Recall, F1-score, Confusion Matrix까지 함께 보며 모델을 평가하는 경험을 쌓고 싶었습니다.
- 실제 의료 문제에서는 어떤 기준으로 모델을 선택해야 하는지, 즉 성능과 안전성 사이의 균형을 판단하는 경험을 하고 싶었습니다.
- Grad-CAM을 통해 모델의 예측 근거를 시각적으로 확인하며, 해석 가능한 AI에 대한 감각도 함께 키우고 싶었습니다.

즉, 이번 프로젝트는 단순한 이미지 분류 실습이 아니라, 의료영상 분류 문제를 실제 판단 기준에 가깝게 다뤄보며 모델 성능과 해석 가능성을 함께 검증하는 프로젝트였습니다.

---

## 문제 정의

- 주제: 흉부 X-Ray 이미지 기반 폐렴 분류
- 문제 유형: 이진 분류
- 입력: Chest X-Ray image
- 출력:
  - NORMAL
  - PNEUMONIA
- 주요 평가 지표:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

이번 프로젝트에서는 전체 정확도뿐 아니라, 실제 폐렴 환자를 정상으로 놓치는 False Negative를 얼마나 줄일 수 있는지도 매우 중요한 기준으로 보았습니다.

---

## 데이터 설명

사용한 데이터셋은 Kaggle의 Chest X-Ray Pneumonia 데이터셋입니다.

데이터 구성은 다음과 같습니다.

- Train
  - NORMAL: 1,341장
  - PNEUMONIA: 3,875장
  - Total: 5,216장
- Original Validation
  - NORMAL: 8장
  - PNEUMONIA: 8장
  - Total: 16장
- Test
  - NORMAL: 234장
  - PNEUMONIA: 390장
  - Total: 624장

초기 validation 데이터가 16장으로 너무 적어 신뢰성이 낮았기 때문에, train 데이터 일부를 추가로 분리해 validation 셋을 확장했습니다.

---

## 데이터 전처리 및 학습 준비

### 1. Validation 재구성

원래 제공된 validation 데이터는 개수가 너무 적어 모델 비교에 적합하지 않다고 판단했습니다.  
따라서 train 데이터의 20%를 추가 validation 용도로 분리해 검증 데이터의 신뢰도를 높였습니다.

- 분리 전 train: 5,216장
- 분리 후 train: 4,172장
- 추가 validation: 1,044장
- 최종 validation: 1,060장

이 과정을 통해 모델 간 성능 차이를 더 안정적으로 비교할 수 있는 환경을 만들었습니다.

### 2. 클래스 불균형 처리

train 데이터는 NORMAL보다 PNEUMONIA 비율이 훨씬 높은 불균형 구조였습니다.  
이를 해결하기 위해 WeightedRandomSampler를 적용하여 학습 배치에서 두 클래스가 보다 균형 있게 반영되도록 구성했습니다.

적용 후 train 클래스 분포는 거의 균형 수준으로 맞춰졌습니다.

- NORMAL: 2,128
- PNEUMONIA: 2,044

### 3. 이미지 전처리 및 증강

사전학습 모델을 활용하기 위해 ImageNet 기준 입력 형태에 맞춰 전처리를 구성했습니다.  
또한 의료영상 분류에서 일반화 성능을 높이기 위해 이미지 변환과 정규화를 함께 적용했습니다.

---

## 모델링 전략

이번 프로젝트에서는 단순히 모델 구조만 비교한 것이 아니라, 같은 백본에 대해 전이학습 전략 자체를 비교했습니다.

비교한 모델은 다음과 같습니다.

### 1. Custom CNN
직접 설계한 CNN을 기준 모델로 사용했습니다.  
사전학습 없이 처음부터 학습하는 구조로, Transfer Learning 모델들과의 차이를 보기 위한 baseline 역할을 했습니다.

### 2. ResNet18
상대적으로 가벼운 구조의 사전학습 모델로, 작은 데이터셋에서도 안정적인 성능을 기대할 수 있어 비교 대상으로 사용했습니다.

- Feature Extraction
- Partial Fine-Tuning
- Full Fine-Tuning

### 3. ResNet50
더 깊은 residual 구조를 활용해 고수준 특징 표현 능력을 비교했습니다.

- Feature Extraction
- Partial Fine-Tuning
- Full Fine-Tuning

### 4. DenseNet121
특징 재사용이 강한 DenseNet 계열 모델로, 의료영상 분류에서 자주 활용되는 구조 중 하나이기 때문에 포함했습니다.

- Feature Extraction
- Partial Fine-Tuning
- Full Fine-Tuning

---

## 추가 구현 요소

### 1. Early Stopping
학습 과정에서 과적합을 방지하고 안정적인 최적 시점을 찾기 위해 Early Stopping 로직을 직접 구현했습니다.

### 2. Grad-CAM
각 모델이 실제로 이미지의 어떤 부위를 보고 폐렴 여부를 판단하는지 시각적으로 확인하기 위해 Grad-CAM을 적용했습니다.

이를 통해 단순히 성능 수치만 보는 것이 아니라, 모델 예측의 근거를 함께 해석할 수 있도록 구성했습니다.

---

## 실험 결과

### 테스트 정확도 비교

- CustomCNN: 0.7019
- ResNet18 Feature Extraction: 0.8942
- ResNet18 Partial Fine-Tuning: 0.9071
- ResNet18 Full Fine-Tuning: 0.9247

- ResNet50 Feature Extraction: 0.8782
- ResNet50 Partial Fine-Tuning: 0.8782
- ResNet50 Full Fine-Tuning: 0.9071

- DenseNet121 Feature Extraction: 0.8494
- DenseNet121 Partial Fine-Tuning: 0.9119
- DenseNet121 Full Fine-Tuning: 0.9359

결과적으로 Full Fine-Tuning 계열이 전반적으로 가장 우수했고, 그중 DenseNet121 Full Fine-Tuning이 가장 높은 정확도를 보였습니다.

---

## 주요 성능 비교

### DenseNet121 Full Fine-Tuning
- Test Accuracy: 0.9359
- Precision: 0.9332
- Recall: 0.9667
- F1-score: 0.9496

전체 성능과 Precision-Recall 균형이 가장 안정적이었으며, Accuracy와 F1-score 모두 가장 높았습니다.

### ResNet18 Full Fine-Tuning
- Test Accuracy: 0.9247
- Recall: 0.9846
- F1-score: 0.9423

DenseNet보다 전체 성능은 조금 낮았지만, Recall이 더 높아 실제 폐렴 환자를 놓칠 가능성이 더 낮았습니다.

### ResNet50 Full Fine-Tuning
- Test Accuracy: 0.9071
- Recall: 0.9872
- F1-score: 0.9300

세 모델 중 Accuracy는 가장 낮았지만 Recall이 가장 높았고, False Negative가 가장 적었습니다.

---

## Confusion Matrix 기반 해석

이번 프로젝트에서는 단순한 Accuracy보다도 실제 의료 현장에서 더 중요한 기준을 함께 고려했습니다.

- FN(False Negative): 실제 폐렴인데 정상으로 예측
- FP(False Positive): 실제 정상인데 폐렴으로 예측

의료 도메인에서는 FN이 훨씬 더 위험합니다.  
즉, 실제 환자를 놓치는 것이 가장 큰 문제이기 때문에, Confusion Matrix를 통해 각 모델의 FN 수를 비교했습니다.

FN 비교 결과는 다음과 같습니다.

- ResNet50_Full: 18
- ResNet50_Partial: 20
- ResNet18_Partial: 21
- DenseNet121_Full: 27
- ResNet18_Full: 29

즉, 전체 성능은 DenseNet121_Full이 가장 좋았지만, 실제 폐렴 환자를 가장 적게 놓친 모델은 ResNet50_Full이었습니다.

---

## 최종 모델 선정

이번 프로젝트에서는 두 가지 관점으로 모델을 해석했습니다.

### 1. 전체 성능 기준 최적 모델
DenseNet121_Full

- 가장 높은 Test Accuracy
- 가장 높은 F1-score
- Precision과 Recall의 균형이 가장 좋음
- 학습 곡선도 안정적

### 2. 의료 안전성 기준 최종 선택 모델
ResNet50_Full

- Recall이 가장 높음
- False Negative가 가장 적음
- 실제 폐렴 환자를 놓칠 가능성이 가장 낮음

따라서 이번 프로젝트에서는 단순 최고 점수 모델을 그대로 선택하지 않고, 의료 문제의 특성을 반영해 ResNet50_Full을 최종 선택 모델로 정리했습니다.

이 점은 단순 모델 성능 비교를 넘어서, 문제 맥락에 맞는 평가 기준을 적용했다는 데 의미가 있습니다.

---

## 이 프로젝트를 통해 갖추게 된 능력

### 1. 의료영상 분류 문제를 실무 관점에서 해석하는 능력

단순히 정확도가 높은 모델을 고르는 것이 아니라, 실제로 어떤 오분류가 더 위험한지까지 함께 고려하면서 문제를 해석하는 경험을 할 수 있었습니다.  
이를 통해 모델 평가를 숫자 자체가 아니라 문제 상황과 연결해서 볼 수 있게 되었습니다.

### 2. 데이터 불균형 대응 능력

WeightedRandomSampler를 적용해 학습 데이터의 클래스 불균형 문제를 완화했습니다.  
이를 통해 실제 분류 문제에서 자주 마주치는 데이터 편향 문제를 다루는 역량을 기를 수 있었습니다.

### 3. Transfer Learning 전략 비교 능력

Feature Extraction, Partial Fine-Tuning, Full Fine-Tuning을 동일한 조건에서 비교하면서, 어떤 전이학습 방식이 현재 데이터셋에 더 적합한지 직접 검증할 수 있었습니다.  
즉, 단순히 사전학습 모델을 가져다 쓰는 수준이 아니라, 학습 전략 자체를 설계하고 비교하는 경험을 쌓을 수 있었습니다.

### 4. 다양한 CNN 백본 모델 활용 능력

Custom CNN, ResNet18, ResNet50, DenseNet121을 직접 구성하고 학습시키면서, 모델 구조별 특성과 성능 차이를 비교할 수 있었습니다.

### 5. PyTorch 기반 비전 학습 파이프라인 구현 능력

Dataset, DataLoader, sampler, 학습 함수, 평가 함수, Early Stopping, 성능 지표 계산, Confusion Matrix, Grad-CAM까지 전체 파이프라인을 직접 구성했습니다.  
이를 통해 PyTorch 기반 이미지 분류 프로젝트를 end-to-end로 구현하는 역량을 강화할 수 있었습니다.

### 6. 해석 가능한 AI 적용 능력

Grad-CAM을 통해 모델이 집중한 영역을 시각화하며 예측 결과를 해석했습니다.  
이 과정을 통해 단순히 분류 성능만 높이는 것이 아니라, 모델의 판단 근거를 함께 설명하는 방향의 실험을 진행할 수 있었습니다.

### 7. 성능과 안전성 사이의 트레이드오프를 판단하는 능력

이번 프로젝트에서는 Accuracy가 가장 높은 모델과 FN이 가장 적은 모델이 서로 달랐습니다.  
이 차이를 해석하고, 최종적으로 어떤 기준이 더 중요한지 판단하는 과정을 통해 모델 선택의 기준을 더 입체적으로 볼 수 있게 되었습니다.

---

## 이 프로젝트에서 중요하게 본 점

이번 프로젝트에서 가장 중요하게 생각한 점은 단순히 최고 점수를 얻는 것이 아니라, 왜 그 모델을 선택해야 하는지를 설명할 수 있어야 한다는 것이었습니다.

그래서 다음과 같은 원칙으로 프로젝트를 진행했습니다.

- validation 데이터의 신뢰성을 먼저 확보하기
- 클래스 불균형 문제를 반드시 해결하기
- baseline부터 transfer learning까지 단계적으로 비교하기
- Accuracy만 보지 않고 Precision, Recall, F1-score를 함께 확인하기
- Confusion Matrix를 통해 의료적으로 더 중요한 오류를 따로 보기
- Grad-CAM으로 모델의 판단 근거를 확인하기
- 성능이 높은 모델과 안전한 모델이 다를 수 있음을 인정하고, 문제 맥락에 맞춰 최종 선택하기

이러한 흐름은 향후 의료 AI, 이상 탐지, 고위험 분류 문제처럼 오분류의 비용이 큰 분야를 다룰 때 매우 중요한 기반이 된다고 생각합니다.

---

## 사용 기술

- Python
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- KaggleHub

---

## 한 줄 정리

이 프로젝트는 흉부 X-Ray 이미지를 활용해 폐렴 여부를 분류하기 위해 데이터 재구성, 불균형 처리, Transfer Learning 전략 비교, Grad-CAM 해석, Confusion Matrix 기반 위험도 평가까지 수행하며 성능과 의료 안전성을 함께 고려한 의료영상 분류 프로젝트입니다.