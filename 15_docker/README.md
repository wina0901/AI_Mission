# Docker 기반 협업 머신러닝 프로젝트

## 프로젝트 개요

이 프로젝트는 Docker를 활용하여 머신러닝 모델 개발과 추론 과정을 연구자 간에 분리하고, 공유 볼륨을 통해 협업하는 환경을 구축하는 프로젝트입니다.

일반적으로 머신러닝 프로젝트에서는 학습 환경과 추론 환경이 서로 다르거나, 연구자가 직접 모델 파일을 전달해야 하는 번거로움이 존재합니다. 이번 프로젝트에서는 Docker Compose를 활용하여 연구자 1과 연구자 2의 역할을 분리하고, 공유 볼륨을 통해 모델과 데이터를 자동으로 전달하는 협업 구조를 구현했습니다.

연구자 1은 모델 학습 및 저장을 담당하고, 연구자 2는 저장된 모델을 활용해 추론과 결과 분석을 수행하도록 구성했습니다.

## 이 프로젝트를 왜 했는가

실제 AI 프로젝트에서는 모델 개발뿐 아니라 연구자 간 협업, 환경 재현성, 배포 가능성도 매우 중요합니다.

이번 프로젝트는 Docker를 활용한 컨테이너 기반 개발 환경과 머신러닝 협업 구조를 경험하기 위해 진행했습니다.

## 문제 정의

- 주제: Docker 기반 머신러닝 협업 환경 구축
- 문제 유형: 컨테이너 기반 ML 파이프라인
- 모델: Linear Regression
- 주요 기술: Docker, Docker Compose, Shared Volume

## 데이터 설명

학생 학업 성취도 예측 데이터를 활용했습니다.

- Train: 7,000건
- Test: 3,000건
- Target: Performance Index

## 프로젝트 진행 과정

### 1. 연구자 1 - 모델 학습

- 데이터 로드
- 전처리 파이프라인 구성
- 모델 학습
- 성능 평가
- 모델 저장

### 2. 전처리

숫자형 데이터

- Median Imputation
- StandardScaler

범주형 데이터

- Most Frequent Imputation
- One-Hot Encoding

### 3. 모델 학습

- Linear Regression
- Validation Split 8:2
- RMSE, R² 평가

### 4. 모델 공유

shared 볼륨에

- model.pkl
- metrics.json
- test.csv

저장

### 5. 연구자 2 - 추론

- model.pkl 로드
- test.csv 로드
- 예측 수행
- result.csv 생성
- 결과 시각화

## Docker Compose 구성

- researcher1 : 학습 컨테이너
- researcher2-jupyter : 추론 컨테이너
- shared volume : 데이터 공유

## 최종 결과

| Metric | Value |
|----------|----------:|
| Validation RMSE | 2.01 |
| Validation R² | 0.9893 |

## 이 프로젝트를 통해 갖추게 된 능력

1. Docker 기반 머신러닝 환경 구축 능력
2. Docker Compose 활용 능력
3. Shared Volume 활용 능력
4. 재현 가능한 ML 파이프라인 구축 능력
5. 협업형 머신러닝 프로젝트 경험

## 개선 방향

- RandomForest, XGBoost 추가 실험
- MLflow 연동
- FastAPI 추론 서버 구축
- Kubernetes 확장

## 사용 기술

- Python
- Docker
- Docker Compose
- Scikit-learn
- Pandas
- NumPy
- Jupyter Notebook

## 한 줄 정리

이 프로젝트는 Docker Compose와 Shared Volume을 활용하여 머신러닝 모델 학습과 추론 환경을 분리하고, 연구자 간 협업이 가능한 재현성 높은 ML 파이프라인을 구축한 프로젝트입니다.
