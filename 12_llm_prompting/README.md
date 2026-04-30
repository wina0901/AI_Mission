# KoBART 기반 한국어 문서 요약 프로젝트

## 프로젝트 개요

이 프로젝트는 한국어 문서를 입력받아 핵심 내용을 짧은 요약문으로 생성하는 Abstractive Summarization 프로젝트입니다.  
Hugging Face Transformers 라이브러리를 활용해 KoBART 기반 사전학습 요약 모델을 불러오고, 한국어 문서 요약 데이터셋에 맞게 fine-tuning한 뒤 ROUGE와 BERTScore를 이용해 정량 평가를 수행했습니다.

단순히 사전학습 모델을 불러와 사용하는 데서 끝나지 않고, 데이터 로드, 텍스트 정제, 길이 통계 분석, Hugging Face Dataset 변환, 토큰화, Seq2SeqTrainer 기반 학습, 생성 결과 확인, 정량 평가까지 전체 문서 요약 파이프라인을 직접 구성했습니다.

---

## 이 프로젝트를 왜 했는가

문서 요약은 긴 글에서 핵심 내용을 빠르게 파악해야 하는 뉴스, 보고서, 논문, 고객 상담 이력, 회의록, 법률 문서 등 다양한 분야에서 활용될 수 있는 중요한 자연어처리 기술입니다.  
특히 정보량이 많은 문서를 그대로 읽기 어려운 상황에서는 자동 요약 모델이 사용자의 시간과 업무 효율을 크게 높일 수 있습니다.

이번 프로젝트는 다음과 같은 이유로 진행했습니다.

- 자연어처리에서 분류나 번역을 넘어, 긴 문서를 짧은 문장으로 재구성하는 생성형 NLP 문제를 경험하고 싶었습니다.
- Hugging Face Transformers를 활용해 실제 사전학습 모델을 fine-tuning하는 전체 과정을 직접 구현해보고 싶었습니다.
- KoBART처럼 한국어에 적합한 encoder-decoder 기반 모델이 문서 요약에서 어떻게 동작하는지 확인하고 싶었습니다.
- ROUGE처럼 표면적 n-gram 일치도를 보는 지표와 BERTScore처럼 의미적 유사도를 보는 지표를 함께 사용해 요약 품질을 평가해보고 싶었습니다.
- 모델이 생성한 요약문을 직접 확인하면서, 정량 평가와 정성 평가를 함께 수행하는 경험을 쌓고 싶었습니다.

즉, 이번 프로젝트는 단순한 모델 호출 실습이 아니라, 한국어 문서 요약 문제를 실제 데이터 전처리부터 모델 fine-tuning, 생성 결과 평가까지 end-to-end로 수행한 프로젝트입니다.

---

## 문제 정의

- 주제: 한국어 문서 자동 요약
- 문제 유형: Abstractive Text Summarization
- 입력: 긴 한국어 문서 본문
- 출력: 문서의 핵심 내용을 담은 짧은 요약문
- 사용 모델: EbanLee/kobart-summary-v3
- 주요 평가 지표:
  - ROUGE-1
  - ROUGE-2
  - ROUGE-L
  - ROUGE-Lsum
  - BERTScore Precision
  - BERTScore Recall
  - BERTScore F1

이번 프로젝트의 핵심은 원문을 단순히 일부 문장으로 잘라내는 것이 아니라, 모델이 문서 내용을 이해하고 새로운 형태의 요약문을 생성하도록 하는 것이었습니다.

---

## 데이터 설명

사용한 데이터는 한국어 문서와 정답 요약문이 함께 제공되는 JSON 형식의 문서 요약 데이터입니다.

데이터 구성은 다음과 같습니다.

- 학습 샘플 수: 56,756개
- 검증 샘플 수: 7,007개
- 입력 컬럼: text
- 정답 컬럼: summary

길이 통계는 다음과 같습니다.

### Train 데이터

| 구분 | text_len | summary_len |
|------|---------:|------------:|
| count | 56,756 | 56,756 |
| mean | 1,170.51 | 122.37 |
| min | 109 | 19 |
| 50% | 1,134 | 120 |
| max | 1,939 | 397 |

### Validation 데이터

| 구분 | text_len | summary_len |
|------|---------:|------------:|
| count | 7,007 | 7,007 |
| mean | 1,087.20 | 126.16 |
| min | 222 | 33 |
| 50% | 1,090 | 125 |
| max | 1,859 | 333 |

길이 분석을 통해 원문은 평균 약 1,000자 이상으로 비교적 긴 문서이며, 요약문은 평균 약 120자 내외로 구성되어 있음을 확인했습니다.  
이를 바탕으로 모델 입력과 출력의 최대 길이를 설정했습니다.

---

## 프로젝트 진행 과정

### 1. 데이터 로드 및 텍스트 정제

먼저 JSON 파일을 불러오고, 문서 본문과 요약문을 각각 추출했습니다.  
이후 모델 학습에 방해가 될 수 있는 불필요한 문자를 줄이기 위해 텍스트 정제 함수를 구성했습니다.

적용한 전처리는 다음과 같습니다.

- 줄바꿈과 탭을 공백으로 변환
- 연속 공백 제거
- 문장부호 앞의 불필요한 공백 제거
- 특수 괄호 문자 정리
- 따옴표 형식 통일
- 숫자 내 쉼표 제거
- 전체 공백 정리

이 과정은 단순한 문자열 정리가 아니라, 요약 모델이 본문과 요약문 사이의 의미 대응 관계를 더 안정적으로 학습할 수 있도록 입력 품질을 정리하는 단계였습니다.

---

### 2. 데이터 길이 통계 분석

전처리된 데이터의 본문 길이와 요약문 길이를 각각 분석했습니다.  
이를 통해 모델 입력 길이와 출력 길이를 무작정 정하는 것이 아니라, 실제 데이터 분포를 기반으로 설정할 수 있었습니다.

이번 프로젝트에서 사용한 길이 설정은 다음과 같습니다.

- max_input_length: 576
- max_target_length: 140

즉, 너무 짧게 잘라 핵심 정보가 사라지지 않도록 하면서도, GPU 메모리와 학습 효율을 고려해 적절한 최대 길이를 설정했습니다.

---

### 3. Hugging Face Dataset 변환

전처리된 train / validation 데이터를 Hugging Face Dataset 형식으로 변환했습니다.

구성 결과는 다음과 같습니다.

- train: 56,756 rows
- validation: 7,007 rows
- features:
  - text
  - summary

이 과정을 통해 Transformers의 Trainer API와 자연스럽게 연결할 수 있는 데이터 구조를 만들었습니다.

---

### 4. 모델 및 토크나이저 로드

사전학습 요약 모델로 EbanLee/kobart-summary-v3를 사용했습니다.

사용한 구성은 다음과 같습니다.

- Tokenizer: PreTrainedTokenizerFast
- Model: BartForConditionalGeneration
- Base model: KoBART 계열 한국어 요약 모델
- Generation config:
  - max_length: 128
  - min_length: 35
  - num_beams: 5

또한 pad_token_id, eos_token_id, bos_token_id, decoder_start_token_id를 명시적으로 설정하여 Seq2Seq 생성 과정에서 토큰 설정 문제가 발생하지 않도록 처리했습니다.

---

### 5. 토큰화 및 모델 입력 변환

본문과 요약문을 각각 모델 입력 형식에 맞게 토큰화했습니다.

토큰화 설정은 다음과 같습니다.

- 입력 본문:
  - max_length = 576
  - truncation = True
- 정답 요약문:
  - max_length = 140
  - truncation = True

요약문은 labels로 저장해 Seq2Seq 학습에서 decoder의 정답 시퀀스로 사용했습니다.  
즉, 원문을 encoder 입력으로 넣고, 요약문을 decoder가 생성해야 할 목표 문장으로 학습하도록 구성했습니다.

---

## 모델 학습

학습은 Hugging Face의 Seq2SeqTrainer를 사용해 진행했습니다.

주요 학습 설정은 다음과 같습니다.

- num_train_epochs: 5
- learning_rate: 2e-5
- train batch size: 16
- eval batch size: 16
- gradient_accumulation_steps: 2
- weight_decay: 0.05
- warmup_ratio: 0.15
- lr_scheduler_type: cosine
- evaluation strategy: epoch
- save strategy: epoch
- predict_with_generate: True
- generation_num_beams: 5
- load_best_model_at_end: True
- early stopping 적용

이 설정은 대형 Seq2Seq 모델을 안정적으로 fine-tuning하기 위한 구성입니다.  
gradient accumulation을 통해 실질적인 batch size를 확보했고, warmup과 cosine scheduler를 적용해 학습 초반 불안정성을 줄였습니다.

---

## 평가 방식

이번 프로젝트에서는 요약 결과를 두 가지 관점에서 평가했습니다.

### 1. ROUGE

ROUGE는 생성 요약문과 정답 요약문 사이의 n-gram 겹침 정도를 측정하는 지표입니다.

사용한 ROUGE 지표는 다음과 같습니다.

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum

이를 통해 생성 요약문이 정답 요약문과 얼마나 비슷한 단어와 문장 흐름을 공유하는지 확인했습니다.

### 2. BERTScore

BERTScore는 문장 표현의 의미적 유사성을 기반으로 생성문과 정답문을 비교하는 지표입니다.  
ROUGE가 표현이 다르면 낮게 나올 수 있는 반면, BERTScore는 의미가 비슷한 경우 더 유연하게 평가할 수 있습니다.

이번 프로젝트에서는 multilingual BERT 기반 BERTScore를 사용했습니다.

---

## 최종 정량 평가 결과

검증 데이터 평가 결과는 다음과 같습니다.

### 전체 Validation 평가

- eval_loss: 2.8831
- eval_ROUGE-1: 0.2760
- eval_ROUGE-2: 0.0989
- eval_ROUGE-L: 0.2727
- eval_ROUGE-Lsum: 0.2724

### 최종 샘플 기반 ROUGE + BERTScore 평가

평가 샘플 수: 200개

| Metric | Score |
|-------|------:|
| ROUGE-1 | 0.2748 |
| ROUGE-2 | 0.0862 |
| ROUGE-L | 0.2732 |
| ROUGE-Lsum | 0.2736 |
| BERTScore Precision | 0.7634 |
| BERTScore Recall | 0.7883 |
| BERTScore F1 | 0.7752 |

이 결과를 통해 다음을 확인할 수 있었습니다.

- ROUGE-1과 ROUGE-L이 약 0.27 수준으로, 모델이 원문의 핵심 키워드와 전반적인 문장 흐름을 어느 정도 반영함
- ROUGE-2는 상대적으로 낮아, 정답 요약문과 완전히 같은 구문을 생성하기보다는 다른 표현으로 요약하는 경향이 있음
- BERTScore F1이 0.7752로 나타나, 표현이 완전히 같지 않아도 의미적으로는 정답 요약문과 유사한 내용을 생성하고 있음을 확인함

즉, 이 모델은 정답 문장과 단어 단위로 완전히 일치하는 요약을 생성하기보다는, 핵심 의미를 유지한 요약을 생성하는 방향으로 학습되었다고 해석할 수 있습니다.

---

## 정성 평가

검증 데이터 샘플에 대해 원문, 정답 요약, 생성 요약을 함께 비교했습니다.

정성 평가에서 확인한 특징은 다음과 같습니다.

- 긴 정치·사회·경제 기사에서 핵심 주제를 비교적 잘 추출함
- 정답 요약보다 더 긴 문장을 생성하는 경우가 있음
- 핵심 인물, 사건, 주장, 결론을 포함하는 경향이 있음
- 일부 문장에서는 원문 표현을 길게 이어 붙이는 경향이 있음
- 압축률은 개선 여지가 있지만, 의미 보존 측면에서는 안정적인 결과를 보임

예를 들어 탈원전 정책과 전기료 인상 관련 기사에서는, 생성 요약이 원문의 핵심 논지인 한전의 경영 부담, 전기료 인상 가능성, 정부의 탈원전 속도 조절 필요성을 비교적 잘 포함했습니다.

즉, 정량 지표뿐 아니라 실제 생성 결과를 직접 확인했을 때도 모델이 문서의 주요 논점을 포착하는 능력을 보였습니다.

---

## 최종 결과 해석

이번 프로젝트의 최종 결과는 다음과 같이 해석할 수 있습니다.

- KoBART 기반 사전학습 요약 모델을 한국어 문서 요약 데이터에 맞게 fine-tuning함
- ROUGE 기준으로 핵심 단어와 문장 흐름을 일정 수준 반영함
- BERTScore 기준으로 의미적 유사성이 비교적 안정적으로 나타남
- 생성 요약문은 정답과 완전히 같은 표현은 아니지만, 문서의 중심 내용을 잘 포함하는 경향을 보임
- 다만 더 짧고 압축적인 요약을 생성하는 부분은 추가 개선 가능성이 있음

따라서 이번 프로젝트는 한국어 문서 요약 문제에서 사전학습 Seq2Seq 모델을 활용해 의미 중심의 요약문을 생성하는 전체 과정을 구현했다는 점에서 의미가 있습니다.

---

## 이 프로젝트를 통해 갖추게 된 능력

### 1. 한국어 문서 요약 파이프라인을 end-to-end로 구현하는 능력

JSON 데이터 로드, 텍스트 정제, 길이 분석, Dataset 변환, 토큰화, 모델 fine-tuning, 생성 결과 평가까지 전체 문서 요약 흐름을 직접 구성할 수 있게 되었습니다.

### 2. Hugging Face Transformers 활용 능력

PreTrainedTokenizerFast, BartForConditionalGeneration, Seq2SeqTrainer, DataCollatorForSeq2Seq, GenerationConfig를 활용해 실제 사전학습 모델을 fine-tuning했습니다.  
이를 통해 Hugging Face 기반 NLP 프로젝트를 실무형 구조로 구현하는 역량을 강화할 수 있었습니다.

### 3. Encoder-Decoder 기반 생성 모델 이해 능력

KoBART 모델을 활용하면서 입력 문서를 encoder가 이해하고, decoder가 요약문을 생성하는 Seq2Seq 기반 생성 구조를 실제 프로젝트에 적용했습니다.

### 4. 생성형 NLP 모델 학습 설정 능력

learning rate, batch size, gradient accumulation, weight decay, warmup ratio, cosine scheduler, beam search, early stopping 등을 적용하며 대형 생성 모델을 안정적으로 학습시키는 방법을 경험했습니다.

### 5. 요약 모델 평가 지표 해석 능력

ROUGE와 BERTScore를 함께 사용하면서, 표면적인 단어 일치도와 의미적 유사도를 구분해 평가할 수 있게 되었습니다.  
이를 통해 생성형 NLP 모델은 하나의 지표만으로 판단하기 어렵고, 여러 평가 관점을 함께 봐야 한다는 점을 이해했습니다.

### 6. 정량 평가와 정성 평가를 함께 수행하는 능력

수치 지표뿐 아니라 실제 생성 요약문을 정답 요약과 비교했습니다.  
이를 통해 모델이 숫자로는 어느 정도 성능을 보이는지, 실제 읽었을 때는 어떤 장단점이 있는지를 함께 분석하는 역량을 기를 수 있었습니다.

### 7. 사전학습 모델을 목적 데이터에 맞게 fine-tuning하는 능력

이미 학습된 KoBART 요약 모델을 새로운 데이터셋에 맞게 추가 학습하면서, 사전학습 모델을 그대로 사용하는 것과 목적 데이터에 맞게 조정하는 것의 차이를 경험할 수 있었습니다.

---

## 이 프로젝트에서 중요하게 본 점

이번 프로젝트에서 가장 중요하게 생각한 점은 단순히 요약 모델을 실행하는 것이 아니라, 실제 문서 요약 프로젝트의 전체 흐름을 직접 구성하고 평가하는 것이었습니다.

그래서 다음과 같은 원칙으로 프로젝트를 진행했습니다.

- 원문과 요약문의 길이 분포를 먼저 확인하기
- 텍스트 정제를 통해 입력 품질을 높이기
- Hugging Face Dataset 형식으로 데이터 흐름을 정리하기
- 모델의 특성에 맞게 입력 길이와 출력 길이를 설정하기
- beam search를 활용해 더 안정적인 요약문 생성하기
- ROUGE와 BERTScore를 함께 사용해 요약 품질을 다각도로 평가하기
- 실제 생성 요약문을 직접 읽으며 정성 평가까지 수행하기

이러한 흐름은 향후 뉴스 요약, 회의록 요약, 고객 상담 요약, 보고서 요약 같은 실무형 NLP 프로젝트를 수행할 때 중요한 기반이 될 수 있다고 생각합니다.

---

## 개선 방향

이번 결과를 바탕으로 향후 다음과 같은 방향으로 성능을 더 높일 수 있다고 판단했습니다.

- 더 큰 validation subset 또는 전체 validation 데이터에 대한 BERTScore 평가
- 요약문 길이 제어를 위한 length penalty 조정
- repetition penalty를 적용해 반복 표현 감소
- num_beams, min_length, max_length 등 generation parameter 튜닝
- 데이터 도메인별 성능 비교
- 문서 길이가 긴 샘플에 대한 별도 성능 분석
- ROUGE 외에 BLEURT, MoverScore 등 추가 의미 기반 평가 지표 도입
- 사람이 직접 평가하는 human evaluation 추가

즉, 현재 모델은 핵심 의미를 비교적 잘 반영하는 요약을 생성했지만, 더 짧고 압축적인 문장 생성과 반복 표현 제어 측면에서는 추가 개선 여지가 있습니다.

---

## 사용 기술

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- KoBART
- ROUGE
- BERTScore
- Pandas
- NumPy

---

## 한 줄 정리

이 프로젝트는 한국어 문서 요약을 위해 KoBART 기반 사전학습 모델을 fine-tuning하고, 데이터 전처리, 토큰화, Seq2SeqTrainer 학습, beam search 생성, ROUGE 및 BERTScore 평가, 정성 분석까지 전 과정을 수행한 생성형 NLP 문서 요약 프로젝트입니다.