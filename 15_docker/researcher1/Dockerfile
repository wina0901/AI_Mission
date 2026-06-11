# 연구자 1 이미지
# 역할: train.csv로 모델을 학습하고 shared/model.pkl을 생성합니다.
FROM python:3.10-slim

WORKDIR /app

# 필요한 패키지 버전을 고정해서 연구자 2 환경과 동일하게 맞춥니다.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 연구자 1 컨테이너 내부에 train/test 데이터를 포함합니다.
# 연구자 2는 이 데이터를 직접 갖고 있지 않고, researcher1 컨테이너 실행 후 shared/test.csv로 전달받습니다.
COPY data /app/data
COPY train_model.py /app/train_model.py

# docker compose에서 ./shared:/app/shared 볼륨을 연결합니다.
CMD ["python", "train_model.py"]
