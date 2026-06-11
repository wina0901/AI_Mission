# 연구자 2 이미지
# 역할: Jupyter Notebook에서 shared/model.pkl과 shared/test.csv를 읽어 추론합니다.
FROM python:3.10-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter Notebook 서버를 외부 브라우저에서 접속할 수 있게 실행합니다.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
