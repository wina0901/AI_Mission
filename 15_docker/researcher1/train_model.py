

from pathlib import Path
import json
import shutil

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 컨테이너 내부 기본 경로입니다.
BASE_DIR = Path("/app")
DATA_DIR = BASE_DIR / "data"
SHARED_DIR = BASE_DIR / "shared"

# 입력/출력 파일 경로를 명확하게 분리합니다.
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = SHARED_DIR / "model.pkl"
METRICS_PATH = SHARED_DIR / "metrics.json"
SHARED_TEST_PATH = SHARED_DIR / "test.csv"


def main() -> None:
    """데이터 로드부터 모델 저장까지 전체 학습 파이프라인을 실행합니다."""
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    train_df = pd.read_csv(TRAIN_PATH)

    # 2. 입력 변수와 목표 변수를 분리합니다.
    target_col = "Performance Index"
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # 3. 컬럼 타입에 따라 전처리 방식을 나눕니다.
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    # 숫자형: 결측치가 생겨도 안전하게 중앙값 대체 후 표준화합니다.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 범주형: 결측치가 생겨도 최빈값 대체 후 원-핫 인코딩합니다.
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer를 사용하면 숫자형/범주형 전처리를 하나의 객체로 관리할 수 있습니다.
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 4. 전처리와 모델을 하나의 Pipeline으로 묶습니다.
    # 이렇게 저장하면 추론 시에도 같은 전처리가 자동으로 적용됩니다.
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )

    # 5. 검증용 데이터를 분리해서 RMSE를 확인합니다.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)
    rmse = root_mean_squared_error(y_valid, valid_pred)
    r2 = r2_score(y_valid, valid_pred)

    # 6. 최종 제출용 모델은 전체 train 데이터로 다시 학습합니다.
    model.fit(X, y)

    # 7. 연구자 2가 사용할 수 있도록 공유 폴더에 모델과 테스트 데이터를 저장합니다.
    joblib.dump(model, MODEL_PATH)
    shutil.copy(TEST_PATH, SHARED_TEST_PATH)

    # 8. 보고서에 쓸 수 있도록 성능 지표도 저장합니다.
    metrics = {
        "model": "LinearRegression with preprocessing pipeline",
        "validation_rmse": float(rmse),
        "validation_r2": float(r2),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(pd.read_csv(TEST_PATH))),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[완료] 연구자 1 학습 컨테이너 실행 완료")
    print(f"- model.pkl 저장 위치: {MODEL_PATH}")
    print(f"- test.csv 공유 위치: {SHARED_TEST_PATH}")
    print(f"- validation RMSE: {rmse:.4f}")
    print(f"- validation R2: {r2:.4f}")


if __name__ == "__main__":
    main()
