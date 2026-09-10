"""v3: 원본 변수 기반 leakage-safe RBF-SVR 모델.

파생변수는 사용하지 않는다. 고정값 결측 처리를 제외한 원핫 인코딩,
스케일링, 타깃 변환은 각 교차검증 학습 fold에서만 학습한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler
from sklearn.svm import SVR


SEED = 42
N_FOLDS = 5
TARGET = "stress_score"
ID_COL = "ID"
OUTPUT = "submission_raw_v3_svr.csv"
# 현재 터미널 위치가 아니라 이 Python 파일이 저장된 폴더를 기준으로 한다.
BASE_DIR = Path(__file__).resolve().parent


def apply_fixed_missing_values(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """데이터 통계량을 사용하지 않는 고정 규칙으로 결측치를 표현한다."""
    df = df.copy()

    # EDA상 고령·비경제활동 집단과 관련된 mean_working 결측을 0으로 표현한다.
    # 0은 실제 관측 근무시간 복원이 아니라 '기록된 근무시간 없음'을 나타내는 값이다.
    df["mean_working"] = df["mean_working"].fillna(0)

    # 범주형 결측은 '없음'으로 단정하지 않고 알 수 없음/무응답 범주로 보존한다.
    for column in categorical_cols:
        df[column] = df[column].fillna("__UNKNOWN__")

    return df


def build_model(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """fold마다 새로 학습할 전처리기와 SVR 파이프라인을 만든다."""
    preprocessor = ColumnTransformer(
        transformers=[
            # 수치형 변수는 이상치 영향이 비교적 작은 RobustScaler로 변환한다.
            ("numeric", RobustScaler(), numeric_cols),
            # 범주 목록은 fit에 들어온 학습 fold에서만 학습한다.
            # 검증/test에 처음 보는 범주가 있으면 오류 없이 0 벡터로 처리한다.
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_cols,
            ),
        ]
    )

    # 타깃 분포를 정규분포 형태로 바꾼 공간에서 RBF-SVR을 학습하고,
    # predict 결과는 TransformedTargetRegressor가 원래 stress_score 척도로 되돌린다.
    target_model = TransformedTargetRegressor(
        regressor=SVR(
            kernel="rbf",
            C=3.963530707518144,
            gamma=1.0631617004546035,
            epsilon=0.0,
            cache_size=1000,
        ),
        transformer=QuantileTransformer(
            n_quantiles=1000,
            output_distribution="normal",
            random_state=SEED,
        ),
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", target_model),
        ]
    )


def main() -> None:
    # 다른 폴더에서 이 파일을 실행해도 open 폴더의 CSV를 정확히 찾는다.
    train = pd.read_csv(BASE_DIR / "train.csv")
    test = pd.read_csv(BASE_DIR / "test.csv")

    # test에 존재하는 원본 변수 중 ID를 제외한 열만 사용한다.
    feature_cols = [column for column in test.columns if column != ID_COL]
    categorical_cols = [
        column
        for column in feature_cols
        if pd.api.types.is_string_dtype(train[column].dtype)
    ]
    numeric_cols = [
        column for column in feature_cols if column not in categorical_cols
    ]

    # 고정값 결측 처리는 train/test의 분포나 통계량을 계산하지 않는다.
    x = apply_fixed_missing_values(train[feature_cols], categorical_cols)
    x_test = apply_fixed_missing_values(test[feature_cols], categorical_cols)
    y = train[TARGET].to_numpy()

    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), start=1):
        x_train = x.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # 매 fold마다 새 파이프라인을 만들기 때문에 전처리 기준이 공유되지 않는다.
        model = build_model(numeric_cols, categorical_cols)
        model.fit(x_train, y_train)

        oof_predictions[valid_idx] = model.predict(x_valid)
        test_predictions += model.predict(x_test) / N_FOLDS

        fold_predictions = np.clip(oof_predictions[valid_idx], 0, 1)
        fold_mae = mean_absolute_error(y_valid, fold_predictions)
        print(f"fold {fold}: mae={fold_mae:.5f}")

    # 대회의 타깃 범위에 맞춰 아주 드물게 벗어나는 예측값을 제한한다.
    oof_predictions = np.clip(oof_predictions, 0, 1)
    test_predictions = np.clip(test_predictions, 0, 1)
    cv_mae = mean_absolute_error(y, oof_predictions)
    print(f"\nOOF MAE: {cv_mae:.5f}")

    submission = pd.read_csv(BASE_DIR / "sample_submission.csv")
    submission[TARGET] = test_predictions
    output_path = BASE_DIR / OUTPUT
    submission.to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
