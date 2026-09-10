"""원본 변수만 사용하는 LightGBM 베이스라인 모델.

파생변수는 만들지 않고 ID를 학습에서 제외한다.
수치형 결측치는 각 학습 fold에서 계산한 중앙값으로 대체한다.
범주형 결측치는 '알 수 없음'을 뜻하는 별도 범주로 처리하며, 범주 목록은
매 교차검증의 학습 fold에서만 구해 검증 fold와 test 데이터에 적용한다.
"""

# 배열 생성과 수치 계산에 사용하는 라이브러리
import numpy as np
# CSV 파일을 읽고 표 형태의 데이터를 다루는 라이브러리
import pandas as pd
# 표 형태 데이터에 강한 부스팅 모델 LightGBM
import lightgbm as lgb
# 대회 평가 지표인 평균 절대 오차(MAE)를 계산하는 함수
from sklearn.metrics import mean_absolute_error
# 데이터를 여러 학습/검증 묶음으로 나누는 교차검증 도구
from sklearn.model_selection import KFold


# 난수 결과를 고정해 코드를 다시 실행해도 같은 결과가 나오도록 한다.
SEED = 42
# 전체 train 데이터를 5개로 나눠 5번 학습한다.
N_FOLDS = 5
# 모델이 예측해야 하는 정답 열의 이름이다.
TARGET = "stress_score"
# ID는 단순 식별자이므로 모델의 입력 변수에서 제외한다.
ID_COL = "ID"
# 예측 결과를 저장할 제출 파일 이름이다.
OUTPUT = "submission_lgbm_raw_v1_median.csv"


def prepare_fold(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """학습 fold 중앙값으로 수치형 NaN을 채우고 범주형 변수를 변환한다."""

    # 원본 DataFrame을 실수로 변경하지 않도록 각각 복사한다.
    train_x = train_x.copy()
    valid_x = valid_x.copy()
    test_x = test_x.copy()

    # 검증/test 정보를 사용하지 않고 현재 학습 fold에서만 중앙값을 계산한다.
    medians = train_x[numeric_cols].median()
    train_x[numeric_cols] = train_x[numeric_cols].fillna(medians)
    valid_x[numeric_cols] = valid_x[numeric_cols].fillna(medians)
    test_x[numeric_cols] = test_x[numeric_cols].fillna(medians)

    # 문자로 된 범주형 변수를 하나씩 처리한다.
    for column in categorical_cols:
        # 범주형 결측치는 '없음'이 아니라 '알 수 없음/무응답'으로 구분한다.
        train_values = train_x[column].fillna("__UNKNOWN__")
        # 현재 학습 fold에 실제로 존재하는 값만 범주 목록으로 만든다.
        categories = sorted(train_values.unique().tolist())
        # LightGBM이 범주형 변수임을 알 수 있도록 category 자료형으로 변환한다.
        train_x[column] = pd.Categorical(train_values, categories=categories)
        # 검증 데이터도 같은 범주 목록으로 변환한다.
        # 학습 fold에 없던 값은 NaN으로 처리되어 새로운 정보를 학습에 섞지 않는다.
        valid_x[column] = pd.Categorical(
            valid_x[column].fillna("__UNKNOWN__"), categories=categories
        )
        # test 데이터도 학습 fold의 범주 목록만 사용해 변환한다.
        test_x[column] = pd.Categorical(
            test_x[column].fillna("__UNKNOWN__"), categories=categories
        )

    # 전처리가 끝난 학습, 검증, test 데이터를 반환한다.
    return train_x, valid_x, test_x


def main() -> None:
    # 정답이 포함된 학습 데이터와 정답이 없는 평가 데이터를 읽는다.
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # test의 원본 열 가운데 ID만 제외한 열을 모델 입력 변수로 사용한다.
    # stress_score는 test에 없기 때문에 자동으로 포함되지 않는다.
    feature_cols = [c for c in test.columns if c != ID_COL]
    # 입력 변수 중 문자열 자료형인 열을 범주형 변수로 분류한다.
    categorical_cols = [
        c for c in feature_cols if pd.api.types.is_string_dtype(train[c].dtype)
    ]
    # 범주형이 아닌 입력 열을 수치형 변수로 분류한다.
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    # X는 모델 입력값, y는 모델이 맞혀야 할 정답값이다.
    x = train[feature_cols]
    y = train[TARGET].to_numpy()
    # x_test는 학습 완료 후 stress_score를 예측할 평가 데이터다.
    x_test = test[feature_cols]

    # 데이터를 무작위로 섞은 뒤 5개 fold로 나눈다.
    # 매번 4개 fold로 학습하고 남은 1개 fold로 성능을 검증한다.
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # 각 train 행이 검증 데이터였을 때의 예측값을 저장할 배열이다.
    # OOF(Out-Of-Fold) 예측으로 전체 교차검증 MAE를 계산할 수 있다.
    oof = np.zeros(len(train))
    # 5개 모델의 test 예측값을 누적해 평균 앙상블하기 위한 배열이다.
    test_predictions = np.zeros(len(test))

    # LightGBM 모델의 학습 옵션이다.
    params = {
        # MAE를 최소화하는 회귀 문제로 학습한다.
        "objective": "mae",
        # 학습 중 검증 성능도 MAE로 측정한다.
        "metric": "mae",
        # 만들 수 있는 부스팅 트리의 최대 개수다.
        "n_estimators": 3000,
        # 트리 하나가 기존 예측을 보정하는 정도다. 작을수록 천천히 학습한다.
        "learning_rate": 0.01,
        # 한 트리가 가질 수 있는 최대 잎 노드 수로 복잡도를 제한한다.
        "num_leaves": 15,
        # 개별 트리의 최대 깊이를 5로 제한해 과적합을 줄인다.
        "max_depth": 5,
        # 하나의 잎 노드에 필요한 최소 데이터 수다.
        "min_child_samples": 25,
        # 각 트리 학습에 전체 행의 80%만 무작위로 사용한다.
        "subsample": 0.8,
        # 매 트리마다 행 샘플링을 수행한다.
        "subsample_freq": 1,
        # 각 트리 학습에 전체 입력 변수의 80%만 무작위로 사용한다.
        "colsample_bytree": 0.8,
        # L1 규제로 불필요하게 복잡한 모델을 억제한다.
        "reg_alpha": 0.5,
        # L2 규제로 큰 가중치를 억제해 과적합을 줄인다.
        "reg_lambda": 0.5,
        # 모델 내부의 무작위 동작도 같은 결과가 나오도록 고정한다.
        "random_state": SEED,
        # LightGBM의 불필요한 학습 메시지를 출력하지 않는다.
        "verbose": -1,
    }

    # 5개의 학습/검증 조합을 차례대로 반복한다.
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), start=1):
        # 현재 fold의 학습 데이터로만 전처리 기준을 계산해 세 데이터에 적용한다.
        x_train, x_valid, fold_test = prepare_fold(
            x.iloc[train_idx],
            x.iloc[valid_idx],
            x_test,
            numeric_cols,
            categorical_cols,
        )
        # 입력 데이터와 같은 인덱스를 사용해 학습/검증 정답도 분리한다.
        y_train, y_valid = y[train_idx], y[valid_idx]

        # 위에서 정의한 옵션으로 새로운 LightGBM 회귀 모델을 만든다.
        model = lgb.LGBMRegressor(**params)
        # 현재 fold의 학습 데이터로 모델을 학습한다.
        model.fit(
            x_train,
            y_train,
            # 검증 데이터와 정답은 성능 관찰 및 조기 종료 판단에만 사용한다.
            eval_X=x_valid,
            eval_y=y_valid,
            eval_metric="mae",
            # LightGBM에 어떤 열이 범주형인지 알려준다.
            categorical_feature=categorical_cols,
            # 검증 MAE가 100회 연속 개선되지 않으면 학습을 일찍 끝낸다.
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        # 가장 성능이 좋았던 트리 개수로 현재 검증 fold를 예측해 제자리에 저장한다.
        oof[valid_idx] = model.predict(x_valid, num_iteration=model.best_iteration_)
        # 현재 모델의 test 예측값을 1/5씩 더해 5개 모델의 평균을 만든다.
        test_predictions += (
            model.predict(fold_test, num_iteration=model.best_iteration_) / N_FOLDS
        )
        # 현재 fold에서 실제 정답과 예측값의 MAE를 계산한다.
        fold_mae = mean_absolute_error(y_valid, oof[valid_idx])
        # fold 번호, 최적 트리 개수, MAE를 터미널에 출력한다.
        print(f"fold {fold}: best_iter={model.best_iteration_} mae={fold_mae:.5f}")

    # stress_score의 실제 범위가 0~1이므로 범위를 벗어난 예측값을 잘라낸다.
    oof = np.clip(oof, 0, 1)
    test_predictions = np.clip(test_predictions, 0, 1)
    # 모든 행의 OOF 예측을 이용해 전체 교차검증 MAE를 출력한다.
    print(f"\nOOF MAE: {mean_absolute_error(y, oof):.5f}")

    # 대회가 요구하는 ID와 열 순서를 유지하기 위해 제출 예시 파일을 읽는다.
    submission = pd.read_csv("sample_submission.csv")
    # stress_score 열을 5개 모델의 평균 test 예측값으로 교체한다.
    submission[TARGET] = test_predictions
    # 행 번호(index)는 제외하고 CSV 제출 파일로 저장한다.
    submission.to_csv(OUTPUT, index=False)
    # 저장이 완료된 파일명을 터미널에 출력한다.
    print(f"wrote {OUTPUT}")


# 이 파일을 직접 실행했을 때만 main 함수를 호출한다.
# 다른 Python 파일에서 import할 때는 자동으로 학습되지 않는다.
if __name__ == "__main__":
    main()
