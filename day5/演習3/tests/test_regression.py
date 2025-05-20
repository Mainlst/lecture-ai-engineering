import json
import os
import time

import pandas as pd
import pickle
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ファイルパス定義
ROOT = os.path.dirname(__file__)
BASE_PATH = os.path.join(ROOT, "../models")
BASE_MET_PATH = os.path.join(BASE_PATH, "baseline_metrics.json")
BASE_MODEL_PATH = os.path.join(BASE_PATH, "baseline_model.pkl")
NEW_MODEL_PATH = os.path.join(BASE_PATH, "titanic_model.pkl")
DATA_PATH = os.path.join(ROOT, "../data/Titanic.csv")


# 許容範囲設定
TOL_ACC = -0.02  # 精度は2%まで低下許容
TOL_TIME = 0.20  # 推論時間は+0.20秒まで許容



def load_metrics():
    """ベースライン指標を読み込み"""
    with open(BASE_MET_PATH) as f:
        return json.load(f)



def load_data():
    """テストデータを読み込み、分割して返す"""
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Survived", axis=1)
    y = df["Survived"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, y_test



def infer_time(model, X):
    """モデルの推論時間を計測"""
    start = time.time()
    model.predict(X)
    end = time.time()
    return end - start



def test_regression():
    """新旧モデルの精度・推論時間を比較し、退化を検知"""
    # ベースラインがない場合はスキップ
    if not os.path.exists(BASE_MODEL_PATH):
        pytest.skip("ベースラインモデルがありません")

    # データとベースライン指標を取得
    base_metrics = load_metrics()
    X_test, y_test = load_data()

    # ベースラインモデル評価
    with open(BASE_MODEL_PATH, "rb") as f:
        base_model = pickle.load(f)
    base_acc = accuracy_score(y_test, base_model.predict(X_test))
    base_time = infer_time(base_model, X_test)

    # 新モデル評価
    with open(NEW_MODEL_PATH, "rb") as f:
        new_model = pickle.load(f)
    new_acc = accuracy_score(y_test, new_model.predict(X_test))
    new_time = infer_time(new_model, X_test)

    # 退化判定（精度・時間）
    assert new_acc >= base_acc + TOL_ACC, (
        f"精度劣化: {base_acc:.3f}→{new_acc:.3f}"
    )
    assert new_time <= base_time + TOL_TIME, (
        f"推論遅延: {base_time:.2f}s→{new_time:.2f}s"
    )
