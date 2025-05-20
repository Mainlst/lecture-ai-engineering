import json, os, time
import numpy as np
from sklearn.metrics import accuracy_score
import pickle, pandas as pd
from sklearn.model_selection import train_test_split
import json, os, time
import pytest

ROOT = os.path.dirname(__file__)
BASE_PATH = os.path.join(ROOT, "../models")
BASE_MET_PATH = os.path.join(BASE_PATH, "baseline_metrics.json")
BASE_MODEL_PATH = os.path.join(BASE_PATH, "baseline_model.pkl")
NEW_MODEL_PATH  = os.path.join(BASE_PATH, "titanic_model.pkl")
DATA_PATH = os.path.join(ROOT, "../data/Titanic.csv")

TOL_ACC  = -0.02  # 2% まで精度低下 OK
TOL_TIME =  0.20  # +0.20 秒まで遅延OK

def load_metrics():
    with open(BASE_MET_PATH) as f:
        return json.load(f)

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Survived", axis=1)
    y = df["Survived"].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)[1:]

def _infer(model, X):
    start = time.time();  model.predict(X);  end = time.time()
    return end - start

def test_regression():
    if not os.path.exists(BASE_MODEL_PATH):
        pytest.skip("ベースラインモデルがありません")
    base_met = load_metrics()
    _, X_test, y_test = load_data()

    # baseline
    with open(BASE_MODEL_PATH, "rb") as f:
        base_model = pickle.load(f)
    base_acc  = accuracy_score(y_test, base_model.predict(X_test))
    base_time = _infer(base_model, X_test)

    # new
    with open(NEW_MODEL_PATH, "rb") as f:
        new_model = pickle.load(f)
    new_acc  = accuracy_score(y_test, new_model.predict(X_test))
    new_time = _infer(new_model, X_test)

    # assert
    assert new_acc  >= base_acc + TOL_ACC,  \
        f"精度劣化: {base_acc:.3f}→{new_acc:.3f}"
    assert new_time <= base_time + TOL_TIME, \
        f"推論遅延: {base_time:.2f}s→{new_time:.2f}s"
