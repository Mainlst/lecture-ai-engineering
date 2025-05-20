import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
import json # JSONを扱うためにインポート
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')


# --- パス定義 ---
# このファイルの場所を基準に、リポジトリのルートや他のファイルへのパスを決定します。
# day5/演習3/tests/test_model.py から見て、
# リポジトリルートは3つ上の階層 (../../..)
# データは ../data/Titanic.csv
# モデルは ../models/titanic_model.pkl
# メトリクスファイルはリポジトリルート直下に current_metrics.json として保存する想定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # リポジトリルート
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")

# メトリクスファイルのパス (リポジトリのルート直下に保存/参照する想定)
CURRENT_METRICS_PATH = os.path.join(BASE_DIR, "current_metrics.json")
# GitHub Actionsでダウンロードされた過去のメトリクスファイルへのパス
# (ワークフローの download-artifact の path とアーティファクト名に合わせる)
PREVIOUS_METRICS_ARTIFACT_DIR = os.path.join(BASE_DIR, "previous_metrics_artifact")
PREVIOUS_METRICS_PATH = os.path.join(PREVIOUS_METRICS_ARTIFACT_DIR, "current_metrics.json") # アップロード時と同じファイル名

# --- グローバル変数 (メトリクス収集用) ---
collected_metrics = {}

@pytest.fixture(scope="session") # セッションスコープでデータを一度だけ読み込む
def raw_data():
    """テスト用データセットを読み込む (ファイルが存在しない場合はダウンロード)"""
    if not os.path.exists(DATA_PATH):
        print(f"データファイル {DATA_PATH} が見つかりません。ダウンロードを試みます...")
        from sklearn.datasets import fetch_openml
        # parser='auto' を指定して警告を抑制
        titanic_data = fetch_openml("titanic", version=1, as_frame=True, parser='auto')
        df = titanic_data.frame
        # fetch_openmlの 'survived' カラムはカテゴリ型 (object) で '0', '1' が入っていることがある
        # これを数値に変換しておく
        # df['Survived'] = pd.to_numeric(df['survived']) # 'survived' は小文字かもしれない

        # カラム名を小文字に統一（より堅牢にするため）
        df.columns = [col.lower() for col in df.columns]

        # 必要なカラムのみ選択 (小文字で)
        required_cols_raw = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
        missing_cols = [col for col in required_cols_raw if col not in df.columns]
        if missing_cols:
            raise ValueError(f"必要なカラム {missing_cols} がDataFrameに存在しません。利用可能なカラム: {df.columns.tolist()}")

        df_selected = df[required_cols_raw]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df_selected.to_csv(DATA_PATH, index=False)
        print(f"データファイルを {DATA_PATH} に保存しました。")
        return df_selected
    
    df_loaded = pd.read_csv(DATA_PATH)
    df_loaded.columns = [col.lower() for col in df_loaded.columns] # 読み込んだCSVもカラム名を小文字に
    return df_loaded


@pytest.fixture(scope="session")
def preprocessor(raw_data): # raw_dataフィクスチャに依存
    """前処理パイプラインを定義"""
    numeric_features = ["age", "pclass", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]

    # raw_data に数値特徴量とカテゴリ特徴量が存在するか確認
    for col in numeric_features + categorical_features:
        if col not in raw_data.columns:
            raise ValueError(f"前処理に必要なカラム '{col}' がデータに存在しません。利用可能なカラム: {raw_data.columns.tolist()}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor_obj = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor_obj


@pytest.fixture(scope="session") # モデル学習もセッションスコープで一度だけ
def trained_model_and_test_data(raw_data, preprocessor): # raw_dataとpreprocessorに依存
    """モデルの学習とテストデータの準備、モデルの保存"""
    if "survived" not in raw_data.columns:
        raise ValueError(f"目的変数 'survived' がデータに存在しません。利用可能なカラム: {raw_data.columns.tolist()}")

    X = raw_data.drop("survived", axis=1)
    y = raw_data["survived"].astype(int) # survivedをint型に変換
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # 層化サンプリングを追加
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')), # class_weightを追加
        ]
    )
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    # trained_model_and_test_data フィクスチャが実行されればモデルは保存されるはず
    assert os.path.exists(MODEL_PATH), f"モデルファイル {MODEL_PATH} が存在しません"


def test_model_accuracy(trained_model_and_test_data):
    """モデルの精度を検証し、メトリクスを収集"""
    model, X_test, y_test = trained_model_and_test_data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"モデルの精度: {accuracy:.4f}")
    collected_metrics["accuracy"] = accuracy
    assert accuracy >= 0.70, f"モデルの精度 ({accuracy:.4f}) が期待値 (0.70) より低いです" # 閾値を少し現実的に


def test_model_inference_time(trained_model_and_test_data):
    """モデルの推論時間を検証し、メトリクスを収集"""
    model, X_test, _ = trained_model_and_test_data # y_testは不要
    
    if X_test.empty:
        print("推論用テストデータが空のため、推論時間テストをスキップします。")
        collected_metrics["inference_time_total_seconds"] = 0
        collected_metrics["inference_time_avg_ms_per_sample"] = 0
        pytest.skip("テストデータが空です。")
        return

    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    inference_time_total = end_time - start_time
    inference_time_avg_ms = (inference_time_total / len(X_test)) * 1000 if len(X_test) > 0 else 0
    
    print(f"総推論時間: {inference_time_total:.4f}秒 ({len(X_test)}件)")
    print(f"1件あたりの平均推論時間: {inference_time_avg_ms:.4f}ミリ秒")
    collected_metrics["inference_time_total_seconds"] = inference_time_total
    collected_metrics["inference_time_avg_ms_per_sample"] = inference_time_avg_ms
    # 非常に小さなデータセットなので、推論時間はかなり短くなるはず
    assert inference_time_total < 5.0, f"総推論時間 ({inference_time_total:.4f}秒) が長すぎます"


def test_model_reproducibility(raw_data, preprocessor): # trained_model_and_test_dataは使わない
    """モデルの再現性を検証 (同じデータ・設定なら同じモデルができるか)"""
    X = raw_data.drop("survived", axis=1)
    y = raw_data["survived"].astype(int)
    X_train, X_test, y_train, _ = train_test_split( # y_testは不要
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor), # preprocessorを再利用
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ]
    )
    model1.fit(X_train, y_train)
    predictions1 = model1.predict(X_test)

    # preprocessor は fit されているので、新しいパイプラインでは再度 fit する必要がある
    # もしくは、preprocessor のインスタンスを毎回新しく作る
    # ここでは、同じ preprocessor インスタンスを使い、再度 fit されることを期待
    
    # 毎回新しい前処理パイプラインインスタンスを作成する方が安全
    numeric_features = ["age", "pclass", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]
    numeric_transformer_new = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    categorical_transformer_new = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor_new = ColumnTransformer(transformers=[("num", numeric_transformer_new, numeric_features),("cat", categorical_transformer_new, categorical_features)])

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor_new), # 新しい preprocessor インスタンス
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ]
    )
    model2.fit(X_train, y_train)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(predictions1, predictions2), "モデルの予測結果に再現性がありません"


def test_performance_regression():
    """過去のモデルと比較して性能が劣化していないか検証"""
    if not os.path.exists(CURRENT_METRICS_PATH):
        pytest.skip(f"現在のメトリクスファイル ({CURRENT_METRICS_PATH}) が見つかりません。")

    with open(CURRENT_METRICS_PATH, "r") as f:
        current_metrics = json.load(f)
    
    current_accuracy = current_metrics.get("accuracy")
    current_inference_time_avg = current_metrics.get("inference_time_avg_ms_per_sample")

    if not os.path.exists(PREVIOUS_METRICS_PATH):
        print(f"過去のメトリクスファイル ({PREVIOUS_METRICS_PATH}) が見つかりません。初回実行または比較対象なしとして扱います。")
        assert current_accuracy is not None, "現在の精度が記録されていません。"
        pytest.skip("過去のメトリクスファイルがないため、性能比較をスキップします。")
        return

    with open(PREVIOUS_METRICS_PATH, "r") as f:
        previous_metrics = json.load(f)

    previous_accuracy = previous_metrics.get("accuracy")
    previous_inference_time_avg = previous_metrics.get("inference_time_avg_ms_per_sample")

    print(f"性能比較 - 精度:      現在={current_accuracy:.4f}, 過去={previous_accuracy:.4f if previous_accuracy else 'N/A'}")
    print(f"性能比較 - 平均推論時間: 現在={current_inference_time_avg:.4f}ms, 過去={previous_inference_time_avg:.4f if previous_inference_time_avg else 'N/A'}ms")

    if previous_accuracy is not None and current_accuracy is not None:
        # 精度が著しく低下していないか (例: 過去の95%未満になったらエラー)
        assert current_accuracy >= previous_accuracy * 0.95, \
            f"精度が許容範囲を超えて低下しました。現在: {current_accuracy:.4f}, 過去: {previous_accuracy:.4f}"

    if previous_inference_time_avg is not None and current_inference_time_avg is not None and previous_inference_time_avg > 0:
        # 推論時間が著しく増加していないか (例: 過去の1.5倍を超えたらエラー)
        assert current_inference_time_avg <= previous_inference_time_avg * 1.5, \
            f"平均推論時間が許容範囲を超えて増加しました。現在: {current_inference_time_avg:.4f}ms, 過去: {previous_inference_time_avg:.4f}ms"

# --- pytest フック (テストセッション終了時に実行) ---
def pytest_sessionfinish(session, exitstatus):
    """テストセッションの最後に収集したメトリクスをファイルに書き出す"""
    if exitstatus == 0 and collected_metrics: # テストが成功し、かつメトリクスが収集されている場合
        # (注意: pytest_sessionfinish は全てのテストが終わった後に呼ばれるので、
        #  ここに来るまでに collected_metrics に値が入っている必要があります)
        print(f"\nテストセッション終了。収集されたメトリクス: {collected_metrics}")
        try:
            with open(CURRENT_METRICS_PATH, "w") as f:
                json.dump(collected_metrics, f, indent=4)
            print(f"メトリクスを {CURRENT_METRICS_PATH} に保存しました！🎉")
        except Exception as e:
            print(f"メトリクスファイルの保存中にエラーが発生しました: {e}")
    elif not collected_metrics:
        print("\nメトリクスは収集されませんでした。")
    else:
        print(f"\nテストが失敗したため ({exitstatus})、メトリクスは保存されませんでした。")