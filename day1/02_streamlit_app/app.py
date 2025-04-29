# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---

# --- ① 追加：履歴を初期化 -----------------------------
if "messages" not in st.session_state:
    # role: "user" | "assistant"
    st.session_state.messages = []
    
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None
pipe = llm.load_model()

# --- Streamlit アプリケーション ---
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)


# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        # ② 既存のチャット履歴を表示
        for chat in st.session_state.messages:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

        # ③ ユーザー入力欄（Enter で送信）
        if prompt := st.chat_input("メッセージを入力してくださいにゃ"):
            # ------------- 送信処理 -------------
            # ユーザー発話を履歴に追加
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            # ④ LLM へ送るプロンプトを構築
            context = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in st.session_state.messages[-6:]  # 直近6往復だけ渡す例
            )
            with st.chat_message("assistant"):
                with st.spinner("思案中です…"):
                    try:
                        resp = pipe(context, max_new_tokens=512)[0]["generated_text"]
                        # Gemma 生成文の後ろに context まで含まれる場合があるので後処理
                        answer = resp[len(context):].strip()
                    except Exception as e:
                        answer = f"エラーが発生したにゃ: {e}"

                    # 画面に表示
                    st.markdown(answer)
                    # 履歴に保存
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")