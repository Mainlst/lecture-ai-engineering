# app.py (fixed)
"""
Gemma Chatbot – Streamlit
* 連続対話
* 評価ページ
* 永続設定 (JSON)
* 画像アイコン保存
* ユーザー名 / LLM 名をメッセージ内に表示 (chat_message に name 引数は無いので Markdown ラベルを挿入)
"""

import json
import os
import shutil
import tempfile
from datetime import datetime

import streamlit as st
import torch
from transformers import pipeline

import ui
import llm
import database
import metrics
import data
from config import MODEL_NAME

# ------------------ 永続設定 ------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_JSON = os.path.join(ROOT_DIR, "user_settings.json")
ICON_DIR = os.path.join(ROOT_DIR, "icons")
os.makedirs(ICON_DIR, exist_ok=True)

DEFAULT_SETTINGS = {
    "assistant_name": "Luna-chan",
    "assistant_persona": "あなたは優しいネコ耳AIアシスタントです。愛を込めてユーザーにご奉仕してください。",
    "assistant_icon_emoji": "😺",
    "assistant_icon_path": None,
    "user_name": "You",
}


def load_settings() -> dict:
    if os.path.exists(SETTINGS_JSON):
        try:
            with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULT_SETTINGS, **data}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(d: dict):
    with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


# ------------------ Streamlit basic ------------------
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# ------------------ Session init ------------------
settings = load_settings()
for k, v in settings.items():
    st.session_state.setdefault(k, v)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("page", "チャット")

# ------------------ external init ------------------
metrics.initialize_nltk()
database.init_db()
data.ensure_initial_data()

# ------------------ LLM ------------------
@st.cache_resource
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.info(f"Using device: {device}")
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        return pipe
    except Exception as e:
        st.sidebar.error(f"モデル読込失敗: {e}")
        return None

pipe = load_model()

# ------------------ Helpers ------------------

def avatar():
    return st.session_state["assistant_icon_path"] or st.session_state["assistant_icon_emoji"]

# ------------------ Sidebar ------------------
PAGES = ["チャット", "履歴閲覧", "サンプルデータ管理", "評価", "設定"]

page_select = st.sidebar.radio("ページ選択", PAGES, index=PAGES.index(st.session_state.page))
st.session_state.page = page_select

st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")

# ------------------ Chat Page ------------------
if st.session_state.page == "チャット":
    st.title("🤖 Gemma 2 Chatbot with Feedback")
    st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
    st.markdown("---")

    if pipe is None:
        st.error("チャット機能を利用できません。モデルを読み込めませんでした。")
    else:
        # 履歴表示
        for m in st.session_state.messages:
            role = m["role"]
            label = st.session_state["user_name"] if role == "user" else st.session_state["assistant_name"]
            av = None if role == "user" else avatar()
            with st.chat_message(role, avatar=av):
                st.markdown(f"**{label}**\n\n{m['content']}")

        # 入力
        if prompt := st.chat_input("メッセージを入力してください"):
            # ユーザー発話
            with st.chat_message("user"):
                st.markdown(f"**{st.session_state['user_name']}**\n\n{prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})

            # コンテキスト
            context = st.session_state["assistant_persona"] + "\n" + "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-6:]
            )

            with st.chat_message("assistant", avatar=avatar()):
                with st.spinner("お返事を考えています…"):
                    try:
                        resp = pipe(context, max_new_tokens=512)[0]["generated_text"]
                        ans = resp[len(context):].strip()
                    except Exception as e:
                        ans = f"エラーが発生した: {e}"
                    st.markdown(f"**{st.session_state['assistant_name']}**\n\n{ans}")
                    st.session_state.messages.append({"role": "assistant", "content": ans})

        if st.button("📝 チャットを終了して評価へ"):
            st.session_state.page = "評価"
            st.rerun()

# ------------------ Other Pages ------------------
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()

elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

elif st.session_state.page == "評価":
    st.header("🔍 チャット評価")
    for m in st.session_state.messages:
        label = st.session_state["user_name"] if m["role"] == "user" else st.session_state["assistant_name"]
        st.markdown(f"**{label}:** {m['content']}")
    st.markdown("---")
    rating = st.slider("このチャットの満足度 (1=😢〜5=😍)", 1, 5, 3)
    comment = st.text_area("コメント", placeholder="自由にご記入ください")
    if st.button("送信！"):
        try:
            database.save_evaluation(rating, comment, st.session_state.messages)
        except AttributeError:
            pass
        st.success("ご協力ありがとうございます♪")

elif st.session_state.page == "設定":
    st.header("🎨 パーソナライズ設定")
    st.write("変更後『💾 保存』を押すと次回起動時も反映されます。")

    st.session_state.user_name = st.text_input("あなたの名前", st.session_state.user_name)
    st.session_state.assistant_name = st.text_input("LLM の名前", st.session_state.assistant_name)
    st.session_state.assistant_icon_emoji = st.text_input("アシスタントアイコン (絵文字)", st.session_state.assistant_icon_emoji)

    uploaded = st.file_uploader("画像アイコンをアップロード (png/jpg)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        ext = os.path.splitext(uploaded.name)[1]
        dst = os.path.join(ICON_DIR, f"icon_{ts}{ext}")
        with open(dst, "wb") as f:
            shutil.copyfileobj(uploaded, f)
        st.session_state.assistant_icon_path = dst
        st.image(dst, width=64)
    elif st.session_state.assistant_icon_path:
        st.image(st.session_state.assistant_icon_path, width=64)

    st.session_state.assistant_persona = st.text_area(
        "キャラクター性格 (システムプロンプト)", st.session_state.assistant_persona, height=120
    )

    if st.button("💾 保存"):
        save_settings({
            "assistant_name": st.session_state.assistant_name,
            "assistant_persona": st.session_state.assistant_persona,
            "assistant_icon_emoji": st.session_state.assistant_icon_emoji,
            "assistant_icon_path": st.session_state.assistant_icon_path,
            "user_name": st.session_state.user_name,
        })
        st.success("設定を保存しました！")
