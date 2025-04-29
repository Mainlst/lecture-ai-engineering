# app.py
import os
import tempfile
import streamlit as st
import ui                   # UIモジュール（履歴表示など）
import llm                  # LLMモジュール（ラッパ）
import database             # データベースモジュール（任意）
import metrics              # 評価指標モジュール（NLTK 初期化など）
import data                 # データモジュール（サンプル投入）
import torch
from transformers import pipeline
from config import MODEL_NAME

# --------------------------------------------------
# 基本設定
# --------------------------------------------------
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --------------------------------------------------
# セッションステート初期化
# --------------------------------------------------

st.session_state.setdefault("messages", [])            # チャット履歴
st.session_state.setdefault("assistant_name", "Luna‑chan")
st.session_state.setdefault("assistant_persona", "あなたは優しいAIアシスタントです。愛を込めてユーザーに給仕してください。")
# アイコンは ① emoji ② 画像パス のいずれかを保存
st.session_state.setdefault("assistant_icon_emoji", "😺")
st.session_state.setdefault("assistant_icon_path", None)

# --------------------------------------------------
# 1. ライブラリの初期化など
# --------------------------------------------------
metrics.initialize_nltk()

database.init_db()

data.ensure_initial_data()

# --------------------------------------------------
# 2. LLM のロード（キャッシュ）
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Gemma / そのほか HuggingFace LLM をロード"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.info(f"Using device: {device}")
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        return pipe
    except Exception as e:
        st.sidebar.error(f"モデルの読み込みに失敗しました: {e}")
        return None

pipe = load_model()

# --------------------------------------------------
# 3. サイドバー（ページナビゲーション）
# --------------------------------------------------

PAGES = ["チャット", "履歴閲覧", "サンプルデータ管理", "評価", "設定"]

st.session_state.setdefault("page", "チャット")

page = st.sidebar.radio(
    "ページ選択",
    PAGES,
    index=PAGES.index(st.session_state.page),
    key="page_selector",
)
st.session_state.page = page  # ラジオボタンの選択結果を保存

st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")

# --------------------------------------------------
# 4. ページごとの UI
# --------------------------------------------------

# ---- ヘルパー：現在のアシスタントアイコンを返す ----

def current_avatar():
    if st.session_state.assistant_icon_path:
        return st.session_state.assistant_icon_path  # 画像ファイルパス
    return st.session_state.assistant_icon_emoji

# ---- チャットページ --------------------------------------------------

if st.session_state.page == "チャット":
    st.title("🤖 Gemma 2 Chatbot with Feedback")
    st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
    st.markdown("---")

    if pipe is None:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
    else:
        # 履歴の描画
        for chat in st.session_state.messages:
            avatar = None
            if chat["role"] == "assistant":
                avatar = current_avatar()
            with st.chat_message(chat["role"], avatar=avatar):
                st.markdown(chat["content"])

        # 入力ボックス
        if prompt := st.chat_input("メッセージを入力してくださいにゃ"):
            # ユーザー発話を即時表示
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # コンテキスト生成（直近 6 往復 + 性格プロンプト）
            context = (
                st.session_state.assistant_persona
                + "\n"
                + "\n".join(
                    f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-6:]
                )
            )

            # LLM 呼び出し
            with st.chat_message("assistant", avatar=current_avatar()):
                with st.spinner("お返事を考えています…"):
                    try:
                        resp = pipe(context, max_new_tokens=512)[0]["generated_text"]
                        answer = resp[len(context) :].strip()
                    except Exception as e:
                        answer = f"エラーが発生したにゃ: {e}"
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        # 終了ボタン
        if st.button("📝 チャットを終了して評価へ"):
            st.session_state.page = "評価"
            st.experimental_rerun()

# ---- 履歴閲覧 --------------------------------------------------------

elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()

# ---- サンプルデータ管理 ---------------------------------------------

elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# ---- 評価ページ ------------------------------------------------------

elif st.session_state.page == "評価":
    st.header("🔍 チャット評価")

    for chat in st.session_state.messages:
        role_label = "👤User" if chat["role"] == "user" else f"{current_avatar()} {st.session_state.assistant_name}"
        st.markdown(f"**{role_label}:** {chat['content']}")

    st.markdown("---")
    rating = st.slider("このチャットの満足度 (1=😢〜5=😍)", 1, 5, 3)
    comment = st.text_area("コメント", placeholder="自由にご記入ください")
    if st.button("送信！"):
        # 任意：DB に保存（実装していない場合はパス）
        try:
            database.save_evaluation(rating, comment, st.session_state.messages)
        except AttributeError:
            pass
        st.success("ご協力ありがとうございますにゃん♪")

# ---- 設定ページ ------------------------------------------------------

elif st.session_state.page == "設定":
    st.header("🎨 パーソナライズ設定")

    # 名前
    st.session_state.assistant_name = st.text_input(
        "LLMの名前", st.session_state.assistant_name
    )

    # アイコン（emoji）
    st.session_state.assistant_icon_emoji = st.text_input(
        "アイコン (絵文字)", st.session_state.assistant_icon_emoji
    )

    # アイコン（画像アップロード）
    uploaded_file = st.file_uploader("画像アイコンをアップロード (png/jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # 一時フォルダに保存してパスを保持
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.assistant_icon_path = temp_path
        st.image(temp_path, width=64)
    elif st.session_state.assistant_icon_path:
        st.image(st.session_state.assistant_icon_path, width=64)

    # キャラクター性格
    st.session_state.assistant_persona = st.text_area(
        "キャラクターの性格・システムプロンプト",
        st.session_state.assistant_persona,
        height=120,
    )

    st.success("設定を保存しました (変更は即時反映されます)")
