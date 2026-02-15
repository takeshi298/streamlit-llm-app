
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1) .env を読み込み（OPENAI_API_KEY を環境変数として設定）
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)


def ask_llm(user_text: str, expert_type: str) -> str:
    """
    入力テキスト と ラジオボタン選択値(expert_type) を引数に取り、
    LLMの回答（文字列）を返す関数
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY が見つかりません。.env に OPENAI_API_KEY=sk-... を設定してください。"
        )

    # 2) 選択値に応じてシステムメッセージを切り替える
    #    ※専門家は自由に設計(A=事業開発、B=Python家庭教師）
    if expert_type == "事業開発コンサル（BizDev）":
        system_message = (
            "あなたは優秀な事業開発コンサルタントです。"
            "ユーザーの状況を前提に、実行可能な施策を箇条書きで提案し、"
            "最後に最初の一手（Next Action）を1つ提示してください。"
        )
    else:  # "Python家庭教師"
        system_message = (
            "あなたは優秀なPython家庭教師です。"
            "初心者にも分かるように、結論→理由→具体例の順で説明し、"
            "必要なら短いコード例も提示してください。"
        )

    # 3) LLM（LangChain）を準備
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.3,
        api_key=api_key,
    )

    # 4) プロンプト（System + Human）を構成
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{user_text}"),
        ]
    )

    chain = prompt | llm

    # 5) 実行 → LangChainの戻りはAIMessageなのでcontentを返す
    result = chain.invoke({"user_text": user_text})
    return result.content


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Streamlit × LangChain LLM App", page_icon="🤖", layout="centered")

st.title("🤖 Streamlit × LangChain LLMアプリ")
st.write(
    """
このWebアプリは、入力フォームにテキストを送信すると、LangChain経由でLLMに質問し、
回答を画面に表示します。  
左の選択（ラジオボタン）で、LLMの振る舞い（専門家タイプ）を切り替えられます。

**使い方**
1. 下のラジオボタンで専門家タイプを選ぶ  
2. 入力フォームに質問/相談を入力  
3. 「送信」を押すと回答が表示されます
"""
)

expert_type = st.radio(
    "専門家タイプを選択してください",
    options=["事業開発コンサル（BizDev）", "Python家庭教師"],
    horizontal=True,
)

user_text = st.text_area(
    "入力フォーム（ここに質問を入力）",
    placeholder="例）水素関連の新規事業の最初の顧客開拓の進め方は？ / Pythonの辞書の使い方を教えて など",
    height=140,
)

col1, col2 = st.columns([1, 2])
with col1:
    submit = st.button("送信", type="primary")
with col2:
    st.caption("※ `.env` に OPENAI_API_KEY を設定してから実行してください。")

if submit:
    if not user_text.strip():
        st.warning("入力テキストを記入してください。")
    else:
        try:
            with st.spinner("LLMに問い合わせ中..."):
                answer = ask_llm(user_text=user_text, expert_type=expert_type)
            st.subheader("回答")
            st.write(answer)
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

