"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import tiktoken
import importlib.util
import pathlib

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import AgentType, initialize_agent

import constants as ct

############################################################
# 設定関連
############################################################
load_dotenv()  # 互換のため残す（ただし Secrets を優先）

def _get_secret(name: str) -> str | None:
    """Secrets > 環境変数 の優先で値を取得"""
    try:
        v = st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        v = None
    return v or os.getenv(name)

# --- OpenAI のキーは必ず明示して使う ---
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not設定されていません。Streamlit の Secrets か環境変数に設定してください。"
    )

# org / project の強制上書きは事故の元なので消しておく（任意）
for var in ("OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT", "OPENAI_PROJECT_ID"):
    if os.getenv(var):
        del os.environ[var]

############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    initialize_session_state()
    initialize_session_id()
    initialize_logger()
    initialize_agent_executor()


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        # 会話履歴の合計トークン数を加算する用の変数
        st.session_state.total_tokens = 0

        # フィードバック関連
        st.session_state.feedback_yes_flg = False
        st.session_state.feedback_no_flg = False
        st.session_state.answer_flg = False
        st.session_state.dissatisfied_reason = ""
        st.session_state.feedback_no_reason_send_flg = False


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_logger():
    """
    ログ出力の設定（/mount/data が書けない環境でも落ちないようにフォールバック）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    if logger.hasHandlers():
        return

    base_dir_candidates = []
    base_dir = "/mount/data"
    try:
        if os.path.exists(base_dir) and os.access(base_dir, os.W_OK | os.X_OK):
            base_dir_candidates.append(base_dir)
    except Exception:
        pass
    base_dir_candidates.append("/tmp")

    for base in base_dir_candidates:
        try:
            log_dir = os.path.join(base, "logs")
            os.makedirs(log_dir, exist_ok=True)
            break
        except PermissionError:
            continue
    else:
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, ct.LOG_FILE)
    log_handler = TimedRotatingFileHandler(log_path, when="D", encoding="utf8")
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, "
        f"session_id={st.session_state.get('session_id', 'N/A')}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    logger.info(f"Logging to: {log_path}")


def initialize_agent_executor():
    """
    画面読み込み時にAgent Executor（AIエージェント機能の実行を担当）を作成
    """
    # ---- app_utils2 を遅延＆フォールバックで import ----
    try:
        import app_utils2 as utils  # もし存在すればこちらを使う
    except Exception:
        mod_path = pathlib.Path(__file__).with_name("app_utils.py")
        spec = importlib.util.spec_from_file_location("app_utils", mod_path)
        utils = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(utils)  # type: ignore

    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでに作成済みならスキップ
    if "agent_executor" in st.session_state:
        return

    # tiktoken の初期化（保険つき）
    try:
        st.session_state.enc = tiktoken.encoding_for_model(getattr(ct, "MODEL", "gpt-4o-mini"))
    except Exception:
        st.session_state.enc = tiktoken.get_encoding(getattr(ct, "ENCODING_KIND", "cl100k_base"))

    # --- Chat LLM を Secrets のキーで明示作成 ---
    # Streaming は別スレッドでのトークン受信により Streamlit のコンテキスト外コールバックを誘発するため無効化
    st.session_state.llm = ChatOpenAI(
        model=ct.MODEL,
        temperature=ct.TEMPERATURE,
        streaming=False,
        api_key=OPENAI_API_KEY,   # ★ 明示
        timeout=60,
        max_retries=0,            # 無駄な再試行で費用が跳ねないように
    )
    logger.info(f"Using OpenAI key ****{OPENAI_API_KEY[-4:]} for Chat & Embeddings")

    # 各Tool用のChainを作成
    st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
    st.session_state.service_doc_chain  = utils.create_rag_chain(ct.DB_SERVICE_PATH)
    st.session_state.company_doc_chain  = utils.create_rag_chain(ct.DB_COMPANY_PATH)
    st.session_state.rag_chain          = utils.create_rag_chain(ct.DB_ALL_PATH)

    # Web検索用のTool
    search = SerpAPIWrapper()

    tools = [
        Tool(name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,  func=utils.run_company_doc_chain,  description=ct.SEARCH_COMPANY_INFO_TOOL_DESCRIPTION),
        Tool(name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,  func=utils.run_service_doc_chain,   description=ct.SEARCH_SERVICE_INFO_TOOL_DESCRIPTION),
        Tool(name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME, func=utils.run_customer_doc_chain, description=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION),
        Tool(name=ct.SEARCH_WEB_INFO_TOOL_NAME,      func=search.run,                    description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION),
    ]

    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )
