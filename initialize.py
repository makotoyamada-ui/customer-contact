"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""
# --- SQLite を pysqlite3 に差し替え（Chromadb が sqlite3 を使う前に実行）---
try:
    import sys
    import pysqlite3 as sqlite3  # wheels 同梱の新しい SQLite
    sys.modules["sqlite3"] = sqlite3
except Exception:
    # 失敗してもアプリを止めない（packages.txt で OS 側を上げる構成に切替可能）
    pass
# --------------------------------------------------------------------------

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
# 未使用なら削ってOK： from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import SerpAPIWrapper  # ← y なし・残す（使ってます）
from langchain_core.tools import Tool
from langchain.agents import AgentType, initialize_agent

import constants as ct


############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # Agent Executorを作成
    initialize_agent_executor()
    # 念のためトークナイザを最終保証
    _ensure_enc()


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        # 会話履歴の合計トークン数を加算する用の変数
        st.session_state.total_tokens = 0

        # フィードバック関連のフラグ群
        st.session_state.feedback_yes_flg = False
        st.session_state.feedback_no_flg = False
        st.session_state.answer_flg = False
        st.session_state.dissatisfied_reason = ""
        st.session_state.feedback_no_reason_send_flg = False

        # 追加：トークナイザの初期値（None で明示）
        st.session_state.enc = None


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

    # 1) /mount/data が書き込めるか先にチェック（親を作ろうとしない）
    base_dir_candidates = []
    base_dir = "/mount/data"
    try:
        if os.path.exists(base_dir) and os.access(base_dir, os.W_OK | os.X_OK):
            base_dir_candidates.append(base_dir)
    except Exception:
        pass  # 判定に失敗したら候補に入れない

    # 2) フォールバック先
    base_dir_candidates.append("/tmp")

    # 3) 使えるベースを選び、logs サブディレクトリを作成
    for base in base_dir_candidates:
        try:
            log_dir = os.path.join(base, "logs")
            os.makedirs(log_dir, exist_ok=True)
            break
        except PermissionError:
            continue
    else:
        # どこも作れない場合は最後の手段として現在ディレクトリ配下に出す
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)

    # 4) ハンドラ設定
    log_path = os.path.join(log_dir, ct.LOG_FILE)
    log_handler = TimedRotatingFileHandler(
        log_path,
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, "
        f"session_id={st.session_state.get('session_id', 'N/A')}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)

    # 5) どこに出しているかを最初に記録
    logger.info(f"Logging to: {log_path}")


def _ensure_enc():
    """
    st.session_state.enc を必ず用意するユーティリティ
    """
    if st.session_state.get("enc") is None:
        try:
            # 推奨：ENCODING_KIND（例: "cl100k_base"）を優先
            st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
        except Exception:
            # フォールバック：モデル名から推定
            model_name = getattr(ct, "MODEL", "gpt-4o-mini")
            try:
                st.session_state.enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                # 最悪の保険：None（呼び出し側でガード）
                st.session_state.enc = None


def _import_utils():
    """
    app_utils2 → app_utils → （最後の保険として）app_utils.py をパス指定で読み込み
    """
    try:
        import app_utils2 as utils
        return utils
    except Exception:
        pass

    try:
        import app_utils2 as utils
        return utils
    except Exception:
        pass

    # 最後の保険：app_utils.py をファイルパスから読み込む
    mod_path = pathlib.Path(__file__).with_name("app_utils.py")
    spec = importlib.util.spec_from_file_location("app_utils", mod_path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


def initialize_agent_executor():
    """
    画面読み込み時にAgent Executor（AIエージェント機能の実行を担当するオブジェクト）を作成
    """
    utils = _import_utils()
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにAgent Executorが作成済みの場合でも enc が無ければ保証してから return
    if "agent_executor" in st.session_state:
        _ensure_enc()
        return

    # まずトークナイザを保証
    _ensure_enc()

    # LLM 構築
    st.session_state.llm = ChatOpenAI(
        model=ct.MODEL,
        temperature=ct.TEMPERATURE,
        streaming=True
    )

    # 各Tool用のChainを作成（create_rag_chain 内で st.session_state.llm を参照）
    st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
    st.session_state.service_doc_chain  = utils.create_rag_chain(ct.DB_SERVICE_PATH)
    st.session_state.company_doc_chain  = utils.create_rag_chain(ct.DB_COMPANY_PATH)
    st.session_state.rag_chain          = utils.create_rag_chain(ct.DB_ALL_PATH)

    # Web検索用のTool
    search = SerpAPIWrapper()

    # Agent Executorに渡すTool一覧
    tools = [
        Tool(
            name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,
            func=utils.run_company_doc_chain,
            description=ct.SEARCH_COMPANY_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,
            func=utils.run_service_doc_chain,
            description=ct.SEARCH_SERVICE_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME,
            func=utils.run_customer_doc_chain,
            description=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_WEB_INFO_TOOL_NAME,
            func=search.run,
            description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION
        ),
    ]

    # Agent Executorの作成
    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )
