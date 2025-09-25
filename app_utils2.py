"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

# =========================
# 安全な constants ローダ
# =========================
import importlib, importlib.util, pathlib

def _load_constants():
    """constants を安全に読み込む（通常 import → 失敗時はファイル直指定でロード）」
    """
    try:
        import constants as ct
        return ct
    except Exception:
        mod_path = pathlib.Path(__file__).with_name("constants.py")
        spec = importlib.util.spec_from_file_location("constants", mod_path)
        ct = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ct)
        return ct

print("DEBUG: enter app_utils")  # 起動トレース

# =========================
# ライブラリの読み込み
# =========================
import os
import tempfile
import uuid
from dotenv import load_dotenv
import streamlit as st
import logging
import sys
import unicodedata
from typing import List
import datetime

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.agent_toolkits import SlackToolkit
from langchain.agents import AgentType, initialize_agent
from sudachipy import tokenizer, dictionary
from docx import Document

# =========================
# 設定関連
# =========================
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

# =========================
# 関数定義
# =========================
def build_error_message(message: str) -> str:
    """エラーメッセージと管理者問い合わせテンプレートの連結"""
    ct = _load_constants()
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def create_rag_chain(db_name: str):
    """
    引数として渡されたDB内を参照するRAGのChainを作成
    Args:
        db_name: RAG化対象のデータを格納するデータベース名
    """
    ct = _load_constants()
    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = []
    # AIエージェント機能を使わない場合（= 全フォルダ集約）
    if db_name == ct.DB_ALL_PATH:
        if not os.path.isdir(ct.RAG_TOP_FOLDER_PATH):
            logger.warning(f"RAG_TOP_FOLDER_PATH not found: {ct.RAG_TOP_FOLDER_PATH}")
        else:
            for folder_name in os.listdir(ct.RAG_TOP_FOLDER_PATH):
                if folder_name.startswith("."):
                    continue
                add_docs(os.path.join(ct.RAG_TOP_FOLDER_PATH, folder_name), docs_all)
    else:
        # 個別DB指定
        folder_path = ct.DB_NAMES.get(db_name)
        if folder_path:
            add_docs(folder_path, docs_all)
        else:
            logger.warning(f"Unknown DB name: {db_name}")

    # Windows互換のための文字列調整
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in list(doc.metadata.keys()):
            doc.metadata[key] = adjust_string(doc.metadata[key])

    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    # --- Embeddings は Secrets のキーで明示作成 ---
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,  # ★ 明示
    )

    # 既存DB読み込み or 新規作成（永続ディレクトリ: 環境変数 > /tmp に退避）
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db")
    can_persist = False
    try:
        os.makedirs(persist_dir, exist_ok=True)
        can_persist = os.access(persist_dir, os.W_OK)
    except Exception:
        can_persist = False

    try:
        if can_persist and os.path.isdir(persist_dir) and os.listdir(persist_dir):
            db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        elif can_persist:
            db = Chroma.from_documents(
                splitted_docs,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
        else:
            # 永続化不可な環境（例: 権限なし）の場合はメモリのみで生成
            db = Chroma.from_documents(splitted_docs, embedding=embeddings)
    except Exception as e:
        # Chroma v0.5 系のテナント初期化失敗などに備え、クリーンな一時ディレクトリで再試行
        fallback_dir = os.path.join(tempfile.gettempdir(), f"chroma_db_{uuid.uuid4().hex}")
        try:
            os.makedirs(fallback_dir, exist_ok=True)
            db = Chroma.from_documents(
                splitted_docs,
                embedding=embeddings,
                persist_directory=fallback_dir,
            )
        except Exception:
            # それでも失敗する場合はメモリのみ
            db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # retriever は必ず作る
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    # プロンプト
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_INQUIRY),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(
        st.session_state.llm, question_answer_prompt
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def add_docs(folder_path: str, docs_all: list):
    """フォルダ内のファイル一覧を取得して読み込む"""
    ct = _load_constants()

    if not os.path.isdir(folder_path):
        return

    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[1]
        if ext in ct.SUPPORTED_EXTENSIONS:
            loader = ct.SUPPORTED_EXTENSIONS[ext](os.path.join(folder_path, file))
            docs_all.extend(loader.load())


def run_company_doc_chain(param: str) -> str:
    """会社に関するデータ参照に特化したTool用関数"""
    ai_msg = st.session_state.company_doc_chain.invoke(
        {"input": param, "chat_history": st.session_state.chat_history}
    )
    st.session_state.chat_history.extend(
        [HumanMessage(content=param), AIMessage(content=ai_msg["answer"])]
    )
    return ai_msg["answer"]


def run_service_doc_chain(param: str) -> str:
    """サービスに関するデータ参照に特化したTool用関数"""
    ai_msg = st.session_state.service_doc_chain.invoke(
        {"input": param, "chat_history": st.session_state.chat_history}
    )
    st.session_state.chat_history.extend(
        [HumanMessage(content=param), AIMessage(content=ai_msg["answer"])]
    )
    return ai_msg["answer"]


def run_customer_doc_chain(param: str) -> str:
    """顧客とのやり取りに関するデータ参照に特化したTool用関数"""
    ai_msg = st.session_state.customer_doc_chain.invoke(
        {"input": param, "chat_history": st.session_state.chat_history}
    )
    st.session_state.chat_history.extend(
        [HumanMessage(content=param), AIMessage(content=ai_msg["answer"])]
    )
    return ai_msg["answer"]


def delete_old_conversation_log(result: str) -> None:
    """古い会話履歴の削除（トークン上限管理）"""
    ct = _load_constants()
    if "enc" not in st.session_state:
        return

    response_tokens = len(st.session_state.enc.encode(result))
    st.session_state.total_tokens += response_tokens

    while (
        st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS
        and len(st.session_state.chat_history) > 1
    ):
        removed_message = st.session_state.chat_history.pop(1)
        removed_tokens = len(st.session_state.enc.encode(removed_message.content))
        st.session_state.total_tokens -= removed_tokens


def execute_agent_or_chain(chat_message: str) -> str:
    """AIエージェント or RAG Chain の実行"""
    ct = _load_constants()
    logger = logging.getLogger(ct.LOGGER_NAME)

    try:
        if st.session_state.agent_mode == ct.AI_AGENT_MODE_ON:
            st_callback = StreamlitCallbackHandler(st.container())
            result = st.session_state.agent_executor.invoke(
                {"input": chat_message}, {"callbacks": [st_callback]}
            )
            response = result["output"]
        else:
            result = st.session_state.rag_chain.invoke(
                {"input": chat_message, "chat_history": st.session_state.chat_history}
            )
            st.session_state.chat_history.extend(
                [HumanMessage(content=chat_message), AIMessage(content=result["answer"])]
            )
            response = result["answer"]

        if response != ct.NO_DOC_MATCH_MESSAGE:
            st.session_state.answer_flg = True
        return response

    except Exception as e:
        err_text = str(e)
        logger.error({"error": err_text})
        rate_or_quota = ("insufficient_quota" in err_text) or ("RateLimitError" in err_text) or ("429" in err_text)
        if rate_or_quota:
            friendly = (
                "OpenAI APIの利用上限に達している可能性があります。時間をおいて再実行するか、管理者に請求・上限の設定を確認してください。"
            )
            return build_error_message(friendly)
        # 上記以外のエラーは一般エラーとして返却
        return build_error_message("ユーザー入力に対しての処理に失敗しました。 このエラーが繰り返し発生する場合は、管理者にお問い合わせください。")


def notice_slack(chat_message: str) -> str:
    """問い合わせ内容のSlackへの通知"""
    ct = _load_constants()

    toolkit = SlackToolkit()
    tools = toolkit.get_tools()
    agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )

    # 従業員情報 / 問い合わせ履歴の読み込み
    loader = CSVLoader(ct.EMPLOYEE_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs = loader.load()
    loader = CSVLoader(ct.INQUIRY_HISTORY_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs_history = loader.load()

    # 文字列調整
    for doc in docs:
        doc.page_content = adjust_string(doc.page_content)
        for key in list(doc.metadata.keys()):
            doc.metadata[key] = adjust_string(doc.metadata[key])
    for doc in docs_history:
        doc.page_content = adjust_string(doc.page_content)
        for key in list(doc.metadata.keys()):
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # データ整形
    docs_all = adjust_reference_data(docs, docs_history)
    docs_all_page_contents = [doc.page_content for doc in docs_all]

    # Retriever 構築（Ensemble: BM25 + embeddings）
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,   # ★ 明示
    )
    slack_persist_dir = os.getenv("CHROMA_PERSIST_DIR_SLACK", "/tmp/chroma_db_slack")
    slack_can_persist = False
    try:
        os.makedirs(slack_persist_dir, exist_ok=True)
        slack_can_persist = os.access(slack_persist_dir, os.W_OK)
    except Exception:
        slack_can_persist = False

    try:
        if slack_can_persist:
            db = Chroma.from_documents(
                docs_all,
                embedding=embeddings,
                persist_directory=slack_persist_dir,
            )
        else:
            db = Chroma.from_documents(docs_all, embedding=embeddings)
    except Exception:
        # Slack 側も同様にクリーンな一時ディレクトリで再試行 → それでもダメならメモリ
        slack_fallback_dir = os.path.join(tempfile.gettempdir(), f"chroma_db_slack_{uuid.uuid4().hex}")
        try:
            os.makedirs(slack_fallback_dir, exist_ok=True)
            db = Chroma.from_documents(
                docs_all,
                embedding=embeddings,
                persist_directory=slack_fallback_dir,
            )
        except Exception:
            db = Chroma.from_documents(docs_all, embedding=embeddings)
    retriever_dense = db.as_retriever(search_kwargs={"k": ct.TOP_K})
    retriever_bm25 = BM25Retriever.from_texts(
        docs_all_page_contents, preprocess_func=preprocess_func, k=ct.TOP_K
    )
    retriever = EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_dense],
        weights=ct.RETRIEVER_WEIGHTS,
    )

    employees = retriever.invoke(chat_message)

    # プロンプト作成
    context = get_context(employees)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", ct.SYSTEM_PROMPT_EMPLOYEE_SELECTION)]
    )
    output_parser = CommaSeparatedListOutputParser()
    format_instruction = output_parser.get_format_instructions()

    messages = prompt_template.format_prompt(
        context=context, query=chat_message, format_instruction=format_instruction
    ).to_messages()

    employee_id_response = st.session_state.llm(messages)
    employee_ids = output_parser.parse(employee_id_response.content)

    target_employees = get_target_employees(employees, employee_ids)
    slack_ids = get_slack_ids(target_employees)
    slack_id_text = create_slack_id_text(slack_ids)

    context = get_context(target_employees)
    now_datetime = get_datetime()

    prompt = PromptTemplate(
        input_variables=["slack_id_text", "query", "context", "now_datetime"],
        template=ct.SYSTEM_PROMPT_NOTICE_SLACK,
    )
    prompt_message = prompt.format(
        slack_id_text=slack_id_text,
        query=chat_message,
        context=context,
        now_datetime=now_datetime,
    )

    agent_executor.invoke({"input": prompt_message})
    return ct.CONTACT_THANKS_MESSAGE


def adjust_reference_data(docs, docs_history):
    """Slack通知用の参照先データの整形"""
    docs_all = []
    for row in docs:
        # 従業員ID
        row_lines = row.page_content.split("\n")
        row_dict = {x.split(": ")[0]: x.split(": ")[1] for x in row_lines if ": " in x}
        employee_id = row_dict.get("従業員ID", "")

        doc = ""
        same_employee_inquiries = []
        for row_history in docs_history:
            row_history_lines = row_history.page_content.split("\n")
            row_history_dict = {
                x.split(": ")[0]: x.split(": ")[1] for x in row_history_lines if ": " in x
            }
            if row_history_dict.get("従業員ID", "") == employee_id:
                same_employee_inquiries.append(row_history_dict)

        new_doc = Document()  # python-docx の Document を流用（page_content は動的属性）

        if same_employee_inquiries:
            doc += "【従業員情報】\n"
            doc += "\n".join(row_lines) + "\n=================================\n"
            doc += "【この従業員の問い合わせ対応履歴】\n"
            for inquiry_dict in same_employee_inquiries:
                for key, value in inquiry_dict.items():
                    doc += f"{key}: {value}\n"
                doc += "---------------\n"
            new_doc.page_content = doc
        else:
            new_doc.page_content = row.page_content
        new_doc.metadata = {}

        docs_all.append(new_doc)

    return docs_all


def get_target_employees(employees, employee_ids: List[str]):
    """問い合わせ内容と関連性が高い従業員情報一覧の取得"""
    target_employees = []
    seen = set()
    target_text = "従業員ID"
    for employee in employees:
        num = employee.page_content.find(target_text)
        employee_id = employee.page_content[num + len(target_text) + 2 :].split("\n")[0]
        if employee_id in employee_ids and employee_id not in seen:
            seen.add(employee_id)
            target_employees.append(employee)
    return target_employees


def get_slack_ids(target_employees):
    """SlackID の一覧を取得"""
    target_text = "SlackID"
    slack_ids = []
    for employee in target_employees:
        num = employee.page_content.find(target_text)
        slack_id = employee.page_content[num + len(target_text) + 2 :].split("\n")[0]
        slack_ids.append(slack_id)
    return slack_ids


def create_slack_id_text(slack_ids):
    """SlackID を「と」で繋いだテキストを生成"""
    out = []
    for sid in slack_ids:
        out.append(f"「{sid}」")
    return "と".join(out)


def get_context(docs) -> str:
    """プロンプトに埋め込むための従業員情報テキスト生成"""
    context = ""
    for i, doc in enumerate(docs, start=1):
        context += "===========================================================\n"
        context += f"{i}人目の従業員情報\n"
        context += "===========================================================\n"
        context += doc.page_content + "\n\n"
    return context


def get_datetime() -> str:
    """現在日時を取得"""
    dt_now = datetime.datetime.now()
    return dt_now.strftime("%Y年%m月%d日 %H:%M:%S")


def preprocess_func(text: str):
    """形態素解析による日本語の単語分割"""
    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text, mode)
    words = [token.surface() for token in tokens]
    return list(set(words))


def adjust_string(s):
    """Windows環境でRAGが正常動作するよう調整"""
    if not isinstance(s, str):
        return s
    if sys.platform.startswith("win"):
        s = unicodedata.normalize("NFC", s)
        s = s.encode("cp932", "ignore").decode("cp932")
    return s
