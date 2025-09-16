"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
print("DEBUG: enter app_utils")  # ★起動トレース

import constants as ct
print("DEBUG: imported config OK")  # config インポート確認

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
print("DEBUG: loaders OK")
from langchain_community.document_loaders.csv_loader import CSVLoader
print("DEBUG: csv loader OK")
from langchain_text_splitters import CharacterTextSplitter
print("DEBUG: splitter OK")
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
print("DEBUG: prompts OK")
from langchain_core.messages import HumanMessage, AIMessage
print("DEBUG: messages OK")
from langchain_core.output_parsers import CommaSeparatedListOutputParser
print("DEBUG: parsers OK")
from langchain_openai import OpenAIEmbeddings
print("DEBUG: embeddings OK")
from langchain_community.vectorstores import Chroma
print("DEBUG: chroma OK")
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, LLMChain
print("DEBUG: chains OK")
from langchain.chains.combine_documents import create_stuff_documents_chain
print("DEBUG: combine OK")
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
print("DEBUG: st callback OK")
from langchain_community.retrievers import BM25Retriever
print("DEBUG: bm25 OK")
from langchain.retrievers import EnsembleRetriever
print("DEBUG: ensemble OK")
from langchain_community.agent_toolkits import SlackToolkit
print("DEBUG: slack toolkit OK")
from langchain.agents import AgentType, initialize_agent
print("DEBUG: agents OK")
from sudachipy import tokenizer, dictionary
print("DEBUG: sudachipy OK")
from docx import Document
print("DEBUG: python-docx OK")

import os
from dotenv import load_dotenv
import streamlit as st
import logging
import sys
import unicodedata
from typing import List
import datetime


############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def create_rag_chain(db_name):
    """
    引数として渡されたDB内を参照するRAGのChainを作成
    Args:
        db_name: RAG化対象のデータを格納するデータベース名
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = []
    # AIエージェント機能を使わない場合の処理（全フォルダ集約）
    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    else:
        # 個別DB指定
        folder_path = ct.DB_NAMES[db_name]
        add_docs(folder_path, docs_all)

    # Windows互換のための文字列調整
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 既存DB読み込み or 新規作成
    persist_dir = ".db"
    if os.path.isdir(persist_dir):
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(splitted_docs, embedding=embeddings, persist_directory=persist_dir)

    # ★どちらの分岐でも retriever を必ず作成
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
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


def add_docs(folder_path, docs_all):
    """
    フォルダ内のファイル一覧を取得して読み込む
    """
    if not os.path.isdir(folder_path):
        return

    files = os.listdir(folder_path)
    for file in files:
        file_extension = os.path.splitext(file)[1]
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](f"{folder_path}/{file}")
        else:
            continue
        docs = loader.load()
        docs_all.extend(docs)


def run_company_doc_chain(param):
    """
    会社に関するデータ参照に特化したTool用関数
    """
    ai_msg = st.session_state.company_doc_chain.invoke(
        {"input": param, "chat_history": st.session_state.chat_history}
    )
    st.session_state.chat_history.extend(
        [HumanMessage(content=param), AIMessage(content=ai_msg["answer"])]
    )
    return ai_msg["answer"]


def run_service_doc_chain(param):
    """
    サービスに関するデータ参照に特化したTool用関数
    """
    ai_msg = st.session_state.service_doc_chain.invoke(
        {"input": param, "chat_history": st.session_state.chat_history}
    )
    st.session_state.chat_history.extend(
        [HumanMessage(content=param), AIMessage(content=ai_msg["answer"])]
    )
    return ai_msg["answer"]


def run_customer_doc_chain(param):
    """
    顧客とのやり取りに関するデータ参照に特化したTool用関数
    """
    ai_msg = st.session_state.customer_doc_chain.invoke(
        {"input": param, "chat_history": st.session_state.chat_history}
    )
    st.session_state.chat_history.extend(
        [HumanMessage(content=param), AIMessage(content=ai_msg["answer"])]
    )
    return ai_msg["answer"]


def delete_old_conversation_log(result):
    """
    古い会話履歴の削除（トークン上限管理）
    """
    response_tokens = len(st.session_state.enc.encode(result))
    st.session_state.total_tokens += response_tokens

    while st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS and len(st.session_state.chat_history) > 1:
        removed_message = st.session_state.chat_history.pop(1)
        removed_tokens = len(st.session_state.enc.encode(removed_message.content))
        st.session_state.total_tokens -= removed_tokens


def execute_agent_or_chain(chat_message):
    """
    AIエージェント or RAG Chain の実行
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

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


def notice_slack(chat_message):
    """
    問い合わせ内容のSlackへの通知
    """
    toolkit = SlackToolkit()
    tools = toolkit.get_tools()
    agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    # 従業員情報 / 問い合わせ履歴の読み込み
    loader = CSVLoader(ct.EMPLOYEE_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs = loader.load()
    loader = CSVLoader(ct.INQUIRY_HISTORY_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs_history = loader.load()

    for doc in docs:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    for doc in docs_history:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # データ整形
    docs_all = adjust_reference_data(docs, docs_history)
    docs_all_page_contents = [doc.page_content for doc in docs_all]

    # Retriever 構築（Ensemble: BM25 + embeddings）
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(docs_all, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})
    bm25_retriever = BM25Retriever.from_texts(
        docs_all_page_contents,
        preprocess_func=preprocess_func,
        k=ct.TOP_K
    )
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=ct.RETRIEVER_WEIGHTS
    )

    employees = retriever.invoke(chat_message)

    # プロンプト作成
    context = get_context(employees)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_EMPLOYEE_SELECTION)
    ])
    output_parser = CommaSeparatedListOutputParser()
    format_instruction = output_parser.get_format_instructions()

    messages = prompt_template.format_prompt(
        context=context,
        query=chat_message,
        format_instruction=format_instruction
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
        slack_id_text=slack_id_text, query=chat_message, context=context, now_datetime=now_datetime
    )

    agent_executor.invoke({"input": prompt_message})
    return ct.CONTACT_THANKS_MESSAGE


def adjust_reference_data(docs, docs_history):
    """
    Slack通知用の参照先データの整形
    """
    docs_all = []
    for row in docs:
        # 従業員IDを取得
        row_lines = row.page_content.split("\n")
        row_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_lines if ": " in item}
        employee_id = row_dict.get("従業員ID", "")

        doc = ""

        # 該当従業員の履歴を抽出
        same_employee_inquiries = []
        for row_history in docs_history:
            row_history_lines = row_history.page_content.split("\n")
            row_history_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_history_lines if ": " in item}
            if row_history_dict.get("従業員ID", "") == employee_id:
                same_employee_inquiries.append(row_history_dict)

        # 新しい Document（python-docx の Document を使い、属性を付与）
        new_doc = Document()

        if same_employee_inquiries:
            doc += "【従業員情報】\n"
            row_data = "\n".join(row_lines)
            doc += row_data + "\n=================================\n"
            doc += "【この従業員の問い合わせ対応履歴】\n"
            for inquiry_dict in same_employee_inquiries:
                for key, value in inquiry_dict.items():
                    doc += f"{key}: {value}\n"
                doc += "---------------\n"
            new_doc.page_content = doc  # 動的属性付与（既存実装踏襲）
        else:
            new_doc.page_content = row.page_content
        new_doc.metadata = {}

        docs_all.append(new_doc)

    return docs_all


def get_target_employees(employees, employee_ids):
    """
    問い合わせ内容と関連性が高い従業員情報一覧の取得
    """
    target_employees = []
    duplicate_check = []
    target_text = "従業員ID"
    for employee in employees:
        num = employee.page_content.find(target_text)
        employee_id = employee.page_content[num + len(target_text) + 2:].split("\n")[0]
        if employee_id in employee_ids:
            if employee_id in duplicate_check:
                continue
            duplicate_check.append(employee_id)
            target_employees.append(employee)
    return target_employees


def get_slack_ids(target_employees):
    """
    SlackIDの一覧を取得
    """
    target_text = "SlackID"
    slack_ids = []
    for employee in target_employees:
        num = employee.page_content.find(target_text)
        slack_id = employee.page_content[num + len(target_text) + 2:].split("\n")[0]
        slack_ids.append(slack_id)
    return slack_ids


def create_slack_id_text(slack_ids):
    """
    SlackIDを「と」で繋いだテキストを生成
    """
    slack_id_text = ""
    for i, id in enumerate(slack_ids):
        slack_id_text += f"「{id}」"
        if i != len(slack_ids) - 1:
            slack_id_text += "と"
    return slack_id_text


def get_context(docs):
    """
    プロンプトに埋め込むための従業員情報テキスト生成
    """
    context = ""
    for i, doc in enumerate(docs, start=1):
        context += "===========================================================\n"
        context += f"{i}人目の従業員情報\n"
        context += "===========================================================\n"
        context += doc.page_content + "\n\n"
    return context


def get_datetime():
    """
    現在日時を取得
    """
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%Y年%m月%d日 %H:%M:%S')


def preprocess_func(text):
    """
    形態素解析による日本語の単語分割
    """
    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text, mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))
    return words


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    """
    if not isinstance(s, str):
        return s
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    return s
