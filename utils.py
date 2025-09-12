"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
import logging
import sys
import unicodedata
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from typing import List
from sudachipy import tokenizer, dictionary
from langchain_community.agent_toolkits import SlackToolkit
from langchain.agents import AgentType, initialize_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from docx import Document
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import LLMChain
import datetime
import constants as ct


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

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
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
    # AIエージェント機能を使わない場合の処理
    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        # 「data」フォルダ直下の各フォルダ名に対して処理
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            # フォルダ内の各ファイルのデータをリストに追加
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    # AIエージェント機能を使う場合の処理
    else:
        # データベース名に対応した、RAG化対象のデータ群が格納されているフォルダパスを取得
        folder_path = ct.DB_NAMES[db_name]
        # フォルダ内の各ファイルのデータをリストに追加
        add_docs(folder_path, docs_all)

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
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

    embeddings = OpenAIEmbeddings()

    # すでに対象のデータベースが作成済みの場合は読み込み、未作成の場合は新規作成する
    if os.path.isdir(db_name):
        db = Chroma(persist_directory=".db", embedding_function=embeddings)
    else:
        db = Chroma.from_documents(splitted_docs, embedding=embeddings, persist_directory=".db")
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

    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def add_docs(folder_path, docs_all):
    """
    フォルダ内のファイル一覧を取得

    Args:
        folder_path: フォルダのパス
        docs_all: 各ファイルデータを格納するリスト
    """
    files = os.listdir(folder_path)
    for file in files:
        # ファイルの拡張子を取得
        file_extension = os.path.splitext(file)[1]
        # 想定していたファイル形式の場合のみ読み込む
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](f"{folder_path}/{file}")
        else:
            continue
        docs = loader.load()
        docs_all.extend(docs)


def run_company_doc_chain(param):
    """
    会社に関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # 会社に関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_service_doc_chain(param):
    """
    サービスに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # サービスに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.service_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_customer_doc_chain(param):
    """
    顧客とのやり取りに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値
    
    Returns:
        LLMからの回答
    """
    # 顧客とのやり取りに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.customer_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]


def delete_old_conversation_log(result):
    """
    古い会話履歴の削除

    Args:
        result: LLMからの回答
    """
    # LLMからの回答テキストのトークン数を取得
    response_tokens = len(st.session_state.enc.encode(result))
    # 過去の会話履歴の合計トークン数に加算
    st.session_state.total_tokens += response_tokens

    # トークン数が上限値を下回るまで、順に古い会話履歴を削除
    while st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS:
        # 最も古い会話履歴を削除
        removed_message = st.session_state.chat_history.pop(1)
        # 最も古い会話履歴のトークン数を取得
        removed_tokens = len(st.session_state.enc.encode(removed_message.content))
        # 過去の会話履歴の合計トークン数から、最も古い会話履歴のトークン数を引く
        st.session_state.total_tokens -= removed_tokens


def execute_agent_or_chain(chat_message):
    """
    AIエージェントもしくはAIエージェントなしのRAGのChainを実行

    Args:
        chat_message: ユーザーメッセージ
    
    Returns:
        LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # AIエージェント機能を利用する場合
    if st.session_state.agent_mode == ct.AI_AGENT_MODE_ON:
        # LLMによる回答をストリーミング出力するためのオブジェクトを用意
        st_callback = StreamlitCallbackHandler(st.container())
        # Agent Executorの実行（AIエージェント機能を使う場合は、Toolとして設定した関数内で会話履歴への追加処理を実施）
        result = st.session_state.agent_executor.invoke({"input": chat_message}, {"callbacks": [st_callback]})
        response = result["output"]
    # AIエージェントを利用しない場合
    else:
        # RAGのChainを実行
        result = st.session_state.rag_chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
        # 会話履歴への追加
        st.session_state.chat_history.extend([HumanMessage(content=chat_message), AIMessage(content=result["answer"])])
        response = result["answer"]

    # LLMから参照先のデータを基にした回答が行われた場合のみ、フィードバックボタンを表示
    if response != ct.NO_DOC_MATCH_MESSAGE:
        st.session_state.answer_flg = True
    
    return response


def notice_slack(chat_message):
    """
    問い合わせ内容のSlackへの通知

    Args:
        chat_message: ユーザーメッセージ

    Returns:
        問い合わせサンクスメッセージ
    """

    # Slack通知用のAgent Executorを作成
    toolkit = SlackToolkit()
    tools = toolkit.get_tools()
    agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    # 担当者割り振りに使う用の「従業員情報」と「問い合わせ対応履歴」の読み込み
    loader = CSVLoader(ct.EMPLOYEE_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs = loader.load()
    loader = CSVLoader(ct.INQUIRY_HISTORY_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs_history = loader.load()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    for doc in docs_history:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # 問い合わせ内容と関連性が高い従業員情報を取得するために、参照先データを整形
    docs_all = adjust_reference_data(docs, docs_history)
    
    # 形態素解析による日本語の単語分割を行うため、参照先データからテキストのみを抽出
    docs_all_page_contents = []
    for doc in docs_all:
        docs_all_page_contents.append(doc.page_content)

    # Retrieverの作成
    embeddings = OpenAIEmbeddings()
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

    # 問い合わせ内容と関連性の高い従業員情報を取得
    employees = retriever.invoke(chat_message)
    
    # プロンプトに埋め込むための従業員情報テキストを取得
    context = get_context(employees)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_EMPLOYEE_SELECTION)
    ])
    # フォーマット文字列を生成
    output_parser = CommaSeparatedListOutputParser()
    format_instruction = output_parser.get_format_instructions()

    # 問い合わせ内容と関連性が高い従業員のID一覧を取得
    messages = prompt_template.format_prompt(context=context, query=chat_message, format_instruction=format_instruction).to_messages()
    employee_id_response = st.session_state.llm(messages)
    employee_ids = output_parser.parse(employee_id_response.content)

    # 問い合わせ内容と関連性が高い従業員情報を、IDで照合して取得
    target_employees = get_target_employees(employees, employee_ids)
    
    # 問い合わせ内容と関連性が高い従業員情報の中から、SlackIDのみを抽出
    slack_ids = get_slack_ids(target_employees)
    
    # 抽出したSlackIDの連結テキストを生成
    slack_id_text = create_slack_id_text(slack_ids)
    
    # プロンプトに埋め込むための（問い合わせ内容と関連性が高い）従業員情報テキストを取得
    context = get_context(target_employees)

    # 現在日時を取得
    now_datetime = get_datetime()

    # Slack通知用のプロンプト生成
    prompt = PromptTemplate(
        input_variables=["slack_id_text", "query", "context", "now_datetime"],
        template=ct.SYSTEM_PROMPT_NOTICE_SLACK,
    )
    prompt_message = prompt.format(slack_id_text=slack_id_text, query=chat_message, context=context, now_datetime=now_datetime)

    # Slack通知の実行
    agent_executor.invoke({"input": prompt_message})

    return ct.CONTACT_THANKS_MESSAGE


def adjust_reference_data(docs, docs_history):
    """
    Slack通知用の参照先データの整形

    Args:
        docs: 従業員情報ファイルの読み込みデータ
        docs_history: 問い合わせ対応履歴ファイルの読み込みデータ

    Returns:
        従業員情報と問い合わせ対応履歴の結合テキスト
    """

    docs_all = []
    for row in docs:
        # 従業員IDの取得
        row_lines = row.page_content.split("\n")
        row_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_lines}
        employee_id = row_dict["従業員ID"]

        doc = ""

        # 取得した従業員IDに紐づく問い合わせ対応履歴を取得
        same_employee_inquiries = []
        for row_history in docs_history:
            row_history_lines = row_history.page_content.split("\n")
            row_history_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_history_lines}
            if row_history_dict["従業員ID"] == employee_id:
                same_employee_inquiries.append(row_history_dict)

        new_doc = Document()

        if same_employee_inquiries:
            # 従業員情報と問い合わせ対応履歴の結合テキストを生成
            doc += "【従業員情報】\n"
            row_data = "\n".join(row_lines)
            doc += row_data + "\n=================================\n"
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



def get_target_employees(employees, employee_ids):
    """
    問い合わせ内容と関連性が高い従業員情報一覧の取得

    Args:
        employees: 問い合わせ内容と関連性が高い従業員情報一覧
        employee_ids: 問い合わせ内容と関連性が「特に」高い従業員のID一覧

    Returns:
        問い合わせ内容と関連性が「特に」高い従業員情報一覧
    """

    target_employees = []
    duplicate_check = []
    target_text = "従業員ID"
    for employee in employees:
        # 従業員IDの取得
        num = employee.page_content.find(target_text)
        employee_id = employee.page_content[num+len(target_text)+2:].split("\n")[0]
        # 問い合わせ内容と関連性が高い従業員情報を、IDで照合して取得（重複除去）
        if employee_id in employee_ids:
            if employee_id in duplicate_check:
                continue
            duplicate_check.append(employee_id)
            target_employees.append(employee)
    
    return target_employees


def get_slack_ids(target_employees):
    """
    SlackIDの一覧を取得

    Args:
        target_employees: 問い合わせ内容と関連性が高い従業員情報一覧

    Returns:
        SlackIDの一覧
    """

    target_text = "SlackID"
    slack_ids = []
    for employee in target_employees:
        num = employee.page_content.find(target_text)
        slack_id = employee.page_content[num+len(target_text)+2:].split("\n")[0]
        slack_ids.append(slack_id)
    
    return slack_ids


def create_slack_id_text(slack_ids):
    """
    SlackIDの一覧を取得

    Args:
        slack_ids: SlackIDの一覧

    Returns:
        SlackIDを「と」で繋いだテキスト
    """
    slack_id_text = ""
    for i, id in enumerate(slack_ids):
        slack_id_text += f"「{id}」"
        # 最後のSlackID以外、連結後に「と」を追加
        if not i == len(slack_ids)-1:
            slack_id_text += "と"
    
    return slack_id_text


def get_context(docs):
    """
    プロンプトに埋め込むための従業員情報テキストの生成
    Args:
        docs: 従業員情報の一覧

    Returns:
        生成した従業員情報テキスト
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

    Returns:
        現在日時
    """

    dt_now = datetime.datetime.now()
    now_datetime = dt_now.strftime('%Y年%m月%d日 %H:%M:%S')

    return now_datetime


def preprocess_func(text):
    """
    形態素解析による日本語の単語分割
    Args:
        text: 単語分割対象のテキスト

    Returns:
        単語分割を実施後のテキスト
    """

    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text ,mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))

    return words


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s