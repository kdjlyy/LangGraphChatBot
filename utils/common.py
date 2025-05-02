import datetime
import os
from dotenv import load_dotenv
import streamlit as st
from langgraph.graph.state import CompiledStateGraph

@st.cache_data
def load_env_vars():
    """
    从 .env 文件读取配置
    :return: json 格式的配置
    """
    load_dotenv(verbose=True)
    # OpenAI API 密钥
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    # OpenAI API 地址
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
    #  可用的模型
    AVAILABLE_MODELS = os.getenv("AVAILABLE_MODELS", None)
    #  模型列表
    AVAILABLE_MODEL_LIST = AVAILABLE_MODELS.split(",") if AVAILABLE_MODELS else []
    # 代码模式使用的模型
    CODE_MODEL = os.getenv("CODE_MODEL", None)
    # 可用的嵌入模型
    AVAILABLE_EMBEDDING_MODELS = os.getenv("AVAILABLE_EMBEDDING_MODELS", None)
    # 嵌入模型列表
    AVAILABLE_EMBEDDING_MODEL_LIST = AVAILABLE_EMBEDDING_MODELS.split(",") if AVAILABLE_EMBEDDING_MODELS else []

    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY in your environment variables.")
    else:
        trans = OPENAI_API_KEY.replace(OPENAI_API_KEY[10:-10], '*' * len(OPENAI_API_KEY[10:-10]))
        print(f'✅ OPENAI_API_KEY: {trans}')

    if not OPENAI_BASE_URL:
        raise ValueError("Please set OPENAI_BASE_URL in your environment variables.")
    else:
        print(f'✅ OPENAI_BASE_URL: {OPENAI_BASE_URL}')

    if not AVAILABLE_MODELS:
        raise ValueError("Please set AVAILABLE_MODELS in your environment variables.")
    else:
        print(f'✅ AVAILABLE_MODELS: {AVAILABLE_MODELS}')

    if not CODE_MODEL:
        raise ValueError("Please set CODE_MODEL in your environment variables.")
    else:
        print(f'✅ CODE_MODEL: {CODE_MODEL}')

    if not AVAILABLE_EMBEDDING_MODELS:
        raise ValueError("Please set AVAILABLE_EMBEDDING_MODELS in your environment variables.")
    else:
        print(f'✅ AVAILABLE_EMBEDDING_MODELS: {AVAILABLE_EMBEDDING_MODELS}')

    return {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "AVAILABLE_MODEL_LIST": AVAILABLE_MODEL_LIST,
        "TEMPERATURE": 0.3,
        "CODE_MODEL": CODE_MODEL,
        "AVAILABLE_EMBEDDING_MODEL_LIST": AVAILABLE_EMBEDDING_MODEL_LIST,
        "SEARCH_NUN": os.getenv("SEARCH_NUN", 3)
    }

def upload_pdf(file):
    """保存上传的文件并返回文件路径"""
    file_path = "upload_files/"
    with open(file_path + file.name, "wb") as f:
        f.write(file.getbuffer())
        return file_path + file.name

def gen_mermaid(graph: CompiledStateGraph, file_name: str):
    """ 生成 graph 对应的 mermaid 文件 """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", file_name))
    with open(path, "w", encoding="utf-8") as file:
        file.write(graph.get_graph().draw_mermaid())
    print(f"⚙️已生成 mermaid 文件 {path}")


def get_current_time() -> str:
    """ 获取当前时间 """
    utc_now = datetime.datetime.utcnow()
    utc_8 = utc_now + datetime.timedelta(hours=8)
    return utc_8.strftime("%Y-%m-%d %H:%M:%S")