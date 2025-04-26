from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def load_model(model_name: str) -> ChatOpenAI:
    """
    加载语言模型

    参数:
        model_name (str): 模型名称

    返回:
        ChatOpenAI实例，用于生成文本和回答问题
    """
    return ChatOpenAI(
        model=model_name,
        temperature=0.0,
    )


def load_embeddings(model_name: str) -> OpenAIEmbeddings:
    """
    加载嵌入模型

    参数:
        model_name (str): 模型名称

    返回:
        OpenAIEmbeddings实例，用于将文本转换为向量表示
    """
    return OpenAIEmbeddings(
        model=model_name
    )


def load_vector_store(model_name: str) -> InMemoryVectorStore:
    """
    创建内存向量存储

    参数:
        model_name (str): 用于生成嵌入的模型名称

    返回:
        InMemoryVectorStore实例，用于存储和检索向量化的文本
    """
    embeddings = load_embeddings(model_name)
    return InMemoryVectorStore(embeddings)
