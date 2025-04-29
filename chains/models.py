import os

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from embedding.ark_embedding import ArkEmbedding


def load_model(model_name: str, temperature: float) -> ChatOpenAI:
    """
    加载语言模型

    参数:
        model_name (str): 模型名称

    返回:
        ChatOpenAI实例，用于生成文本和回答问题
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )


def load_embeddings(model_name: str) -> OpenAIEmbeddings:
    """
    加载嵌入模型

    参数:
        model_name (str): 模型名称

    返回:
        OpenAIEmbeddings 实例，用于将文本转换为向量表示
    """
    return OpenAIEmbeddings(
        model=model_name
    )



def load_ark_embeddings(model_name: str) -> ArkEmbedding:
    """
    加载火山方舟嵌入模型

    参数:
        model_name (str): 模型名称

    返回:
        ArkEmbedding 实例，用于将文本转换为向量表示
    """
    return ArkEmbedding(
        api_base=os.getenv("ARK_BASE_URL", ""),
        api_key=os.getenv("ARK_API_KEY", ""),
        model_name=model_name
    )

def load_vector_store(model_name: str) -> InMemoryVectorStore:
    """
    创建内存向量存储

    参数:
        model_name (str): 用于生成嵌入的模型名称

    返回:
        InMemoryVectorStore实例，用于存储和检索向量化的文本
    """
    if model_name.startswith("doubao-embedding"):
        embeddings = load_ark_embeddings(model_name)
    else:
        embeddings = load_embeddings(model_name)

    return InMemoryVectorStore(embeddings)
