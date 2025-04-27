import os
from langchain.schema import Document
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import TextLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.graph.state import StateGraph, CompiledStateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from graph.graph_state import GraphState
from chains.summary import SummaryChain
from chains.generate import GenerateChain

def route_question(state: GraphState) -> str:
    """
    根据操作类型路由到相应的处理节点。

    参数:
        state (GraphState): 当前图的状态

    返回:
        str: 下一个要调用的节点名称
    """
    print("--- 🤖 正在根据类型选择分支 ---")
    if state['type'] == 'websearch':
        return "extract_keywords"
    if state['type'] == 'file':
        return "file_process"
    elif state['type'] == 'chat':
        return "generate"

def generate(state: GraphState) -> GraphState:
    """
    根据文档和对话历史生成答案。

    参数:
        state (GraphState): 当前图的状态

    返回:
        state (GraphState): 返回添加了LLM生成内容的新状态
    """
    print("--- 🤖 正在生成回答 ---")
    chain = GenerateChain(state["model_name"], state["temperature"])
    messages = state["messages"]
    state["messages"] = chain.invoke({
        "question": messages[-1].content,
        "history": messages[:-1],
        "documents": state["documents"]
    })
    return state

def file_process(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    处理文件

    参数:
        state (GraphState): 当前图的状态
        config (RunnableConfig): 可运行配置

    返回:
        state (GraphState): 返回图状态，将文档添加 config 中的向量存储
    """

    print("--- 🤖 开始处理文件 ---")
    vector_store = config["configurable"]["vectorstore"]

    for doc in state["documents"]:
        file_path: str = doc.page_content
        if os.path.exists(file_path):
            print(f"--- 📄 文件路径: {file_path}")
            split_docs: list[Document] = None
            if file_path.endswith(".txt") or file_path.endswith(".md"):
                # 处理文本或Markdown文件
                docs = TextLoader(file_path, autodetect_encoding=True).load()
                # 文本分割
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", " ", ".", ",", "\u200B", "\uff0c", "\u3001", "\uff0e", "\u3002", ""],
                    chunk_size=512,
                    chunk_overlap=256,
                    add_start_index=True
                )
                split_docs = splitter.split_documents(docs)
            else:
                # 使用 marker-pdf 处理其他文件
                converter = PdfConverter(artifact_dict=create_model_dict())
                rendered = converter(file_path)
                docs, _, _ = text_from_rendered(rendered)
                splitter = MarkdownHeaderTextSplitter(
                    [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
                    strip_headers = False
                )
                split_docs = splitter.split_text(docs)

            # 将处理后的文档添加到向量存储中
            vector_store.add_documents(split_docs)
        else:
            print(f"--- 📄 文件路径不存在: {file_path}")
    return state

def extract_keywords(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    从问题中提取关键词。

    参数:
        state (GraphState): 当前图的状态
        config (RunnableConfig): 可运行配置

    返回:
        state (GraphState): 返回添加了提取关键词的新状态
    """

    print("--- 🤖 正在提取关键词 ---")
    chain = SummaryChain(state["model_name"], state["temperature"])
    messages = state["messages"]
    query = chain.invoke({"question": messages[-1].content, "history": messages[:-1]})

    if state["type"] == "websearch":
        # 将生成的搜索查询添加到消息列表中，下一个节点将会使用
        state["messages"] = query
    elif state["type"] == "file":
        # 使用生成的搜索查询在向量数据库中搜索
        # docs = config["configurable"]["vectorstore"].max_marginal_relevance_search(query.content, 5)
        docs = config["configurable"]["vectorstore"].similarity_search(query.content, 5)
        print(f"--- 📄 召回结果:")
        for idx, doc in enumerate(docs):
            print(f"============================ doc-{idx + 1}  {doc.metadata} ============================ ")
            print(doc.page_content)
        state["documents"] = docs
    return state

def decide_to_generate(state: GraphState) -> str:
    """
    决定是进行网络搜索还是直接生成回答。

    参数:
        state (GraphState): 当前图的状态

    返回:
        str: 下一个要调用的节点名称
    """

    if state["type"] == "websearch":
        print("--- 🌐 需要进行网络搜索 ---")
        return "websearch"
    elif state["type"] == "file":
        print("--- ⭐ 无需搜索，直接生成答案 ---")
        return "generate"

def web_search(state: GraphState) -> GraphState:
    """
    基于问题进行网络搜索。

    参数:
        state (GraphState): 当前图的状态

    返回:
        state (GraphState): 返回添加了网络搜索结果的新状态
    """

    print("---🌐 正在进行网络搜索 ---")
    web_search_tool = TavilySearchResults(k = 3)
    documents = state["documents"]
    try:
        docs = web_search_tool.invoke({"query": state["messages"][-1].content})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        state["documents"] = documents
    except:
        pass
    print(f"🌐 搜索结果:\n{documents}")
    return state

def create_graph() -> CompiledStateGraph:
    """
    创建并配置状态图工作流。

    返回:
        CompiledStateGraph: 编译好的状态图
    """

    workflow = StateGraph(GraphState)
    # 添加节点
    workflow.add_node("websearch", web_search)
    workflow.add_node("extract_keywords", extract_keywords)
    workflow.add_node("file_process", file_process)
    workflow.add_node("generate", generate)
    # 添加边
    workflow.set_conditional_entry_point(
        route_question,
        {
            "extract_keywords": "extract_keywords",
            "generate": "generate",
            "file_process": "file_process",
        },
    )
    workflow.add_edge("file_process", "extract_keywords")
    workflow.add_conditional_edges(
        "extract_keywords",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("generate", END)

    # 创建图，并使用 `MemorySaver()` 在内存中保存状态
    return workflow.compile(checkpointer=MemorySaver())

def stream_graph_updates(graph: CompiledStateGraph, user_input: GraphState, config: dict):
    """
    流式处理图更新并返回最终结果。

    参数:
        graph (CompiledStateGraph): 编译好的状态图
        user_input (GraphState): 用户输入的状态
        config (dict): 配置字典

    返回:
        generator: 生成器对象，逐步返回图更新的内容
    """

    for chunk, _ in graph.stream(user_input, config, stream_mode="messages"):
        yield chunk.content
