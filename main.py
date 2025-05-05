import uuid
import datetime
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.messages import AIMessage, HumanMessage
from chains.models import load_vector_store
from graph.graph import create_graph, stream_graph_updates, GraphState
from utils.common import *
from utils import pretty

def main():
    # langchain.debug = True  # 启用langchain调试模式，可以获得如完整提示词等信息
    load_dotenv(verbose=True)  # 加载环境变量配置

    # 创建状态图以及对话相关的设置
    config = {"configurable": {
        "thread_id": uuid.uuid4().hex,
        "vectorstore": load_vector_store("BAAI/bge-large-zh-v1.5")}
    }
    state = GraphState(
        model_name="Qwen/QwQ-32B",
        temperature=0.0,
        type="chat",
        documents=[Document(page_content="upload_files/test.pdf")],
    )
    graph = create_graph()

    # 生成 mermaid 图
    # gen_mermaid(graph, "graph.mmd")

    # 对话
    while True:
        user_input = input("User: ")
        if user_input.strip() == "":
            continue
        if user_input.lower() in ["e", "exit", 'q', "quit"]:
            break
        state["messages"] = [HumanMessage(user_input)]
        # 流式获取AI的回复
        for answer in stream_graph_updates(graph, state, config):
            print(answer, end="")
        print()

    # 打印对话历史
    pretty.ALogger("[main]").title("History")
    for message in graph.get_state(config).values["messages"]:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"
        pretty.ALogger("[main]").title(prefix)
        print(f"{message.content}")

if __name__ == "__main__":
    main()
