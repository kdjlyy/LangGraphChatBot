import os

from langgraph.graph.state import CompiledStateGraph

def gen_mermaid(graph: CompiledStateGraph, file_name: str):
    """ 生成 graph 对应的 mermaid 文件 """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", file_name))
    with open(path, "w", encoding="utf-8") as file:
        file.write(graph.get_graph().draw_mermaid())
    print(f"⚙️已生成 mermaid 文件 {path}")