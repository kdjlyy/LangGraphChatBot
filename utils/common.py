import os
import datetime
import pytz

from langgraph.graph.state import CompiledStateGraph

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