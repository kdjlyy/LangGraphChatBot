from typing import Literal, Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    定义图状态的类型字典。
    用于表示图中的状态信息。
    """
    model_name: str                             # 使用的模型名称
    temperature: float                          #  模型温度
    embedding_model_name: str                   # 使用的嵌入模型名称
    type: Literal["websearch", "file", "chat"]  # 操作类型，包括联网搜索、上传文件和聊天
    messages: Annotated[list, add_messages]     # 消息列表，使用add_messages注解处理消息追加
    documents: Optional[list] = []              # 文档列表，默认为空列表
