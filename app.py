import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

import uuid
import datetime
from dotenv import load_dotenv
from langchain.schema import Document
import streamlit as st
from streamlit_extras.bottom_container import bottom
from chains.models import load_vector_store
from graph.graph import create_graph, stream_graph_updates, GraphState
from utils.common import *

# 设置上传文件的存储路径
file_path = "upload_files/"
# 加载环境变量
load_dotenv(verbose=True)

def upload_pdf(file):
    """保存上传的文件并返回文件路径"""
    with open(file_path + file.name, "wb") as f:
        f.write(file.getbuffer())
        return file_path + file.name

# 设置页面配置信息
st.set_page_config(
    page_title="AI 聊天机器人",
    page_icon="🤖",
    layout="wide"
)

# 初始化会话状态变量，创建图
if "graph" not in st.session_state:
    st.session_state.graph = create_graph()
# 初始化会话ID和向量存储
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": uuid.uuid4().hex, "vectorstore": load_vector_store("BAAI/bge-m3")}}
# 初始化对话历史记录
if "history" not in st.session_state:
    st.session_state.history = []
# 初始化上传状态、模型名称和对话类型
if "settings" not in st.session_state:
    st.session_state.settings = {"uploaded": False, "model_name": "Qwen/QwQ-32B", "type": "chat"}

# 显示应用标题
st.subheader("我可以帮你:blue[写代码、读文件、联网搜索解决各种问题]，欢迎向我提问～ 😸", divider="gray")

# 定义可选的模型
model_options = {"通义千问": "Qwen/QwQ-32B", "DeepSeek R1": "deepseek-ai/DeepSeek-R1"}
with st.sidebar:
    # 侧边栏设置部分
    st.header("设置")
    # 模型选择下拉框
    st.session_state.settings["model_name"] = model_options[st.selectbox("选择模型", model_options, index=list(model_options.values()).index(st.session_state.settings["model_name"]))]

    st.divider()

    # 显示版本信息
    st.caption(f"{datetime.datetime.now().strftime('%Y.%m')} - [LangGraphChatBot](https://github.com/kdjlyy/LangGraphChatBot)")
# 定义对话类型选项
type_options = {"⭐️ 离线对话": "chat", "🌐 联网搜索": "websearch", "⌨️ 代码模式": "code"}
question = None
with bottom():
    # 底部容器，包含工具选择、文件上传和输入框
    st.session_state.settings["type"] = type_options[st.radio("工具选择", type_options.keys(), horizontal=True, label_visibility="collapsed", index=list(type_options.values()).index(st.session_state.settings["type"]))]
    # 文件上传组件, pdf、doc、xlsx 格式的文件可能造成系统资源不足
    uploaded_file = st.file_uploader("上传文件", type=["txt", "md"], accept_multiple_files=False, label_visibility="collapsed")
    # 聊天输入框
    question = st.chat_input('输入您要询问的内容，shift + enter 换行')

# 显示历史对话内容
for message in st.session_state.history:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

# 处理用户提问
if question:
    # 显示用户问题
    with st.chat_message("user"):
        st.markdown(question)

    # 准备请求状态
    state = []
    message = [{"role": "system", "content": f"当前日期是：{get_current_time()}"}, {"role": "user", "content": question}]
    if st.session_state.settings["type"] == "code":
        # 代码模式使用专门的代码模型
        state = {"model_name": "Qwen/QwQ-32B", "messages": message, "type": "chat", "documents": []}
    else:
        # 其他模式使用选择的模型
        state = {"model_name": st.session_state.settings["model_name"], "messages": message, "type": st.session_state.settings["type"], "documents": []}

    # 处理文件上传
    if uploaded_file:
        state["type"] = "file"
        if not st.session_state.settings["uploaded"]:
            # 保存上传的文件
            file_path = upload_pdf(uploaded_file)
            # 添加文档到请求
            state["documents"].append(Document(page_content=file_path))
            st.session_state.settings["uploaded"] = True

    # 获取AI回答并以流式方式显示
    answer = st.chat_message("assistant").write_stream(stream_graph_updates(st.session_state.graph, state, st.session_state.config))

    # 将对话保存到历史记录
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})
