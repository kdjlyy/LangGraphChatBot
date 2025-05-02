import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
import uuid
from langchain.schema import Document
from streamlit_extras.bottom_container import bottom
from chains.models import load_vector_store, load_rerank
from graph.graph import create_graph, stream_graph_updates
from utils.common import *

# 加载 .env 到环境变量
load_dotenv(verbose=True)

# 设置页面配置信息
st.set_page_config(
    page_title="AI 聊天机器人",
    page_icon="🤖",
    layout="wide"
)

# Fixme: 减少加载频率
env = load_env_vars()
# 定义对话类型选项
type_options = {"⭐️ 离线对话": "chat", "🌐 联网搜索": "websearch", "⌨️ 代码模式": "code"}

# 初始化上传状态、模型名称和对话类型
if "settings" not in st.session_state:
    # 默认为离线对话
    st.session_state.settings = {"type": "chat", "uploaded": False, "search_num": env["SEARCH_NUN"]}
    st.session_state.embedding_selectbox_disable = False
# 初始化会话ID和向量存储
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": uuid.uuid4().hex, "vectorstore": None, "rerank": None}}
# 初始化会话状态变量，创建图
if "graph" not in st.session_state:
    st.session_state.graph = create_graph()
# 初始化对话历史记录
if "history" not in st.session_state:
    st.session_state.history = []

# 显示应用标题
st.subheader("我可以帮你:blue[写代码、读文件、联网搜索解决各种问题]，欢迎向我提问～ 😸", divider="gray")

# 定义可选的模型
model_options = {"通义千问": "Qwen/QwQ-32B", "DeepSeek R1": "deepseek-ai/DeepSeek-R1"}

# 重新创建图
def rebuild_graph():
    st.session_state.graph = create_graph()

# 侧边栏设置部分
with st.sidebar:
    st.header("设置")
    st.divider()

    # 模型选择下拉框
    env['CURRENT_MODEL'] = st.selectbox(
        label="选择模型",
        options=env["AVAILABLE_MODEL_LIST"],
        index=0,
        help="选择 LLM 模型的种类",
        on_change=rebuild_graph
    )
    # 模型温度滑动条
    env['TEMPERATURE'] = st.slider(
        label="模型温度",
        min_value=0.0, max_value=1.0, value=0.0,
        help="模型温度（Temperature）参数用于控制模型输出的多样性和确定性。高 Temperature 增加多样性但可能降低确定性，低 Temperature 则增加确定性但可能降低多样性。"
    )
    st.divider()
    st.selectbox(
        key="embedding_model_selectbox",
        label="选择嵌入模型",
        options=env["AVAILABLE_EMBEDDING_MODEL_LIST"],
        index=2,
        help="选择嵌入模型的种类",
    )

    st.session_state.settings["model_name"] = env['CURRENT_MODEL']
    st.session_state.settings["temperature"] = env['TEMPERATURE']

    if not st.session_state.config["configurable"]["vectorstore"]:
        st.session_state.config["configurable"]["vectorstore"] = load_vector_store(st.session_state.embedding_model_selectbox)

    if not st.session_state.config["configurable"]["rerank"]:
        st.session_state.config["configurable"]["rerank"] = load_rerank()
    st.divider()

    # 自定义链接
    st.caption(f"{datetime.datetime.now().strftime('%Y.%m')} - [LangGraphChatBot](https://github.com/kdjlyy/LangGraphChatBot)")

question = None

with bottom():
    # 底部容器，包含工具选择、文件上传和输入框
    st.session_state.settings["type"] = type_options[st.radio("工具选择", type_options.keys(), horizontal=True, label_visibility="collapsed", index=list(type_options.values()).index(st.session_state.settings["type"]))]
    # 文件上传组件, pdf、doc、xlsx 格式的文件可能造成系统资源不足
    # uploaded_file = st.file_uploader("上传文件", type=["txt", "md", "pdf", "doc", "xls", "xlsx"], accept_multiple_files=False, label_visibility="collapsed")
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
        state = {"model_name": env["CODE_MODEL"], "temperature": st.session_state.settings["temperature"],
                 "messages": message, "type": "chat", "documents": [],  "search_num": env["SEARCH_NUN"]}
    else:
        # 其他模式使用选择的模型
        state = {"model_name": st.session_state.settings["model_name"], "temperature": st.session_state.settings["temperature"],
                 "messages": message, "type": st.session_state.settings["type"], "documents": [], "search_num": env["SEARCH_NUN"]}

    # 处理文件上传
    if uploaded_file:
        state["type"] = "file"
        if not st.session_state.settings["uploaded"]:
            # 保存上传的文件
            file_path = upload_pdf(uploaded_file)
            # 添加文档到请求
            state["documents"].append(Document(page_content=file_path))
            st.session_state.settings["uploaded"] = True
        else:
            st.error("请刷新页面后再上传文件")

    # 获取AI回答并以流式方式显示
    answer = st.chat_message("assistant").write_stream(stream_graph_updates(st.session_state.graph, state, st.session_state.config))

    # 将对话保存到历史记录
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})
