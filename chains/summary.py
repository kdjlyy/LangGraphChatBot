from langchain.prompts import ChatPromptTemplate
from chains.models import load_model

class SummaryChain:
    """
    一个用于生成搜索查询的类。
    从用户问题和聊天记录中提取关键词，并生成高效的搜索查询。
    """
    def __init__(self, model_name):
        """
        初始化 SummaryChain 类，并加载指定的语言模型。

        参数:
            model_name (str): 要加载的语言模型的名称。
        """
        self.llm = load_model(model_name)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a professional assistant specializing in extracting keywords from user questions and chat histories. "
            "Extract keywords and connect them with spaces to output a efficient and precise search query. "
            "First, you need to take the user question itself as the first keyword, "
            "then extract the keyword from the user question and append it to the keyword list."
            "Be careful not answer the question directly, just output the search query.\n\nHistories: {history}\n\nQuestion: {question}"
        )
        self.chain = self.prompt | self.llm

    def invoke(self, input_data):
        """
        使用提供的输入数据调用链以生成搜索查询。

        参数:
            input_data (dict): 包含 'history' 和 'question' 键的字典。

        返回:
            str: 链生成的搜索查询。
        """
        return self.chain.invoke(input_data)
