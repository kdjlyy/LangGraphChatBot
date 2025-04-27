from langchain.prompts import ChatPromptTemplate
from chains.models import load_model

class GenerateChain:
    """
    一个用于生成问答任务响应的类。
    它使用语言模型和提示模板来处理输入数据。
    """
    def __init__(self, model_name, temperature):
        """
        初始化 GenerateChain 类，并加载指定的语言模型。

        参数:
            model_name (str): 要加载的语言模型的名称。
        """
        self.llm = load_model(model_name, temperature)
        self.prompt = ChatPromptTemplate.from_template(
            "You are an assistant for question-answering tasks. "
            "Use the following documents or chat histories to answer the question. "
            "If the documents or chat histories is empty, answer the question based on your own knowledge. "
            "If you don't know the answer, just say that you don't know."
            "\n\nDocuments: {documents}\n\nHistories: {history}\n\nQuestion: {question}")

        self.chain = self.prompt | self.llm

    def invoke(self, input_data):
        """
        使用提供的输入数据调用链以生成响应。

        参数:
            input_data (dict): 包含 'documents'、'history' 和 'question' 键的字典。

        返回:
            str: 链生成的响应。
        """
        return self.chain.invoke(input_data)
