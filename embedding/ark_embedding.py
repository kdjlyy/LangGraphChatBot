from typing import List, Dict, Any, Optional

from .base_embedding import BaseEmbedding
from openai import AsyncOpenAI, OpenAI


class ArkEmbedding(BaseEmbedding):
    """Ark 嵌入模型实现类。

    该类实现了 BaseEmbedding 接口，提供与 OpenAI 兼容的嵌入功能。
    使用官方 SDK 进行 API 调用，确保了接口调用的可靠性和兼容性。

    Attributes:
        client (AsyncOpenAI): OpenAI 异步客户端实例，用于进行 API 调用。
        dimensions (Dict[str, int]): 不同模型的嵌入维度映射。
    """

    # 模型维度映射
    dimensions = {
        "doubao-embedding-text-240715": 2560,
        "doubao-embedding-large-text-240915": 4096,
    }

    def __init__(
            self,
            model_name: str,
            api_base: str,
            api_key: Optional[str] = None,
            **kwargs: Dict[str, Any]
    ) -> None:
        """初始化 Ark 嵌入模型实例。

        Args:
            model_name: 要使用的模型名称，如 'text-embedding-ada-002'。
            api_base: API 服务的基础 URL。
            api_key: OpenAI API 密钥，用于认证。
            **kwargs: 额外的模型配置参数。
        """
        super().__init__(model_name, api_base, api_key, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为嵌入向量。

        Args:
            text: 要嵌入的查询文本

        Returns:
            表示文本的嵌入向量

        Raises:
            OpenAIError: 当 API 调用失败时抛出
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            **self.kwargs
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文档文本转换为嵌入向量列表。

        Args:
            texts: 要嵌入的文档文本列表

        Returns:
            表示文档的嵌入向量列表

        Raises:
            OpenAIError: 当 API 调用失败时抛出
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            **self.kwargs
        )
        # 按照输入顺序排序结果
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def get_dimension(self) -> int:
        """获取嵌入向量的维度。

        Returns:
            嵌入向量的维度
        """
        return self.dimensions.get(self.model_name, 2560)  # 默认返回1536维