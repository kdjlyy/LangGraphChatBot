from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseEmbedding(ABC):
    """所有嵌入模型的基类。

    这个抽象类定义了所有嵌入模型实现必须遵循的接口。
    它提供了一个统一的方式来与不同类型的嵌入模型进行交互。

    Args:
        model_name (str): 要使用的模型名称
        api_base (str): API 端点的基础 URL
        api_key (Optional[str]): 认证所需的 API 密钥（如果需要）
        **kwargs: 额外的模型特定配置选项
    """

    def __init__(
            self,
            model_name: str,
            api_base: str,
            api_key: Optional[str] = None,
            **kwargs: Dict[str, Any]
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.kwargs = kwargs

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为嵌入向量。

        Args:
            text (str): 要嵌入的查询文本

        Returns:
            List[float]: 表示文本的嵌入向量
        """
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文档文本转换为嵌入向量列表。

        Args:
            texts (List[str]): 要嵌入的文档文本列表

        Returns:
            List[List[float]]: 表示文档的嵌入向量列表
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """获取嵌入向量的维度。

        Returns:
            int: 嵌入向量的维度
        """
        pass