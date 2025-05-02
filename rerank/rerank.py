import requests
from langchain_core.documents import Document


class Rerank:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.rerank_model = model
        self.base_url = base_url
        self.api_key = api_key
        print(f'rerank 初始化成功：{self.rerank_model}, {self.base_url}, {self.api_key}')

    def rerank(self, results: list[Document], query: str, k = 5) -> list[Document]:
        """
        使用云端重排序模型模型对检索结果进行重排序

        :param results: 原始检索结果
        :param query: 查询
        :param k: 返回的top-k结果数
        :return: 重排序后的结果
        """
        texts = [item.page_content for item in results]

        url = self.base_url

        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": texts,
            "top_n": k,
            "return_documents": False,
            "max_chunks_per_doc": 512,
            "overlap_tokens": 256
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        try:
            if response.status_code == 200:
                response = response.json()
                indices = [item['index'] for item in response['results']]
                return [results[i] for i in indices]
            else:
                print(f"❌ {response.json()}")
        except:
            print(f'Error:network error status_code={response.status_code}')
