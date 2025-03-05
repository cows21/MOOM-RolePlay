import os
from typing import Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

import requests


class Text2vec_Large_ChineseEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None, local_model=None):
        super().__init__(config)

        self.jump_server_url = "http://10.4.148.51:13334/get_style_embedding"
        self.local_model = local_model

    def embed(self, text):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        embedding_model = self.local_model["text_encoder"]

        sentence_embeddings = embedding_model.encode(text)
        return sentence_embeddings

        # try:
        #     response = requests.post(self.jump_server_url, json=data)

        #     # 检查响应状态
        #     if response.status_code == 200:
        #         # 返回 OpenAI 的结果
        #         response_c = response.json()
        #         # print(response_c)
        #         # print(type(response_c))  # list
        #         return response_c
        #     else:
        #         # 处理错误响应
        #         raise Exception(f"Error from jump server: {response.status_code} - {response.text}")

        # except Exception as e:
        #     raise Exception(f"Failed to call jump server: {str(e)}")
