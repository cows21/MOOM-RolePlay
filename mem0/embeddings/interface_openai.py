import os
from typing import Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

import requests


class Interface_OpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.jump_server_url = "http://10.4.148.51:13334/get_style_embedding"

    def embed(self, text):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        config_dict = {
            "model": self.config.model,
            "api_key": self.config.api_key,
            "openai_base_url": self.config.openai_base_url,
        }

        data = {
            "config": config_dict,
            "text": text
        }

        try:
            response = requests.post(self.jump_server_url, json=data)

            # 检查响应状态
            if response.status_code == 200:
                # 返回 OpenAI 的结果
                response_c = response.json()
                # print(response_c)
                # print(type(response_c))  # list
                return response_c
            else:
                # 处理错误响应
                raise Exception(f"Error from jump server: {response.status_code} - {response.text}")

        except Exception as e:
            raise Exception(f"Failed to call jump server: {str(e)}")
