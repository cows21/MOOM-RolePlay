import os
import json
import requests
from typing import Dict, List, Optional

from mem0.llms.base import LLMBase
from mem0.configs.llms.base import BaseLlmConfig


def get_style_chat(request_body):
    url = "http://10.4.148.51:12225/llm_create"
    retry_times = 5
    while retry_times > 0:
        try:
            print('in get_style_chat')
            response = requests.post(url=url, json=request_body, verify=False)
            print('response.status_code:', response.status_code)
            if response.status_code == 200:
                print('response.content', json.loads(response.content))
                return json.loads(response.content)
            else:
                retry_times -= 1
        except Exception as e:
            retry_times -= 1
            print(e, 'in get_style_chat')
    return 'error happened'


class Interface_OpenAI(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-mini"

    def _parse_response(self, response, tools):
        """s
        处理响应，基于是否使用了工具。
        """
        if tools:
            processed_response = {
                "content": response["choices"][0]["message"]["content"],
                "tool_calls": [],
            }

            if response["choices"][0]["message"].get("tool_calls"):
                for tool_call in response["choices"][0]["message"]["tool_calls"]:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"]),
                        }
                    )
            return processed_response
        else:
            return response["choices"][0]["message"]["content"]

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        基于提供的消息生成响应，使用 get_style_chat 间接调用 OpenAI。
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # 调用 get_style_chat 函数代替原来的 OpenAI API 调用
        response = get_style_chat(params)
        if response == 'error happened':
            raise Exception("Error in calling get_style_chat")
        return self._parse_response(response, tools)
