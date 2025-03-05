import os
import re
import json
from typing import Dict, List, Optional

from openai import OpenAI

from mem0.llms.base import LLMBase
from mem0.configs.llms.base import BaseLlmConfig

def extract_json_from_text(text):
    # 匹配以 { 开始，以 } 结束的整个 JSON 子字符串，允许跨行内容
    json_pattern = r'\{[\s\S]*\}'
    json_matches = re.findall(json_pattern, text)

    extracted_jsons = []

    for match in json_matches:
        try:
            # 尝试将匹配的字符串解析为 JSON
            json_obj = json.loads(match)
            extracted_jsons.append(json_obj)
        except json.JSONDecodeError:
            # 忽略无法解析为 JSON 的部分
            continue

    # 返回最后一个匹配到的 JSON 对象
    return extracted_jsons[-1] if extracted_jsons else None

def extract_last_json_string(text):
    # 使用正则表达式匹配所有夹在```json与```之间的字符串
    matches = re.findall(r'```json(.*?)```', text, re.DOTALL)
    # 返回最后一个匹配的字符串，如果没有匹配则返回空字符串
    return matches[-1] if matches else ""


class Qwen72B_api(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "qwen"

        ip_url = 'http://10.142.5.99:8012' + '/v1'
        openai_api_key = "EMPTY"
        openai_api_base = ip_url
        # 假设text有多条,需要遍历/拼接
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using OpenAI.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """

        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "timeout": 600
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        print("in mem0 72b api")
        print('params')
        print(params)
        response = self.client.chat.completions.create(**params)

        ans = self._parse_response(response, tools)
        print('this is ans!')
        print(ans)

        if '```json' in ans:
            # print('使用extract_last_json_string函数')
            ans = extract_last_json_string(ans)
        else:
            # print('使用extract_json_from_text函数')
            ans = extract_json_from_text(ans)
        print('after process')
        print(ans)
        print(isinstance(ans, str))

        return ans
