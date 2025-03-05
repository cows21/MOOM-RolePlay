import os
import re
import json
import requests
from typing import Dict, List, Optional
from mem0.llms.base import LLMBase
from mem0.configs.llms.base import BaseLlmConfig

def remove_code_block_tags(text):
    # 使用正则表达式匹配并移除 ```json 和 ``` 
    pattern = r'```json\s*([\s\S]*?)\s*```'
    cleaned_text = re.sub(pattern, r'\1', text)
    return cleaned_text

# 以下函数暂时还没用
def convert_to_openai_format(data):
    # 初始化标准结构
    converted_data = {
        'model': 'gpt-4o-mini',
        'messages': [],
        'temperature': 0.7,
        'max_tokens': 256,
        'top_p': 1
    }

    # 处理config部分
    if 'config' in data:
        config = data['config']
        converted_data['model'] = config.get('model', 'gpt-4o-mini')  # 默认使用gpt-4
        converted_data['temperature'] = config.get('temperature', 0.7)
        converted_data['max_tokens'] = config.get('max_tokens', 256)
        converted_data['top_p'] = config.get('top_p', 1)

    # 处理messages部分
    if 'messages' in data:
        for message in data['messages']:
            role = message.get('role')
            content = message.get('content', '')

            # 移除不必要的$符号
            content = re.sub(r'\$', '', content)
            
            # 将消息加入标准格式
            converted_data['messages'].append({'role': role, 'content': content})
    
    if 'tools' in data:
        converted_data['tools'] = data['tools']

    if 'tool_choice' in data:
        converted_data['tool_choice'] = data['tool_choice'] 
    
    return converted_data


class Interface_OpenAI(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)
        if not self.config.model:
            self.config.model = "gpt-4o-mini"
        self.jump_server_url = "http://10.4.148.54:22225/llm_create"
        # self.jump_server_url = "http://10.4.148.51:13334/get_style_chat"
        
    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        print('this is in interface openai response')
        print(response)
        response = response['response']
        print(response["choices"][0]["message"]["content"], type(response["choices"][0]["message"]["content"]))
        if tools:
            processed_response = {
                # "content": remove_code_block_tags(response["choices"][0]["message"]["content"]),
                "content": response["choices"][0]["message"]["content"],
                "tool_calls": [],
            }
            if response["choices"][0]["message"]["tool_calls"]:
                for tool_call in response["choices"][0]["message"]["tool_calls"]:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"]),
                        }
                    )

            return processed_response
        else:
            # x_str = remove_code_block_tags(response["choices"][0]["message"]["content"])
            x_str = response["choices"][0]["message"]["content"]
            return x_str
            # return response.choices[0].message.content
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        config_dict = {
            "model": self.config.model,
            "openrouter_base_url": self.config.openrouter_base_url,
            "api_key": self.config.api_key,
            "openai_base_url": self.config.openai_base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "route": self.config.route,
            "site_url": self.config.site_url,
            "app_name": self.config.app_name,
        }

        if not isinstance(response_format, str):
            response_format = None  # 或者设置为一个默认字符串值，比如 'json'


        # 组装要发送给跳板机的请求数据
        data = {
            "config": config_dict,
            "messages": messages,
            "response_format": response_format,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        data = convert_to_openai_format(data)
        print("以下是发给openai的数据")
        print(data)

        # 调用跳板机上的 /get_style_chat API
        try:
            response = requests.post(self.jump_server_url, json=data)

            # 检查响应状态
            if response.status_code == 200:
                # 返回 OpenAI 的结果
                # print(type(response))
                x = response.json()
                return self._parse_response(response.json(), tools)
            else:
                # 处理错误响应
                raise Exception(f"Error from jump server: {response.status_code} - {response.text}")

        except Exception as e:
            raise Exception(f"Failed to call jump server: {str(e)}")