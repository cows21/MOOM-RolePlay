import os
import re
import json
import requests
from typing import Dict, List, Optional
from mem0.llms.base import LLMBase
from mem0.configs.llms.base import BaseLlmConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

def chat_once(messages, ip_url):
    openai_api_key = "EMPTY"
    openai_api_base = ip_url
    # 假设text有多条,需要遍历/拼接
    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    )
    response = client.chat.completions.create(
        #model="gpt-4-1106-preview",  # 指定使用 gpt-4 模型
        model='qwen',
        #model='gpt-3.5-turbo',
        messages=messages,
        temperature = 1.0,
        timeout=600
    )

    ans = response.choices[0].message.content

    return ans

def lsl_qwen_70b(prompt):
    try:
        prompt = prompt
        ip_url = 'http://10.142.5.99:8012' + '/v1'
        messages = []
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})

        try:
            re_polish_content = chat_once(messages, ip_url)
            return re_polish_content
        except Exception as e:
            print(f"{e}: {ip_url}")
            return None
        
    except Exception as e:
        pass



class Qwen72B_api(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None, local_model=None):
        super().__init__(config)
        if not self.config.model:
            self.config.model = "gpt-4o-mini"
        self.jump_server_url = "http://10.4.148.51:12225/get_style_chat"
        self.local_model = local_model
    
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
            response_format = None  # 或者设置为一个默认字符串值，比如 'json']

        response = lsl_qwen_70b(messages[0]['content'])
        print(messages[0]['content'])
        print("in mem0 qwen72b api")
        print(response)
        # print("在mem0-Qwen2 里试用qwen2，以下是response")
        # print(response)
        return response