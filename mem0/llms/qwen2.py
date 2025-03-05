import os
import re
import json
import requests
from typing import Dict, List, Optional
from mem0.llms.base import LLMBase
from mem0.configs.llms.base import BaseLlmConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# def extract_json_from_text(text):
#     # 匹配最外层的 JSON 子字符串（以 { 开始，以 } 结束）
#     json_pattern = r'\{.*?\}'
#     json_matches = re.findall(json_pattern, text)
    
#     extracted_jsons = []
    
#     for match in json_matches:
#         try:
#             # 尝试将匹配的字符串解析为 JSON
#             json_obj = json.loads(match)
#             extracted_jsons.append(json_obj)
#         except json.JSONDecodeError:
#             # 忽略无法解析为 JSON 的部分
#             continue
    
#     # 返回最后一个匹配到的 JSON 对象
#     return json.dumps(extracted_jsons[-1], ensure_ascii=False) if extracted_jsons else {}

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

def remove_code_block_tags(text):
    # 使用正则表达式匹配并移除 ```json 和 ``` 
    pattern = r'```json\s*([\s\S]*?)\s*```'
    cleaned_text = re.sub(pattern, r'\1', text)
    return cleaned_text

def extract_last_json_string(text):
    # 使用正则表达式匹配所有夹在```json与```之间的字符串
    matches = re.findall(r'```json(.*?)```', text, re.DOTALL)
    # 返回最后一个匹配的字符串，如果没有匹配则返回空字符串
    return matches[-1] if matches else ""

def model_process(model, tokenizer, device, messages, max_new_tokens=1024):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print('in qwen2 mem0 model_process')
    print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.split('assistant')[-1]
    # print('提取json前')
    # print(response)

    if '```json' in response:
        # print('使用extract_last_json_string函数')
        response = extract_last_json_string(response)
    else:
        # print('使用extract_json_from_text函数')
        response = extract_json_from_text(response)
    # print('提取json后')
    # print(response)

    return response



class Qwen2(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None, local_model=None):
        super().__init__(config)
        if not self.config.model:
            self.config.model = "gpt-4o-mini"
        self.jump_server_url = "http://10.4.148.51:12225/get_style_chat"
        self.local_model = local_model
        
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
                "content": remove_code_block_tags(response["choices"][0]["message"]["content"]),
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
            x_str = remove_code_block_tags(response["choices"][0]["message"]["content"])
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

        model = self.local_model["model"]
        tokenizer = self.local_model["tokenizer"]
        device = self.local_model["device"]

        response = model_process(model=model, tokenizer=tokenizer, device=device, messages=messages)
        print("在mem0-Qwen2 里试用qwen2，以下是response")
        print(response)
        return response

        # 调用跳板机上的 /get_style_chat API
        # try:
        #     response = requests.post(self.jump_server_url, json=data)

        #     # 检查响应状态
        #     if response.status_code == 200:
        #         # 返回 OpenAI 的结果
        #         # print(type(response))
        #         x = response.json()
        #         return self._parse_response(response.json(), tools)
        #     else:
        #         # 处理错误响应
        #         raise Exception(f"Error from jump server: {response.status_code} - {response.text}")

        # except Exception as e:
        #     raise Exception(f"Failed to call jump server: {str(e)}")