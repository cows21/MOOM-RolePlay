from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import re
import uuid
import uvicorn
import requests
import json

app = FastAPI(
    title="JSON 转换 API",
    description="将 OpenAI API 格式的 JSON 数据转换为自定义格式。",
    version="1.0.0",
)

# 定义输入的消息格式
class InputMessage(BaseModel):
    role: str
    content: str

# 定义输入的整体 JSON 结构
class OpenAIInputJSON(BaseModel):
    model: str
    messages: List[InputMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

# 定义输出的消息格式
class OutputMessage(BaseModel):
    sender_name: str
    text: str


def transform_openai_json_for_store(openai_json: OpenAIInputJSON):
    # 初始化变量
    role_meta = {}
    prompt = {}
    session_id = None
    messages = []
    
    user_name = None
    primary_bot_name = None

    for message in openai_json.messages:
        role = message.role
        content = message.content.strip()
        
        if role == "system":
            # 解析 system 内容: "s1_s2_d1"
            match = re.match(r"([^_]+)_([^_]+)_(\d+)", content)
            if match:
                s1, s2, d1 = match.groups()
                user_name = s1
                primary_bot_name = s2
                session_id = d1
                role_meta = {
                    "user_name": user_name,
                    "primary_bot_name": primary_bot_name
                }
                prompt = {
                    user_name: {},
                    primary_bot_name: {}
                }
            else:
                raise ValueError("System message content does not match the expected format 's1_s2_d1'.")
        
        elif role == "user":
            if not user_name or not primary_bot_name:
                raise ValueError("System message must be provided before user messages.")
            
            # 按行分割内容
            lines = content.split('\n')
            for line in lines:
                # 每行应为 "Sender:Message"
                parts = line.split(':', 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid message format: '{line}'. Expected 'Sender:Message'.")
                sender, text = parts
                sender = sender.strip()
                text = text.strip()
                
                # 验证发送者
                if sender not in [user_name, primary_bot_name]:
                    raise ValueError(f"Unknown sender '{sender}'. Expected '{user_name}' or '{primary_bot_name}'.")
                
                messages.append({
                    "sender_name": sender,
                    "text": text
                })
        else:
            # 忽略其他角色
            continue

    if not role_meta:
        raise ValueError("No system message found to extract role_meta.")
    
    if not messages:
        raise ValueError("No user messages found to extract conversations.")

    # 构建输出的 JSON 结构
    transformed_json = {
        'prompt': prompt,
        'role_meta': role_meta,
        'session_id': session_id,
        'messages': messages
    }
    return transformed_json

def transform_openai_json_for_rag(openai_json: OpenAIInputJSON):
    session_id = None
    for message in openai_json.messages:
        role = message.role
        content = message.content.strip()
        print('a')
        if role == "system":
            # 解析 system 内容: "s1_s2_d1"
            print('aa')
            match = re.match(r"(\d+)_(\d+)_(\d+)", content)
            print('ab')
            if match:
                d1, d2, d3 = match.groups()
                print('ac')
                print(d1, d2, d3)
                session_id = d1
                print('ad')
                topk = d2
                print('ae')
                current_round = d3
                print('af')
            else:
                raise ValueError("System message content does not match the expected format 'd1_d2_d3'.")
            print('b')
        elif role == "user":
            # 按行分割内容
            rag_dia = content
        else:
            # 忽略其他角色
            continue
    
    print('c')
    # 构建输出的 JSON 结构
    transformed_json = {
        'session_id': session_id,
        'dialogue': rag_dia,
        'top_k': topk,
        'round': current_round
    }
    print('d')
    return transformed_json

@app.post("/store/")
async def store_mem(input_json: OpenAIInputJSON):
    """
    将 OpenAI API 格式的 JSON 数据转换为自定义格式。

    - **input_json**: OpenAI API 格式的 JSON 数据
    """
    try:       
        transformed = transform_openai_json_for_store(input_json)
        data = {
            "session_id": transformed['session_id'],
            "messages": transformed
        }

        store_memory_url = 'http://127.0.0.1:8231/store-memory/'
        kv_mem = None
        try:
            # 发送 POST 请求，将数据编码为 JSON
            response = requests.post(url=store_memory_url, json=data)
            if response.status_code == 200:
                # print('Respzonse content:', response.text)  # 打印响应文本
                kv_mem = response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to store data when store kv, status code:", response.status_code)
                return None
        except Exception as e:
            print("Exception occurred:", str(e))
            return None   
        
        store_summmary_url = "http://127.0.0.1:8231/store-summary/"

        summary_mem = None
        # 构建发送的数据，符合 DialogueInput 的定义
        try:
            # 发送 POST 请求，将数据编码为 JSON
            response = requests.post(url=store_summmary_url, json=data)
            # print(response)
            if response.status_code == 200:
                summary_mem = response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to store summary, status code:", response.status_code)
                return None
        except Exception as e:
            print("Exception occurred:", str(e))
            return None
        
        return [kv_mem, summary_mem]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag/")
async def rag_mem(input_json: OpenAIInputJSON):
    try:
        # transformed_json = {
        #     'session_id': session_id,
        #     'dialogue': rag_dia,
        #     'top_k': topk,
        #     'round': current_round
        # }
        print(0)
        rag_memory_url = "http://127.0.0.1:8231/rag-memory/"
        transformed_json = transform_openai_json_for_rag(input_json)
        kv_mem = None
        print(1)
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=rag_memory_url, params=transformed_json, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                kv_mem = response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
        except Exception as e:
            print("Exception occurred:", str(e))
        
        api_url = "http://127.0.0.1:8231/rag-summary/"
        sum_mem = None
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=transformed_json, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                sum_mem = response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
        except Exception as e:
            print("Exception occurred:", str(e))

        print(kv_mem)
        print(sum_mem)

        rag_result = {
            "kv": kv_mem,
            "summary_1": sum_mem[0],
            "summary_2": sum_mem[1],
            "summary_3": sum_mem[2]
        }

        rag_str = json.dumps(rag_result, ensure_ascii=False)

        choice_item = {
            "index": 0,
            "messages":{
                "role": "assistant",
                "content": rag_str
            }
        }

        openai_result = {
            "id": transformed_json['session_id'],
            "choices":[choice_item]
        }

        return openai_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # 运行应用
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6459, help='display an integer')
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
