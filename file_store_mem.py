import json
import requests
import json
import time
from utils.dialogue_processor import OriginJsonlDialogueReader, OriginDialogueProcessor

class JsonlProcessor:
    def __init__(self, filename):
        # Initialize the processor with the jsonl filename
        self.filename = filename
        self.data = []
        self.load_data()

    def load_data(self):
        # Load json data from the file
        with open(self.filename, 'r', encoding='utf-8') as file:
            for line in file:
                self.data.append(json.loads(line))
    
    def get_line_data(self, line_number):
        # Return the data at the specified line number
        return self.data[line_number]
    
    def split_messages(self, line_number, r):
        # Split messages in the specified line into chunks of length r
        original_data = self.get_line_data(line_number)
        messages = original_data["messages"]
        message_chunks = [messages[i:i + r] for i in range(0, len(messages), r)]
        
        # Create new entries based on the original data but with split messages
        split_data = []
        for chunk in message_chunks:
            new_entry = original_data.copy()
            new_entry["messages"] = chunk
            split_data.append(new_entry)
        
        return split_data
    
def store_memory(session_id, messages):
    api_url = "http://127.0.0.1:8231/store-memory/"
    # 构建发送的数据，符合 DialogueInput 的定义
    data = {
        "session_id": session_id,
        "messages": messages
    }

    try:
        # 发送 POST 请求，将数据编码为 JSON
        response = requests.post(url=api_url, json=data)
        if response.status_code == 200:
            # print('Respzonse content:', response.text)  # 打印响应文本
            return response.json()  # 如果响应是 JSON，直接返回解析后的字典
        else:
            print("Failed to store data, status code:", response.status_code)
            return None
    except Exception as e:
        print("Exception occurred:", str(e))
        return None
    
def store_mem0(session_id, messages):
    api_url = "http://127.0.0.1:8231/store-mem0/"
    # 构建发送的数据，符合 DialogueInput 的定义
    data = {
        "session_id": session_id,
        "messages": messages
    }

    try:
        # 发送 POST 请求，将数据编码为 JSON
        response = requests.post(url=api_url, json=data)
        if response.status_code == 200:
            # print('Respzonse content:', response.text)  # 打印响应文本
            return response.json()  # 如果响应是 JSON，直接返回解析后的字典
        else:
            print("Failed to store data, status code:", response.status_code)
            return None
    except Exception as e:
        print("Exception occurred:", str(e))
        return None

def store_memorybank(session_id, messages):
    api_url = "http://127.0.0.1:8231/store-memorybank/"
    # 构建发送的数据，符合 DialogueInput 的定义
    data = {
        "session_id": session_id,
        "messages": messages
    }

    try:
        # 发送 POST 请求，将数据编码为 JSON
        response = requests.post(url=api_url, json=data)
        if response.status_code == 200:
            # print('Respzonse content:', response.text)  # 打印响应文本
            return response.json()  # 如果响应是 JSON，直接返回解析后的字典
        else:
            print("Failed to store data, status code:", response.status_code)
            return None
    except Exception as e:
        print("Exception occurred:", str(e))
        return None

def store_memochat(session_id, messages):
    api_url = "http://127.0.0.1:8231/store-memochat/"
    # 构建发送的数据，符合 DialogueInput 的定义
    data = {
        "session_id": session_id,
        "messages": messages
    }

    try:
        # 发送 POST 请求，将数据编码为 JSON
        response = requests.post(url=api_url, json=data)
        if response.status_code == 200:
            # print('Respzonse content:', response.text)  # 打印响应文本
            return response.json()  # 如果响应是 JSON，直接返回解析后的字典
        else:
            print("Failed to store data, status code:", response.status_code)
            return None
    except Exception as e:
        print("Exception occurred:", str(e))
        return None

def get_memory(session_id):
    api_url = "http://10.198.34.66:8231/retrieve-memory/"
    params = {'session_id': session_id}  # 将 session_id 设置为查询参数
    retry_times = 1  # 设置重试次数，比如重试3次

    while retry_times > 0:
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=params, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                return response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
                retry_times -= 1
        except Exception as e:
            print("Exception occurred:", str(e))
            retry_times -= 1

    return "Error occurred in get_memory"  # 所有尝试失败后返回错误消息

def get_rag(session_id, dialogue, top_k, round, importance_freeze=False):
    api_url = "http://10.198.34.66:8231/rag-memory/"
    params = {'session_id': session_id, 'dialogue':dialogue, 'top_k':top_k, 'round':round, 'importance_freeze': importance_freeze}  
    retry_times = 1  # 设置重试次数，比如重试3次

    while retry_times > 0:
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=params, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                return response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
                retry_times -= 1
        except Exception as e:
            print("Exception occurred:", str(e))
            retry_times -= 1

    return "Error occurred in get_memory"  # 所有尝试失败后返回错误消息

# rag_summary(session_id: int, dialogue:str, top_k: int, manager: MemoryManager = Depends(get_memory_manager))
def get_sum_rag(session_id, dialogue, top_k, round, importance_freeze=False):
    api_url = "http://10.198.34.66:8231/rag-summary/"
    params = {'session_id': session_id, 'dialogue':dialogue, 'top_k':top_k, 'round':round, 'importance_freeze': importance_freeze}  
    retry_times = 1  # 设置重试次数，比如重试3次

    while retry_times > 0:
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=params, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                return response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
                retry_times -= 1
        except Exception as e:
            print("Exception occurred:", str(e))
            retry_times -= 1

    return "Error occurred in get_memory"  # 所有尝试失败后返回错误消息

def store_summary(session_id, messages):
    api_url = "http://127.0.0.1:8231/store-summary/"
    # 构建发送的数据，符合 DialogueInput 的定义
    data = {
        "session_id": session_id,
        "messages": messages
    }
    try:
        # 发送 POST 请求，将数据编码为 JSON
        response = requests.post(url=api_url, json=data)
        # print(response)
        if response.status_code == 200:
            return response.json()  # 如果响应是 JSON，直接返回解析后的字典
        else:
            print("Failed to store summary, status code:", response.status_code)
            return None
    except Exception as e:
        print("Exception occurred:", str(e))
        return None

def get_rag_mem0(session_id, dialogue, top_k, round):
    api_url = "http://10.198.34.66:8231/rag-mem0/"
    params = {'session_id': session_id, 'dialogue':dialogue, 'top_k':top_k, 'round':round}  
    retry_times = 1  # 设置重试次数，比如重试3次

    while retry_times > 0:
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=params, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                return response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
                retry_times -= 1
        except Exception as e:
            print("Exception occurred:", str(e))
            retry_times -= 1

    return "Error occurred in get_memory"  # 所有尝试失败后返回错误消息

def get_retrieve_mem0(session_id, limit=100):
    api_url = "http://10.198.34.66:8231/retrieve-mem0/"
    params = {'session_id': session_id, 'limit':limit}  
    retry_times = 1  # 设置重试次数，比如重试3次

    while retry_times > 0:
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=params, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                return response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
                retry_times -= 1
        except Exception as e:
            print("Exception occurred:", str(e))
            retry_times -= 1

    return "Error occurred in get_memory"  # 所有尝试失败后返回错误消息

def delete_retrieve_mem0(session_id):
    api_url = "http://10.198.34.66:8231/delete-mem0/"
    params = {'session_id': session_id}  
    retry_times = 1  # 设置重试次数，比如重试3次

    while retry_times > 0:
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=params, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                return response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
                retry_times -= 1
        except Exception as e:
            print("Exception occurred:", str(e))
            retry_times -= 1

    return "Error occurred in get_memory"  # 所有尝试失败后返回错误消息

def get_rag_memorybank(session_id, dialogue, top_k, round):
    api_url = "http://10.198.34.66:8231/rag-memorybank/"
    params = {'session_id': session_id, 'dialogue':dialogue, 'top_k':top_k, 'round':round}  
    retry_times = 1  # 设置重试次数，比如重试3次

    while retry_times > 0:
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=params, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                return response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
                retry_times -= 1
        except Exception as e:
            print("Exception occurred:", str(e))
            retry_times -= 1

    return "Error occurred in get_memory"  # 所有尝试失败后返回错误消息

def get_rag_memochat(session_id, dialogue, top_k, round):
    api_url = "http://10.198.34.66:8231/rag-memochat/"
    params = {'session_id': session_id, 'dialogue':dialogue, 'top_k':top_k, 'round':round}  
    retry_times = 1  # 设置重试次数，比如重试3次

    while retry_times > 0:
        try:
            # 使用 GET 方法发送请求，并通过 params 参数传递 session_id
            response = requests.get(url=api_url, params=params, verify=False)
            if response.status_code == 200:
                # print('Response content:', response.text)  # 打印响应文本
                return response.json()  # 如果响应是 JSON，直接返回解析后的字典
            else:
                print("Failed to retrieve data, status code:", response.status_code)
                retry_times -= 1
        except Exception as e:
            print("Exception occurred:", str(e))
            retry_times -= 1

    return "Error occurred in get_memory"  # 所有尝试失败后返回错误消息

def store_memory_loop(session_id, dialogue_id=0, file_name='x.jsonl', split_num=20, mode='importance_data'):
    processor = JsonlProcessor(file_name)
    split_data = processor.split_messages(dialogue_id, split_num)  

    # processor2 = OriginDialogueProcessor(split_data[0])
    # dia = processor2.split_by_turns(20)


    for r, d in enumerate(split_data):
        user_name = d['role_meta']['user_name']
        bot_name = d['role_meta']['primary_bot_name']
        
        mes = None
        for m in d['messages']:
            if m['sender_name'] == user_name:
               mes = m['text']
               break
        
        if mode == 'importance_data':
            if mes is None:
                store_memory(session_id, d)
                store_summary(session_id, d)     
            else:
                get_rag(session_id, mes, 3, r)
                x = get_sum_rag(session_id, mes, 3, r)
                print('the result of summary rag')
                print(x)

                store_memory(session_id, d)
                store_summary(session_id, d)
        elif mode == 'memorybank':
            store_memorybank(session_id, d)
        elif mode == 'mem0':  
            store_mem0(session_id, d)
        elif mode == 'memochat':
            store_memochat(session_id, d)

        print(f'----{mode} 当前的轮数与信息----') 
        print("session_id", session_id, "round", r)
        print(mes)

if __name__ == "__main__":
    file_path = 'data/ZH-4O_dataset.jsonl'
    for i in range(1, 29):
        store_memory_loop(session_id=i, dialogue_id=i, file_name=file_path, split_num=20, mode='importance_data')
 