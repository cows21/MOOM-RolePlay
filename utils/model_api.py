import requests
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
        # model='qwen_32b',
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

def store_kv_memory(session_id, messages):
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

def get_kv_rag(session_id, dialogue, top_k, round, importance_freeze=False):
    api_url = "http://10.198.34.66:8231/rag-memory/"
    params = {'session_id': session_id, 'dialogue':dialogue, 'top_k':top_k, 'round':round, 'importance_freeze':importance_freeze}  
    # params = {'session_id': session_id, 'dialogue':dialogue, 'top_k':top_k, 'round':round} 
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

def get_sum_rag(session_id, dialogue, top_k, round, importance_freeze=False):
    api_url = "http://10.198.34.66:8231/rag-summary/"
    params = {'session_id': session_id, 'dialogue':dialogue, 'top_k':top_k, 'round':round, 'importance_freeze':importance_freeze}  
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