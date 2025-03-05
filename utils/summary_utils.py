from memory_summary.dialogue_init import memory_load, embedding_load, llm_api_load, vectordb_load
from memory_summary.embedding import BGEEmbedding
import json
import os
import traceback
import sys
import time

# 用于summary的data_loader, by lsl
def dialogue_load(dialogue_path, test_sample):
    if not dialogue_path:
        return test_sample
    #初始化导入已有的对话数据进去
    with open(dialogue_path, 'r') as f:
        dialogue = json.load(f)
        for message in dialogue['messages']:
            test_sample['messages'].append(
                {"role": message['sender_name'], "content": message['text']}
            )
    return test_sample

# 用于memory初始化, by lsl（在原代码中，用了@st.cache_resource修饰，缓存长期存在的
# 资源，这里暂时做不到，需要每调用一次接口就重新加载一次数据）
def memory_init(session_id, _embed_model, _llm_api, store_dir, chunk_sizes = [6,5,5]):

    print('--------------this is in memory_init-----------------')
    print('summary store dir', store_dir)

    vector_paths = [
        f'{session_id}_vec_1.pkl',
        f'{session_id}_vec_2.pkl',
        f'{session_id}_vec_3.pkl',
    ]

    content_paths = [
        f'{session_id}_text_1.json',
        f'{session_id}_text_2.json',
        f'{session_id}_text_3.json',
    ]

    vector_paths = [os.path.join(store_dir, path) for path in vector_paths]
    content_paths = [os.path.join(store_dir, path) for path in content_paths]

    #embed_model = embedding_load()
    #llm_api = llm_api_load()
    vector_db = vectordb_load(vector_paths, content_paths, chunk_sizes, session_id, _embed_model, _llm_api)

    return vector_db

# 用于summary的bge加载, by lsl
def embedding_load():
    model_path = '/nvme/lisongling/models/bge-base-zh-v1.5'
    bge = BGEEmbedding(model_path)
    return bge

def chat_once(messages, select_bot = '', vector_db = None):
    #这里加记忆
    query = messages[-1]["role"] + '：' + messages[-1]["content"]
    mem1, mem2, mem3 = vector_db.query(query, 2, [0.6, 0.4 ,0.2])
    select_mems = []
    select_mems.extend([mem['content'] for mem in mem1])
    select_mems.extend([mem['content'] for mem in mem2[: 2]])
    select_mems.extend([mem['content'] for mem in mem3[: 1]])
    
    total_mems = '\n以下是一些参考信息，包含了这两个人的部分历史对话以及部分对话的概括，你可以参考其中的信息，选择相关有用的部分来回答新的问题：' + '\n'.join(select_mems)
    
    # test_sample["prompt"][select_bot]["补充设定"] += total_mems
        
    return messages

def chat_once_retry(messages, mode, select_bot, vector_db):
    try:
        messages = chat_once(messages, mode, select_bot, vector_db)
        return messages
    except Exception as e:
        traceback.print_exc()
        print(f'===({e})===')
        sys.stdout.flush()
        time.sleep(1)
        return chat_once_retry(messages, mode, select_bot, vector_db)
    