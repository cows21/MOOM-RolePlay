import json
from .embedding import BGEEmbedding, BaseEmbeddings
from typing import List
from .llm_api import LLMAPI
from .vectorDB import MemoryVectorDB
import time
import tqdm

def embedding_load():
    model_path = '/nvme/lisongling/models/bge-base-zh-v1.5'
    bge = BGEEmbedding(model_path)
    return bge

def llm_api_load(model, tokenzier, api_url='http://10.142.5.99:8012'  + '/v1', use_api=False):
    return LLMAPI(model=model, tokenzier=tokenzier, api_url=api_url, use_api=use_api)

def vectordb_load(vector_paths: List[str],
        content_paths: List[str],
        chunk_sizes: List[int], 
        session_id: int,
        emb_model:BaseEmbeddings,
        llm_api:LLMAPI):
    return MemoryVectorDB(vector_paths, content_paths, chunk_sizes, session_id, emb_model, llm_api)

def memory_load(dialogue_path, chunk_sizes=[6,5,5]):
    with open(dialogue_path, 'r') as f:
        dialogues = json.load(f)
    # vector_paths = [
    #     '/home/lisongling/sensetime_character_beta/long_term_memory/test/test_vec_1.pkl',
    #     '/home/lisongling/sensetime_character_beta/long_term_memory/test/test_vec_2.pkl',
    #     '/home/lisongling/sensetime_character_beta/long_term_memory/test/test_vec_3.pkl',
    # ]
    # content_paths = [
    #     '/home/lisongling/sensetime_character_beta/long_term_memory/test/test_text_1.json',
    #     '/home/lisongling/sensetime_character_beta/long_term_memory/test/test_text_2.json',
    #     '/home/lisongling/sensetime_character_beta/long_term_memory/test/test_text_3.json',
    # ]

    vector_paths = [
        '/nvme/chenweishu/code2/lsl_summary/UI/long_term_memory/test/test_vec_1.pkl',
        '/nvme/chenweishu/code2/lsl_summary/UI/long_term_memory/test/test_vec_2.pkl',
        '/nvme/chenweishu/code2/lsl_summary/UI/long_term_memory/test/test_vec_3.pkl',
    ]
    content_paths = [
        '/nvme/chenweishu/code2/lsl_summary/UI/long_term_memory/test/test_text_1.json',
        '/nvme/chenweishu/code2/lsl_summary/UI/long_term_memory/test/test_text_2.json',
        '/nvme/chenweishu/code2/lsl_summary/UI/long_term_memory/test/test_text_3.json',
    ]

    names = list(dialogues['prompt'].keys())
    embed_model = embedding_load()
    llm_api = llm_api_load()
    vector_db = vectordb_load(vector_paths, content_paths, chunk_sizes, names[0], names[1], embed_model, llm_api)

    
    for message in tqdm.tqdm(dialogues["messages"], desc='memory loading'):
        text = message['sender_name'] + '：' + message['text']
        now_time = int(time.time())
        vector_db.put(text, now_time)
    
    vector_db.__memory_write__()
    return vector_db
    
