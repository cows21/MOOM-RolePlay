from app_module import MemoryManager, DialogueInput
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from FlagEmbedding import BGEM3FlagModel, FlagReranker 
from sentence_transformers import SentenceTransformer
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from memory_summary.dialogue_init import memory_load, embedding_load, llm_api_load, vectordb_load

os.environ['OPENAI_API_KEY'] = "xxx"

mem0_config = {
    "llm": {
        "provider": "qwen2",
        "config": {
            "model": "qwen",
            "temperature": 1.0
        }
    },
    "embedder": {
        "provider": "text2vec_large_chinese",
        "config": {
            "model": "text-embedding-3-large"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "embedding_model_dims": 1024,  # 3072
            "path": 'mem0_pool',
            "on_disk": True
        }
    },
    "version": "v1.1",
}

app = FastAPI()

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

# 模型和分词器的初始化
model_id = 'model/qwen2-0621-7B-LTM-FT/checkpoint-800'
model_mem = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True)
tokenizer_mem = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model_id_combine = 'model/Qwen1.5-14B-Chat'
model_combine = AutoModelForCausalLM.from_pretrained(model_id_combine, torch_dtype="auto", device_map="auto", trust_remote_code=True)
tokenizer_combine = AutoTokenizer.from_pretrained(model_id_combine, trust_remote_code=True)

model_id_bge = 'model/FlagEmbedding/BAAI/bge-m3'
bge = BGEM3FlagModel(model_id_bge, use_fp16=True)

model_id_reranker = 'model/bge_reranker'
reranker = FlagReranker(model_id_reranker, use_fp16=True)

model_id_text_encoder = 'model/text2vec-large-chinese'
text_encoder = SentenceTransformer(model_id_text_encoder)

# model = {"mem":model_mem, "timestamp":model_time, "combine":model_combine}
# tokenizer = {"mem":tokenizer_mem, "timestamp":tokenizer_time, "combine":tokenizer_combine}

# 以下和summary有关
bge_summary = embedding_load()
llm_api = llm_api_load(model_combine, tokenizer_combine, use_api=False)
# llm_api = llm_api_load(model_combine, tokenizer_combine, use_api=False)

model = {"mem":model_mem, "combine":model_combine, "bge":bge, "reranker":reranker, "error":model_combine, "summary":llm_api, "bge_summary": bge_summary}
tokenizer = {"mem":tokenizer_mem, "combine":tokenizer_combine, "error":tokenizer_combine}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prompt_mem_file_path = 'prompt/memory_extract_prompt_template.txt'
with open(prompt_mem_file_path, 'r', encoding='utf-8') as prompt_file:
    prompt_mem_template = prompt_file.read()

# prompt_time_file_path = 'prompt/memory_time_stamp.txt'
# with open(prompt_time_file_path, 'r', encoding='utf-8') as prompt_file:
#     prompt_time_template = prompt_file.read()

prompt_combine_file_path = 'model/prompt/memory_combine.txt'
with open(prompt_combine_file_path, 'r', encoding='utf-8') as prompt_file:
    prompt_combine_template = prompt_file.read()

prompt_error_file_path = 'prompt/error_process.txt'
with open(prompt_error_file_path, 'r', encoding='utf-8') as prompt_file:
    prompt_error_template = prompt_file.read()

prompt_template ={"mem":prompt_mem_template, "combine":prompt_combine_template, "error":prompt_error_template}

mem0_local_model = {
    "model": model_combine,
    "tokenizer": tokenizer_combine,
    "device": device,
    "text_encoder": text_encoder
}


experiments_name = 'dec27th_cap9999_add_forprobe_allapi_false'

mem0_config["vector_store"]["config"]["path"] = os.path.join(mem0_config["vector_store"]["config"]["path"], experiments_name)
    
# 定义一个依赖函数，用于获取 MemoryManager 实例
def get_memory_manager():
    return MemoryManager(model=model, tokenizer=tokenizer, device=device, prompt_template=prompt_template, 
                         mem_save_folder = './memory_pool', sum_save_folder = './summary_mem_store', 
                         importance_save_folder = './importance_pool', run_importance=True, grade_update_mode = 'add',
                         mem_store_capacity=9999, mem0_config=mem0_config, mem0_local_model=mem0_local_model, 
                         experiments_name=experiments_name, memochat_use_api=False, kv_use_api=False, membank_use_api=False, summary_chunk_sizes=[6,5,5])

@app.post("/store-memory/")
async def store_memory_api(input_data: DialogueInput, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.store_memory(input_data)

@app.get("/retrieve-memory/")
async def retrieve_memory_api(session_id: int, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.retrieve_memory(session_id)

@app.get("/rag-memory/")
async def rag_memory(session_id: int, dialogue:str, top_k: int, round: int, importance_freeze: bool = False, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.rag_memory(session_id=session_id, dialogue=dialogue, top_k=top_k, round=round, importance_freeze=importance_freeze)

@app.post("/store-summary/")
async def store_summary(input_data: DialogueInput, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.store_summary(input_data)

@app.get("/rag-summary/")
async def rag_summary(session_id: int, dialogue:str, top_k: int, round:int, importance_freeze: bool = False, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.rag_summary(session_id=session_id, dialogue=dialogue, top_k=top_k, round=round, importance_freeze=importance_freeze)

@app.post("/store-mem0-kv/")
async def store_memor0_kv_api(input_data: DialogueInput, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.store_mem0_kv(input_data)

@app.post("/store-mem0/")
async def store_memor0_api(input_data: DialogueInput, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.store_mem0(input_data)

@app.get("/rag-mem0/")
async def rag_mem0(session_id: int, dialogue:str, top_k: int, round:int, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.rag_mem0(session_id=session_id, dialogue=dialogue, top_k=top_k, round=round)

@app.get("/retrieve-mem0/")
async def retrieve_mem0(session_id: int, limit:int, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.retrieve_mem0(session_id=session_id, limit=limit)

@app.get("/delete-mem0/")
async def delete_mem0(session_id: int, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.delete_mem0(session_id=session_id)

@app.post("/store-memorybank/")
async def store_memorybank(input_data: DialogueInput, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.store_memorybank(input_data)

@app.get("/rag-memorybank/")
async def rag_memorybank(session_id: int, dialogue:str, top_k: int, round: int, importance_freeze: bool = False, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.rag_memorybank(session_id=session_id, dialogue=dialogue, top_k=top_k, round=round, importance_freeze=importance_freeze)

@app.post("/store-memochat/")
async def store_memochat(input_data: DialogueInput, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.store_memochat(input_data)

@app.get("/rag-memochat/")
async def rag_memochat(session_id: int, dialogue:str, top_k: int, round: int, importance_freeze: bool = False, manager: MemoryManager = Depends(get_memory_manager)):
    return await manager.rag_memochat(session_id=session_id, dialogue=dialogue, top_k=top_k, round=round, importance_freeze=importance_freeze)