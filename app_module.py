from utils.dialogue_processor import OriginJsonlDialogueReader, OriginDialogueProcessor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from utils.json_extract import clean_json_string, time_extract, parse_to_json, clean_combined_json, time_standard
from utils.json_extract import memory_time_standard, split_comma_separated_values, extract_json_block
from utils.prompt_processor import build_memory_prompt_cws, build_time_prompt_cws, build_combine_prompt_cws, build_error_prompt_cws
from utils.memory_combine_time import memory_update_through_time
from utils.rag import RetrievalAugmentedGeneration
from utils.standarlize_value import MemoryStandardizer
from utils.summary_utils import dialogue_load, memory_init
from utils.forget_mechanism import ImportanceHandler
from utils.model_api import lsl_qwen_70b
from openai import OpenAI
from memory_summary.dialogue_init import memory_load, embedding_load, llm_api_load, vectordb_load
from memory_summary.embedding import BGEEmbedding
from memorybank.membank_summary_prompt import summarize_content_prompt, summarize_overall_prompt, summarize_overall_personality, summarize_person_prompt
from memochat.memo_infer_chinese import convert_jsonl_to_json, get_model_answers
import random
import logging
from utils.memory_combine_new import MemoryCombineManager
import copy
import tqdm
import time

from mem0 import Memory
from openai import OpenAI

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json
import os
import re

import asyncio

os.environ['OPENAI_API_KEY'] = 'default'

def init(l):
    global lock
    lock = l

def model_process(model, tokenizer, device, prompt, max_new_tokens=1024):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# def setup_logging(output_file_path):
#     # 创建一个logger
#     logger = logging.getLogger('memory_dialogue_logger')
#     logger.setLevel(logging.DEBUG)  # 设置日志记录级别

#     # 清除已存在的handlers，避免重复记录
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # 创建一个handler，用于将日志写入文件
#     fh = logging.FileHandler(output_file_path, encoding='utf-8')
#     fh.setLevel(logging.DEBUG)

#     # 创建一个handler，用于将日志输出到控制台
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)

#     # 定义handler的输出格式
#     # formatter = logging.Formatter('%(asctime)s - %(levelname)s \n%(message)s')
#     formatter = logging.Formatter('%(message)s')
#     fh.setFormatter(formatter)
#     ch.setFormatter(formatter)

#     # 给logger添加handler
#     logger.addHandler(fh)
#     logger.addHandler(ch)

#     return logger

def setup_logging(output_file_path):
    # 创建一个logger
    logger = logging.getLogger('memory_dialogue_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志记录级别

    # 清除已存在的handlers，避免重复记录
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个handler，用于将日志写入文件
    fh = logging.FileHandler(output_file_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # 创建一个handler，用于将日志输出到控制台
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s \n%(message)s')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    # logger.addHandler(ch)

    return logger

class DialogueInput(BaseModel):
    messages: dict
    session_id: int = Field(default=0)

class MemoryManager:
    def __init__(self, model, tokenizer, device, prompt_template, mem0_config = None,
                 mem_save_folder = './memory_pool', sum_save_folder = './summary_mem_store', 
                 importance_save_folder = './importance_pool', run_importance = True, experiments_name='default', 
                 grade_update_mode = 'exp', mem_store_capacity=100, mem0_local_model=None,
                 membank_save_folder='./membank_pool', membank_use_api=False, kv_use_api=False,
                 memochat_save_folder='./memochat_pool', memochat_use_api=False,
                 split_memory_dialogue_num=20, summary_chunk_sizes=[6,5,5]):
        # 这里可以定义一些共享的变量，比如内存存储的数据结构

        self.mem_save_folder = os.path.join(mem_save_folder, experiments_name)
        os.makedirs(self.mem_save_folder, exist_ok=True)

        self.sum_save_folder = os.path.join(sum_save_folder, experiments_name)
        os.makedirs(self.sum_save_folder, exist_ok=True)

        self.importance_save_folder = os.path.join(importance_save_folder, experiments_name)
        os.makedirs(self.importance_save_folder, exist_ok=True)

        self.membank_save_folder = os.path.join(membank_save_folder, experiments_name)
        os.makedirs(self.membank_save_folder, exist_ok=True)

        self.memochat_save_folder = os.path.join(memochat_save_folder, experiments_name)
        os.makedirs(self.memochat_save_folder, exist_ok=True)
        
        self.device = device

        self.chunk_sizes = summary_chunk_sizes
        self.split_memory_dialogue_num = split_memory_dialogue_num
        self.mem_store_capacity = mem_store_capacity

        # self.summary_rag_threshold = [0.6,0.5,0.4]
        self.summary_rag_threshold = [0, 0, 0]
        # 模型和分词器的初始化
        self.model_mem = model['mem']
        self.tokenizer_mem = tokenizer['mem']

        self.model_combine = model['combine']
        self.tokenizer_combine = tokenizer['combine']

        self.model_error = model['error']
        self.tokenizer_error = tokenizer['error']

        # self.model_error = self.model_combine
        # self.tokenizer_error = self.tokenizer_combine

        # print('compare model equal')
        # print('model', model['error'] == model['combine'])
        # print('tokenizer', tokenizer['error'] == tokenizer['combine'])

        self.model_bge = model['bge']
        self.model_reranker = model['reranker']

        self.prompt_mem_template = prompt_template['mem']
        self.prompt_combine_template = prompt_template['combine']
        self.prompt_error_template =  prompt_template['error']

        self.memory_standarder = MemoryStandardizer(self.model_bge)

        self.model_summary = model['summary']
        self.model_bge_summary = model['bge_summary']
        
        if mem0_config is not None:
            # mem0_config["vector_store"]["config"]["path"] = os.path.join(mem0_config["vector_store"]["config"]["path"], experiments_name)
            self.mem0_config = mem0_config
            self.mem0_memory = Memory.from_config(config_dict=mem0_config, local_model=mem0_local_model)
        
        self.memorybank_use_api = membank_use_api
        self.kv_use_api = kv_use_api
        self.memochat_use_api = memochat_use_api
        
        self.run_importance = run_importance
        self.grade_update_mode = grade_update_mode

    async def store_memory(self, input_data: DialogueInput):
        try:
            if self.run_importance:
                importance_handeler = ImportanceHandler(session_id=input_data.session_id, importance_dir=self.importance_save_folder)
                importance_handeler.initialize_importance()
            log_output_file = os.path.join(self.mem_save_folder, f"{input_data.session_id}_output.log")
            memory_file_path = os.path.join(self.mem_save_folder, f"{input_data.session_id}.jsonl")
            # 设置日志记录
            logger = setup_logging(log_output_file)
            processor = OriginDialogueProcessor(input_data.messages)
            # 获取对话轮次信息等（根据需要选择使用）
            turns = processor.count_total_turns()
            logger.info(f"对话总轮次数：{turns}")

            role_info=  processor.get_role_info()
            user_name =  role_info[0]['user_name']
            bot_name = role_info[0]['primary_bot_name']
            user_info = role_info[1][user_name]
            bot_info = role_info[1][bot_name]
            logger.info('---------------角色信息---------------')
            logger.info(user_name)
            logger.info(user_info)
            logger.info(bot_name)
            logger.info(bot_info)

            dialogues = processor.split_by_turns(self.split_memory_dialogue_num)
            dialogue_memory_list = []
            used_infos = []
            history_mem = None
            fail_mem_extract = False

            start_round = 0
            if os.path.exists(memory_file_path):
                with open(memory_file_path, 'r', encoding='utf-8') as file:
                    dialogue_memory_list = json.load(file)
                    logger.info('--------CHECK exsting file---------')
                    if dialogue_memory_list is not None and len(dialogue_memory_list) != 0:
                        logger.info(dialogue_memory_list[-1])
                        history_mem = dialogue_memory_list[-1]['final_memory']
                        start_round = dialogue_memory_list[-1]['current_round']
                    else:
                        logger.info('dialogue_memory_list is none or empty')

            for dia_i, dia in enumerate(dialogues):
                logger.info('--------dia_i---------')
                logger.info(dia_i)

                logger.info('---------------')
                logger.info(dia)
                ## 提取当前对话的memory
                prompt_mem = build_memory_prompt_cws(user_name, bot_name, dia, self.prompt_mem_template)
                if self.kv_use_api is False:
                    response_mem = model_process(self.model_mem, self.tokenizer_mem, self.device, prompt_mem)
                else:
                    response_mem = lsl_qwen_70b(prompt_mem)
                logger.info('-----新总结的memory(raw)----------')
                logger.info(extract_json_block(response_mem))
                # print('送到json中的数据', response_mem)
                if self.kv_use_api is False:
                    cleaned_new_mem = clean_json_string(response_mem)
                else:
                    if response_mem[0:3] == '```':
                        cleaned_new_mem = clean_json_string(response_mem)
                    else:
                        response_mem2 = response_mem
                        cleaned_new_mem = json.loads(response_mem2)
                # print('before cleaned_new_mem', cleaned_new_mem)
                cleaned_new_mem = split_comma_separated_values(cleaned_new_mem)
                # print('after cleaned_new_mem', cleaned_new_mem)
                logger.info("-----新总结的memory(清洗后)------")
                logger.info(cleaned_new_mem)

                standardized_memory = self.memory_standarder.standardize_regex(cleaned_new_mem)
                logger.info("-----新总结的memory(标准化后)------")
                logger.info(standardized_memory)

                # error 部分还需修改
                # prompt_error =  build_error_prompt_cws(content=dia, prompt=self.prompt_error_template)
                # error_note = model_process(self.model_error, self.tokenizer_error, self.device, prompt_error)
                # error_note = extract_json_block(error_note)
                # logger.info("-----找到的错误信息------")
                # logger.info(error_note)

                if history_mem is not None:
                    print('history_mem is not None!!')
                    model_dict = dict()
                    tokenizer_dict = dict()
                    model_dict["inner_combine"] = self.model_combine
                    tokenizer_dict["inner_combine"] = self.tokenizer_combine
                    model_dict["bge"] = self.model_bge
                    combine_manager = MemoryCombineManager(model_dict=model_dict, tokenizer_dict=tokenizer_dict, device=self.device)
                    updated_memory1 = combine_manager.merge_memories(long_term_memory=history_mem, new_memory=standardized_memory)
                    updated_memory2 = combine_manager.resolve_conflicts_hobby(long_term_memory=history_mem, new_memory=updated_memory1)
                    updated_memory3 = combine_manager.resolve_conflicts_plan(long_term_memory=history_mem, new_memory=updated_memory2)
                    new_history_mem = updated_memory3

                    if self.run_importance:
                        importance_handeler.alignment_kv(new_history_mem, dia_i + start_round + 1)
                else:
                    new_history_mem = cleaned_new_mem

                    if self.run_importance:
                        importance_handeler.alignment_kv(new_history_mem, dia_i + start_round + 1)

                logger.info("------（格式化后）合并后的memory------")
                logger.info(new_history_mem)
                history_mem = new_history_mem
                dialogue_memory_list.append({'dialogue': dia, 'current_round':dia_i + start_round + 1,  'new_memory':copy.deepcopy(cleaned_new_mem), 'final_memory': copy.deepcopy(history_mem)})

            print('memory_file_path', memory_file_path)
            with open(memory_file_path, 'w', encoding='utf-8') as f:
                dialogue_memory_list_to_save = []
                if len(dialogue_memory_list) > 0:
                    dialogue_memory_list_to_save.append(dialogue_memory_list[-1])
                json.dump(dialogue_memory_list_to_save, f, ensure_ascii=False, indent=4)

            return history_mem
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        except KeyboardInterrupt:
            print("捕获到 Ctrl+C")
    
    async def retrieve_memory(self, session_id: int):
        try:
            # 检索指定session_id的记忆
            memory_file_path = os.path.join(self.mem_save_folder, f"{session_id}.jsonl")
            print(memory_file_path, "retrieve")
            with open(memory_file_path, 'r') as file:
                memory_data = json.load(file)
            return JSONResponse(content=memory_data)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Memory data not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def rag_memory(self, session_id: int, dialogue:str, top_k: int, round: int, importance_freeze: bool = False):
        try:
            # 检索指定session_id的记忆
            print('in rag memory')
            if self.run_importance:
                importance_handeler = ImportanceHandler(session_id=session_id, importance_dir=self.importance_save_folder)
                importance_handeler.initialize_importance()
            memory_file_path = os.path.join(self.mem_save_folder, f"{session_id}.jsonl")
            with open(memory_file_path, 'r') as file:
                memory_data = json.load(file)
            
            if len(memory_data) == 0:
                return 'memory data is empty'

            if len(memory_data[-1]['final_memory']) == 0:
                return 'memory data is empty'

            print('rag memory FLAG 1')
            rag = RetrievalAugmentedGeneration(self.model_bge, self.model_reranker, memory_data[-1]['final_memory'], session_id)
            rag.encode()  # Encode the order list

            print('rag memory FLAG 2')
            indices = rag.search(dialogue, top_k)  # Search for the top 2 similar items

            inhibition_list = None
            print('rag memory FLAG 3')
            if self.grade_update_mode == 'add_inhibition':
                # 抑制模式下，前topk还是增强，之后的topk会抑制
                print('rag memory FLAG 3.1')
                indices, inhibition_indices = rag.search(dialogue, top_k, inihibition_flag=True)
                print('rag memory FLAG 3.2')
                rag_dict = dict()
                for (key, vi) in rag.locate(indices):
                    if key in rag_dict:
                        # 如果键已存在，添加值到对应的列表
                        rag_dict[key].append(memory_data[-1]['final_memory'][key][vi])
                    else:
                        # 如果键不存在，创建一个新列表并添加值
                        rag_dict[key] = [memory_data[-1]['final_memory'][key][vi]]
                print('rag memory FLAG 3.3')
                inhibition_list = list()
                for (key, vi) in rag.locate(inhibition_indices):
                    inhi_str = key + ":" + memory_data[-1]['final_memory'][key][vi]
                    inhi_str = re.sub(r'\(\d+轮对话前\)', '', inhi_str).strip()
                    inhibition_list.append(inhi_str)
                print('inhibition_list')
                print(inhibition_list)
                print('rag memory FLAG 3.4')
            else:
                rag_dict = dict()   
                for (key, vi) in rag.locate(indices):
                    if key in rag_dict:
                        # 如果键已存在，添加值到对应的列表
                        rag_dict[key].append(memory_data[-1]['final_memory'][key][vi])
                    else:
                        # 如果键不存在，创建一个新列表并添加值
                        rag_dict[key] = [memory_data[-1]['final_memory'][key][vi]]

            print('rag memory FLAG 4')
            if self.run_importance and not importance_freeze:
                print('-----------here RAG---------')
                for key in rag_dict:
                    for v in rag_dict[key]:
                        importance_handeler.update_importance_kv(key=key, value=v, round=round)
                importance_handeler.update_grade(c=round, mode=self.grade_update_mode, inhibition_infos=inhibition_list)
                importance_handeler.filter_memory(top_n=self.mem_store_capacity)
                memory_file_path = os.path.join(self.mem_save_folder, f"{session_id}.jsonl")
                importance_handeler.update_kv_file(kv_file_name=memory_file_path)
                
            return JSONResponse(content=rag_dict)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Memory data not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def update_memory(self, session_id:int):
        pass

    async def store_summary(self, input_data: DialogueInput):
        if self.run_importance:
            importance_handeler = ImportanceHandler(session_id=input_data.session_id, importance_dir=self.importance_save_folder)
            importance_handeler.initialize_importance()
        print('in store summary func')
        log_output_file = os.path.join(self.sum_save_folder, f"{input_data.session_id}_output.log")
        processor = OriginDialogueProcessor(input_data.messages)
        print('after processor in store func')

        # 设置日志记录
        logger = setup_logging(log_output_file)
        processor = OriginDialogueProcessor(input_data.messages)
        # 获取对话轮次信息等（根据需要选择使用）
        turns = processor.count_total_turns()
        logger.info(f"对话总轮次数：{turns}")
        print('after logger init')

        role_info=  processor.get_role_info()
        user_name =  role_info[0]['user_name']
        bot_name = role_info[0]['primary_bot_name']
        user_info = role_info[1][user_name]
        bot_info = role_info[1][bot_name]

        print('before summary mem init')
        vector_db = memory_init(session_id=input_data.session_id, _embed_model=self.model_bge_summary, _llm_api=self.model_summary, 
                                chunk_sizes = self.chunk_sizes, store_dir=self.sum_save_folder)
        print('after summary mem init')

        dialogues = processor.split_by_turns(1)

        for dia_i, dia in enumerate(dialogues):
            now_time = int(time.time())
            vector_db.put(text=dia, memory_time=now_time)

            summary_text = vector_db.get_text()
            importance_handeler.alignment_summary(summary=summary_text, chunk_sizes=self.chunk_sizes, round=1 + len(summary_text[0])*self.chunk_sizes[0]/self.split_memory_dialogue_num)
        vector_db.__memory_write__()
        print('summary success in store_summary')

    async def rag_summary(self, session_id: int, dialogue:str, top_k: int, round: int, importance_freeze: bool = False):
        if self.run_importance:
            importance_handeler = ImportanceHandler(session_id=session_id, importance_dir=self.importance_save_folder)
            importance_handeler.initialize_importance()
        vector_db = memory_init(session_id=session_id, _embed_model=self.model_bge_summary, _llm_api=self.model_summary, 
                                chunk_sizes = self.chunk_sizes, store_dir=self.sum_save_folder)
        res1, res2, res3 = vector_db.query_together(dialogue, top_k, self.summary_rag_threshold)
        
        inhibition_list=None
        if self.grade_update_mode == 'add_inhibition':
            res1_2, res2_2, res3_2 = vector_db.query_together(dialogue, 2 * top_k, self.summary_rag_threshold)
            inhi1 = [item['content'] for item in res1_2 if item not in res1]
            inhi2 = [item['content'] for item in res2_2 if item not in res2]
            inhi3 = [item['content'] for item in res3_2 if item not in res3]
            inhibition_list = inhi1 + inhi2 + inhi3
        # importance_handeler.update_importance_summary(summary=[res1, res2, res3], round=round)
        # importance_handeler.update_grade(c=round, mode=self.grade_update_mode)
        # importance_handeler.filter_memory(top_n=30)
        if self.run_importance and not importance_freeze:
            for r1 in res1:
                importance_handeler.update_importance_summary(summary=r1['content'], round=round)
            for r2 in res2:
                importance_handeler.update_importance_summary(summary=r2['content'], round=round)
            for r3 in res3:
                importance_handeler.update_importance_summary(summary=r3['content'], round=round)
            importance_handeler.update_grade(c=round, mode=self.grade_update_mode, inhibition_infos=inhibition_list)
            importance_handeler.filter_memory(top_n=self.mem_store_capacity)
            # print("------------rag summary------------")
            # print("res1")
            # print(res1)
            # print('res2')
            # print(res2)
            # print('res3')
            # print(res3)
            # print("-----更新后的importance_handeler--------")

            # summary_txt_path1 = os.path.join(self.sum_save_folder, f"{session_id}_text_1.json")
            # summary_vec_path1 = os.path.join(self.sum_save_folder, f"{session_id}_vec_1.pkl")
            # importance_handeler.update_summary_file(summary_txt_path1, summary_vec_path1, self.chunk_sizes[1])  

            # summary_txt_path2 = os.path.join(self.sum_save_folder, f"{session_id}_text_2.json")
            # summary_vec_path2 = os.path.join(self.sum_save_folder, f"{session_id}_vec_2.pkl")
            # importance_handeler.update_summary_file(summary_txt_path2, summary_vec_path2, self.chunk_sizes[2])  
            # print(importance_handeler.importance_data)
        return res1, res2, res3

    async def store_mem0_kv(self, input_data: DialogueInput):
        log_output_file = os.path.join(self.mem_save_folder, f"{input_data.session_id}_output.log")
        memory_file_path = os.path.join(self.mem_save_folder, f"{input_data.session_id}.jsonl")
        # 设置日志记录
        logger = setup_logging(log_output_file)
        processor = OriginDialogueProcessor(input_data.messages)
        # 获取对话轮次信息等（根据需要选择使用）
        turns = processor.count_total_turns()
        logger.info(f"对话总轮次数：{turns}")

        role_info=  processor.get_role_info()
        user_name =  role_info[0]['user_name']
        bot_name = role_info[0]['primary_bot_name']
        user_info = role_info[1][user_name]
        bot_info = role_info[1][bot_name]
        logger.info('---------------角色信息---------------')
        logger.info(user_name)
        logger.info(user_info)
        logger.info(bot_name)
        logger.info(bot_info)

        dialogues = processor.split_by_turns(20)
        dialogue_memory_list = []
        used_infos = []
        history_mem = None
        fail_mem_extract = False

        if os.path.exists(memory_file_path):
            with open(memory_file_path, 'r', encoding='utf-8') as file:
                dialogue_memory_list = json.load(file)
                logger.info('--------CHECK exsting file---------')
                if dialogue_memory_list is not None and len(dialogue_memory_list) != 0:
                    logger.info(dialogue_memory_list[-1])
                    history_mem = dialogue_memory_list[-1]['final_memory']
                else:
                    logger.info('dialogue_memory_list is none or empty')

        for dia_i, dia in enumerate(dialogues):
            logger.info('--------dia_i---------')
            logger.info(dia_i)

            logger.info('---------------')
            logger.info(dia)
            ## 提取当前对话的memory
            prompt_mem = build_memory_prompt_cws(user_name, bot_name, dia, self.prompt_mem_template)
            response_mem = model_process(self.model_mem, self.tokenizer_mem, self.device, prompt_mem)
            logger.info('-----新总结的memory(raw)----------')
            logger.info(extract_json_block(response_mem))
            cleaned_new_mem = clean_json_string(response_mem)
            cleaned_new_mem = split_comma_separated_values(cleaned_new_mem)
            logger.info("-----新总结的memory(清洗后)------")
            logger.info(cleaned_new_mem)

            standardized_memory = self.memory_standarder.standardize_regex(cleaned_new_mem)
            logger.info("-----新总结的memory(标准化后)------")
            logger.info(standardized_memory)        

        # 要看看如何将mem0的记忆存储！！！
        new_mem = cleaned_new_mem
        for k in new_mem:
            for v in new_mem[k]:
                self.mem0_memory.add(v, user_id=input_data.session_id, metadata={"category": k})
        
        info = self.mem0_memory.get_all(user_id=input_data.session_id)
        print("this is info of mem0!!!!")
        print(info)

    async def store_mem0(self, input_data: DialogueInput):
        log_output_file = os.path.join(self.mem_save_folder, f"{input_data.session_id}_output.log")
        memory_file_path = os.path.join(self.mem_save_folder, f"{input_data.session_id}.jsonl")
        # 设置日志记录
        logger = setup_logging(log_output_file)
        processor = OriginDialogueProcessor(input_data.messages)
        # 获取对话轮次信息等（根据需要选择使用）
        turns = processor.count_total_turns()
        logger.info(f"对话总轮次数：{turns}")

        role_info=  processor.get_role_info()
        user_name =  role_info[0]['user_name']
        bot_name = role_info[0]['primary_bot_name']
        user_info = role_info[1][user_name]
        bot_info = role_info[1][bot_name]
        logger.info('---------------角色信息---------------')
        logger.info(user_name)
        logger.info(user_info)
        logger.info(bot_name)
        logger.info(bot_info)

        dialogues = processor.split_by_turns(20)
        dialogue_memory_list = []
        used_infos = []
        history_mem = None
        fail_mem_extract = False

        for dia_i, dia in enumerate(dialogues):
            logger.info('--------dia_i---------')
            logger.info(dia_i)

            logger.info('---------------')
            logger.info(dia)
            ## 提取当前对话的memory
            # print(self.mem0_config)
            # print('nov_11st user_id', input_data.session_id)
            self.mem0_memory.add(dia, user_id=input_data.session_id)      
        
        info = self.mem0_memory.get_all(user_id=input_data.session_id)
        print("this is info of mem0!!!!")
        print(info)

    async def rag_mem0(self, session_id: int, dialogue:str, top_k: int, round: int):
        result = self.mem0_memory.search(query=dialogue, user_id=session_id, limit=top_k)
        print(result)
        return result
    
    async def retrieve_mem0(self, session_id: int, limit: int):
        info = self.mem0_memory.get_all(user_id=session_id, limit=limit)
        print("this is info of mem0!!!!")
        print(info)
        return info

    async def delete_mem0(self, session_id: int):
        info = self.mem0_memory.delete_all(user_id=session_id)
        return info

    async def store_memorybank(self, input_data: DialogueInput):
        log_output_file = os.path.join(self.membank_save_folder, f"{input_data.session_id}_output.log")
        memory_file_path = os.path.join(self.membank_save_folder, f"{input_data.session_id}.jsonl")
        # 设置日志记录
        logger = setup_logging(log_output_file)
        processor = OriginDialogueProcessor(input_data.messages)
        # 获取对话轮次信息等（根据需要选择使用）
        turns = processor.count_total_turns()
        logger.info(f"对话总轮次数：{turns}")

        role_info=  processor.get_role_info()
        user_name =  role_info[0]['user_name']
        bot_name = role_info[0]['primary_bot_name']
        user_info = role_info[1][user_name]
        bot_info = role_info[1][bot_name]
        logger.info('---------------角色信息---------------')
        logger.info(user_name)
        logger.info(user_info)
        logger.info(bot_name)
        logger.info(bot_info)

        dialogues = processor.split_by_turns(20)
        history_mem = []
      
        if os.path.exists(memory_file_path):
            with open(memory_file_path, 'r', encoding='utf-8') as file:
                dialogue_memory_list = json.load(file)
                logger.info('--------CHECK exsting file for memory bank---------')
                if dialogue_memory_list is not None and len(dialogue_memory_list) != 0:
                    logger.info('--------CHECK success in memory bank---------')
                    history_mem = dialogue_memory_list
                else:
                    logger.info('dialogue_memory_list for memory_bank is none or empty')

        # from memorybank.membank_summary_prompt import summarize_content_prompt, summarize_overall_prompt, summarize_overall_personality, summarize_person_prompt
        for dia_i, dia in enumerate(dialogues):
            mem_group = len(history_mem) // 10
            mem_bank_i = dict()
            
            prompt_history = summarize_content_prompt(dia)
            if self.memorybank_use_api is True:
                response_history = lsl_qwen_70b(prompt_history)
            else:
                response_history = model_process(self.model_combine, self.tokenizer_mem, self.device, prompt_history)

            prompt_personality = summarize_person_prompt(dia, user_name, bot_name)
            if self.memorybank_use_api is True:
                response_personality = lsl_qwen_70b(prompt_personality)
            else:
                response_personality = model_process(self.model_combine, self.tokenizer_mem, self.device, prompt_personality)
                

            mem_bank_i['history'] =  response_history.split("\n总结：\n\n")[-1]
            mem_bank_i['personality'] = response_personality.split("回复策略为：")[-1]

            if len(history_mem) % 10 == 9:
                prompt_history_all = summarize_overall_prompt(history_mem[mem_group * 10: (mem_group+1) * 10])
                if self.memorybank_use_api is True:
                    response_history_all = lsl_qwen_70b(prompt_history_all)
                else:
                    response_history_all = model_process(self.model_combine, self.tokenizer_mem, self.device, prompt_history_all)

                prompt_personality_all = summarize_overall_personality(history_mem[mem_group * 10: (mem_group+1) * 10], user_name, bot_name)
                if self.memorybank_use_api is True:
                    response_personality_all = lsl_qwen_70b(prompt_personality_all)
                else:
                    response_personality_all = model_process(self.model_combine, self.tokenizer_mem, self.device, prompt_personality_all)

                mem_bank_i['history_overall'] =  response_history_all.split("\n总结：\n\n")[-1]
                mem_bank_i['personality_overall'] = response_personality_all.split("高度概括。总结为：\n\n")[-1]
            
            history_mem.append(mem_bank_i)

        with open(memory_file_path, 'w', encoding='utf-8') as f:
            json.dump(history_mem, f, ensure_ascii=False, indent=4)
    
    async def rag_memorybank(self, session_id: int, dialogue:str, top_k: int, round: int, importance_freeze: bool = False):
        try:
            memory_file_path = os.path.join(self.membank_save_folder, f"{session_id}.jsonl")
            with open(memory_file_path, 'r') as file:
                memory_data = json.load(file)

            if len(memory_data) == 0:
                return 'memory data is empty'
            
            order_list = list()
            for m_item in memory_data:
                order_list.append(m_item['history'])
                order_list.append(m_item['personality'])

            rag = RetrievalAugmentedGeneration(embedding_model=self.model_bge, reranker_model=self.model_reranker, json_data=None, session_id=session_id, order_list=order_list)
            rag.encode() 
            indices = rag.search(dialogue, top_k)

            rag_ls = list()  
            for i in indices:
                rag_ls.append(order_list[i])
            
            return JSONResponse(content=rag_ls)
    
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Memory data not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # store_memochat只支持把整个对话输进去
    async def store_memochat(self, input_data: DialogueInput):
        try:
            local_check = False
            memochat_prompt_path = 'memochat/memo_chat_prompt_chinese.json'
            prompts = json.load(open(memochat_prompt_path, "r"))

            entry_id = input_data.session_id
            role_meta = input_data.messages.get('role_meta')

            conversations = [
                {"from": msg["sender_name"], "value": msg["text"]}
                for msg in input_data.messages.get('messages', [])
            ]

            converted_entry = {
                "id": entry_id,
                "origin_dialogue": input_data.messages,
                "conversations": conversations,
                "role_meta": role_meta
            }

            converted_dia = [converted_entry]
            ans_item = get_model_answers(self.model_combine, self.tokenizer_combine, local_check, converted_dia, prompts, self.memochat_use_api)

            new_answer_file = self.memochat_save_folder + f"/memochat_{entry_id}.json"
            json.dump(ans_item, open(os.path.expanduser(new_answer_file), "w"), indent=2, ensure_ascii=False)
        except KeyboardInterrupt:
            print("捕获到 Ctrl+C")

    async def rag_memochat(self, session_id: int, dialogue:str, top_k: int, round: int, importance_freeze: bool = False):
        def extract_between(text, start, end):
            # 获取起始和结束子串的索引
            start_idx = text.find(start)
            end_idx = text.find(end, start_idx + len(start))
            
            # 检查起始和结束子串是否存在
            if start_idx != -1 and end_idx != -1:
                return text[start_idx + len(start):end_idx]
            return ""

        def remove_id(text):
            # 正则表达式匹配开头的括号及其中的数字
            return re.sub(r'^\(\d+\)\s*', '', text)
        
        memochat_file_path = self.memochat_save_folder + f"/memochat_{session_id}.json"

        with open(memochat_file_path, 'r', encoding='utf-8') as file:
            memochat_data = json.load(file)
        
        for line in range(len(memochat_data)):
            for i in range(len(memochat_data[line]["conversations"])-1, 0, -1):
                if memochat_data[line]["conversations"][i].get("thinking"):
                    last_thinking = memochat_data[line]["conversations"][i]["thinking"]
                    break
            
            start = 'nTopic Options:'
            end = '\\n```\\n\\n```\\n任务介绍：'
            split_thinking = extract_between(last_thinking, start, end)
            split_thinking2 = split_thinking.split('\\n')

            remove_id_thinking = []
            for s_t in split_thinking2:
                m = remove_id(s_t)
                remove_id_thinking.append(m)

            order_list = remove_id_thinking[2:]
        
        rag = RetrievalAugmentedGeneration(embedding_model=self.model_bge, reranker_model=self.model_reranker, json_data=None, session_id=session_id, order_list=order_list)
        rag.encode() 
        indices = rag.search(dialogue, top_k)

        rag_ls = list()  
        for i in indices:
            rag_ls.append(order_list[i])
        
        return JSONResponse(content=rag_ls)

if __name__ == '__main__':

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_mem_file_path = 'prompt/memory_extract_prompt_template.txt'
    with open(prompt_mem_file_path, 'r', encoding='utf-8') as prompt_file:
        prompt_mem_template = prompt_file.read()

    prompt_combine_file_path = 'prompt/memory_combine.txt'
    with open(prompt_combine_file_path, 'r', encoding='utf-8') as prompt_file:
        prompt_combine_template = prompt_file.read()

    prompt_error_file_path = 'prompt/error_process.txt'
    with open(prompt_error_file_path, 'r', encoding='utf-8') as prompt_file:
        prompt_error_template = prompt_file.read()

    prompt_template ={"mem":prompt_mem_template, "combine":prompt_combine_template, "error":prompt_error_template}

    # (self, model, tokenizer, device, prompt_template, mem_save_folder = './memory_pool')

    bge_summary = embedding_load()
    llm_api = llm_api_load()

    model = {"mem":None, "combine":None, "bge":None, "reranker":None, "error":None, "summary":llm_api, "bge_summary": bge_summary}
    tokenizer = {"mem":None, "combine":None, "error":None}

    manager = MemoryManager(model=model, tokenizer=tokenizer, prompt_template=prompt_template, device=device)

    