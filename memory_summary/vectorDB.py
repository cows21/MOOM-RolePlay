import os
import json
from typing import Dict, List, Optional
import numpy as np
import pickle
from tqdm import tqdm
import uuid
import time
from .embedding import BaseEmbeddings
from .llm_api import LLMAPI
import logging

# logging.basicConfig(
#     filename="result/summary_debug.log",  # 指定日志文件路径
#     level=logging.DEBUG,  # 设置日志级别
#     format="%(asctime)s [%(levelname)s] %(message)s",  # 日志格式
#     datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式
# )

# logging.basicConfig(
#     filename=None,  # 指定日志文件路径
#     level=logging.DEBUG,  # 设置日志级别
#     format="%(asctime)s [%(levelname)s] %(message)s",  # 日志格式
#     datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式
# )

class BaseMemoryVectorDB:
    def __init__(self, 
        vec_path: str,
        content_path: str, 
        emb_model:BaseEmbeddings,
):
        #向量存储路径和文本存储路径 分别是pickle和json格式
        self.vec_path = vec_path
        self.content_path = content_path
        self.emb_model = emb_model
        self.cur_chunk = []
        self.vector_db = {}
        self.text_db = {}

        if os.path.exists(vec_path):
            with open(vec_path, 'rb') as vec_f:
                self.vector_db = pickle.load(vec_f)
        if os.path.exists(content_path):
            with open(content_path, 'r') as content_f:
                self.text_db = json.load(content_f)

    def put_text(self, text:str, start_time:int, end_time: int, id:str, num:int):
        self.text_db[id] = {
            'start_time' : start_time,
            'end_time' : end_time,
            'content' : text,
            'num': num
        }

    def put_embedding(self, text:str, id:str):
        embedding = self.emb_model.get_embedding(text)
        self.vector_db[id] = embedding

    def __memory_write__(self):
        #把当前记忆存储到文件中去
        with open(self.vec_path, 'wb') as vec_f:
            pickle.dump(self.vector_db, vec_f)

        with open(self.content_path, 'w', encoding='utf-8') as content_f:
            json.dump(self.text_db, content_f, ensure_ascii=False)

    def query(self, query: str, topk: int = 1, threshold:float = 0.6) -> List[str]:
        query_embedding = self.emb_model.get_embedding(query)
        similarities = []
        for id, embedding in self.vector_db.items():
            #计算query和数据库的每个vector的相似度
            similarities.append((id, self.emb_model.cosine_smilarity(query_embedding, embedding)))
        
        similarities.sort(key = lambda x: x[1], reverse=True)
        top_k_ids = similarities[:topk]

        top_k_texts = [self.text_db[id] for id, smi in top_k_ids if smi > threshold]

        return top_k_texts
    
    def query_with_grade(self, rank_flag: int, query: str, topk: int = 1, threshold:float = 0.6):
        query_embedding = self.emb_model.get_embedding(query)
        similarities = []
        for id, embedding in self.vector_db.items():
            #计算query和数据库的每个vector的相似度
            similarities.append((id, self.emb_model.cosine_smilarity(query_embedding, embedding)))
        
        similarities.sort(key = lambda x: x[1], reverse=True)
        top_k_ids = similarities[:topk]

        top_k_texts_and_smi = [[self.text_db[id], smi, rank_flag] for id, smi in top_k_ids if smi > threshold]

        return top_k_texts_and_smi
    
    def get_text(self):
        return self.text_db
    
    def get_embedding(self):
        return self.vector_db


    
class MemoryVectorDB:
    def __init__(self,
        vector_paths: List[str],
        content_paths: List[str],
        chunk_sizes: List[int], 
        session_id: int,
        emb_model:BaseEmbeddings,
        llm_api:LLMAPI
):
        self.vec_db_l0 = BaseMemoryVectorDB(vector_paths[0], content_paths[0], emb_model)
        self.vec_db_l1 = BaseMemoryVectorDB(vector_paths[1], content_paths[1], emb_model)
        self.vec_db_l2 = BaseMemoryVectorDB(vector_paths[2], content_paths[2], emb_model)

        self.session_id = session_id
        #chunk_size[0]是用来表示多少轮对话最为一个块存储，0级检索单位为块
        #chunk_size[1]和chunk_size[2]是用来表示多少块/1级summary做一次summary
        self.chunk_sizes = chunk_sizes
        self.emb_model = emb_model

        self.llm_api = llm_api

    def put(self, text:str, memory_time:int):
        # logging.debug(f"Input text: {text}, memory_time: {memory_time}")
        cur0 = self.vec_db_l0.get_text()
        cur1 = self.vec_db_l1.get_text()
        cur2 = self.vec_db_l2.get_text()

        # logging.debug(f"Current Level 0 memory: {cur0}")
        # logging.debug(f"Current Level 1 memory: {cur1}")
        # logging.debug(f"Current Level 2 memory: {cur2}")
        if len(cur0) > 0:
            last_key0 = list(cur0.keys())[-1]
            # logging.debug(f"Last key in Level 0 memory: {last_key0}")
        if len(cur1) > 0: 
            last_key1 = list(cur1.keys())[-1]
            # logging.debug(f"Last key in Level 1 memory: {last_key1}")
        if len(cur2) > 0: 
            last_key2 = list(cur2.keys())[-1]
            # logging.debug(f"Last key in Level 2 memory: {last_key2}")

        c1 = False
        c1_text = ''
        c2 = False
        c2_text = ''

        if len(cur0) > 0 and cur0[last_key0]['num'] < self.chunk_sizes[0]:
            cur0[last_key0]["content"] = cur0[last_key0]["content"] + text
            cur0[last_key0]["num"] = cur0[last_key0]["num"] + 1
            cur0[last_key0]["start_time"] = cur0[last_key0]["start_time"]
            cur0[last_key0]["end_time"] = memory_time
            # logging.debug(f"Updated Level 0 memory (key: {last_key0}): {cur0[last_key0]}")
            if cur0[last_key0]['num'] == self.chunk_sizes[0]:
                self.vec_db_l0.put_embedding(text=cur0[last_key0]["content"], id=last_key0)
                # logging.debug(f"Level 0 memory chunk completed: {cur0[last_key0]}")
                c1 = True            
                c1_text = cur0[last_key0]["content"]
        else:
            id0 = f'level0_rag_memory_{self.session_id}_{int(time.time())}_{uuid.uuid4()}'
            content0 = text
            num0 = 1
            start_time0 = memory_time
            end_time0 = memory_time
            self.vec_db_l0.put_text(content0, start_time=start_time0, end_time=end_time0, id=id0, num=num0)
            # logging.debug(f"New Level 0 memory added (key: {id0}, content: {content0})")

            
        if c1 and len(cur1) == 0:
            # 保留上一个记忆的最后一个元素，为了连贯
            id1 = f'level1_rag_memory_{self.session_id}_{int(time.time())}_{uuid.uuid4()}'
            content1 = c1_text
            num1 = 1
            start_time1 = memory_time
            end_time1 = memory_time
            self.vec_db_l1.put_text(content1, start_time=start_time1, end_time=end_time1, id=id1, num=num1)
            # logging.debug(f"New Level 1 memory added (key: {id1}, content: {c1_text})")

        elif c1 and cur1[last_key1]['num'] < self.chunk_sizes[1]:
            cur1[last_key1]["content"] = cur1[last_key1]["content"] + '\n' + c1_text
            cur1[last_key1]["num"] = cur1[last_key1]["num"] + 1
            cur1[last_key1]["start_time"] = cur1[last_key1]["start_time"]
            cur1[last_key1]["end_time"] = memory_time
            # logging.debug(f"Updated Level 1 memory (key: {last_key1}): {cur1[last_key1]}")

            if cur1[last_key1]['num'] == self.chunk_sizes[1]:
                summary = self.llm_api.get_dialogue_summary(cur1[last_key1]["content"])
                cur1[last_key1]["content"] = summary
                self.vec_db_l1.put_embedding(text=cur1[last_key1]["content"], id=last_key1)
                # logging.debug(f"Level 1 memory chunk completed: {cur1[last_key1]}")

                c2 = True
                c2_text = cur1[last_key1]["content"]

                # 保留上一个记忆的最后一个元素，为了连贯
                id1 = f'level1_rag_memory_{self.session_id}_{int(time.time())}_{uuid.uuid4()}'
                content1 = c1_text
                num1 = 1
                start_time1 = memory_time
                end_time1 = memory_time
                self.vec_db_l1.put_text(content1, start_time=start_time1, end_time=end_time1, id=id1, num=num1)

        if c2 and len(cur2) == 0:
            id2 = f'level2_rag_memory_{self.session_id}_{int(time.time())}_{uuid.uuid4()}'
            content2 = c2_text
            num2 = 1
            start_time2 = memory_time
            end_time2 = memory_time
            self.vec_db_l2.put_text(content2, start_time=start_time2, end_time=end_time2, id=id2, num=num2)
            # logging.debug(f"New Level 2 memory added (key: {id2}, content: {c2_text})")

        elif c2 and cur2[last_key2]['num'] < self.chunk_sizes[2]:
            cur2[last_key2]["content"] = cur2[last_key2]["content"] + '\n' + c2_text
            cur2[last_key2]["num"] = cur2[last_key2]["num"] + 1
            cur2[last_key2]["start_time"] = cur2[last_key2]["start_time"]
            cur2[last_key2]["end_time"] = memory_time
            # logging.debug(f"Updated Level 2 memory (key: {last_key2}): {cur2[last_key2]}")

            if cur2[last_key2]['num'] == self.chunk_sizes[2]:
                summary = self.llm_api.get_summaries_summary(cur2[last_key2]["content"])
                cur2[last_key2]["content"] = summary
                self.vec_db_l2.put_embedding(text=cur2[last_key2]["content"], id=last_key2)

                id2 = f'level2_rag_memory_{self.session_id}_{int(time.time())}_{uuid.uuid4()}'
                content2 = c2_text
                num2 = 1
                start_time2 = memory_time
                end_time2 = memory_time
                self.vec_db_l2.put_text(content2, start_time=start_time2, end_time=end_time2, id=id2, num=num2)

                # logging.debug(f"Level 2 memory chunk completed: {cur2[last_key2]}")

    def __memory_write__(self):
        self.vec_db_l0.__memory_write__()
        self.vec_db_l1.__memory_write__()
        self.vec_db_l2.__memory_write__()

    def copy(self, new_vector_paths, new_content_paths):
        return MemoryVectorDB(
            new_vector_paths,
            new_content_paths,
            self.chunk_sizes,
            self.session_id,
            self.emb_model,
            self.llm_api
        )

    def query(self, query: str, topk: int = 1, thresholds: List[float] = [0.6,0.5,0.4]) -> List[List[str]]:
        if len(thresholds) != 3:
            raise ValueError("Thresholds should contain exactly 3 values corresponding to the 3 levels.")
        res_0 = self.vec_db_l0.query(query, topk, thresholds[0])
        res_1 = self.vec_db_l1.query(query, topk, thresholds[1])
        res_2 = self.vec_db_l2.query(query, topk, thresholds[2])
        return [res_0, res_1, res_2]
    
    def query_together(self, query: str, topk: int = 1, thresholds: List[float] = [0.6,0.5,0.4]):
        if len(thresholds) != 3:
            raise ValueError("Thresholds should contain exactly 3 values corresponding to the 3 levels.")
        res_0 = self.vec_db_l0.query_with_grade(0, query, topk, thresholds[0])
        res_1 = self.vec_db_l1.query_with_grade(1, query, topk, thresholds[1])
        res_2 = self.vec_db_l2.query_with_grade(2, query, topk, thresholds[2])
        print('in query_together function')
        print('res_0')
        print(res_0)
        print('res_1')
        print(res_1)
        print('res_2')
        print(res_2)

        res = res_0 + res_1 + res_2
        res.sort(key = lambda x: x[1], reverse=True)
        res = res[:topk]
        print('重新排序后')
        print(res)
        result = [[],[],[]]
        for r in res:
            res_class = r[2]
            result[res_class].append(r[0])
        return result
    
    def get_text(self):
        x = [self.vec_db_l0.get_text(), self.vec_db_l1.get_text(), self.vec_db_l2.get_text()]
        return x