import numpy as np
import os
import heapq
from FlagEmbedding import BGEM3FlagModel,FlagReranker
import random

def top_k_indices(lst, k):
    """
    获取top_k评分的索引
    
    lst 评分列表
    
    k top_k的阈值
    """
    # 首先获取最大的k个数的值
    # print('k', k)
    # print('lst')
    # print(lst, type(lst[0]))
    largest_values = heapq.nlargest(k, lst)
    
    # 建立值到索引的映射
    value_to_indices = {}
    for index, num in enumerate(lst):
        # num_key = np.array2string(num, precision=6)
        # print('num_key')
        # print(num_key)
        if num in largest_values:
            value_to_indices.setdefault(num, []).append(index)
    
    # 根据最大值列表，从映射中获取所有索引
    # print('largest_values')
    # print(largest_values, type(largest_values[0]))
    # print('value_to_indices')
    # print(value_to_indices)

    indices = [index for value in largest_values for index in value_to_indices[value]]
    
    return indices[:k]  # 如果有重复值，确保只返回k个索引

class RetrievalAugmentedGeneration:
    def __init__(self, embedding_model, reranker_model, json_data, session_id, order_list=None):      
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.session_id = session_id
        self.json_data = json_data
        self.index_table = {}
        self.order_list = []
        self.encoded_list = []
        self.score_threshold = 0.1

        
        # Process JSON data to create order list and index table
        if order_list is None:
            idx = 0
            for key, values in json_data.items():
                num_values = len(values)
                self.index_table[key] = (idx, num_values)
                idx += num_values
                for value in values:
                    self.order_list.append(f"{key}:{value}")
        else:
            self.order_list = order_list
    
    def encode(self):
        # Placeholder for encoding method using BGE model
        self.encoded_list = self.embedding_model.encode([item for item in self.order_list], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"] 
    
    def search(self, query, k, inihibition_flag=False):
        # print('in search function')
        # 初筛
        q_embeddings = self.embedding_model.encode(query)["dense_vecs"]
        p_embeddings = np.array(self.encoded_list)
        scores = q_embeddings @ p_embeddings.T
        
        top_k_first_select_id = top_k_indices(scores, 2 * k)
        
        # 过滤掉低于阈值的索引和分数
        filtered_ids_and_scores = [(idx, scores[idx]) for idx in top_k_first_select_id if scores[idx] > self.score_threshold]
        select_content = [self.order_list[idx] for idx, _ in filtered_ids_and_scores]
        original_indices = [idx for idx, _ in filtered_ids_and_scores]  # 保留原始索引

        pairs = [[query, content] for content in select_content]
        reranker_scores = self.reranker_model.compute_score(pairs, normalize=True)

        # 获取重排序后的top_k结果的原始索引
        reranked_top_k_ids = top_k_indices(reranker_scores, k)

        select_content_id = [original_indices[id] for id in reranked_top_k_ids]  # 将重排序后的索引映射回encoded_list中的原始索引
        # print('select_content_id')
        # print(select_content_id)
        if inihibition_flag is False:
            return select_content_id
        else:
            inhibition_id = [item for item in original_indices if item not in select_content_id]
            return select_content_id, inhibition_id

    def locate(self, indices):
        # Locate the original positions in the JSON data
        # print('in locate')
        # print(self.index_table)
        result = []
        for index in indices:
            for key, (start_idx, count) in self.index_table.items():
                if start_idx <= index < start_idx + count:
                    value_index = index - start_idx
                    result.append((key, value_index))
                    break
            # print(result)
        return result
    
    def remove_brackets(self):
        # Method to remove brackets from the values in the JSON data
        for key, values in self.json_data.items():
            self.json_data[key] = [value.split('(')[0] for value in values]
    
    def update_json(self, new_session_id, new_json_data):
        if new_session_id != self.session_id:
            return "Session ID mismatch."
        else:
            self.json_data = new_json_data
            self.order_list.clear()
            self.index_table.clear()
            # Reinitialize the order list and index table
            self.__init__(self.embedding_model, self.reranker_model, new_json_data, new_session_id)
            return "JSON data updated successfully."


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" #0,1,4,6,7
    embedding_model = BGEM3FlagModel('model/bge_m3',use_fp16=True)
    reranker_model = FlagReranker('model/bge_reranker', use_fp16=True)
    json_data = {"喜欢的食物": ["苹果", "可乐"], "人际关系": ["与张明是好朋友"], "高中学校": ["湖北中学"], "姓名": ["张虎"]}
    session_id = 1

    rag = RetrievalAugmentedGeneration(embedding_model, reranker_model, json_data, session_id)
    print("rag.order_list")
    print(rag.order_list)
    rag.encode()  # Encode the order list
    indices = rag.search("你叫什么", 2)  # Search for the top 2 similar items
    print(rag.locate(indices))  # Locate the positions in the original JSON