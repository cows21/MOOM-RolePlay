import os
import numpy as np
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
import torch

class BaseEmbeddings:
    def __init__(self, model_path:str, is_api:bool) -> None:
        self.model_path = model_path
        self.is_api = is_api

    def get_embedding(self, text:str) -> List[float]:
        raise NotImplementedError
    
    @classmethod
    def cosine_smilarity(cls, vec1:List[float], vec2:List[float]) -> float:
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # Calculate norms
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        norm_product = norm_vec1 * norm_vec2
        if not norm_product:
            return 0
        
        # Calculate cosine similarity
        cosine_sim = dot_product / (norm_vec1 * norm_vec2)
        
        return cosine_sim
    
class BGEEmbedding(BaseEmbeddings):
    def __init__(self, model_path:str, is_api:bool = False) -> None:
        super().__init__(model_path, is_api)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()

    def get_embedding(self, text:str) -> List[float]:
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            model_output = model_output[0]
            sentence_embeddings = model_output[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.squeeze()
        return sentence_embeddings.numpy().tolist()
    

if __name__ == '__main__':
    model_path = '/nvme/lisongling/models/bge-base-zh-v1.5'
    bge = BGEEmbedding(model_path)
    text1 = '你想去吃点什么呀？'
    text2 = '喜欢的食物'
    text3 = '喜欢的电影'
    text4 = '喜欢的音乐'
    embedding1 = bge.get_embedding(text1)
    embedding2 = bge.get_embedding(text2)
    embedding3 = bge.get_embedding(text3)
    embedding4 = bge.get_embedding(text4)
    print(BGEEmbedding.cosine_smilarity(embedding1, embedding2))
    print(BGEEmbedding.cosine_smilarity(embedding1, embedding3))
    print(BGEEmbedding.cosine_smilarity(embedding1, embedding4))
