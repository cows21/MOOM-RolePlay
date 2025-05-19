import json
import re
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer
import torch
import torch.nn as nn
import numpy as np
import requests
import glob
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel, FlagReranker 
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os
import jieba
from rouge_chinese import Rouge

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


# TODO 原先代码在tmp/test.py里
# 任务描述：
# 为了方便复现实验，最后对外的接口应该是将所有参数传入某一个函数，即可进行一次实验
# bertscore记忆实验的变量有：label文件夹名，模型记忆来源（例如kv，mem0，kv，kv以及sum），实验类型名（reference与candiadate类型）

def model_process(model, tokenizer, prompt, max_new_tokens=1024):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def remove_suffix(s):
    """
    去除字符串末尾形如"(xx轮对话前)"的后缀内容
    
    参数:
    - s: 输入字符串
    
    返回:
    - 去除后缀后的字符串
    """
    pattern =  r'\((\d+轮对话前)\)'
    match = re.search(pattern, s)
    if match:
        s = re.sub(pattern, '', s)
    return s

def get_mem_list(memory_data, user_name, mem_class="final_memory"):
    """
    根据记忆数据生成记忆列表，每条记忆为字符串形式
    
    参数:
    - memory_data: 包含记忆数据的字典
    - user_name: 用户名称，用于个性化输出
    - mem_class: 要提取的记忆类别，默认 "final_memory"
    
    返回:
    - mem_list: 包含所有记忆的字符串列表
    """
    mem_list = list()  
    id_card = ['姓名', '年龄', '生日', '性别', '性取向', '民族或种族', '国籍', '星座', '生肖', 'MBTI', '性癖']
    
    for key in memory_data[-1][mem_class]:
        if key == '发生事件':  
            for value in memory_data[-1][mem_class][key]:
                clean_v = remove_suffix(value)
                mem_list.append(clean_v)
        elif '喜欢如何称呼' in key or '喜欢被称呼的昵称' in key:
            for value in memory_data[-1][mem_class][key]:
                v = key + ': ' + value
                mem_list.append(v)
        elif key in id_card:  
            for value in memory_data[-1][mem_class][key]:
                v = user_name + '的' + key + '是' + value
                mem_list.append(v)
        elif '喜欢' in key:  
            for value in memory_data[-1][mem_class][key]:
                v = user_name + '喜欢' + value
                mem_list.append(v)
        elif '讨厌' in key:  
            for value in memory_data[-1][mem_class][key]:
                v = user_name + '讨厌' + value
                mem_list.append(v)
        else:  
            for value in memory_data[-1][mem_class][key]:
                mem_list.append(value)
    
    return mem_list

def extract_b_from_colon_string(input_string):
    """
    提取符合 'A:B' 格式字符串中的 'B' 部分，对不符合格式的字符串不做处理。

    参数:
        input_string (str): 输入的字符串。

    返回:
        str: 提取出的 'B' 部分，或者返回原始字符串（如果不符合 'A:B' 格式）。
    """
    if input_string.count(':') == 1:
        # 将字符串按 ':' 分割，并返回第二部分
        parts = input_string.split(':', 1)
        return parts[1]
    return input_string  # 对不符合格式的字符串，不做处理

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

def bge_similar(bge_model, sentence1, sentence_ls2):
    sentence_pairs = [[sentence1, s2] for s2 in sentence_ls2]
    scores = bge_model.compute_score(sentence_pairs, max_passage_length=128, weights_for_different_modes=[0.4, 0.2, 0.4], use_fp16=True,)
    # Assuming the "colbert+sparse+dense" score is used to determine similarity
    score = max(scores['colbert+sparse+dense'])
    score_index = scores['colbert+sparse+dense'].index(score)
    return score, score_index

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
        ip_url = 'http://10.142.5.99:8012'  + '/v1'
        # ip_url = 'http://10.142.5.82:27777'  + '/v1'
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

def dict_add(dict_longkey, dict_shotkey, shortkey, suffix=''):
    dict_a = dict_longkey.copy()
    for short_k in shortkey:
        long_k = short_k + suffix
        if short_k not in dict_shotkey:
            raise ValueError(f"key {short_k} missed in dict_shotkey {dict_shotkey} in dict_add function!!!!")
        if long_k not in dict_longkey:
            raise ValueError(f"key {long_k} missed in dict_longkey {dict_longkey} in dict_add function!!!!")
        dict_a[long_k] = dict_longkey[long_k] + dict_shotkey[short_k]
    return dict_a 

class MemoryEvaluator:
    def __init__(self, bertscore=None, judge_model=None, bge_model=None, m3e_model=None):
        """
        初始化 MemoryEvaluator 类。

        参数:
        - model_path (str): BERT 模型路径
        """
        if bertscore is not None:
            self.scorer = bertscore

        self.label = dict()
        self.output= dict()
        self.dataset_set = ['mem0', 'importance_data', 'memochat', 'memory_bank']

        if judge_model is not None:
            self.judge_model = judge_model['model']
            self.judge_tokenizer = judge_model['tokenizer'] 

        if bge_model is not None:
            self.bge = bge_model["bge"]
            self.bge_reranker = bge_model["bge_reranker"]
            
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.bge = self.bge.to(device)
            # self.bge = nn.DataParallel(self.bge)
        
        if m3e_model is not None:
            self.m3e = m3e_model

    def read_label(self, directory):
        '''
        读取label到self.label
        '''
        # 构建符合格式的文件匹配模式
        pattern = f"{directory}/*.jsonl"
        # 使用 glob 查找符合模式的文件
        files_names = glob.glob(pattern)
        self.label = dict()
        for f_n in files_names:
            with open(f_n, 'r', encoding='utf-8') as file:
                label_id = int(f_n.split('_')[-1].split('.')[0])
                self.label[label_id] = json.load(file)   

    def read_importance_file(self, directory):
        '''
        读取importance_file到self.output中
        '''
        pattern = f"{directory}/*.jsonl"
        # 使用 glob 查找符合模式的文件
        files_names = glob.glob(pattern)
        self.output = dict()
        for f_n in files_names:
            with open(f_n, 'r', encoding='utf-8') as file:
                output_id = int(f_n.split('/')[-1].split('_')[0])
                self.output[output_id] = json.load(file)

    def read_mem0_memory(self):
        '''
        读取mem0结果到self.output中
        '''
        if len(self.label) < 1:
            raise ValueError("在读取mem0前，必须先读取label")
        
        id_cards = list(self.label.keys())
        self.output = dict()
        for id_card in id_cards:
            mem = get_rag_mem0(session_id=id_card, dialogue='你喜欢吃什么', top_k=500, round=1)
            self.output[id_card] = mem

    def read_memochat_memory(self, directory):
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
        
        pattern = f"{directory}/*.json"
        # 使用 glob 查找符合模式的文件
        file_paths = glob.glob(pattern)
        self.output = dict()

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            output_id = int(file_path.split('/')[-1].split('_')[-1].split('.')[0])
            
            for line in range(len(data)):
                for i in range(len(data[line]["conversations"])-1, 0, -1):
                    if data[line]["conversations"][i].get("thinking"):
                        last_thinking = data[line]["conversations"][i]["thinking"]
                        break
            
                start = 'nTopic Options:'
                end = '\\n```\\n\\n```\\n任务介绍：'
                split_thinking = extract_between(last_thinking, start, end)
                split_thinking2 = split_thinking.split('\\n')

                remove_id_thinking = []
                for s_t in split_thinking2:
                    m = remove_id(s_t)
                    remove_id_thinking.append(m)
                
                # 这里要去掉第一个为空的列表，以及第一个为'NOTO. None of the others.'的列表
                self.output[output_id] = remove_id_thinking[2:]

    def read_memorybank_memory(self, directory):
        '''
        读取memorybank_file到self.output中
        '''
        pattern = f"{directory}/*.jsonl"
        # 使用 glob 查找符合模式的文件
        files_names = glob.glob(pattern)
        self.output = dict()
        for f_n in files_names:
            with open(f_n, 'r', encoding='utf-8') as file:
                output_id = int(f_n.split('/')[-1].split('.')[0])
                self.output[output_id] = json.load(file)

    def process_importance_data(self, no_summary=True):
        '''
        对importancedata，将output中的信息转换为一个扁平dict
        '''
        k_l = self.output.keys()
        formalized_output = dict()
        for k in k_l:
            output_list = []
            for x in self.output[k]:
                if no_summary:
                    if isinstance(no_summary, int) and x.count(':') == 1:
                        x_cut = extract_b_from_colon_string(x)
                        if len(x_cut) >= no_summary:
                            print("在nosummary函数中，原本内容：", x, '更改后内容', x_cut)
                            x = x_cut          
                    else:
                        x = extract_b_from_colon_string(x)
                output_list.append(x)
            formalized_output[k] = output_list
        return formalized_output
    
    def process_mem0(self):
        '''
        对储存了mem0的output，将记忆信息转换为一个扁平dict
        '''
        k_l = self.output.keys()
        formalized_output = dict()
        for k in k_l:
            output_list = []
            for x in self.output[k]['memories']:
                output_list.append(x['memory'])
            formalized_output[k] = output_list
        
        return formalized_output
    
    def process_memochat(self):
        '''
        对储存了memochat的output，将记忆信息转换为一个扁平dict
        '''
        return self.output.copy() 

    def process_memorybank_memory(self):
        '''
        对储存了memochat的output，将记忆信息转换为一个扁平dict
        '''
        k_l = self.output.keys() 
        formalized_output = dict()
        for k in k_l:
            output_list = []
            for o_l in self.output[k]:
                for o_key in o_l:
                    output_list.append(o_l[o_key].strip())
            formalized_output[k] = output_list
        return formalized_output

    def process_label(self):
        '''
        将label中的信息转换为一个扁平列表
        '''
        k_l = self.label.keys()
        formalized_label = dict()
        for k in k_l:
            label_list = []
            for v_l in self.label[k]['memory_human_label'].values():
                label_list.extend(v_l)
            formalized_label[k] = label_list   
        
        return formalized_label
    
    def add_grade_llm_juge(self, LLMgrade, tmp_dict, llm_thres):
        if LLMgrade.isdigit():
            tmp_dict["LLMgrade"] = int(LLMgrade)
            # self.score_dict["LLMgrade_all"] = self.score_dict["LLMgrade_all"] + int(LLMgrade)
            self.score_dict = dict_add(self.score_dict, tmp_dict, self.score_key + ["LLMgrade"], '_all')
            self.l_llm_dict["l_LLM_all"] = self.l_llm_dict["l_LLM_all"] + 1

            for i in range(0, len(llm_thres)):
                if int(LLMgrade) <= llm_thres[i]:
                    self.score_dict = dict_add(self.score_dict, tmp_dict, self.score_key + ["LLMgrade"], '_' + str(i))
                    self.l_llm_dict["l_LLM_" + str(i)] = self.l_llm_dict["l_LLM_" + str(i)] + 1
                    break

    def cal_indicator(self, label_dir, prompt_template, importance_dir=None, bertscore_candidate = 'label', no_summary=True,
                      dataset='importance_data', llm_thres=5, rogue_mode='rouge-2', llm_judge=None, indicator_mode='bertscore', 
                      rouge_candidate='label', memochat_dir=None, memorybank_dir=None, bertsocre_max_standard='F1'):
        if dataset not in self.dataset_set:
            raise ValueError(f'dataset:{dataset}不符合规定')
        
        if dataset == 'importance_data' and importance_dir==None:
            raise ValueError(f"需要importance_data文件夹位置")
        
        self.read_label(label_dir)
        label = self.process_label()

        if dataset == 'importance_data':
            self.read_importance_file(importance_dir)
            mem_list = self.process_importance_data(no_summary=no_summary)
        elif dataset == 'mem0':
            self.read_mem0_memory()
            mem_list = self.process_mem0()
        elif dataset == 'memochat':
            self.read_memochat_memory(memochat_dir)
            mem_list = self.process_memochat()
        elif dataset == 'memory_bank':
            self.read_memorybank_memory(directory=memorybank_dir)
            mem_list = self.process_memorybank_memory()

        id_cards = self.label.keys()

        l = 0  # 全部的信息条数
        self.score_dict = {}  # 所有数值指标的均值
        self.score_key = []
        score_llm_key = []
        self.l_llm_dict = {}
        l_llm_key = []

        if indicator_mode == "bertscore":
            self.score_key = ["Precision", "Recall", "F1"]
        elif indicator_mode == "bge":
            self.score_key = ["bge"]
        elif indicator_mode == "m3e":
            self.score_key = ["m3e"]
        elif indicator_mode == 'rouge':
            self.score_key = ["Precision", "Recall", "F1"]
            rouge = Rouge()
        else:
            raise ValueError(f"indicator_mode {indicator_mode} not exist")

        for s in self.score_key:
            self.score_dict[s] = 0

        if llm_judge is not None:
            if not isinstance(llm_thres, list):
                llm_thres = [llm_thres]
            if llm_thres[0] != 0:
                llm_thres.insert(0, 0)
            # llm_thres.append(9999999999999)
            score_key_id = [str(i) for i in range(len(llm_thres))]
            score_key_id.append('all')

            # score_llm_key.append('LLMgrade_all')
            # self.score_key.append('LLMgrade')
            for s_k_i in score_key_id:
                score_llm_key.append('LLMgrade' + '_' + s_k_i)
            
            for s_k in self.score_key:
                for s_k_i in score_key_id:
                    score_llm_key.append(s_k + '_' + s_k_i)

            # l_llm_key = ["l_LLM_all"]
            for s_k_i in score_key_id:
                l_llm_key.append("l_LLM" + '_' + s_k_i)

            for s in score_llm_key:
                self.score_dict[s] = 0

            for i in l_llm_key:
                self.l_llm_dict[i] = 0
                    
        for id_card in id_cards:
            if indicator_mode == "bertscore":
                if bertscore_candidate == 'label':
                    candidate = label[id_card]
                    reference = mem_list[id_card]
                else:
                    candidate = mem_list[id_card]
                    reference = label[id_card]
                for c in candidate:
                    (P_i, R_i, F1_i), refer_i = self.scorer.score_reference([c], [reference])
                    tmp_dict = {}
                    tmp_dict["Precision"] = P_i
                    tmp_dict["Recall"] = R_i
                    tmp_dict["F1"] = F1_i
                    
                    self.score_dict = dict_add(self.score_dict, tmp_dict, self.score_key)
                    l = l + 1
                    # print(l)

                    if llm_judge is not None:
                        bertsocre_max_standard_dict = {"F1":2, "R":1, "P":0}
                        bertsocre_max_id = bertsocre_max_standard_dict[bertsocre_max_standard]
                        prompt = prompt_template.replace("{Adialogue}", reference[refer_i[bertsocre_max_id]]).replace("{Bdialogue}", c)
                        if llm_judge == 'api':
                            LLMgrade = lsl_qwen_70b(prompt)
                        else:
                            LLMgrade = model_process(tokenizer=self.judge_tokenizer, model=self.judge_model, prompt=prompt).split('\n')[-1]
                        tmp_dict["LLMgrade"] = LLMgrade

                        print("记忆信息", reference[refer_i[2]], "label", c, "LLM比较分数", LLMgrade)
                        self.add_grade_llm_juge(LLMgrade, tmp_dict, llm_thres)
                            
            elif indicator_mode == "bge":
                label_i = label[id_card]
                mem_i = mem_list[id_card]  
                for l_i in label_i:
                    grade_i, _ = bge_similar(self.bge, l_i, mem_i)
                    tmp_dict = {}
                    tmp_dict["bge"] = grade_i
                    
                    self.score_dict = dict_add(self.score_dict, tmp_dict, self.score_key)
                    l = l + 1

                    if llm_judge is not None:
                        prompt = prompt_template.replace("{Adialogue}", reference[refer_i[2]]).replace("{Bdialogue}", c)
                        if llm_judge == 'api':
                            LLMgrade = lsl_qwen_70b(prompt)
                        else:
                            LLMgrade = model_process(tokenizer=self.judge_tokenizer, model=self.judge_model, prompt=prompt).split('\n')[-1]
                        tmp_dict["LLMgrade"] = LLMgrade

                        self.add_grade_llm_juge(LLMgrade, tmp_dict, llm_thres)

            elif indicator_mode == "m3e":
                label_i = label[id_card]
                mem_i = mem_list[id_card]  
                
                label_i_emb = self.m3e.encode(label_i)
                mem_i_emb = self.m3e.encode(mem_i)

                similarity = util.cos_sim(label_i_emb, mem_i_emb)
                row_max, row_indices = torch.max(similarity, dim=1)

                m3e_grade = torch.sum(row_max)
                tmp_dict = {}
                tmp_dict['m3e'] = m3e_grade

                self.score_dict = dict_add(self.score_dict, tmp_dict, self.score_key)
                l = l + len(row_max)
                print(l)
                if llm_judge is not None:
                    for j, label_sentence in enumerate(label_i):     
                        max_mem = mem_i[row_indices[j]]
                        prompt = prompt_template.replace("{Adialogue}", max_mem).replace("{Bdialogue}", label_sentence)
                        if llm_judge == 'api':
                            LLMgrade = lsl_qwen_70b(prompt)
                        else:
                            LLMgrade = model_process(tokenizer=self.judge_tokenizer, model=self.judge_model, prompt=prompt).split('\n')[-1]
                        tmp_dict["LLMgrade"] = LLMgrade
                        tmp_dict['m3e'] = row_max[j]
                        self.add_grade_llm_juge(LLMgrade, tmp_dict, llm_thres)

            elif indicator_mode == "rouge":               
                if rouge_candidate == 'label':
                    candidate = label[id_card]
                    reference = mem_list[id_card]
                else:
                    candidate = mem_list[id_card]
                    reference = label[id_card]
                
                # 遍历每一个candidate，计算BERT评分
                for c in candidate:
                    c = ' '.join(jieba.cut(c))
                    c_l = [c] * len(reference)
                    reference_ls = [' '.join(jieba.cut(sentence)) for sentence in reference]
                    scores = rouge.get_scores(c_l, reference_ls)
                    max_index, max_value = max(enumerate(scores), key=lambda x: x[1][rogue_mode]['f'])

                    # 获取 rouge-2 的 r, p, f 值
                    rouge_2_values = max_value[rogue_mode]
                    tmp_dict = {}
                    tmp_dict["Precision"] = rouge_2_values['r']
                    tmp_dict["Recall"] = rouge_2_values['p']
                    tmp_dict["F1"] = rouge_2_values['f']
                    
                    self.score_dict = dict_add(self.score_dict, tmp_dict, self.score_key)
                    l = l + 1

                    if llm_judge is not None:
                        prompt = prompt_template.replace("{Adialogue}", reference[max_index]).replace("{Bdialogue}", c)
                        if llm_judge == 'api':
                            LLMgrade = lsl_qwen_70b(prompt)
                        else:
                            LLMgrade = model_process(tokenizer=self.judge_tokenizer, model=self.judge_model, prompt=prompt).split('\n')[-1]
                        tmp_dict["LLMgrade"] = LLMgrade

                        self.add_grade_llm_juge(LLMgrade, tmp_dict, llm_thres)

        print('最终结果')
        print('个数', l)
        print('分数', end=' ')
        for k in self.score_key:
            if '_' not in k:
                print(k, self.score_dict[k]/l, end=' ')
        print()
        
        if llm_judge is not None:
            print('所有有关大模型的结果')
            print("个数", self.l_llm_dict["l_LLM_all"], "整体大模型分数:", self.score_dict["LLMgrade_all"]/self.l_llm_dict["l_LLM_all"])
            print('大模型能评分的'+ indicator_mode +'数值结果')
            for i, thres in enumerate(llm_thres):
                if i==0:
                    print('大模型0分的结果：', end=' ')
                else:
                    print(f'大模型{llm_thres[i-1]}到{llm_thres[i]}分的结果：', end=' ')
                print("个数", self.l_llm_dict["l_LLM_" + str(i)], end=' ')
                print("大模型分数", self.score_dict['LLMgrade' + '_' + str(i)]/self.l_llm_dict["l_LLM_" + str(i)], end=' ')
                for k in self.score_key:   
                    l_k = k + '_' + str(i)
                    print(k, self.score_dict[l_k]/self.l_llm_dict["l_LLM_" + str(i)], end=' ')
                print()

    def cal_indicator_qwen(self, label_dir, prompt_template, importance_dir=None, bertscore_candidate = 'label', no_summary=True,
                      dataset='importance_data', llm_thres=5, rogue_mode='rouge-2', llm_judge=None, indicator_mode='bertscore', 
                      rouge_candidate='label', memochat_dir=None, memorybank_dir=None, bertsocre_max_standard='F1'):
        if dataset not in self.dataset_set:
            raise ValueError(f'dataset:{dataset}不符合规定')
        
        if dataset == 'importance_data' and importance_dir==None:
            raise ValueError(f"需要importance_data文件夹位置")
        
        self.read_label(label_dir)
        label = self.process_label()

        if dataset == 'importance_data':
            self.read_importance_file(importance_dir)
            mem_list = self.process_importance_data(no_summary=no_summary)
        elif dataset == 'mem0':
            self.read_mem0_memory()
            mem_list = self.process_mem0()
        elif dataset == 'memochat':
            self.read_memochat_memory(memochat_dir)
            mem_list = self.process_memochat()
        elif dataset == 'memory_bank':
            self.read_memorybank_memory(directory=memorybank_dir)
            mem_list = self.process_memorybank_memory()

        id_cards = self.label.keys()

        l = 0  # 全部的信息条数
        self.score_dict = {}  # 所有数值指标的均值
        self.score_key = []
        score_llm_key = []
        self.l_llm_dict = {}
        l_llm_key = []

        if indicator_mode == "bertscore":
            self.score_key = ["Precision", "Recall", "F1"]
        elif indicator_mode == "bge":
            self.score_key = ["bge"]
        elif indicator_mode == "m3e":
            self.score_key = ["m3e"]
        elif indicator_mode == 'rouge':
            self.score_key = ["Precision", "Recall", "F1"]
            rouge = Rouge()
        else:
            raise ValueError(f"indicator_mode {indicator_mode} not exist")

        for s in self.score_key:
            self.score_dict[s] = 0

        if llm_judge is not None:
            if not isinstance(llm_thres, list):
                llm_thres = [llm_thres]
            if llm_thres[0] != 0:
                llm_thres.insert(0, 0)
            # llm_thres.append(9999999999999)
            score_key_id = [str(i) for i in range(len(llm_thres))]
            score_key_id.append('all')

            # score_llm_key.append('LLMgrade_all')
            # self.score_key.append('LLMgrade')
            for s_k_i in score_key_id:
                score_llm_key.append('LLMgrade' + '_' + s_k_i)
            
            for s_k in self.score_key:
                for s_k_i in score_key_id:
                    score_llm_key.append(s_k + '_' + s_k_i)

            # l_llm_key = ["l_LLM_all"]
            for s_k_i in score_key_id:
                l_llm_key.append("l_LLM" + '_' + s_k_i)

            for s in score_llm_key:
                self.score_dict[s] = 0

            for i in l_llm_key:
                self.l_llm_dict[i] = 0
                    
        for id_card in id_cards:
            if indicator_mode == "bertscore":
                
                # memory作为候选candidate，label作为参考reference，此时应该调用recall。
                # 每一个memory按照recall，然后计算分数包含分数，归类到某一个label中去
                # 最后对所有label，取最高分
                candidate = mem_list[id_card]
                reference = label[id_card]

                label_grade_dict = dict()
                for c in candidate:
                    (P_i, R_i, F1_i), refer_i = self.scorer.score_reference([c], [reference])
                    tmp_dict = {}
                    tmp_dict["Precision"] = P_i
                    tmp_dict["Recall"] = R_i
                    tmp_dict["F1"] = F1_i
                    
                    self.score_dict = dict_add(self.score_dict, tmp_dict, self.score_key)
                    l = l + 1

                    if llm_judge is not None:
                        bertsocre_max_standard_dict = {"F1":2, "R":1, "P":0}
                        bertsocre_max_id = bertsocre_max_standard_dict[bertsocre_max_standard]
                        prompt = prompt_template.replace("{Adialogue}", c).replace("{Bdialogue}", reference[refer_i[bertsocre_max_id]])
                        if llm_judge == 'api':
                            LLMgrade = lsl_qwen_70b(prompt)
                        else:
                            LLMgrade = model_process(tokenizer=self.judge_tokenizer, model=self.judge_model, prompt=prompt).split('\n')[-1]
                        tmp_dict["LLMgrade"] = LLMgrade

                        print("个数", l, "记忆信息", c,  "label", reference[refer_i[2]], "LLM比较分数", LLMgrade)

                        if reference[refer_i[2]] not in label_grade_dict:
                            label_grade_dict[reference[refer_i[2]]] = [[],[],[],[],[],[]]

                        label_grade_dict[reference[refer_i[2]]][int(LLMgrade)].append((c, tmp_dict))
                
                for l_key in label_grade_dict:
                    for i in range(5, -1, -1):
                        if len(label_grade_dict[l_key][i]) > 0:
                            self.add_grade_llm_juge(str(i), label_grade_dict[l_key][i][0][1], llm_thres)
                            break
                            
        print('数据集', dataset)
        print('最终结果')
        print('个数', l)
        print('分数', end=' ')
        for k in self.score_key:
            if '_' not in k:
                print(k, self.score_dict[k]/l, end=' ')
        print()
        
        if llm_judge is not None:
            print('所有有关大模型的结果')
            print("个数", self.l_llm_dict["l_LLM_all"], "整体大模型分数:", self.score_dict["LLMgrade_all"]/self.l_llm_dict["l_LLM_all"])
            print('大模型能评分的'+ indicator_mode +'数值结果')
            for i, thres in enumerate(llm_thres):
                if i==0:
                    print('大模型0分的结果：', end=' ')
                else:
                    print(f'大模型{llm_thres[i-1]}到{llm_thres[i]}分的结果：', end=' ')
                print("个数", self.l_llm_dict["l_LLM_" + str(i)], end=' ')
                print("大模型分数", self.score_dict['LLMgrade' + '_' + str(i)]/self.l_llm_dict["l_LLM_" + str(i)], end=' ')
                for k in self.score_key:   
                    l_k = k + '_' + str(i)
                    print(k, self.score_dict[l_k]/self.l_llm_dict["l_LLM_" + str(i)], end=' ')
                print()

if __name__ == '__main__':
    prompt0_5 = "请你帮我评判A句子是否准确包含B句子中蕴含的信息，A句子：{Adialogue}, B句子：{Bdialogue}\n请你只输出0-5的一个数字，0代表A句子与B句子毫无关系，5代表A句子完全包含了B句子想要表达的信息。注意，只输出0-5的一个数字，不要输出其它。"

    bertmodel_path = 'model/bert-base-chinese'
    judge_model_path = 'model/Qwen1.5-14B-Chat'

    label_dir = 'label'  
    importance_dir = 'importance_pool/dec5th_all_32b'
    memorybank_dir = ''

    bge_path = 'model/FlagEmbedding/BAAI/bge-m3'
    bge_reranker_path = 'model/bge_reranker'

    bertscore_model = BERTScorer(model_type=bertmodel_path)

    x = MemoryEvaluator(bertscore=bertscore_model)

    x.cal_indicator_qwen(label_dir=label_dir, prompt_template=prompt0_5, dataset='importance_data', importance_dir='importance_pool/test_experiment', indicator_mode='bertscore', llm_judge='api', llm_thres=[0, 3, 5], bertsocre_max_standard='R', no_summary=5)



    
