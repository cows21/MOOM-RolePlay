import json
import os
import re
import math
import copy
import pickle
from collections import OrderedDict
import random

def remove_suffix(text):
    # 只匹配括号内为数字+“轮对话前”的部分，并将其替换为空字符串
    x = re.sub(r'\(\d+轮对话前\)', '', text).strip()
    # print("remove_suffix 后的结果", x)
    return x

def match_dialogue_rounds(text):
    # 使用正则表达式匹配 "(数字轮对话前)" 的格式
    match = re.search(r'\(\d+轮对话前\)', text)
    
    # 如果找到匹配，返回匹配的字符串，否则返回空字符串
    if match:
        return match.group(0)
    else:
        return ''
    
def split_string_on_colon(text):
    # 使用 split 方法并指定 maxsplit=1，这样只会根据第一个冒号进行分割
    parts = text.split(":", 1)
    
    # 如果字符串中没有冒号，返回原字符串和空字符串
    if len(parts) == 1:
        return parts[0], ""
    else:
        return parts[0], parts[1]

class ImportanceHandler:
    def __init__(self, session_id, importance_dir='importance_pool'):
        """
        初始化方法，接受jsonl文件路径。
        """
        self.importance_dir = importance_dir
        self.session_id = session_id
        self.importance_file_path = self._get_importance_file_path()
        # importance_data是一个字典，key是type为字符串的信息，value是字典，格式如下：
        # {"birth": (int)入库轮数, "score":(float) 重要性分数, "retrieve":(list of list) 被调用的所有轮数, "type_mem":(string) 来源于哪一种记忆}
        # type暂时有"kv", "summary", 两种
        self.importance_data = dict()
        self.trajectory_keys = [
            "职业相关", "经济相关", "健康相关", "社会地位", "生活习惯", "发生事件"
        ]
        self.initialize_importance

    def _get_importance_file_path(self):
        """
        根据原始文件路径生成对应的重要性文件路径。
        """
        importance_file_name = f"{self.session_id}_importance.jsonl"
        return os.path.join(self.importance_dir, importance_file_name)
    
    def initialize_importance(self):
        """
        初始化重要性标注文件，如果文件不存在则创建。
        """
        print("this is importance_file_path" ,self.importance_file_path)
        if os.path.exists(self.importance_file_path):
            print('yes, importance_file_path exist')
            with open(self.importance_file_path, 'r', encoding='utf-8') as f:
                self.importance_data = json.load(f)
        else:
            print('importance_file_path does not exist, creating a new one...')
            # 如果文件不存在，创建新文件并写入一个空字典
            self.importance_data = {}
            with open(self.importance_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.importance_data, f, ensure_ascii=False, indent=4)
            print('New file created with an empty dictionary')

    def alignment_kv(self, memory, round):
        """
        importance与memory的对齐，包括删去不再存在的记忆，加入新的记忆
        memory格式：一系列的key与value，key是字符串，value是字符串列表
        memory格式示例：{"key1":["v1_1", "v1_2"], "key2":["v2_1"]}
        """
        memory_n = copy.deepcopy(memory)
        if os.path.exists(self.importance_file_path):
            with open(self.importance_file_path, 'r', encoding='utf-8') as f:
                self.importance_data = json.load(f)

            for key, _ in memory_n.items():
                if key in self.trajectory_keys:
                    for i in range(len(memory_n[key])):
                        new_v = remove_suffix(memory_n[key][i])
                        memory_n[key][i] = new_v
            
            # print('--------去除前缀后的memory-------------')
            # print(memory)
                        
            for key, values in memory_n.items():
                for v in values:
                    memory_info = f"{key}:{v}"
                    if memory_info not in self.importance_data:
                        d = {"birth": round, "score":0, "retrieve":[], "type_mem":"kv"}
                        self.importance_data[memory_info] = d

            importance_data_cp = copy.deepcopy(self.importance_data)
            for key in importance_data_cp:
                if importance_data_cp[key]["type_mem"] == "kv":
                    info = key.split(':')
                    if len(info) != 2:
                        print("when split info, something error happens. This is info:", info)
                    else:
                        k = info[0]
                        v = info[1]
                        if k not in memory_n:
                            del self.importance_data[key]
                        else:
                            if v not in memory_n[k]:
                                del self.importance_data[key]

        self._save_importance()

    def alignment_summary(self, summary, chunk_sizes, round):
        """
        importance与summary的对齐，包括删去不再存在的记忆，加入新的记忆
        summary的格式如下：[s1, s2, s3]
        s1, s2, s3的格式相似，结构如下：
        {"id": {"start_time": int, "end_time": int, "content": string, "num": int}}
        """
        # print('-------------IN alignment_summary-------------')
        flag = [chunk_sizes[0], 1, 1]
        summary_n = copy.deepcopy(summary)
        for i in range(3):
            last_key = None
            if len(summary_n[i]) > 0:
                if i == 0:
                    last_key = list(summary_n[i].keys())[-1]
                    l_content = last_key
                else:
                    if len(summary_n[i]) > 1:
                        last_key = list(summary_n[i].keys())[-1]
                        l_content = list(summary_n[i].keys())[-2]

                # if last_key is not None:
                #     print("数据ID")
                #     print(last_key)
                #     print("summary_n[i][last_key]['num'] 与 chunk_sizes[i]")
                #     print(summary_n[i][last_key]['num'], chunk_sizes[i])

                if last_key is not None and summary_n[i][last_key]['num'] == flag[i]:
                    d = {"birth": round, "score":0, "retrieve":[], "type_mem":"summary"}
                    self.importance_data[summary_n[i][l_content]['content']] = d
                    # print("要在importance文件中录入以下内容")
                    # print(summary_n[i][l_content]['content'], d)
                    # print("当前self.importance_data")
                    # print(self.importance_data)

        self._save_importance()

    def _save_importance(self):
        """"
        保存重要性文件内容。
        """
        with open(self.importance_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.importance_data, f, ensure_ascii=False)

    def update_importance_kv(self, key, value, round):
        """
        更新重要性文件，记录某条信息被查询的次数。
        """
        if key in self.trajectory_keys:
            value = remove_suffix(value)
        memory_info = f"{key}:{value}"

        # print('@@@@@@@@@在update_importance_memory内部@@@@@@@@@@')
        # print('要更新的summary', memory_info)
        # print('当前importance_data')
        # print(self.importance_data)
        # print('@@@@@@@@@一次结束@@@@@@@@@@')

        if memory_info in self.importance_data:
            self.importance_data[memory_info]["retrieve"].append(round)
        else:
            if key in self.trajectory_keys:
                value = remove_suffix(value)
            d = {"birth": round, "score":0, "retrieve":[], "type_mem":"kv"}
            self.importance_data[memory_info] = d

        # 保存更新后的重要性文件
        self._save_importance()

    def update_importance_summary(self, summary, round):
        """
        更新重要性文件，记录某条信息被查询的次数。
        """
        summary_info = summary

        # print('@@@@@@@@@在update_importance_summary内部@@@@@@@@@@')
        # print('要更新的summary', summary_info)
        # print('当前importance_data')
        # print(self.importance_data)
        # print('@@@@@@@@@一次结束@@@@@@@@@@')

        if summary_info in self.importance_data:
            self.importance_data[summary_info]["retrieve"].append(round)
        else:
            d = {"birth": round, "score":0, "retrieve":[], "type_mem":"summary"}
            self.importance_data[summary_info] = d

        # 保存更新后的重要性文件
        self._save_importance()

    def update_kv_file(self, kv_file_name):
        try:
            # 尝试打开并读取文件
            with open(kv_file_name, 'r', encoding='utf-8') as file:
                try:
                    dialogue_memory_list = json.load(file)
                except json.JSONDecodeError as e:
                    raise ValueError(f"无法解析JSON文件 {kv_file_name}: {e}")

                # 检查文件内容结构是否符合预期
                if not dialogue_memory_list or 'final_memory' not in dialogue_memory_list[-1]:
                    raise KeyError(f"{kv_file_name} 文件中缺少 'final_memory' 键")

                final_memory = dialogue_memory_list[-1]['final_memory']
                new_final_memory = dict()

                prefix_dict = {}
                for k, vs in final_memory.items():
                    for v in vs:
                        message = k + ':' + remove_suffix(v)
                        prefix_dict[message] = match_dialogue_rounds(v)

                # print('以下是储存的字典')
                # print(prefix_dict)

                for mes in self.importance_data:
                    if self.importance_data[mes]['type_mem'] == 'kv':
                        key, value = split_string_on_colon(mes)
                        if mes in prefix_dict:
                            # print('需要检索后缀的')
                            # print(value, prefix_dict[mes])
                            n_v = value + prefix_dict[mes]
                        else:
                            n_v = value
                        
                        if key in new_final_memory:
                            new_final_memory[key].append(n_v)
                        else:
                            new_final_memory[key] = [n_v]

                # 更新final_memory
                dialogue_memory_list[-1]['final_memory'] = new_final_memory

            # 尝试将更新后的数据写回文件
            with open(kv_file_name, 'w', encoding='utf-8') as file:
                # print('FINAL MEMORY!!!!!!!!!!')
                # print(dialogue_memory_list[-1]['final_memory'])
                json.dump(dialogue_memory_list, file, ensure_ascii=False, indent=4)

        except FileNotFoundError:
            print(f"文件 {kv_file_name} 未找到。请检查文件路径。")
        except KeyError as e:
            print(f"字典中缺少预期的键：{e}")
        except Exception as e:
            print(f"发生了一个意外错误：{e}")

    def update_summary_file(self, summary_text_file, summary_vector_file, reserve_num):
        print('---------in update_summary_file function---------')
        try:
            # 检查 summary_text_file 是否存在
            if not os.path.isfile(summary_text_file):
                raise FileNotFoundError(f"文本摘要文件 {summary_text_file} 不存在。")
            
            # 读取 summary_text_file
            with open(summary_text_file, 'r', encoding='utf-8') as file:
                try:
                    summary_txt_list = json.load(file)
                    if not isinstance(summary_txt_list, dict):
                        raise ValueError(f"{summary_text_file} 的内容应为字典格式。")
                except json.JSONDecodeError as e:
                    raise ValueError(f"无法解析 JSON 文件 {summary_text_file}: {e}")
            
            # 检查 summary_vector_file 是否存在
            if not os.path.isfile(summary_vector_file):
                raise FileNotFoundError(f"向量摘要文件 {summary_vector_file} 不存在。")
            
            # 读取 summary_vector_file
            with open(summary_vector_file, 'rb') as file:
                try:
                    summary_vec_list = pickle.load(file)
                    if not isinstance(summary_vec_list, dict):
                        raise ValueError(f"{summary_vector_file} 的内容应为字典格式。")
                except pickle.UnpicklingError as e:
                    raise ValueError(f"无法解析 Pickle 文件 {summary_vector_file}: {e}")
            
            # 初始化新的摘要列表
            new_summary_txt_ls = {}
            new_summary_vec_ls = {}
            
            total_entries = len(summary_txt_list)
            if reserve_num < 0:
                raise ValueError(f"reserve_num 不应该小于0")
            
            if reserve_num > total_entries:
                return 
            
            # 计算保留的起始索引
            l_txt = total_entries - reserve_num - 1
            
            # 迭代 summary_txt_list
            for index, (key_id, entry) in enumerate(summary_txt_list.items()):
                if not isinstance(entry, dict) or 'content' not in entry:
                    print(f"警告: 键 {key_id} 的条目格式不正确或缺少 'content' 键，跳过。")
                    continue
                
                content = entry['content']
                
                if index > l_txt:
                    if content in self.importance_data:
                        new_summary_txt_ls[key_id] = entry
                        if key_id in summary_vec_list:
                            new_summary_vec_ls[key_id] = summary_vec_list[key_id]
                        else:
                            print(f"警告: key_id {key_id} 存在于文本摘要中，但不存在于向量摘要中。")
                else:
                    new_summary_txt_ls[key_id] = entry
                    if key_id in summary_vec_list:
                        new_summary_vec_ls[key_id] = summary_vec_list[key_id]
                    else:
                        print(f"警告: key_id {key_id} 存在于文本摘要中，但不存在于向量摘要中。")
            
            # 写回更新后的文本摘要
            with open(summary_text_file, 'w', encoding='utf-8') as file:
                json.dump(new_summary_txt_ls, file, ensure_ascii=False, indent=4)
                print(f"成功更新文本摘要文件 {summary_text_file}。")
            
            # 写回更新后的向量摘要
            with open(summary_vector_file, 'wb') as file:
                pickle.dump(new_summary_vec_ls, file)
                print(f"成功更新向量摘要文件 {summary_vector_file}。")
        
        except FileNotFoundError as e:
            print(f"文件未找到错误: {e}")
        except ValueError as e:
            print(f"值错误: {e}")
        except KeyError as e:
            print(f"键错误: {e}")
        except Exception as e:
            print(f"发生了一个意外错误: {e}")

    def get_grade_exp(self, birth, retrieve, c):
        '''
        birth (float): 记忆诞生轮数
        retrieve (List(int)): 被检索的轮数，升序排列
        c (int): 当前轮数
        '''
        x = (c - birth) * 2 / c
        h = 0
        sig = 1
        if len(retrieve) > 0:
           for a in retrieve:
               h_i =  1/math.exp(sig * (c - a))   
               h = h + h_i
        f = pow(h, x)
        return f  

    def get_grade_iter(self, birth, retrieve, c, score):  
        '''
        TODO 这个函数暂时还是有bug，一些被检索到的数据分数却是0
        birth (float): 记忆诞生轮数
        retrieve (List(int)): 被检索的轮数，升序排列
        c (int): 当前轮数
        score(float): 当前分数
        '''
        decline_beta = 0.9
        new_score = score * decline_beta
        if retrieve[-1] == c:
            new_score = new_score + 1 + (c - birth)/(c)
            # print('有记忆被检索到了，增加权重！')
            # print('轮数', c, '旧分数', score, '新分数', new_score)
        return new_score
    
    def get_grade_exp_add(self, birth, retrieve, c):
        time_alpha = 0.1
        time_gamma = 1
        epsilon = 0.000001
        time_socre = time_alpha * (1 / (math.exp(time_gamma * (c - birth)) - (1 - epsilon)))
        # print("forget by add !!!!!!")

        retrieve_alpha = 1 -  time_alpha
        retrieve_delta = 1
        retrieve_score = 0
        for r in retrieve:
            r_s_i = 1 / (retrieve_delta * (c - r + epsilon))
            retrieve_score = retrieve_score + r_s_i

        retrieve_score = retrieve_alpha * retrieve_score

        score = time_socre + retrieve_score
        return score
    
    def get_grade_exp_only_time(self, birth, retrieve, c):
        time_alpha = 0.1
        time_gamma = 1
        epsilon = 0.000001
        time_socre = time_alpha * (1 / (math.exp(time_gamma * (c - birth)) - (1 - epsilon)))
        score = time_socre 
        return score
       
    def forgetting_curve(t, S):
        """
        Calculate the retention of information at time t based on the forgetting curve.

        :param t: Time elapsed since the information was learned (in days，如果被检索则置为0).
        :type t: float
        :param S: Strength of the memory.(每次被检索加1)
        :type S: float
        :return: Retention of information at time t.
        :rtype: float
        Memory strength is a concept used in memory models to represent the durability or stability of a memory trace in the brain. 
        In the context of the forgetting curve, memory strength (denoted as 'S') is a parameter that 
        influences the rate at which information is forgotten. 
        The higher the memory strength, the slower the rate of forgetting, 
        and the longer the information is retained.
        """
        return math.exp(-t / 5*S)

    def get_ebbinghaus_score(self, birth, retrieve, c):
        if len(retrieve) == 0:
            prob = self.forgetting_curve(c-birth, 1)
        else:
            prob = self.forgetting_curve(c-retrieve[-1], len(retrieve)+1)
        
        random_number = random.random()
        if random_number < prob:
            return 999999
        else:
            return 0

    def update_grade(self, c, mode='exp', inhibition_infos=None):
        try:
            for key, value in self.importance_data.items():
                b = self.importance_data[key]['birth']
                r = self.importance_data[key]['retrieve']
                if mode == 'iter':
                    self.importance_data[key].setdefault('score', 0)
                    cur_score = self.importance_data[key]['score']
                    self.importance_data[key]['score'] = self.get_grade_iter(b, r, c, cur_score)
                elif mode == 'exp':
                    self.importance_data[key]['score'] = self.get_grade_exp(b, r, c)
                elif mode == 'add':
                    self.importance_data[key]['score'] = self.get_grade_exp_add(b, r, c)
                elif mode == 'ebbinghaus_forget':
                    self.importance_data[key]['score'] = self.get_ebbinghaus_score(b, r, c)
                elif mode == 'add_inhibition':
                    self.importance_data[key]['score'] = self.get_grade_exp_add(b, r, c)
                    if inhibition_infos is not None:
                        if key in inhibition_infos:
                            origin_score = self.importance_data[key]['score']
                            if origin_score > 0:
                                after_score = self.importance_data[key]['score'] / 2
                            else:
                                after_score = self.importance_data[key]['score'] * 2
                            print(f'we found info to be inhibited, key:{key}, origin score:{origin_score}, after:{after_score}')
                            self.importance_data[key]['score'] = after_score
                        
            self._save_importance()
        except Exception as e:
            # 打印错误信息或记录到日志文件
            print(f"An error occurred: {e}")

    def filter_memory(self, top_n=10):
        filtered_data = OrderedDict()
        for key, value in self.importance_data.items():
            filtered_data[key] = value

        # 步骤3: 对过滤后的信息按查询次数进行排序，并只保留分数最大的前top_n个信息
        sorted_data = dict(sorted(filtered_data.items(), key=lambda item: item[1]['score'], reverse=True)[:top_n])

        # 返回被删除的信息列表
        deleted_data = [item for item in self.importance_data.keys() if item not in sorted_data]

        # 更新重要性文件，只保留top_n个信息
        self.importance_data = sorted_data
        # print('----------IN FILTER MEMORY----------')
        # print(self.importance_data)
        self._save_importance()

        return deleted_data

if __name__ == '__main__':
    importance_dir_1 = 'importance_pool'
    handler = ImportanceHandler(3, importance_dir=importance_dir_1)

    handler.initialize_importance()
    important_kv_file = 'memory_pool/3.jsonl'