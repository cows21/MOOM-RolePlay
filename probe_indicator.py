import json
import requests
import random
import re
import time
from openai import OpenAI
from utils.model_api import store_memorybank, get_rag_memorybank, store_kv_memory, get_kv_rag, get_sum_rag, store_summary, lsl_qwen_70b, store_mem0, get_rag_mem0

def insert_probe(data_line, user_probe, bot_probe, position):
    messages = data_line["messages"].copy()
    
    user_name = data_line['role_meta']['user_name']
    bot_name = data_line['role_meta']['primary_bot_name']

    user_probe_dict = dict()
    user_probe_dict["sender_name"] = user_name
    user_probe_dict["text"] = user_probe
    user_probe_dict["flag"] = "PROBE"

    bot_probe_dict = dict()
    bot_probe_dict["sender_name"] = bot_name
    bot_probe_dict["text"] = bot_probe
    bot_probe_dict["flag"] = "PROBE"

    messages.insert(position, bot_probe_dict)
    messages.insert(position, user_probe_dict)

    return messages

def generate_conversation(messages):
    """
    生成整个对话的文本，每条消息后添加换行。
    返回:
    - 格式化的对话字符串。
    """
    conversation = []
    stop_i = 6
    for message in messages:
        text = message['text']
        sender = message['sender_name']
        formatted_text = f"{sender}: {text}\n"
        conversation.append(formatted_text)
        stop_i -= 1
        if stop_i ==0:
            break
    return ''.join(conversation)

def probe_memory(memory_method):
    file_path = 'data/ZH-4O_dataset.jsonl'
    probe_file_path = 'probe/probe_test.json'
    line_number = 1
    user_probe = '我想吃苹果'
    bot_probe_ls = ['哦', '好的', "我知道了"]
    dia_turn = 2
    split_num = 20
    top_k = 3
    memory_method = memory_method
    

    data_all = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_all.append(json.loads(line))
    
    with open(probe_file_path, 'r', encoding='utf-8') as file:
        probe_data = json.load(file)

    sample_num = 0
    all_grade = 0

    for probe_id, p_d_i in enumerate(probe_data):
        # 探针信息注入数据
        for probe_item in p_d_i["probe"]:
            messages = insert_probe(data_all[line_number], probe_item["content"], random.choice(bot_probe_ls), probe_item["position"])
            split_data = [messages[i:i + split_num] for i in range(0, len(messages), split_num)] 

        rag_i = 0
        session_id = line_number * 1000 + probe_id
        for r, d in enumerate(split_data):
            dia_dict = data_all[line_number].copy()
            dia_dict['messages'] = d
            if memory_method == 'importance_data':
                store_kv_memory(messages=dia_dict, session_id=session_id)
                store_summary(messages=dia_dict, session_id=session_id)
            elif memory_method == 'mem0':
                store_mem0(messages=dia_dict, session_id=session_id)
            elif memory_method == 'memorybank':
                store_memorybank(messages=dia_dict, session_id=session_id)


            # 探针问题
            if rag_i < len(p_d_i["query"]):
                if p_d_i["query"][rag_i]["position"] // 20 == r:
                    # rag（可变，选用哪个记忆存储方法）
                    if memory_method == 'importance_data':
                        kv_result = get_kv_rag(session_id=session_id, dialogue=p_d_i["query"][rag_i]["content"], top_k=top_k, round=r, importance_freeze=True)
                        summary_result = get_sum_rag(session_id=session_id, dialogue=p_d_i["query"][rag_i]["content"], top_k=top_k, round=r, importance_freeze=True)
                    elif memory_method == 'mem0':
                        mem0_result = get_rag_mem0(session_id=session_id, dialogue=p_d_i["query"][rag_i]["content"], top_k=top_k, round=r)
                    elif memory_method == 'memorybank':
                        memorybank_result = get_rag_memorybank(session_id=session_id, dialogue=p_d_i["query"][rag_i]["content"], top_k=top_k, round=r)

                    result_ls = list()

                    if memory_method == 'importance_data':
                        for key_kv in kv_result:
                            for value_item in kv_result[key_kv]:
                                result_ls.append(key_kv + ':' + value_item)
                        
                        for summary_level in summary_result:
                            for summary_item in summary_level:
                                result_ls.append(summary_item['content'])
                    elif memory_method == 'mem0':
                        for mem0_item in mem0_result['memories']:
                            result_ls.append(mem0_item['memory'])
                    elif memory_method == 'memorybank':
                        result_ls = memorybank_result
                    
                    prompt_generate_dia_template = '你现在正在做角色扮演，你扮演的角色是{bot_name}。用户扮演的角色是{user_name}。请你根据我给你的记忆信息以及近期对话回应{user_name}，参考他们之前的对话内容：{dia}。{user_name}的消息：{user_chat}，相关记忆信息：{memory}。请直接给你的回答。'
                    memory_info = str(result_ls)
                    prompt_generate_dia = prompt_generate_dia_template.replace('{user_chat}', p_d_i["query"][rag_i]["content"])
                    prompt_generate_dia = prompt_generate_dia.replace('{memory}', memory_info)
                    prompt_generate_dia = prompt_generate_dia.replace('{bot_name}', dia_dict['role_meta']['primary_bot_name'])
                    prompt_generate_dia = prompt_generate_dia.replace('{user_name}', dia_dict['role_meta']['user_name'])
                    prompt_generate_dia = prompt_generate_dia.replace('{dia}', generate_conversation(dia_dict['messages']))
                    prompt_generate_dia = prompt_generate_dia.replace('{bot_info}', str(dia_dict['prompt'][dia_dict['role_meta']['primary_bot_name']]))
                    print("生成对话的prompt")
                    print(prompt_generate_dia)

                    answer_from_model = lsl_qwen_70b(prompt_generate_dia)

                    prompt_indicator_template = "我这里有针对消息“{user_chat}”的两个回答，请你帮我评判A回答是否准确包含B回答中蕴含的信息，A回答：{Adialogue}, B回答：{Bdialogue}\n请你只输出0-5的一个数字，0代表B回答的信息在A回答完全没有体现，5代表A回答完全包含了B回答想要表达的信息。A回答肯能会包含其他信息，这不应该干扰你的打分，你只要关注A回答是否准确包含B回答中蕴含的信息。你可以给出打分的简短理由，但是你给出的分数需要出现在最后，并且用{}包围，且只输出0-5的一个数字。"
                    
                    prompt_indicator = prompt_indicator_template.replace('{Adialogue}', answer_from_model)
                    
                    prompt_indicator = prompt_indicator.replace('{Bdialogue}', p_d_i["query"][rag_i]["answer"])
                    prompt_indicator = prompt_indicator.replace('{user_chat}', p_d_i["query"][rag_i]["content"])
                    
                    grade_str_item = lsl_qwen_70b(prompt_indicator)

                    match_braces = re.search(r'\{([^{}]*)\}(?!.*\{)', grade_str_item)
                    if match_braces:
                        grade_int_item = match_braces.group(1)
                    else:
                        grade_int_item = None

                    if grade_int_item.isdigit():
                        print('label answer:', p_d_i["query"][rag_i]["answer"])
                        print('model answer:', answer_from_model)
                        print('分数以及原因:', grade_str_item)
                        all_grade = all_grade + int(grade_int_item)
                        sample_num += 1

                    rag_i = rag_i + 1 

                    if rag_i == len(p_d_i["query"]):
                        break
        
    print("最终分数", all_grade / sample_num)

if __name__ == "__main__":
    file_path = 'data/ZH-4O_dataset.jsonl'
    method_ls = ["importance_data", "mem0", "memorybank"]
    
    for m in method_ls:
        start_time = time.time()
        y = probe_memory(memory_method=m)
        end_time = time.time()
        print(f"当前方法{m}的程序运行时间：{end_time - start_time} 秒")
    z = 20


    