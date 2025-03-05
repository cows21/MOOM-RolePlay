# -*- coding: utf-8 -*-
import sys 
sys.path.append('../memory_bank')
# from azure_client import LLMClientSimple
import openai, json, os
import argparse
import copy
import requests
from openai import OpenAI

def model_process(model, tokenizer, device, prompt, max_new_tokens=1024, use_api=False):
    if not use_api:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    else:
        return lsl_qwen_70b(prompt)

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
        ip_url = 'http://10.142.5.99:8012' + '/v1'
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

def summarize_content_prompt(content,user_name,boot_name,language='en'):
    prompt = '请总结以下的对话内容，尽可能精炼，提取对话的主题和关键信息。如果有多个关键事件，可以分点总结。对话内容：\n' if language=='cn' else 'Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately. Dialogue content:\n'
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        # prompt += f"\n用户：{query.strip()}"
        # prompt += f"\nAI：{response.strip()}"
        prompt += f"\n{user_name}：{query.strip()}"
        prompt += f"\n{boot_name}：{response.strip()}"
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt

def summarize_overall_prompt(content,language='en'):
    prompt = '请高度概括以下的事件，尽可能精炼，概括并保留其中核心的关键信息。概括事件：\n' if language=='cn' else "Please provide a highly concise summary of the following event, capturing the essential key information as succinctly as possible. Summarize the event:\n"
    for date,summary_dict in content:
        summary = summary_dict['content']
        prompt += (f"\n时间{date}发生的事件为{summary.strip()}" if language=='cn' else f"At {date}, the events are {summary.strip()}")
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt

def summarize_overall_personality(content,language='en'):
    prompt = '以下是用户在多段对话中展现出来的人格特质和心情，以及当下合适的回复策略：\n' if language=='cn' else "The following are the user's exhibited personality traits and emotions throughout multiple dialogues, along with appropriate response strategies for the current situation:"
    for date,summary in content:
        prompt += (f"\n在时间{date}的分析为{summary.strip()}" if language=='cn' else f"At {date}, the analysis shows {summary.strip()}")
    prompt += ('\n请总体概括用户的性格和AI恋人最合适的回复策略，尽量简洁精炼，高度概括。总结为：' if language=='cn' else "Please provide a highly concise and general summary of the user's personality and the most appropriate response strategy for the AI lover, summarized as:")
    return prompt

def summarize_person_prompt(content,user_name,boot_name,language):
    prompt = f'请根据以下的对话推测总结{user_name}的性格特点和心情，并根据你的推测制定回复策略。对话内容：\n' if language=='cn' else f"Based on the following dialogue, please summarize {user_name}'s personality traits and emotions, and devise response strategies based on your speculation. Dialogue content:\n"
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        # prompt += f"\n用户：{query.strip()}"
        # prompt += f"\nAI：{response.strip()}"
        prompt += f"\n{user_name}：{query.strip()}"
        prompt += f"\n{boot_name}：{response.strip()}"

    prompt += (f'\n{user_name}的性格特点、心情、{boot_name}的回复策略为：' if language=='cn' else f"\n{user_name}'s personality traits, emotions, and {boot_name}'s response strategy are:")
    return prompt

def summarize_memory(memory_dir, model, tokenizer, device, name=None, language='cn', use_api=False):
    boot_name = 'AI'
    memory = json.loads(open(memory_dir,'r',encoding='utf8').read())
    all_prompts,all_his_prompts, all_person_prompts = [],[],[]
    for k,v in memory.items():
        if name != None and k != name:
            continue
        user_name = k
        print(f'Updating memory for user {user_name}')
        print('---------------this is the whole memory--------------------')
        print(memory[user_name])
        if v.get('history') == None:
            print('找不到history')
            continue
        print('开始更新history')
        history = v['history']
        if v.get('summary') == None:
            memory[user_name]['summary'] = {}
        if v.get('personality') == None:
            memory[user_name]['personality'] = {}
        print('history长度')
        print(len(history))
        hi_id = 0
        for date, content in history.items():
            print('目前history ID')
            print(hi_id)
            hi_id = hi_id + 1
            # print(f'Updating memory for date {date}')
            his_flag = False if (date in v['summary'].keys() and v['summary'][date]) else True
            person_flag = False if (date in v['personality'].keys() and v['personality'][date]) else True
            hisprompt = summarize_content_prompt(content,user_name,boot_name,language)
            print('============this is v[summary]===========')
            print(v['summary'])

            print('this is hisproompt')
            print(hisprompt)
            person_prompt = summarize_person_prompt(content,user_name,boot_name,language)
            print('this is person prompt')
            print(person_prompt)
            print('his_flag:', his_flag, 'person_flag:', person_flag)
            if his_flag:
                his_summary = model_process(model=model, tokenizer=tokenizer, device=device, prompt=hisprompt,language=language, use_api=use_api)
                memory[user_name]['summary'][date] = {'content':his_summary}
            if person_flag:
                person_summary = model_process(model=model, tokenizer=tokenizer, device=device, prompt=person_prompt,language=language, use_api=use_api)
                print('---------------person summary---------------')
                print(person_summary)
                memory[user_name]['personality'][date] = person_summary
        
        overall_his_prompt = summarize_overall_prompt(list(memory[user_name]['summary'].items()),language=language)
        overall_person_prompt = summarize_overall_personality(list(memory[user_name]['personality'].items()),language=language)
        memory[user_name]['overall_history'] = model_process(model=model, tokenizer=tokenizer, device=device, prompt=overall_his_prompt,language=language, use_api=use_api)
        memory[user_name]['overall_personality'] = model_process(model=model, tokenizer=tokenizer, device=device, prompt=overall_person_prompt,language=language, use_api=use_api)
        print('---------------overall_his_prompt---------------')
        print(overall_his_prompt)
        print('---------------最终的总结history---------------')
        print(memory[user_name]['overall_history'])
        print('---------------overall_person_prompt---------------')
        print(overall_person_prompt)
        print('---------------最终的总结personality---------------')
        print(memory[user_name]['overall_personality'])
 
    with open(memory_dir,'w',encoding='utf8') as f:
        print(f'Sucessfully update memory for {name}')
        json.dump(memory,f,ensure_ascii=False)
    return memory

if __name__ == '__main__':
    summarize_memory('MemoryBank-SiliconFriend-main/memories/update_memory_0717_cn.json',language='cn')


                


