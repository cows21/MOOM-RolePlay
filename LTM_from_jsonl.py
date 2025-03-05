import os
import json

from utils.dialogue_processor import OriginJsonlDialogueReader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel
from utils.json_extract import clean_json_string, time_extract, parse_to_json, clean_combined_json, time_standard
from utils.json_extract import memory_time_standard, split_comma_separated_values, extract_json_block
from utils.prompt_processor import build_memory_prompt_cws, build_time_prompt_cws, build_combine_prompt_cws
from utils.memory_combine_time import memory_update_through_time
from openai import OpenAI
import random
import logging
from utils.memory_combine_new import MemoryCombineManager
import copy


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

def setup_logging(output_file_path):
    # 创建一个logger
    logger = logging.getLogger('memory_dialogue_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志记录级别

    # 创建一个handler，用于将日志写入文件
    fh = logging.FileHandler(output_file_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # 创建一个handler，用于将日志输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s \n%(message)s')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main():
    output_dir = './outputdata/all_0812/'
    log_output_file = output_dir + 'output.log'
    # 设置日志记录
    logger = setup_logging(log_output_file)

    # 指定 JSONL 文件路径
    file_path = './data/20240626_long_meta_revise.jsonl'
    # file_path = 'data/guantouqiang.jsonl'
    
    # 创建读取器实例
    reader = OriginJsonlDialogueReader(file_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    # 模型和分词器的初始化
    model_id_mem = 'model/qwen2-0621-7B-LTM-FT/checkpoint-800'
    model_mem = AutoModelForCausalLM.from_pretrained(model_id_mem, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer_mem = AutoTokenizer.from_pretrained(model_id_mem, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id_time = 'model/qwen2-0704-7B-LTM-timestamp-FT/checkpoint-400'
    model_time = AutoModelForCausalLM.from_pretrained(model_id_time, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer_time = AutoTokenizer.from_pretrained(model_id_time, trust_remote_code=True)

    model_id_combine = 'model/Qwen1.5-14B-Chat'
    model_combine = AutoModelForCausalLM.from_pretrained(model_id_combine, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer_combine = AutoTokenizer.from_pretrained(model_id_combine, trust_remote_code=True)

    model_id_bge = 'model/FlagEmbedding/BAAI/bge-m3'
    model_bge = BGEM3FlagModel(model_id_bge, use_fp16=True)
  
    prompt_mem_file_path = 'prompt/memory_extract_prompt_template.txt'
    with open(prompt_mem_file_path, 'r', encoding='utf-8') as prompt_file:
        prompt_mem_template = prompt_file.read()

    prompt_time_file_path = 'prompt/memory_time_stamp.txt'
    with open(prompt_time_file_path, 'r', encoding='utf-8') as prompt_file:
        prompt_time_template = prompt_file.read()

    prompt_combine_file_path = 'model/memory_api3/prompt/memory_combine.txt'
    with open(prompt_combine_file_path, 'r', encoding='utf-8') as prompt_file:
        prompt_combine_template = prompt_file.read()
    
    dialogue_count = 0
    while dialogue_count < 100:
        if dialogue_count < 12:
            dialogue_count += 1
            continue

        processor = reader.read_dialogue_at_line(dialogue_count)
        if processor is None:
            break  # 当没有更多行时结束循环
        
        # 获取对话轮次信息等（根据需要选择使用）
        turns = processor.count_total_turns()
        logger.info(f"对话总轮次数：{turns}")

        # {'别名', '性别', '身份', '详细设定'}
        role_info=  processor.get_role_info()
        user_name =  role_info[0]['user_name']
        bot_name = role_info[0]['primary_bot_name']
        user_info = role_info[1][user_name]
        bot_info = role_info[1][bot_name]
        logger.info(user_name)
        logger.info(user_info)
        logger.info(bot_name)
        logger.info(bot_info)

        dialogues = processor.split_by_turns(20)
        dialogue_memory_list = []
        used_infos = []
        history_mem = None
        fail_mem_extract = False
        for dia in dialogues:
            logger.info('---------------')
            logger.info(dia)
            ## 提取当前对话的memory
            prompt_mem = build_memory_prompt_cws(user_name, bot_name, dia, prompt_mem_template)
            response_mem = model_process(model_mem, tokenizer_mem, device, prompt_mem)
            # logger.info('-----新总结的memory(raw)----------')
            # logger.info(extract_json_block(response_mem))
            cleaned_new_mem = clean_json_string(response_mem)
            cleaned_new_mem = split_comma_separated_values(cleaned_new_mem)
            logger.info("-----新总结的memory(处理后)------")
            logger.info(cleaned_new_mem)
            
            # 暂时删去了时间模块

            if history_mem is not None:
                model_dict = dict()
                tokenizer_dict = dict()
                model_dict["inner_combine"] = model_combine
                tokenizer_dict["inner_combine"] = tokenizer_combine
                model_dict["bge"] = model_bge
                combine_manager = MemoryCombineManager(model_dict=model_dict, tokenizer_dict=tokenizer_dict, device=device)
                updated_memory1 = combine_manager.merge_memories(long_term_memory=history_mem, new_memory=cleaned_new_mem)
                updated_memory2 = combine_manager.resolve_conflicts_hobby(long_term_memory=history_mem, new_memory=updated_memory1)
                updated_memory3 = combine_manager.resolve_conflicts_plan(long_term_memory=history_mem, new_memory=updated_memory2)
                new_history_mem = updated_memory3
            else:
                new_history_mem = cleaned_new_mem

            # 下面这行是为了去掉一些融合时不恰当的格式
            # new_history_mem = clean_combined_json(cleaned_new_mem, new_history_mem)

            logger.info("------（格式化后）合并后的memory------")
            logger.info(new_history_mem)
            history_mem = new_history_mem
            dialogue_memory_list.append({'dialogue': dia,  'new_memory':copy.deepcopy(cleaned_new_mem), 'final_memory': copy.deepcopy(history_mem)})

        dialogue_count += 1
        if fail_mem_extract is True:
            continue
        with open(output_dir + f'{dialogue_count}.json', 'w', encoding='utf-8') as f:
            json.dump(dialogue_memory_list, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
    # dia1 = processor.get_conversation_by_turns(0, 19)
    # dia2 = processor.get_conversation_by_turns(20, 39)
    # dialogues.append(dia1)
    # dialogues.append(dia2)