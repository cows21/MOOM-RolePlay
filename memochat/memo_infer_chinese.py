import os
import json
import torch
import re
from random import sample
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoModelForSeq2SeqLM
from openai import OpenAI

q_pre = "<s>\n"
qa_link = "\n"
MaxLen = 2048
TarLen = 512
TaskTarLen = {
    "chatting_dialogsum": MaxLen,
    "chatting_alpacagpt4": MaxLen,
    "writing_topiocqa": TarLen // 2,
    "writing_dialogsum": TarLen,
    "retrieval_dialogsum": 32,
    "retrieval_topiocqa": 32
}

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
        model='qwen_32b',
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

def replace_placeholders(data, user_replacement, bot_replacement):
    """
    递归遍历 JSON 对象，替换 {user} 和 {bot} 占位符为指定的字符串。
    
    :param data: 原始 JSON 数据（可以是字典、列表、字符串）
    :param user_replacement: 用于替换 {user} 的字符串
    :param bot_replacement: 用于替换 {bot} 的字符串
    :return: 处理过的 JSON 数据
    """
    if isinstance(data, dict):
        # 如果是字典，递归处理键值对
        return {key: replace_placeholders(value, user_replacement, bot_replacement) for key, value in data.items()}
    
    elif isinstance(data, list):
        # 如果是列表，递归处理每个元素
        return [replace_placeholders(element, user_replacement, bot_replacement) for element in data]
    
    elif isinstance(data, str):
        # 如果是字符串，进行替换
        return data.replace("{user}", user_replacement).replace("{bot}", bot_replacement)
    
    else:
        # 其他类型的数据直接返回
        return data

def convert_jsonl_to_json(input_file, line_floor=1, line_ceil=1):
    result = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i < line_floor:
                continue
            if i >= line_ceil:
                break
            data = json.loads(line.strip())
            
            # 提取id
            entry_id = i
            role_meta = data.get('role_meta')
            
            # 提取conversations
            conversations = [
                {"from": msg["sender_name"], "value": msg["text"]}
                for msg in data.get('messages', [])
            ]
            # "role_meta": {"user_name": "发情母螳螂", "primary_bot_name": "陆易斯"}
            
            # 构建目标格式的字典
            converted_entry = {
                "id": entry_id,
                "origin_dialogue": line,
                "conversations": conversations,
                "role_meta": role_meta
            }
            
            # 添加到结果列表中
            result.append(converted_entry)
    return result

def normalize_chatting_outputs(model_outputs):
    """post processing function of chatting response"""
    def white_space_fix(text):
        lines = text.split("\n")
        result = []
        for line in lines:
            result.append(' '.join(line.split()))
        output = '\n'.join(result)
        return output
    return white_space_fix(model_outputs)

@torch.inference_mode()
def get_model_answers(model, tokenizer, local_check, ques_jsons, prompts, model_api):
    output_data = []
    # "role_meta": {"user_name": "发情母螳螂", "primary_bot_name": "陆易斯"}
    origin_prompts = prompts

    for d in ques_jsons:
        user_name = d["role_meta"]["user_name"]
        bot_name = d["role_meta"]["primary_bot_name"]
        
        new_d = d
        prompts = replace_placeholders(origin_prompts, user_name, bot_name)

        history = {
            "Recent Dialogs": ["user: 你好!", "bot: 你好! 今天我能为你做些什么?"], 
            "Related Topics": [], 
            "Related Summaries": [], 
            "Related Dialogs": [], 
            "User Input": "",
        }
        memo = {
            "NOTO": [{"summary": "None of the others.", "dialogs": []}]
        }
        

        for l_i in range(len(new_d["conversations"])):
            if l_i % 2 == 1:
                bot_thinking = {"retrieval": "", "summarization": ""}
                print("=" * 20 + "start of turn {}".format(l_i // 2 + 1) + "=" * 20)
                if new_d["conversations"][l_i - 1]["from"] == user_name:
                    user = "user: " + new_d["conversations"][l_i - 1]["value"]
                    bot = "bot: " + new_d["conversations"][l_i]["value"]
                else:
                    user = "user: " + new_d["conversations"][l_i]["value"]
                    if l_i + 1 >= len(new_d["conversations"]):
                        bot = "bot: "
                    else:
                        bot = "bot: " + new_d["conversations"][l_i+1]["value"]
                print(user + "\n\n")

                # create summary if recent dialogs exceed threshold
                if len(" ### ".join(history["Recent Dialogs"]).split(" ")) > (MaxLen // 2) or len(history["Recent Dialogs"]) >= 10:
                    history, memo, bot_thinking = run_summary(history, model, tokenizer, memo, local_check, bot_thinking, prompts, model_api)

                # retrieve most related topics for every new user input
                history["User Input"] = user
                history["Bot Input"] = bot
                if len(memo.keys()) > 1:
                    history, bot_thinking = run_retrieval(history, model, tokenizer, memo, local_check, bot_thinking, prompts, model_api)
                
                # generate bot response
                system_insturction = prompts["chatting"]["system"]
                task_instruction = prompts["chatting"]["instruction"]
                task_case = "```\n相关证据:\n" + "\n".join(["({}) {}".format(r_tsd_i + 1, {
                                "相关话题": history["Related Topics"][r_tsd_i], 
                                "相关摘要": history["Related Summaries"][r_tsd_i], 
                                "相关对话": history["Related Dialogs"][r_tsd_i]
                            }) for r_tsd_i in range(len(history["Related Topics"]))]) + "\n\n最近的对话:\n" + \
                            " ### ".join([hrd.replace("\n", " ") for hrd in history["Recent Dialogs"]]) + "\n```\n\nUser Input:\n" + history["User Input"] + " ### " + history["Bot Input"]
                qs = q_pre + system_insturction + task_case + task_instruction + qa_link
                print('this is qs')
                print(qs)
                # 使用模型生成的output
                # outputs = gen_model_output(model, tokenizer, qs, local_check, "chatting_dialogsum")
                # outputs = normalize_chatting_outputs(outputs)
                # history["Recent Dialogs"] += [user, "bot: " + outputs]
                # print("bot: " + outputs + "\n")
                # print("=" * 20 + "end of turn {}".format(l_i // 2 + 1) + "=" * 20)
                # print("\n\n\n\n")
                # new_d["conversations"][l_i]["thinking"] = json.dumps(bot_thinking, ensure_ascii=False)
                # new_d["conversations"][l_i]["value"] = outputs

                # 只提取记忆
                history["Recent Dialogs"] += [user, bot]
                print(user, bot)
                print("=" * 20 + "end of turn {}".format(l_i // 2 + 1) + "=" * 20)
                new_d["conversations"][l_i]["thinking"] = json.dumps(bot_thinking, ensure_ascii=False)
                new_d["conversations"][l_i]["value"] = bot

        output_data.append(d)
    return output_data

def run_eval(model, tokenizer, local_check, question_file, answer_dir, prompt_path, line_floor=1, line_ceil=1, model_api=False):
    # 加载提示文件
    prompts = json.load(open(prompt_path, "r"))

    # 从问题文件中读取数据
    ques_jsons = convert_jsonl_to_json(question_file, line_floor, line_ceil)

    chunk_size = 1  # 每次处理一条
    ans_handles = []
    file_counter = 1  # 文件编号计数器

    for i in range(0, len(ques_jsons), chunk_size):
        ques_i = ques_jsons[i: i + chunk_size]
        print("this is ques_i!!!!!!!!!!!")
        print(ques_i)
        ans_item = get_model_answers(model, tokenizer, local_check, ques_i, prompts, model_api)
        new_answer_file = answer_dir + f"/memochat_{file_counter}.json"
        json.dump(ans_item, open(os.path.expanduser(new_answer_file), "w"), indent=2, ensure_ascii=False)
        file_counter += 1

    return 0

def run_summary(history, model, tokenizer, memo, local_check, bot_thinking, prompts, model_api):
    """We assume there's no too long input from user, e.g. over 1500 tokens"""
    system_insturction = prompts["writing_dialogsum"]["system"]
    task_instruction = prompts["writing_dialogsum"]["instruction"]
    history_log = "\n\n```\nTask Conversation:\n" + "\n".join(["(line {}) {}".format(h_i + 1, h.replace("\n", " ")) for h_i, h in enumerate(history["Recent Dialogs"][2:])])
    qs = q_pre + system_insturction.replace("LINE", str(len(history["Recent Dialogs"]) - 2)) + history_log + "\n```" + task_instruction.replace("LINE", str(len(history["Recent Dialogs"]) - 2)) + qa_link
    print("#" * 20 + "summarizing" + "#" * 20)
    print(qs)
    print("#" * 20 + "summarizing" + "#" * 20)
    sum_history = gen_model_output(model, tokenizer, qs, local_check, "writing_dialogsum", model_api)
    sum_history = normalize_model_outputs(sum_history)
    print("#" * 20 + "summarization" + "#" * 20)
    print(sum_history)
    print("#" * 20 + "summarization" + "#" * 20)
    for s in sum_history:
        memo[s["topic"]] = memo.get(s["topic"], []) + [{"summary": s["summary"], "dialogs": history["Recent Dialogs"][2:][(s["start"] - 1):s["end"]]}]
    if local_check:
        memo["test_topic{}".format(len(memo.keys()))] = [{"summary": "test_summary{}".format(len(memo.keys())), "dialogs": history["Recent Dialogs"][2:][2:4]}]
    if len(sum_history) == 0:
        si_0, si_1 = sample(list(range(len(history["Recent Dialogs"][2:]))), 2)
        memo["NOTO"].append({"summary": "Partial dialogs about: {} or {}.".format(history["Recent Dialogs"][2:][si_0], history["Recent Dialogs"][2:][si_1]), "dialogs": history["Recent Dialogs"][2:]})
    history["Recent Dialogs"] = history["Recent Dialogs"][-2:]
    bot_thinking["summarization"] = {"input": qs, "output": sum_history}
    return history, memo, bot_thinking

def gen_model_output(model, tokenizer, input_qs, local_check, task_type, model_api):
    if local_check:
        from faker import Faker
        fake = Faker(locale="en")
        return fake.text(2000)
    
    if model_api == True:
        answer = lsl_qwen_70b(input_qs)
        return answer

    if "writing" in task_type:
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.encode("]", add_special_tokens=False)[0]]
    elif "retrieval" in task_type:
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[0], tokenizer.encode(" ", add_special_tokens=False)[0]]
    else:
        eos_token_ids = [tokenizer.eos_token_id]
    input_ids = tokenizer([input_qs], max_length=(MaxLen - TarLen), truncation=True, add_special_tokens=False).input_ids
    target_len = min(len(input_ids[0]) + TaskTarLen[task_type], MaxLen)
    repetition_penalty_value = 1.0
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.2,
        max_length=target_len,
        eos_token_id=eos_token_ids,
        repetition_penalty=repetition_penalty_value
    )

    output_ids = output_ids[0][len(input_ids[0]):]
    model_outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return model_outputs

def normalize_model_outputs(model_text):
    """post processing function of memo writing task"""
    extracted_elements = [re.sub(r'\s+', ' ', mt.replace('"', '').replace("'", "")) for mt in re.findall(r"'[^']*'|\"[^\"]*\"|\d+", model_text)]
    model_outputs = []
    ti = 0
    while ti + 7 < len(extracted_elements):
        if extracted_elements[ti] == "topic" and extracted_elements[ti + 2] == "summary" and extracted_elements[ti + 4] == "start" and extracted_elements[ti + 6] == "end":
            try:
                model_outputs.append({"topic": extracted_elements[ti + 1], "summary": extracted_elements[ti + 3], "start": int(extracted_elements[ti + 5]), "end": int(extracted_elements[ti + 7])})
            except:
                pass
        ti += 1
    return model_outputs

def run_retrieval(history, model, tokenizer, memo, local_check, bot_thinking, prompts, model_api):
    topics = []
    for k, v in memo.items():
        for vv in v:
            topics.append((k, vv["summary"], vv["dialogs"]))
    system_insturction = prompts["retrieval"]["system"]
    task_instruction = prompts["retrieval"]["instruction"]
    task_case = "```\nQuery Sentence:\n" + history["User Input"][6:] + "\nTopic Options:\n" + \
                "\n".join(["({}) {}".format(v_i + 1, v[0] + ". " + v[1]) for v_i, v in enumerate(topics)]) + "\n```"
    qs = q_pre + system_insturction.replace("OPTION", str(len(topics))) + task_case + task_instruction.replace("OPTION", str(len(topics))) + qa_link
    print("#" * 20 + "retrieving" + "#" * 20)
    print(qs)
    print("#" * 20 + "retrieving" + "#" * 20)
    outputs = gen_model_output(model, tokenizer, qs, local_check, "retrieval_dialogsum", model_api)
    print("#" * 20 + "retrieval" + "#" * 20)
    print(outputs)
    print("#" * 20 + "retrieval" + "#" * 20)
    outputs = outputs.split("#")
    chosen_topics = []
    for output in outputs:
        try:
            index_ = int(output) - 1
        except:
            continue
        if index_ < len(topics) and "NOTO" not in topics[index_][0]:
            chosen_topics.append(topics[index_])
    if local_check:
        chosen_topics = sample(topics, min(len(topics) - 1, 2))
    if len(chosen_topics) > 0:
        history["Related Topics"] = [ct[0] for ct in chosen_topics]
        history["Related Summaries"] = [ct[1] for ct in chosen_topics]
        history["Related Dialogs"] = [" ### ".join(ct[2]) for ct in chosen_topics]
    else:
        history["Related Topics"] = []
        history["Related Summaries"] = []
        history["Related Dialogs"] = []
    bot_thinking["retrieval"] = {"input": qs, "output": outputs}
    return history, bot_thinking

if __name__ == "__main__":
    local_check = False
    model_api = True
    if not local_check and not model_api:
        model_id_combine = '/nvme/lisongling/models/Qwen1.5-14B-Chat'
        model_combine = AutoModelForCausalLM.from_pretrained(model_id_combine, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        tokenizer_combine = AutoTokenizer.from_pretrained(model_id_combine, trust_remote_code=True)
    else:
        model_combine = None
        tokenizer_combine = None

    question_file_path = 'data/ZH-4O_dataset.jsonl.jsonl'
    answer_file_path = 'memochat/answer_file.json'
    prompt_file_path = 'memochat/memo_chat_prompt_chinese.json'
    result = run_eval(model=model_combine, tokenizer=tokenizer_combine, local_check=local_check, question_file=question_file_path, 
             answer_dir='memochat/memory_32B', prompt_path=prompt_file_path, line_floor=1, line_ceil=21, model_api=model_api)
    print(result)