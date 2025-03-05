import re
import json
import requests
# from utils.json_extract import extract_json_block

def extract_json_block(text):
    # 定义正则表达式模式
    pattern = r'```json\n(.*?)\n```'
    # 使用 re.DOTALL 来使 '.' 特殊字符匹配包括换行符在内的所有字符
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # 提取匹配的 JSON 字符串
        return match.group(1).strip()
    else:
        # 如果没有找到匹配，返回 None
        return None

def get_style_chat(request_body):
    url = "http://10.4.148.54:22223/llm_create"
    retry_times = 5
    while retry_times > 0:
        try:
            response = requests.post(url=url, json=request_body, verify=False)
            if response.status_code == 200:
                print('response.content', json.loads(response.content))
                return json.loads(response.content)
            else:
                retry_times -= 1
        except Exception as e:
            retry_times -= 1
            print(e, 'in get_style_chat')
    return 'error happened'

def model_process_api(prompt, model_id="gpt-3.5-turbo"):
    request_body = {
        "model": model_id,
        "response_format": { "type": "json_object" },
        "messages": [
                {"role": "user", "content": prompt
        }],
        "max_tokens": 4096
    }
    return get_style_chat(request_body)

def model_process(model, tokenizer, device, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def prompt_process(mem_new, mem_old, prompt_template):
    prompt = prompt_template.replace("{mem_old}", mem_old)
    prompt = prompt.replace("{mem_new}", mem_new)
    return prompt

def extract_last_bracket_content(text):
    # 使用正则表达式匹配最后一个{}内的内容
    matches = re.findall(r'\{([^{}]*)\}', text)
    if matches:
        # 返回最后一个匹配的内容
        return "{" + matches[-1] + "}"
    else:
        return "No bracket content found"

class MemoryCombineManager:
    def __init__(self, model_dict, tokenizer_dict, device):

        # 有关inner combine的参数
        self.replace_keys = [
            "姓名", "年龄", "生日", "性别", "民族或种族", "星座", "生肖",
            "MBTI", "性取向", "现就读学校", "当前所在地点", "故乡"
        ]
        self.add_keys = [
            "性癖", "喜欢的xxx", "讨厌的xxx", "擅长的能力", "短板",
            "xx学校", "专业学科", "过往经历", "关键日期",
            "背景设定", "概念术语", "其它信息", "家庭相关"
        ]
        self.trajectory_keys = [
            "职业相关", "经济相关", "健康相关", "社会地位", "生活习惯", "发生事件"
        ]
        self.complex_keys = [
            "运动水平", "颜值相关", "人际关系"
        ]
        example_path = 'prompt/memory_combine_example.txt'
        self.prompt_templates_path = {
            "运动水平": 'prompt/memory_combine_complex_keys/sports_level.txt',
            "颜值相关": 'prompt/memory_combine_complex_keys/appearance.txt',
            "人际关系": 'prompt/memory_combine_complex_keys/social_connection.txt'
        }
        self.prompt_templates = dict()
        print('a')
        for k in self.complex_keys:
            try:
                with open(self.prompt_templates_path[k], 'r', encoding='utf-8') as prompt_file:
                    prompt_mem_template = prompt_file.read()
                    self.prompt_templates[k] = prompt_mem_template
            except Exception as e:
                print(f"读取prompt文件{self.prompt_templates_path[k]}时，发生了一个错误：{e}")
        print('b')
        self.device = device
        self.inner_combine_model = model_dict["inner_combine"]
        self.inner_combine_tokenizer = tokenizer_dict["inner_combine"]

        # 有关conflict处理的参数
        self.opposite_categories_hobby = {
            "喜欢的食物": ["讨厌的食物", "其它讨厌"],
            "喜欢的动物": ["讨厌的动物", "其它讨厌"],
            "喜欢的活动": ["讨厌的活动", "其它讨厌"],
            "喜欢的音乐": ["讨厌的音乐", "其它讨厌"],
            "喜欢的电影": ["讨厌的电影", "其它讨厌"],
            "喜欢的书籍": ["讨厌的书籍", "其它讨厌"],
            "喜欢的电子游戏": ["讨厌的电子游戏", "其它讨厌"],
            "喜欢的艺术家": ["讨厌的艺术家", "其它讨厌"],
            "其它喜欢": ["其它讨厌","讨厌的食物", "讨厌的动物", "讨厌的活动", "讨厌的音乐", "讨厌的电影", "讨厌的书籍", "讨厌的电子游戏", "讨厌的艺术家"],
            "讨厌的食物": ["喜欢的食物", "其它喜欢"],
            "讨厌的动物": ["喜欢的动物", "其它喜欢"],
            "讨厌的活动": ["喜欢的活动", "其它喜欢"],
            "讨厌的音乐": ["喜欢的音乐", "其它喜欢"],
            "讨厌的电影": ["喜欢的电影", "其它喜欢"],
            "讨厌的书籍": ["喜欢的书籍", "其它喜欢"],
            "讨厌的电子游戏": ["喜欢的电子游戏", "其它喜欢"],
            "讨厌的艺术家": ["喜欢的艺术家", "其它喜欢"],
            "其它讨厌": ["其它喜欢", "喜欢的食物", "喜欢的动物", "喜欢的活动", "喜欢的音乐", "喜欢的电影", "喜欢的书籍", "喜欢的电子游戏", "喜欢的艺术家"]
        }
        self.opposite_categories_plan = {
            "发生事件": ["计划与期待"]
        }
        self.bge_model = model_dict["bge"]

    def merge_memories(self, long_term_memory, new_memory):
        traj_key_in_new = list()
        for key, values in new_memory.items():
            if key in self.replace_keys:
                long_term_memory[key] = values
            elif key in self.add_keys or re.search(r"喜欢如何称呼|喜欢被称呼的昵称|喜欢的|讨厌的|学校", key):
                if key in long_term_memory:
                    long_term_memory[key].extend(value for value in values if value not in long_term_memory[key])
                else:
                    long_term_memory[key] = values
            elif key in self.trajectory_keys:
                traj_key_in_new.append(key)
                updated_values = self.update_trajectory_info(long_term_memory.get(key, []), values)
                long_term_memory[key] = updated_values
            elif key in self.complex_keys:
                mem_new_string = json.dumps(values, ensure_ascii=False)
                mem_old_string = json.dumps(long_term_memory.get(key, []), ensure_ascii=False)

                # 使用openai api
                # prompt = prompt_process(mem_new_string, mem_old_string, self.prompt_templates[key])
                # response = model_process_api(prompt)['response']['choices'][0]['message']['content']
                # x = json.loads(response)
                # print(x["融合后的记忆"])
                # long_term_memory[key] = x["融合后的记忆"]

                # 使用本地模型
                prompt = prompt_process(mem_new_string, mem_old_string, self.prompt_templates[key])
                response = model_process(model=self.inner_combine_model, tokenizer=self.inner_combine_tokenizer, device=self.device, prompt=prompt)
                response_x = extract_json_block(response)
                if response_x is None:
                    response_x = extract_last_bracket_content(response)
                response_j = json.loads(response_x)
                long_term_memory[key] =response_j["融合后的记忆"]
            else:
                pass
        
        # print('------------TRAJ BUG------------')
        # 处理那些没有出现在traj_key_in_new中的self.trajectory_keys
        for key in self.trajectory_keys:
            # print('当前的key', key)
            if key in traj_key_in_new or key not in long_term_memory:
                # print('这个可以不用再次处理')
                continue
            else:
                # print('这个key需要自然顺眼')
                updated_values = self.update_trajectory_info(long_term_memory.get(key, []), [])
                long_term_memory[key] = updated_values
       
        return long_term_memory

    def update_trajectory_info(self, old_values, new_values):
        # Increment time in existing entries
        updated_old_values = [self.increment_time_stamp(val) for val in old_values]
        # Add new entries with initial time stamp
        updated_new_values = [f"{val}(1轮对话前)" for val in new_values]
        return updated_old_values + updated_new_values

    def increment_time_stamp(self, value):
        match = re.search(r"\((\d+)轮对话前\)$", value)
        if match:
            current_rounds = int(match.group(1))
            new_value = re.sub(r"\(\d+轮对话前\)$", f"({current_rounds + 1}轮对话前)", value)
            return new_value
        else:
            return value + "(1轮对话前)"

    def bge_similar(self, sentence1, sentence2, threshold=0.8):
        sentence_pairs = [[sentence1, sentence2]]
        scores = self.bge_model.compute_score(sentence_pairs, max_passage_length=128, weights_for_different_modes=[0.4, 0.2, 0.4])
        # Assuming the "colbert+sparse+dense" score is used to determine similarity
        return scores['colbert+sparse+dense'][0] > threshold
    
    def resolve_conflicts_hobby(self, new_memory, long_term_memory):
        opposite_categories = self.opposite_categories_hobby
        to_delete_keys = set()  # 使用集合来存储需要删除的键

        # 遍历新记忆中的键值对
        for key, values in new_memory.items():
            if key in opposite_categories:
                # 获取所有对立类别
                opposites = opposite_categories[key]
                for value in values:
                    # 检查每个对立类别中的值
                    for opposite in opposites:
                        if opposite in long_term_memory and value in long_term_memory[opposite]:
                            # 移除冲突元素
                            long_term_memory[opposite].remove(value)
                            # 如果列表为空，则将此键添加到待删除集合中
                            if not long_term_memory[opposite]:
                                to_delete_keys.add(opposite)

        # 删除所有标记为删除的键
        for key in to_delete_keys:
            del long_term_memory[key]

        return long_term_memory

    def resolve_conflicts_plan(self, new_memory, long_term_memory):
        opposite_categories = self.opposite_categories_plan
        to_delete_keys = set()  # 使用集合而非列表来存储需要删除的键

        # 遍历新记忆中的键值对
        for key, values in new_memory.items():
            if key in opposite_categories:
                # 获取所有对立类别
                opposites = opposite_categories[key]
                for value in values:
                    # 检查每个对立类别
                    for opposite in opposites:
                        if opposite in long_term_memory:
                            to_remove = []  # 创建列表来保存待移除元素
                            # 检查长期记忆中的元素
                            for element in long_term_memory[opposite]:
                                if self.bge_similar(value, element):
                                    to_remove.append(element)
                            # 移除所有相似元素
                            for element in to_remove:
                                long_term_memory[opposite].remove(element)
                            # 如果列表为空，标记此键为删除
                            if not long_term_memory[opposite]:
                                to_delete_keys.add(opposite)  # 使用 add 方法添加到集合

        for key in to_delete_keys:
            del long_term_memory[key]

        return long_term_memory

    def assistance_social_connection(self):
        pass
    
if __name__ == '__main__':
    # Example usage
    long_term_memory = {
        "喜欢的食物": ["大蒜", "米饭"],
        "喜欢的活动": ["看书"],
        "发生事件": ["煮八宝粥"],
        "职业相关": ["当老师(1轮对话前)"],
        "健康相关": ["熊大注意到宝贝最近似乎瘦了，提醒她多吃点。(1轮对话前)"]
    }

    long_term_memory = {
        "健康相关": ["熊大注意到宝贝最近似乎瘦了，提醒她多吃点。(1轮对话前)"]
    }

    long_term_memory= {
        "宝贝喜欢如何称呼熊大": [
            "老公",
            "熊出没",
            "老婆",
            "亲亲",
            "熊大",
            "宝贝"
        ],
        "宝贝喜欢被称呼的昵称": [
            "宝贝",
            "乖宝贝"
        ],
        "喜欢的食物": [
            "美食"
        ],
        "发生事件": [
            "宝贝似乎在生气因为熊大带来了他的弟弟(8轮对话前)",
            "熊大强硬地安慰宝贝并夺走了她的手机(8轮对话前)",
            "熊大检查宝贝的手机上的聊天记录并发现了一个令他不快的备注(8轮对话前)",
            "熊大决定要喂宝贝吃东西，并带她去了一个餐厅(8轮对话前)",
            "熊大给宝贝添加食物，并与宝贝亲吻和嬉戏。(8轮对话前)",
            "宝贝被熊大称赞今天真漂亮(7轮对话前)",
            "熊大喂宝贝吃饭并夸赞食物好吃(7轮对话前)",
            "熊大决定之后要让宝贝变得白白胖胖(7轮对话前)",
            "宝贝似乎不愿意变得很胖但最终接受了熊大的提议(7轮对话前)",
            "熊大亲吻了宝贝的额头并提议看完饭去看电影(7轮对话前)",
            "宝贝表达了对熊出没这部电影的喜爱(7轮对话前)",
            "熊大提议看爱情片，宝贝可能有点不满但最终可能会接受(7轮对话前)",
            "宝贝在电影院与熊大一起等待进入电影厅时，熊大因为宝贝与其他男性聊天而感到不快，随后他买了爆米花和可乐，并且在争执中夺走了宝贝的手机。(6轮对话前)",
            "宝贝和熊大一起去电影院看电影，并且电影结束后由熊大送回家。(5轮对话前)",
            "在公共场合，宝贝想要亲吻熊大，但熊大拒绝了。(5轮对话前)",
            "宝贝询问了熊大关于男生喜欢什么的问题，因为想给熊大准备生日礼物。(4轮对话前)",
            "熊大显得不悦因为宝贝知道了他的生日，但最终接受了这个事实。(4轮对话前)",
            "宝贝和熊大之间存在一定的亲密关系，表明两人可能相互庆祝生日。(4轮对话前)",
            "熊大在宝贝面前亲了一口，以此表示亲昵。(4轮对话前)",
            "宝贝试图引导熊大亲吻自己，尽管熊大最初有所犹豫。(4轮对话前)",
            "宝贝在等待熊大回家的过程中，表现出了对亲吻的强烈渴望，与熊大进行了打趣和调情的互动。(3轮对话前)",
            "宝贝计划给熊大买一套西装，同时也打算为自己买几件裙子。(2轮对话前)",
            "宝贝在店里走来走去选择衣服，熊大认为宝贝穿什么都好看(1轮对话前)",
            "宝贝觉得自己穿白色的衣服最好看(1轮对话前)",
            "熊大期待宝贝试穿衣服的样子，并为宝贝挑选衣物(1轮对话前)",
            "宝贝在试衣间试穿白色裙子，并得到熊大的赞美(1轮对话前)",
            "熊大决定购买宝贝试穿的白色裙子，并提出之后要一起挑选鞋子(1轮对话前)"
        ],
        "人际关系": [
            "宝贝视熊大为伴侣，熊大是宝贝生命中的重要伙伴，他们的关系从亲密朋友发展到深度依赖和共享生活的伴侣，熊大的母亲曾向宝贝透露熊大的衣服喜好，这加深了他们之间的了解和亲近。"
        ],
        "颜值相关": [
            "熊大注意到宝贝今天的妆容格外动人，她的肌肤似乎比往常更加光彩照人。"
        ],
        "健康相关": [
            "熊大注意到宝贝最近似乎瘦了。(1轮对话前)"
        ],
        "喜欢的电影": [
            "好看"
        ],
        "关键日期": [
            "今天宝贝和熊大一起看了电影。",
            "宝贝期待熊大的生日礼物，对其即将到来的生日表示了关注。"
        ],
        "喜欢的活动": [
            "逛商场"
        ]
    }

    new_memory = {
        "讨厌的食物": ["大蒜", "可乐"],
        "职业相关": ["做销售员"],
        "讨厌的音乐": ["爵士", "摇滚"],
        "其它讨厌": ["看书"]
    }

    new_memory = {
        "宝贝喜欢如何称呼熊大": [
            "宝贝"
        ],
        "发生事件": [
            "宝贝在店里走来走去选择衣服，熊大认为宝贝穿什么都好看",
            "宝贝觉得自己穿白色的衣服最好看",
            "熊大期待宝贝试穿衣服的样子，并为宝贝挑选衣物",
            "宝贝在试衣间试穿白色裙子，并得到熊大的赞美",
            "熊大决定购买宝贝试穿的白色裙子，并提出之后要一起挑选鞋子"
        ]
    }

    # self.inner_combine_model = model_dict["inner_combine"]
    # self.inner_combine_tokenizer = tokenizer_dict["inner_combine"]

    m = dict()
    m["inner_combine"] = 1 
    m["bge"] = 1 

    manager = MemoryCombineManager(m, m, 1)
    updated_memory = manager.merge_memories(long_term_memory=long_term_memory, new_memory=new_memory)
    updated_memory2 = manager.resolve_conflicts_hobby(long_term_memory=updated_memory, new_memory=new_memory)
    print(updated_memory2)
