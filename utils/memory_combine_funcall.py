#TODO 这一份代码并没有完善
import json
import openai
# 跳过 - 什么也不做
def skip_memory(old_memory, key, value):
    return old_memory

# 删除 - 从旧记忆中删除信息
def delete_memory(old_memory, key, value):
    if key in old_memory and value in old_memory[key]:
        old_memory[key].remove(value)
    return old_memory

# 修改 - 修改旧记忆中的信息
def modify_memory(old_memory, key, value, new_value):
    if key in old_memory and value in old_memory[key]:
        index = old_memory[key].index(value)
        old_memory[key][index] = new_value
    return old_memory

# 增加 - 增加补充信息到旧记忆中
def add_memory(old_memory, key, additional_value):
    if key in old_memory:
        old_memory[key].append(additional_value)
    else:
        old_memory[key] = [additional_value]
    return old_memory

openai.api_key = 'your-api-key'

functions = [
    {
        "name": "skip_memory",
        "description": "跳过这个记忆条目，不做任何处理。",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "delete_memory",
        "description": "删除旧记忆中的这个信息，以避免与新记忆冲突。",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "modify_memory",
        "description": "修改旧记忆中的信息以避免冲突。",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "old_value": {"type": "string"},
                "new_value": {"type": "string"}
            },
            "required": ["key", "old_value", "new_value"]
        }
    },
    {
        "name": "add_memory",
        "description": "添加补充信息到旧记忆中。",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "additional_value": {"type": "string"}
            },
            "required": ["key", "additional_value"]
        }
    }
]

# 使用 OpenAI 接口帮助决定调用哪个函数
def choose_function(new_memory, old_memory, key, value):
    prompt = f"新记忆: {key}: {value}。旧记忆: {old_memory.get(key, '无此条目')}. 应该如何处理？"
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "你是一个处理记忆冲突的助手。"},
            {"role": "user", "content": prompt}
        ],
        functions=functions,
        function_call="auto"  # 自动选择并返回合适的函数和参数
    )
    
    return response["choices"][0]["message"]["function_call"]

def append_new_memory(old_memory, key, value):
    if key in old_memory:
        old_memory[key].append(value)
    else:
        old_memory[key] = [value]
    return old_memory

def process_memories(new_memory, old_memory):
    for key, values in new_memory.items():
        for value in values:
            # 步骤 2: 调用OpenAI接口选择合适的处理函数
            func_call = choose_function(new_memory, old_memory, key, value)

            # 步骤 3: 根据OpenAI返回的结果调用相应的处理函数
            if func_call["name"] == "skip_memory":
                pass  # 跳过该信息
            elif func_call["name"] == "delete_memory":
                old_memory = delete_memory(old_memory, key, value)
            elif func_call["name"] == "modify_memory":
                # 你可以在此处进一步优化，提供替代的新信息
                new_value = "some_new_value"  # 修改后的值
                old_memory = modify_memory(old_memory, key, value, new_value)
            elif func_call["name"] == "add_memory":
                additional_value = "some_additional_value"  # 增加的补充信息
                old_memory = add_memory(old_memory, key, additional_value)
            
            # 步骤 4: 将新记忆信息添加到旧记忆中
            old_memory = append_new_memory(old_memory, key, value)
    
    return old_memory

if __name__ == '__main__':
    memory_data = {'人际关系': ['默认角色与顾秋拾之间存在一种复杂的关系，顾秋拾似乎对默认角色有着特殊的关注，而默认角色则对此表现出明显的不耐烦和挑衅的态度。(一小时前)(2段对话前)'], '发生事件': ['在对话中，顾秋拾试图对默认角色施加控制并警告其服从，而默认角色则进行了反抗和挑衅。(一小时前)(2段对话前)', '默认角色拒绝了顾秋拾递给他的精致糕点，并表示自己并不饿。(一小时前)(1段对话前)', '顾秋拾尝试让默认角色开心，提议出去玩，包括划船，但未提及结果。(当前对话)'], '背景设定': ['根据对话中的信息，默认角色可能与皇室有所关联，因顾秋拾提到默认角色收到了朝廷的婚契。(一小时前)(2段对话前)'], '其他信息': ['默认角色在与顾秋拾的互动中表现出独立自主的性格，虽然在对方的压制下显得稍微屈服，但并未真正认同对方的权威。(一小时前)(2段对话前)'], '喜欢的食物': ['糕点(一小时前)(1段对话前)', '出去玩, 划船']}