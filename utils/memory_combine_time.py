import json
import re

def memory_combine_hard_1(old_memory, new_memory):
    override_keys = ["姓名", "年龄", "生日", "性别", "性取向", "民族或种族", "国籍", "星座", "生肖", "MBTI", "运动水平", "小学学校", "初中学校", "大学学校", "研究生学校", "博士学校", "专业学科", "现就读学校", "当前所在地点", "故乡"]

    # 合并记忆的逻辑
    for key, value in new_memory.items():
        if key in override_keys:
            # 如果key在override_keys列表中，进行覆盖操作
            old_memory[key] = value
        else:
            # 如果key不在override_keys列表中，进行追加操作
            if key in old_memory:
                old_memory[key].extend(value)  # 旧记忆已有该key，追加新记忆
            else:
                old_memory[key] = value  # 旧记忆中没有该key，直接添加
        
    return old_memory

def extract_time_descriptions(memory):
    """
    Extracts the time descriptions from the memory items.
    从记忆信息中提取时间信息
    
    :param memory: Dict containing the memory data with time and dialogue round information.
    :return: Dict with extracted time descriptions for updating.
    """
    time_descriptions = {}
    
    for key, events in memory.items():
        extracted_times = []
        for event in events:
            # Split the event to isolate the time description
            parts = event.split('(')
            time_desc = parts[1].split(')')[0]  # Extract the time description from the first parenthesis
            extracted_times.append(time_desc)
        
        time_descriptions[key] = extracted_times
    
    return time_descriptions

def chinese_to_arabic(chinese_num):
    ## 将汉字数字转换成阿拉伯数字
    chinese_arabic_map = {
        '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '几': -1,
        '零': 0
    }
    if chinese_num in chinese_arabic_map:
        return chinese_arabic_map[chinese_num]
    return int(chinese_num)  # for Arabic numerals already in the string

def normalize_time_units(time_desc):
    """
    Normalizes time units in the given time description using lookaround assertions,
    to ensure accurate replacement even with non-Western scripts.
    """
    replacements = {
        r'(?<=\d)分(?!\w)': '分钟',  # Matches '分' preceded by a digit and not followed by a word character
        r'(?<=\d)时(?!\w)': '小时',  # Matches '时' preceded by a digit and not followed by a word character
        r'(?<=\d)周(?!\w)': '星期'   # Matches '周' preceded by a digit and not followed by a word character
    }
    for old, new in replacements.items():
        time_desc = re.sub(old, new, time_desc)
    return time_desc

def parse_time_description(time_desc, time_units):
    """
    Parses a time description into its numeric and unit components using predefined time units.
    将时间信息分为数值和单位，例：一天 -> 一、天
    """
    time_desc = normalize_time_units(time_desc) # Normalize units before parsing
    # Create a regex pattern that includes all the time units
    units_pattern = '|'.join(time_units)  # Join all units with '|' to create a pattern like '分钟|小时|天|...'
    pattern = rf'(\d+|\D+)(?:{units_pattern})'  # Updated to handle digits and non-digits
    
    match = re.match(pattern, time_desc)
    if match:
        # The numeric part will be in group 1, and the unit follows directly
        numeric_part = match.group(1)
        # The unit is extracted by removing the numeric part from the full match
        unit = time_desc[len(numeric_part):].rstrip('前')  # Remove '前' if present
        return (numeric_part, unit)
    return (None, None)

def time_add(time_desc, duration, suffix=''):
    """
    Updates a single time description with a duration.
    将time_desc, duration相加

    :param time_desc: Current time description (e.g., '两天').
    :param duration: Duration to add (e.g., '一小时').
    :param time_units: Ordered list of time units.
    :return: Updated time description.
    """
    # Split the current description and duration into value and unit
    time_units = ['秒', '分钟', '小时', '天', '星期', '月', '年']

    # Remove suffix '前' before processing
    if time_desc.endswith(suffix):
        time_desc = time_desc[:-1]  # Remove the last character '前'

    # Extract numeric and unit parts
    num_desc, unit_desc = parse_time_description(time_desc, time_units)
    num_dur, unit_dur = parse_time_description(duration, time_units)
    # Convert Chinese numbers to Arabic
    value_desc = chinese_to_arabic(num_desc)
    value_dur = chinese_to_arabic(num_dur)
    
    # Compare units and calculate the result
    if time_units.index(unit_desc) > time_units.index(unit_dur):
        result = time_desc
    elif time_units.index(unit_desc) < time_units.index(unit_dur):
        result = duration
    else:
        if value_desc == -1 or value_dur == -1:
            result = f'几{unit_desc}'
        else:
            total_value = value_desc + value_dur
            result = f'{total_value}{unit_desc}'

    # Add the '前' suffix back to the result
    return result + suffix

def update_time_descriptions(time_descriptions, duration):
    """
    Updates all time descriptions with a given duration.
    得到更新后的时间信息

    :param time_descriptions: Dict of time descriptions to update.
    :param duration: Duration to add to each time description.
    :param time_units: Ordered list of time units.
    :return: Updated dict of time descriptions.
    """
    updated_descriptions = {}
    for key, times in time_descriptions.items():
        updated_times = [time_add(time, duration, suffix='前') for time in times]
        updated_descriptions[key] = updated_times
    return updated_descriptions

def replace_time_in_memory(original_memory, updated_time_descriptions):
    """
    Replace the time descriptions in the original memory with updated times.
    根据更新的时间信息填充旧的记忆信息
    
    :param original_memory: Dict of original memory entries with detailed descriptions and times.
    :param updated_time_descriptions: Dict of updated time descriptions.
    :return: Updated memory with new time descriptions.
    """
    updated_memory = {}
    for key, events in original_memory.items():
        updated_events = []
        for idx, event in enumerate(events):
            # Extract the part before the first parenthesis (activity or event description)
            event_base = event.split('(')[0]
            # Extract the dialogue round information which is the second part in parenthesis
            dialogue_info = event.split(')')[1] + ')'
            # Use the updated time description from the corresponding list
            new_time_desc = updated_time_descriptions[key][idx]
            # Construct the updated event with new time and existing dialogue info
            updated_event = f"{event_base}({new_time_desc}){dialogue_info}"
            updated_events.append(updated_event)
        updated_memory[key] = updated_events
    return updated_memory


def time_stamp_update(original_memory, duration):
    old_time_stamp = extract_time_descriptions(original_memory)
    new_time_stamp = update_time_descriptions(old_time_stamp, duration)
    updated_memory = replace_time_in_memory(original_memory, new_time_stamp)
    return updated_memory

def memory_update_through_time(memory, dialogue_duration):
    """
    Update the long-term memory information based on the duration of the new dialogue.
    根据时间更新记忆信息
    
    :param memory: Dict containing the memory data with time and dialogue round information.
    :param dialogue_duration: Duration of the new dialogue in terms of narrative time.
    :return: Updated memory dictionary.
    """
    updated_memory = {}
    
    for key, events in memory.items():
        updated_events = []
        for event in events:
            if '(' in event:
                # Extract time description and dialogue round
                parts = event.split('(')
                base_event = parts[0]
                time_desc = '(' + parts[1]
                dialogue_round = '(' + parts[2]
                
                # Update dialogue round
                round_number = int(dialogue_round.split('段对话前')[0][1:])
                updated_dialogue_round = f'({round_number + 1}段对话前)'

                # Reconstruct the updated event
                updated_event = f'{base_event}{time_desc}{updated_dialogue_round}'
            else:
                updated_event = f'{event}(0秒前)(1段对话前)'
            updated_events.append(updated_event)
        updated_memory[key] = updated_events
    
    # Update the time
    updated_memory = time_stamp_update(updated_memory, dialogue_duration)

    return updated_memory

if __name__ == '__main__':
    memory_data = {'人际关系': ['默认角色与顾秋拾之间存在一种复杂的关系，顾秋拾似乎对默认角色有着特殊的关注，而默认角色则对此表现出明显的不耐烦和挑衅的态度。(一小时前)(2段对话前)'], '发生事件': ['在对话中，顾秋拾试图对默认角色施加控制并警告其服从，而默认角色则进行了反抗和挑衅。(一小时前)(2段对话前)', '默认角色拒绝了顾秋拾递给他的精致糕点，并表示自己并不饿。(一小时前)(1段对话前)', '顾秋拾尝试让默认角色开心，提议出去玩，包括划船，但未提及结果。(当前对话)'], '背景设定': ['根据对话中的信息，默认角色可能与皇室有所关联，因顾秋拾提到默认角色收到了朝廷的婚契。(一小时前)(2段对话前)'], '其他信息': ['默认角色在与顾秋拾的互动中表现出独立自主的性格，虽然在对方的压制下显得稍微屈服，但并未真正认同对方的权威。(一小时前)(2段对话前)'], '喜欢的食物': ['糕点(一小时前)(1段对话前)', '出去玩, 划船']}

    x =  memory_update_through_time(memory_data, '几分钟')
    print(x)


