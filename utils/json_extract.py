import json
import re

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

def clean_json_string(json_string):
    try:
        # 按照指定的格式寻找JSON字符串的开始和结束位置
        start = json_string.find("```json\n") + len("```json\n")
        end = json_string.rfind("\n```")
        json_string = json_string[start:end].strip()
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # 如果在解析过程中发生错误，打印错误信息并继续抛出异常
        print(f"clean_json_string函数 解析错误: {e}")
        raise

def split_comma_separated_values(json_data):
    # 遍历所有的键值对
    for key, value in json_data.items():
        if isinstance(value, list):  # 确保值是一个列表
            new_list = []
            for item in value:
                if ',' in item:  # 检查元素中是否包含逗号
                    # 拆分字符串并去除额外的空格
                    split_items = [i.strip() for i in item.split(',')]
                    new_list.extend(split_items)  # 将拆分后的项添加到新列表中
                else:
                    new_list.append(item)
            json_data[key] = new_list  # 更新原来的列表为新列表
    return json_data

def time_extract(text, prefix=None):
    if prefix is not None:
        pos = text.find(prefix)
        if pos != -1:
            # 提取前缀后的子串
            text = text[pos + len(prefix):]
        else:
            print("前缀不在字符串中")

    # 使用正则表达式匹配所有大括号内的内容
    matches = re.findall(r'\{([^{}]*)\}', text)
    # 返回最后一个匹配的结果，如果没有匹配内容则返回空字符串
    return matches[-1] if matches else ""

# def time_standard(text):
#     # 对提取出来的时间的标准化（使用replace）
#     new_time_string_cleaned = text.replace("个", "")
#     new_time_string_cleaned = text.replace("周", "星期")
#     new_time_string_cleaned = new_time_string_cleaned.replace("多", "")
#     new_time_string_cleaned = new_time_string_cleaned.replace("几千", "几")
#     new_time_string_cleaned = new_time_string_cleaned.replace("几万", "几")
#     new_time_string_cleaned = new_time_string_cleaned.replace("几亿", "几")
#     new_time_string_cleaned = new_time_string_cleaned.replace("一夜", "几小时")
#     new_time_string_cleaned = new_time_string_cleaned.replace("一晚上", "几小时")
#     new_time_string_cleaned = new_time_string_cleaned.replace("十几", "几")
#     return new_time_string_cleaned

def select_larger_time_unit(time_str):
    """
    Select the larger time unit from a string containing two time expressions.
    For example, '三年几小时' should return '三年'.
    """
    # Define unit priorities
    units_priority = {'秒': 1, '分钟': 2, '小时': 3, '天': 4, '星期': 5, '月': 6, '年': 7}
    
    # Find all time units and their descriptions in the string
    matches = re.findall(r'(\d+|几|一|二|两|三|四|五|六|七|八|九|十)?(秒|分钟|小时|天|星期|月|年)', time_str)
    
    # Sort the matches by the priority of their units, keeping the highest first
    if matches:
        sorted_matches = sorted(matches, key=lambda x: units_priority[x[1]], reverse=True)
        # Return the highest priority time unit with its prefix
        x = ''.join(sorted_matches[0])
        x = x
        return x
    else:
        return None

def time_standard(s):
    s = s.replace("一个半小时", "90分钟")
    s = s.replace("许多", "几")
    s = s.replace("多", "")
    s = s.replace("个", "")
    s = s.replace("個", "")
    s = s.replace("一天半", "两天")
    s = s.replace("半小时", "30分钟")
    s = s.replace("半年", "几月")
    s = s.replace("半月", "15天")
    s = s.replace("一学期", "几月")
    s = s.replace("一晚上", "几小时")
    s = s.replace("一整夜", "几小时")
    s = s.replace("一早晨", "几小时")
    s = s.replace("一晚", "几小时")
    s = s.replace("一夜", "几小时")
    s = s.replace("时辰", "小时")
    s = s.replace("几十", "20")
    s = s.replace("十几", "15")
    s = s.replace("几百", "200")
    s = s.replace("几千", "2000")
    s = s.replace("几万", "20000")
    s = s.replace("一两", "几")
    s = s.replace("两三", "几")
    s = s.replace("三四", "几")
    s = s.replace("四五", "几")
    s = s.replace("五六", "几")
    s = s.replace("六七", "几")
    s = s.replace("七八", "几")
    s = s.replace("八九", "几")
    s = s.replace("左右", "")
    s = s.replace("大约", "")
    s = s.replace("大概", "")
    s = s.replace("日", "天")
    s = s.replace("整", "")
    s = s.replace("数", "几")
    s = s.replace("周", "星期")
    s = s.replace("禮拜", "星期")
    s = s.replace("一节课", "一小时")
    s = s.replace("半", "")
    s = s.replace("十五", "15")

    s = s.replace("一节课", "一小时")
    
    # 去掉三年零几天这种描述
    s = re.sub(r'零(几|一|二|两|三|四|五|六|七|八|九|十|十?[一二三四五六七八九]|两三|三四|\d+)(个)?(秒|分钟|小时|天|星期|月|年)', '', s)

    # 去掉三年几小时这种描述
    s = select_larger_time_unit(s)
    
    return s

# 函数：标准化时间描述
def standardize_time_expression(expression):
    time_units = ['秒', '分钟', '小时', '天', '星期', '月', '年']
    # 正则检查是否符合格式：时间数量（数字或汉字‘几’）+ 时间单位 + '前'
    if re.match(r'(\d+|几|一|二|两|三|四|五|六|七|八|九|十)+[' + ''.join(time_units) + ']前', expression):
        return expression
    # 查找字符串中的时间单位
    found_units = [unit for unit in time_units if unit in expression]
    if found_units:
        # 取第一个找到的时间单位进行修正
        return f"几{found_units[0]}前"
    # 如果没有找到时间单位，则默认使用小时
    return "几小时前"

def standardize_round_expression(expression):
    text = expression.replace("段对话后", "段对话前")
    pattern = r"(\d+)段对话.*"
    # 尝试匹配并处理文本
    match = re.match(pattern, text)
    if match:
        # 如果匹配成功，则提取数字并构造新字符串
        number = match.group(1)
        return f"{number}段对话前"
    else:
        # 如果不符合规定的格式，返回默认值
        return "1段对话前"

# 函数：处理字符串中的时间描述
def process_time_descriptions(text):
    # 正则表达式匹配括号内的内容
    pattern = re.compile(r'\((.*?)\)')
    parts = pattern.findall(text)
    if len(parts) == 2:
        new_parts = [standardize_time_expression(parts[0]), standardize_round_expression(parts[1])]
        for old, new in zip(parts, new_parts):
            text = text.replace(f"({old})", f"({new})")
    elif len(parts) == 1:
        text = text.replace(f"({parts[0]})", '')
    elif len(parts) >= 3:
        part_dialogue_before = None
        part_time = None

        # 寻找包含“段对话前”的括号内容
        for part in parts:
            if '段对话前' in part:
                part_dialogue_before = part

        # 寻找包含时间单位的括号内容
        time_units = ['秒', '分钟', '小时', '天', '星期', '月', '年']
        for part in parts:
            if any(unit in part for unit in time_units):
                part_time = standardize_time_expression(part)
                break
        
        # 处理找到的括号内容
        if part_dialogue_before and part_time:
            new_parts = [part_time, part_dialogue_before]
        elif part_dialogue_before:
            new_parts = ['几小时前', part_dialogue_before]
        else:
            # 没有找到符合条件的括号内容，删除所有括号
            return re.sub(r'\(.*?\)', '', text)

        # 替换找到的括号内容
        text = re.sub(r'\(.*?\)', '', text)  # 先删除所有括号
        text = text + f"({new_parts[0]})" + f"({new_parts[1]})"

    return text

# 标准化memory内的时间
def memory_time_standard(memory):
    for key, values in memory.items():
        memory[key] = [process_time_descriptions(value) for value in values]
    return memory

def remove_bracketed_info(s):
    return re.sub(r'\(.*?\)', '', s)

# 暂时使用的函数，现在的模型会把b加上一些（新信息）这种干扰后续处理的后缀
def clean_combined_json(b, c):
    # 生成b的一个变体，其中的字符串都被清除了括号信息，以便正确识别和比较
    b_flat = {key: [remove_bracketed_info(value) for value in values] for key, values in b.items()}
    # 转换为单一的集合，以方便查找
    b_values_flat = set(sum(b_flat.values(), []))

    for key, values in c.items():
        new_values = []
        for value in values:
            # 移除可能的括号信息
            clean_value = remove_bracketed_info(value)
            if clean_value in b_values_flat:
                # 如果清理后的值属于b中的值，则添加清理后的值
                new_values.append(clean_value)
            else:
                # 否则保留原始未清理前的值
                new_values.append(value)
        c[key] = new_values
    return c

## 以下来自于api2的main函数，这些函数用来解析不太规则的json格式，同时去除无用信息
## 调用parse_to_json即可

def parse_to_json(raw_data):
    json_str = extract_json(raw_data)
    if json_str is None:
        print("No valid JSON object found.")
        return None

    try:
        new_data = parse_custom_json(json_str)
        clean_data = clean_invalid_data(new_data)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON. Error: {e}")
        return None

    if not clean_data:
        print("All data was invalid and has been ignored.")
        return None

    return clean_data

def generate_full_match_phrases():
    """Generate list of phrases for full matching."""
    return [
        "none", "null", "NONE", "NULL", "empty", "EMPTY", 
        "未有", "没有", "无", "暂无", "尚未有", "尚无", "尚未", "未曾有", "未曾", "未", "不明", "缺失", "缺乏", "不", "没", "不明确"
    ]

def is_phrase_invalid(value, full_match_phrases, pattern):
    """Check if the value is invalid based on the criteria."""
    # Split the value by commas
    parts = re.split(r'[，,]', value)
    invalid_parts = 0
    for part in parts:
        # Strip whitespace around the part
        trimmed_part = part.strip()
        # Check full match
        if trimmed_part in full_match_phrases:
            invalid_parts += 1
        # Check regex pattern match
        elif re.search(pattern, trimmed_part):
            invalid_parts += 1

    # If all parts are invalid, the whole value is considered invalid
    return invalid_parts == len(parts)

def clean_invalid_data(data):
    full_match_phrases = generate_full_match_phrases()
    # Building the regex pattern
    prefixes = ["对话", "对话中", "对话里", "上文", "上文中"]
    negations = ["未有", "没有", "无", "暂无", "尚未有", "尚无", "尚未", "未曾有", "未曾", "未", "不明", "缺失", "缺乏", "不", "没", "无法"]
    adjectives = ["相关", "明显", "直接", "具体", "准确", "确切", "详细", "全部", "完整", "部分", "全面", "具体", "实际", "精确", "精准", "特定", "明确", "间接", "深入"]
    articles = ["的", "地"]
    nouns = ["提及", "描述", "信息", "记录", "说明", "数据", "内容", "资料", "情况", "细节", "涉及", "出现", "检测", "给出", "提供", "提到", "揭示", "确定"]

    prefix_pattern = '|'.join(prefixes) if prefixes else ''
    negation_pattern = '|'.join(negations)
    adjective_pattern = '|'.join(adjectives) if adjectives else ''
    article_pattern = '|'.join(articles) if articles else ''
    noun_pattern = '|'.join(nouns) if nouns else ''
    
    pattern = fr"({prefix_pattern})?({negation_pattern})((?:{adjective_pattern})?(?:{article_pattern})?(?:{noun_pattern})?)"

    clean_data = {}
    for key, values in data.items():
        if not isinstance(values, list):
            values = [values]  # Ensure values are in list format

        filtered_values = [value for value in values if not is_phrase_invalid(value, full_match_phrases, pattern)]
        
        if filtered_values:
            clean_data[key] = filtered_values

    return clean_data

def extract_json(raw_data):
    """ 提取最外层的 JSON 对象 """
    counter = 0
    json_start_index = None
    for index, char in enumerate(raw_data):
        if char == '{':
            if counter == 0:
                json_start_index = index
            counter += 1
        elif char == '}':
            counter -= 1
            if counter == 0 and json_start_index is not None:
                return raw_data[json_start_index:index + 1]
    return None

def parse_custom_json(data):
    """ 解析JSON数据，处理单个字符串值和字符串数组 """
    # 匹配键和可能是数组或单个字符串的值
    pattern = r'"([^"]+)"\s*[:：]\s*(\[[^\]]*\]|"[^"]*"|[^",\s]+)'
    matches = re.finditer(pattern, data, re.DOTALL)
    
    parsed_data = {}
    for match in matches:
        key = match.group(1)
        value_string = match.group(2)
        # 检查值是否为数组
        if value_string.startswith('[') and value_string.endswith(']'):
            # 解析数组中的各个项
            values = re.findall(r'"([^"]+)"', value_string)
            parsed_data[key] = values
        else:
            # 处理单个字符串值，移除可能的引号
            single_value = re.match(r'"?(.*?)"?$', value_string).group(1)
            parsed_data[key] = [single_value]

    return parsed_data

if __name__ == '__main__':
    b = {'喜欢的食物': ['糕点'], '发生事件': ['默认角色拒绝了顾秋拾递给他的精致糕点，并表示自己并不饿。']}
    c = {'人际关系': ['默认角色与顾秋拾之间存在一种复杂的关系，顾秋拾似乎对默认角色有着特殊的关注，而默认角色则对此表现出明显的不耐烦和挑衅的态度。(几分钟前)(1段对话前)'], '发生事件': ['默认角色在与顾秋拾的互动中表现出独立和直言不讳的性格，尽管在对方的强势态度下显得有些无奈和痛苦，但最终选择挑战顾秋拾的耐心。(几分钟前)(1段对话前)', '默认角色拒绝了顾秋拾递给他的精致糕点，并表示自己并不饿。(刚刚)']}

    b = {'姓名': ['梦宝'], '性别': ['女'], '默认角色喜欢如何称呼顾秋拾': ['本宫'], '默认角色喜欢被称呼的昵称': ['梦宝'], '人际关系': ['顾秋拾是梦宝的爱人，他们之间有着深厚的感情。'], '计划与期待': ['梦宝和顾秋拾期待着未来，并计划着要生一个大胖小子。'], '发生事件': ['梦宝与顾秋拾庆祝他们的大喜之日，顾秋拾使尽浑身解数表达对梦宝的爱意，而梦宝也回应了顾秋拾的爱意。', '在一天的活动结束后，顾秋拾为梦宝脱下鞋子，并把她扶到床上，为她的夜间休息做准备。']}
    c = {'人际关系': ['默认角色与顾秋拾之间存在一种复杂的关系，顾秋拾似乎对默认角色有着特殊的关注，而默认角色则对此表现出明显的不耐烦和挑衅的态度。(几天前)(4段对话前)', '默认角色拒绝了顾秋拾递给他的精致糕点，并表示自己不想吃。(几天前)(3段对话前)', '默认角色与顾秋拾结婚，成为夫妻。(几天前)(1段对话前)', '顾秋拾是梦宝的爱人，他们之间有着深厚的感情.'], '喜欢的食物': ['糕点(几天前)(3段对话前)', '出去玩, 划船(几天前)(2段对话前)', '顾秋拾希望以后能经常带默认角色去湖中心的小岛上看看。(几天前)(1段对话前)'], '发生事件': ['顾秋拾试图让默认角色开心地吃东西, 默认角色最终接受了糕点并表现出满意(几天前)(2段对话前)', '顾秋拾带领默认角色去御花园散步(几天前)(2段对话前)', '默认角色制作了花圈并装饰自己(几天前)(2段对话前)', '顾秋拾提议划船并交给默认角色桨(几天前)(2段对话前)', '默认角色显示给顾秋拾自己的划船技能。(几天前)(1段对话前)', '默认角色与顾秋拾成亲，结为夫妻。(几天前)(1段对话前)', '梦宝与顾秋拾庆祝他们的大喜之日, 顾秋拾表达了对梦宝的爱意', '顾秋拾为梦宝脱鞋并准备她夜间休息(几天前)'], '姓名': ['梦宝', '梦宝'], '性取向': ['异性恋(几天前)(1段对话前)'], '默认角色喜欢如何称呼顾秋拾': ['本宫', '梦宝喜欢被称呼为本宫'], '默认角色喜欢被称呼的昵称': ['梦宝', '梦宝'], '计划与期待': ['顾秋拾希望以后能经常带默认角色去湖中心的小岛上看看。(几天前)(1段对话前)', '梦宝和顾秋拾期待着未来，计划生孩子'], '关键日期': ['婚后洞房之夜(几天前)(1段对话前)']}

    b = {'喜欢的活动': ['出去玩, 划船'], '发生事件': ['默认角色拒绝吃糕点, 顾秋拾试图让默认角色开心地吃东西, 默认角色同意尝试划船']}
    c = {'发生事件': ['顾秋拾尝试让默认角色开心，提议出去玩，包括划船，但未提及结果。(当前对话)'], '背景设定': ['根据对话中的信息，默认角色可能与皇室有所关联，因顾秋拾提到默认角色收到了朝廷的婚契。(一小时前)(2段对话前)']}

    new_history_mem = clean_combined_json(b, c)
    print(new_history_mem)

    x = remove_bracketed_info('顾秋拾尝试让默认角色开心，提议出去玩，包括划船，但未提及结果。(当前对话)')
    print(x)
    