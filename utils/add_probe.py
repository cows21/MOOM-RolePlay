import random

def insert_random_info(dialogue_text, info_file, character_name, used_infos):
    dialogues = dialogue_text.split('\n')
    with open(info_file, 'r', encoding='utf-8') as file:
        infos = [line.strip() for line in file.readlines() if line.strip() not in used_infos]
    
    character_dialogues = [(index, line) for index, line in enumerate(dialogues) if line.startswith(character_name)]
    
    if character_dialogues and infos:  # 确保还有可用信息和对话
        selected_info = random.choice(infos)
        info_index, selected_dialogue = random.choice(character_dialogues)

        info_to_insert = selected_info.split(', ')[2]
        new_dialogue = f"{selected_dialogue} {info_to_insert}"
        dialogues[info_index] = new_dialogue

        new_dialogue_text = '\n'.join(dialogues)
        used_infos.append(selected_info)  # 更新已用信息列表

        return new_dialogue_text, selected_info.split(', ')[:2], used_infos
    else:
        return dialogue_text, None, used_infos

if __name__ == '__main__':
    # 示例对话
    dialogue_text = """西弗勒斯·斯内普: 记住，任何情况下，都不能强迫或威胁任何人进行非法行为。
    君墨: （亲）
    西弗勒斯·斯内普: (回应)唔，你这个小家伙，总是这么直接。
    君墨: 知道啦～
    西弗勒斯·斯内普: (摸摸他的头)好了，时间差不多了，我们该走了。
    君墨: (牵你的手)好的，老公~"""

    # 示例信息文件路径
    info_file = 'C:\\Users\\tangjinyi\\Documents\\cws_code\\memory_api3\\prompt\\prob_set.txt'

    # 使用函数
    updated_dialogue, info_tags = insert_random_info(dialogue_text, info_file, '西弗勒斯·斯内普')
    print(updated_dialogue)
    print(info_tags)