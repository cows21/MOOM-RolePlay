# content格式
# content:[{”history”,”personality”}]
# 每第10，20…个还有”overall_history”, ”overall_personality“两个key

def summarize_content_prompt(dia):
    prompt = '请总结以下的对话内容，尽可能精炼，提取对话的主题和关键信息。如果有多个关键事件，可以分点总结。对话内容：\n' 
    prompt += dia
    prompt += ('\n总结：')
    return prompt

def summarize_overall_prompt(content):
    prompt = '请高度概括以下的事件，尽可能精炼，概括并保留其中核心的关键信息。概括事件：\n' 
    for turn,summary_dict in enumerate(content):
        summary = summary_dict['history']
        prompt += (f"\n第{turn}轮对话发生的事件为{summary.strip()}")
    prompt += ('\n总结：')
    return prompt

def summarize_overall_personality(content, user_name, bot_name):
    prompt = '以下是用户在多段对话中展现出来的人格特质和心情，以及当下合适的回复策略：\n' 
    for turn,summary_dict in enumerate(content):
        summary = summary_dict['personality']
        prompt += (f"\n第{turn}轮对话的分析为{summary.strip()}" )
    prompt += (f'\n请总体概括{user_name}的性格和{bot_name}最合适的回复策略，尽量简洁精炼，高度概括。总结为：')
    return prompt

def summarize_person_prompt(dia, user_name, bot_name):
    prompt = f'请根据以下的对话推测总结{user_name}的性格特点和心情，并根据你的推测制定回复策略。对话内容：\n' 
    prompt += dia
    prompt += (f'\n{user_name}的性格特点、心情、{bot_name}的回复策略为：')
    return prompt
