def build_memory_prompt_cws(user_name, bot_name, content, prompt):
    prompt = prompt.replace("<user_name>", user_name)
    prompt = prompt.replace("<bot_name>", bot_name)
    prompt = prompt.replace("{messages}", content)
    return prompt

def build_time_prompt_cws(content, prompt):
    prompt = prompt.replace("{messages}", content)
    return prompt

def build_combine_prompt_cws(mem_old, mem_new, prompt):
    prompt = prompt.replace("{mem_old}", mem_old)
    prompt = prompt.replace("{mem_new}", mem_new)
    return prompt

def build_error_prompt_cws(content, prompt):
    prompt = prompt.replace("{messages}", content)
    return prompt

def build_combine_prompt_cws_autodate(mem_old, mem_new, new_mem_time, prompt):
    prompt = prompt.replace("{mem_old}", mem_old)
    prompt = prompt.replace("{mem_new}", mem_new)
    prompt = prompt.replace("{new_mem_time}", new_mem_time)
    return prompt