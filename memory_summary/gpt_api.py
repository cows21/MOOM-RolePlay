import json
from multiprocessing import Pool
import random

# encoding: utf-8
import openai
from openai import OpenAI
from tqdm import tqdm
import os
import logging

os.chdir(os.path.dirname(__file__))
logging.basicConfig(filename='GPT_api_exception.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_keys = [
        "sk-proj-wADJ3CPzi4NJpNMJX5DlT3BlbkFJmTtRrOrfL7EQYshoM9nO",
        "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"
            ]

removable_errors = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.RateLimitError
)

# TODO: 写成多进程
# 1. 一问一答
# 2. 多问多答
def chat_once(messages, key):
    '''

    Args:
        messages: [{"role": "assistant", "content": ans},{"role": "assistant", "content": ans},{"role": "assistant", "content": ans}]

    Returns:
        result / 被包好的result
    '''
    # 假设text有多条,需要遍历/拼接
    client = OpenAI(api_key=key,)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # 指定使用 gpt-4 模型
        #model='gpt-3.5-turbo',
        messages=messages
    )
    ans = response.choices[0].message.content

    messages.append({"role": "assistant", "content": ans})
    return messages

def get_useful_key(shared_keys):
    keys = list(shared_keys.keys())
    key = random.choice(keys)
    while shared_keys[key] <= 0:
        #choose one key which can be used
        key = random.choice(keys)
    shared_keys[key] -= 1
    return key

def is_removable(e):
    if isinstance(e, openai.RateLimitError) and 'You exceeded your current quota' not in e.message:
        return False
    if isinstance(e, removable_errors):
        return True
    return False

def handle_exception(e, key, shared_keys):
    logging.info({'error': e})
    if is_removable(e):
        shared_keys.pop(key, None)  # 移除密钥
    else:
        if key in shared_keys.keys():
            shared_keys[key] += 1
            
    if len(shared_keys) == 0:
        raise Exception("No useful api keys left!")


def chat_once_retry(messages, shared_keys):
    retry_limit = 5
    while len(shared_keys) > 0 and retry_limit > 0:
        retry_limit -= 1
        try:
            key = get_useful_key(shared_keys)
            messages = chat_once(messages, key)
            shared_keys[key] += 1
            return messages
        except Exception as e:
            handle_exception(e, key, shared_keys)
    if len(shared_keys) <= 0:
        logging.info({'error': "No useful api keys left!"})
        raise Exception("No useful api keys left!")
    logging.info({'error' : "retry too many times"})
    raise Exception("retry too many times!")