from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
import torch
import gc
from .utils import Qwen_load
from typing import List
from openai import OpenAI

dialogue_summary_prompt_template = """
在对话区中我会给你一段对话，请你用20-100个字总结这段对话的摘要。
你的总结要精简一些，能表达清楚意思即可。同时要尽可能覆盖对话中的关键名词、时间、动作等要点。并按照格式区的格式输出。
对话区{{
{content}
}}
格式区{{
这段内容的摘要是：
[具体摘要内容]
例如{{
这段内容的摘要是：
[小红喜欢吃红富士苹果。小兰不喜欢吃苹果，她更喜欢吃菠萝。他们说果唯伊水果店的水果最好。]
}}
}}
"""

summaries_summary_prompt_template = """
在内容区中我会给你一段较长的文本，它包含了多个对话摘要，请你用100-200个字对这些摘要文本再总结一次摘要。
对内容区的摘要文本的再次摘要可以再精简、抽象一些，过滤掉重复相似的要点，总结出这些摘要的核心内容。同时要尽可能覆盖内容中的关键名词、时间、动作等要点。并按照格式区的格式输出。
内容区{{
{content}
}}
格式区{{
这段内容的摘要是：
[具体摘要内容]
例如{{
这段内容的摘要是：
[小红喜欢吃红富士苹果。小兰不喜欢吃苹果，她更喜欢吃菠萝。他们说果唯伊水果店的水果最好。]
}}
}}
"""


class LLMAPI:
    def __init__(self, model, tokenzier, api_url, use_api=False):
        self.model =  model
        self.tokenizer = tokenzier    
        self.device = self.model.device

        self.api_url = api_url
        self.use_api = use_api
        
    def chat(self, query, history  = None):
        if not history:
            history = [{"role": "system", "content": "You are a helpful assistant."}]
        history.append({
            "role": "user", "content": query
        })

        if self.use_api is True:
            openai_api_key = "EMPTY"
            client = OpenAI(api_key=openai_api_key,base_url=self.api_url)
            response_ori = client.chat.completions.create(
                #model="gpt-4-1106-preview",  # 指定使用 gpt-4 模型
                model='qwen_32b',
                #model='gpt-3.5-turbo',
                messages=history,
                temperature = 1.0,
                timeout=600
            )
            response = response_ori.choices[0].message.content
        else:

            text = self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs,
                    max_new_tokens = 512
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs, generated_ids)
            ]
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            history = history.append({
                "role": "assistant", "content": response
            })
            del model_inputs
            del generated_ids
            torch.cuda.empty_cache()
            gc.collect()
        return response, history
    
    def get_dialogue_summary(self, text: str):
        #用来获取对话的summary
        prompt = dialogue_summary_prompt_template.format(
            content = text
        )
        summary, _ = self.chat(prompt)
        return summary


    def get_summaries_summary(self, text: str):
        #用来获取summaries的summary
        prompt = summaries_summary_prompt_template.format(
            content = text
        )
        summary, _ = self.chat(prompt)
        return summary