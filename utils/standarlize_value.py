from FlagEmbedding import BGEM3FlagModel
import re
import os

class MemoryStandardizer:
    def __init__(self, bge_model):
        self.model = bge_model
        self.keywords = {
            "性别": ["男", "女"],
            "性取向": ["男", "女", "异性恋", "同性恋"],
            "星座": ["白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座", "天秤座", "天蝎座", "射手座", "魔羯座", "水瓶座", "双鱼座"],
            "生肖": ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"],
            "MBTI": ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"]
        }

        self.keywords_regex = {
            "性别": (["男", "女"], ["男", "女"]),
            "性取向": (["男", "女", "异性恋", "同性恋"], ["男", "女", "异性恋", "同性恋"]),
            "星座": (["白羊", "金牛", "双子", "巨蟹", "狮子", "处女", "天秤", "天蝎", "射手", "魔羯", "水瓶", "双鱼"], 
                     ["白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座", "天秤座", "天蝎座", "射手座", "魔羯座", "水瓶座", "双鱼座"]),
            "生肖": (["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"], ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]),
            "MBTI": (["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"],
                     ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"])
        }

        self.compiled_patterns = {
            key: {re.compile(re.escape(word)): rep for word, rep in zip(*vals)}
            for key, vals in self.keywords_regex.items()
        }

    def standardize_bge(self, memory_dict, keys_to_be_processed=None, threshold=0.4):
        new_memory = {}
        for key, values in memory_dict.items():
            if keys_to_be_processed is None or len(keys_to_be_processed) == 0 or key in keys_to_be_processed:
                if key in self.keywords:
                    candidate_keywords = self.keywords[key]
                    best_matches = self.find_best_matches_bge(values, candidate_keywords, threshold)
                    new_memory[key] = best_matches
                else:
                    new_memory[key] = values
            else:
                new_memory[key] = values
        return new_memory

    def find_best_matches_bge(self, elements, candidate_keywords, threshold):
        # 生成所有可能的句子对
        sentence_pairs = [[element, candidate] for element in elements for candidate in candidate_keywords]
        scores = self.model.compute_score(sentence_pairs, max_passage_length=128, weights_for_different_modes=[0.4, 0.2, 0.4])
        colbert_scores = scores['colbert+sparse+dense']

        print('sentence_pairs')
        print(sentence_pairs)
        print('scores')
        print(scores)
        
        # 分析每个元素与所有候选关键词的相似度，找到最佳匹配
        standardized_elements = []
        index = 0
        for element in elements:
            highest_score = 0
            best_match = element  # 默认保留原始元素
            for candidate in candidate_keywords:
                score = colbert_scores[index]
                if score > threshold and score > highest_score:
                    highest_score = score
                    best_match = candidate
                index += 1
            standardized_elements.append(best_match)
        return standardized_elements
    
    def standardize_regex(self, memory_dict, keys_to_be_processed=None):
        new_memory = {}
        for key, values in memory_dict.items():
            if keys_to_be_processed and key not in keys_to_be_processed:
                new_memory[key] = values
                continue
            if key in self.compiled_patterns:
                regex_patterns = self.compiled_patterns[key]
                new_values = self.apply_regex(values, regex_patterns)
                new_memory[key] = new_values
            else:
                new_memory[key] = values
        return new_memory

    def apply_regex(self, elements, regex_patterns):
        standardized_elements = []
        for element in elements:
            matches = {pattern: pattern.search(element) for pattern in regex_patterns}
            match_count = sum(1 for match in matches.values() if match)
            if match_count == 1:
                for pattern, match in matches.items():
                    if match:
                        element = regex_patterns[pattern]
                        break
            standardized_elements.append(element)
        return standardized_elements


if __name__ == '__main__':
    # 示例
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
    # model_path = 'model/FlagEmbedding/BAAI/bge-m3'
    # bge_model = BGEM3FlagModel(model_path, use_fp16=True)

    # memory_data = {"性别": ["男性", "女性人", "不是男的"], "星座": ["狮子", "巨蟹"]}
    # memory_data2 = {"性别": ["不是男的"]}
    # standardizer = MemoryStandardizer(bge_model)
    # standardized_memory = standardizer.standardize_bge(memory_data2)

    bge_model = None
    memory_data = {"性别": ["男性化的女性", "未明性别", "男性"], "星座": ["我是金牛座的", "我是狮子"]}
    # memory_data = {"性别": ["男性化的女性", "未明性别"]}
    standardizer = MemoryStandardizer(bge_model)
    standardized_memory = standardizer.standardize_regex(memory_data)
    print(standardized_memory)
