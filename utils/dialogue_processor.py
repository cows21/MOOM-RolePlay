import json

class OriginDialogueProcessor:
    """
    处理单个对话的类，包括生成对话文本、分割对话轮次、统计轮次等功能。
    """
    def __init__(self, dialogue_data):
        """
        初始化对话处理器。
        参数:
        - dialogue_data: 包含对话信息的字典。
        """
        self.dialogue_data = dialogue_data
        self.messages = dialogue_data['messages']
        self.role_meta = dialogue_data['role_meta']
        self.prompt = dialogue_data['prompt']
        self._total_turns = None  # 缓存对话的总轮次数
        
    def generate_conversation(self):
        """
        生成整个对话的文本，每条消息后添加换行。
        返回:
        - 格式化的对话字符串。
        """
        conversation = []
        for message in self.messages:
            text = message['text']
            sender = message['sender_name']
            formatted_text = f"{sender}: {text}\n"
            conversation.append(formatted_text)
        return ''.join(conversation)

    def split_by_turns(self, num_turns=1):
        """
        按对话轮次分割对话，根据指定的轮数来组织数据。
        参数:
        - num_turns: 每个列表元素包含的轮次数。
        返回:
        - 分割后的对话轮次列表，每个元素是指定轮次数的格式化对话文本。
        """
        rounds = []
        current_speaker = None
        current_round = []
        
        for message in self.messages:
            if message['sender_name'] != current_speaker:
                if current_round:
                    rounds.append(current_round)
                current_round = [message]
                current_speaker = message['sender_name']
            else:
                current_round.append(message)
        
        if current_round:
            rounds.append(current_round)

        # 组合指定轮数的对话文本
        combined_rounds = []
        for i in range(0, len(rounds), num_turns):
            combined_text = []
            for j in range(i, min(i + num_turns, len(rounds))):
                for message in rounds[j]:
                    combined_text.append(f"{message['sender_name']}: {message['text']}\n")
            combined_rounds.append(''.join(combined_text))
        
        return combined_rounds

    def count_total_turns(self):
        """
        计算对话的总轮次数。
        返回:
        - 对话的总轮次数。
        """
        if self._total_turns is None:
            self._total_turns = len(self.split_by_turns())
        return self._total_turns

    def get_conversation_by_turns(self, start_turn, end_turn):
        """
        根据起始和终止轮次号截取对话。
        参数:
        - start_turn: 起始轮次号。
        - end_turn: 终止轮次号。
        返回:
        - 截取的对话字符串。
        """
        rounds = self.split_by_turns()
        selected_rounds = rounds[start_turn:end_turn+1]
        conversation = []
        for round_text in selected_rounds:
            conversation.append(round_text)
        return ''.join(conversation)

    def get_role_info(self):
        """
        获取对话中角色的基本信息，包括名字和prompt字段中的额外信息。
        返回:
        - 角色信息列表。
        """
        role_info = self.role_meta.copy()  # 创建角色基础信息的副本
        character_info = self.prompt.copy()
        return role_info, character_info

class OriginJsonlDialogueReader:
    """
    从jsonl文件读取对话数据的类。
    """
    def __init__(self, filename):
        """
        初始化文件阅读器。
        参数:
        - filename: jsonl文件的路径。
        """
        self.filename = filename
    
    def read_dialogue_at_line(self, line_number):
        """
        读取文件中指定行数的对话数据。
        参数:
        - line_number: 指定的行数。
        返回:
        - 对话处理器实例，包含读取的对话数据。
        """
        with open(self.filename, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == line_number:
                    dialogue_data = json.loads(line)
                    return OriginDialogueProcessor(dialogue_data)
        return None
