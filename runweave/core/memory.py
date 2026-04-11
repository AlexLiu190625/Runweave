#定义记忆类，用于存储记忆

class Memory:
    #初始化记忆
    def __init__(self):
        self.memory = []

    #添加记忆
    def add(self, role: str, content: str) -> None:
        self.memory.append({"role": role, "content": content})

    #转换为消息列表
    def to_messages(self):
        return list(self.memory)