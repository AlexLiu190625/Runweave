import os
from openai import OpenAI
#定义模型类，用于调用模型

class Model:
    #初始化模型
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        self.model_name = model_name

    #生成
    def generate(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content