#最简单的agent实现
from runweave.core.model import Model
from runweave.core.memory import Memory
from runweave.core.tool import Tool
from runweave.executor.python_executor import FakePythonExecutor


SYSTEM_PROMPT = """You are an agent that solves tasks step by step.

CRITICAL RULES:
1. You MUST output exactly ONE thought or small action per response.
2. You MUST NOT solve the entire task in a single response.
3. After each response you will see an observation, then you respond again.
4. Only when you are completely done, start your response with "Final:"
   followed by the final answer.

Example of correct multi-step behavior:
  Step 1 response: "I need to calculate 2+2 first. The answer is 4."
  Step 2 response: "Now I need to recall the capital of France. It is Paris."
  Step 3 response: "Final: 2+2 is 4 and the capital of France is Paris."

Do NOT compress multiple steps into one response.
"""

class Agent:
    #初始化agent
    def __init__(self, model: Model, memory: Memory, tools: dict):
        self.model = model
        self.memory = memory
        self.tools = tools
        self.max_steps = 20
        self.is_final_answer = False
        self.step_count = 0
        self.python_executor = FakePythonExecutor(tools)
        self.memory.add("system", SYSTEM_PROMPT)

    def _valid_final_answer(self, step: str) -> bool:
        return step.strip().startswith("Final:")

    def run(self, task: str):
        # data flow per step:
        #   memory ──to_messages()──> model
        #   model ──generate()──> step (LLM's words)
        #   step ──add as assistant──> memory
        #   step ──check Final──> maybe break
        #   step ──execute──> result (observation from env)
        #   result ──add as user──> memory
        self.memory.add("user", task)
        last_step = None
        while self.step_count < self.max_steps and not self.is_final_answer:
            self.step_count += 1
            #生成step
            step = self.model.generate(self.memory.to_messages())
            print(f"\n--- step {self.step_count} ---")
            print(f"step: {step}")
            print("-" * 32)
            #判断是否是最终答案
            self.is_final_answer = self._valid_final_answer(step)
            if self.is_final_answer:
                self.memory.add("assistant", step)
                last_step = step
                break
            else:
                #更新记忆
                self.memory.add("assistant", step)
            #执行step
            result = self.python_executor.execute(step)
            #更新结果：
            self.memory.add("user", f"observation: {result}")
            
        return last_step