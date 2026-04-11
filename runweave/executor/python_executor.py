#定义一个假的python执行器，用于测试


class FakePythonExecutor:
    """A fake executor for Stage 1.

    It doesn't actually execute Python code. It just looks at the
    LLM's output and returns a hardcoded observation. This is enough
    to test the agent loop end-to-end before we write a real executor
    in Stage 2.
    """

    def __init__(self, tools: dict):
        self.tools = tools

    def execute(self, code: str) -> str:
        return f"[stub] pretended to execute: {code[:50]}..."