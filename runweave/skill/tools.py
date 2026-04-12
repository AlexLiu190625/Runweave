from __future__ import annotations

from typing import TYPE_CHECKING

from smolagents import Tool

if TYPE_CHECKING:
    from runweave.skill.loader import SkillLoader


class LoadSkillTool(Tool):
    """加载指定 skill 的完整指令文档。"""

    name = "load_skill"
    description = (
        "加载指定 skill 的详细指令。"
        "当你的任务匹配某个可用 skill 时，先调用此工具获取完整指令，再按指令执行。"
    )
    inputs = {
        "skill_name": {
            "type": "string",
            "description": "要加载的 skill 名称（来自可用 Skills 列表）",
        },
    }
    output_type = "string"

    def __init__(self, loader: SkillLoader) -> None:
        self.loader = loader
        self._loaded: list[str] = []
        super().__init__()

    def forward(self, skill_name: str) -> str:
        self._loaded.append(skill_name)
        return self.loader.load_skill(skill_name)

    def get_loaded_and_reset(self) -> list[str]:
        """返回去重的已加载 skill 列表，并重置记录。"""
        result = list(dict.fromkeys(self._loaded))
        self._loaded.clear()
        return result


class ReadSkillResourceTool(Tool):
    """读取 skill 目录下的参考文件。"""

    name = "read_skill_resource"
    description = (
        "读取 skill 目录下的文件（references/、assets/ 等）。"
        "当 skill 指令中引用了额外文件时使用。"
    )
    inputs = {
        "skill_name": {
            "type": "string",
            "description": "skill 名称",
        },
        "path": {
            "type": "string",
            "description": "相对于 skill 目录的文件路径，如 references/FORMS.md",
        },
    }
    output_type = "string"

    def __init__(self, loader: SkillLoader) -> None:
        self.loader = loader
        super().__init__()

    def forward(self, skill_name: str, path: str) -> str:
        return self.loader.read_resource(skill_name, path)


class RunSkillScriptTool(Tool):
    """执行 skill 目录下 scripts/ 中的脚本。"""

    name = "run_skill_script"
    description = (
        "执行 skill 目录下 scripts/ 中的脚本，返回脚本输出。"
        "当 skill 指令要求运行脚本时使用。"
    )
    inputs = {
        "skill_name": {
            "type": "string",
            "description": "skill 名称",
        },
        "script": {
            "type": "string",
            "description": "scripts/ 下的脚本文件名，如 validate.py",
        },
        "args": {
            "type": "string",
            "description": "传给脚本的命令行参数（可为空字符串）",
        },
    }
    output_type = "string"

    def __init__(self, loader: SkillLoader) -> None:
        self.loader = loader
        super().__init__()

    def forward(self, skill_name: str, script: str, args: str) -> str:
        return self.loader.run_script(skill_name, script, args)
