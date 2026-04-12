from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolagents import Tool

from runweave.skill.tools import LoadSkillTool, ReadSkillResourceTool, RunSkillScriptTool


@dataclass
class SkillMeta:
    """skill 的元信息，从 SKILL.md frontmatter 解析而来。"""

    name: str
    description: str
    path: Path  # skill 目录的绝对路径


class SkillLoader:
    """扫描 skills 目录，解析 SKILL.md，提供 skill 目录和 tools。"""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir.resolve()
        self._skills: dict[str, SkillMeta] = {}
        self._scan()

    # ── 公开方法 ──────────────────────────────────────────

    def get_catalog(self) -> str:
        """生成 skill 目录文本，用于注入 system prompt。"""
        if not self._skills:
            return ""
        lines = [
            "## 可用 Skills",
            "当任务匹配以下 skill 时，请调用 load_skill(skill_name) 加载详细指令后再执行。",
            "",
        ]
        for meta in self._skills.values():
            lines.append(f"- **{meta.name}**: {meta.description}")
        return "\n".join(lines)

    def get_tools(self) -> list[Tool]:
        """返回 skill 相关的 Tool 实例。"""
        return [
            self.load_skill_tool,
            ReadSkillResourceTool(self),
            RunSkillScriptTool(self),
        ]

    @property
    def load_skill_tool(self) -> LoadSkillTool:
        """返回 LoadSkillTool 单例，用于追踪 skill 加载记录。"""
        if not hasattr(self, "_load_skill_tool"):
            self._load_skill_tool = LoadSkillTool(self)
        return self._load_skill_tool

    def load_skill(self, name: str) -> str:
        """读取指定 skill 的 SKILL.md 正文（不含 frontmatter）。"""
        meta = self._skills.get(name)
        if meta is None:
            return f"错误：未找到名为 '{name}' 的 skill。可用 skills: {', '.join(self._skills)}"
        content = (meta.path / "SKILL.md").read_text(encoding="utf-8")
        return self._strip_frontmatter(content)

    def read_resource(self, skill_name: str, path: str) -> str:
        """读取 skill 目录下的文件，带路径安全校验。"""
        meta = self._skills.get(skill_name)
        if meta is None:
            return f"错误：未找到名为 '{skill_name}' 的 skill。"
        # 路径安全：解析后必须在 skill 目录内
        target = (meta.path / path).resolve()
        if not str(target).startswith(str(meta.path)):
            return "错误：路径越界，不允许访问 skill 目录之外的文件。"
        if not target.is_file():
            return f"错误：文件不存在: {path}"
        return target.read_text(encoding="utf-8")

    def run_script(self, skill_name: str, script: str, args: str) -> str:
        """执行 skill 目录下 scripts/ 中的脚本，带安全校验和超时。"""
        meta = self._skills.get(skill_name)
        if meta is None:
            return f"错误：未找到名为 '{skill_name}' 的 skill。"
        # 脚本必须在 scripts/ 子目录内
        script_path = (meta.path / "scripts" / script).resolve()
        if not str(script_path).startswith(str(meta.path / "scripts")):
            return "错误：脚本路径越界，只允许执行 scripts/ 目录下的文件。"
        if not script_path.is_file():
            return f"错误：脚本不存在: scripts/{script}"
        # 构建命令
        cmd = [str(script_path)]
        if args and args.strip():
            cmd.extend(args.strip().split())
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(meta.path),
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\n[stderr]\n{result.stderr}" if result.stderr else ""
                output += f"\n[exit code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return "错误：脚本执行超时（30 秒限制）。"

    # ── 内部方法 ──────────────────────────────────────────

    def _scan(self) -> None:
        """扫描 skills_dir 下所有包含 SKILL.md 的子目录。"""
        if not self.skills_dir.is_dir():
            return
        for skill_md in self.skills_dir.glob("*/SKILL.md"):
            frontmatter = self._parse_frontmatter(
                skill_md.read_text(encoding="utf-8")
            )
            if frontmatter and "name" in frontmatter and "description" in frontmatter:
                self._skills[frontmatter["name"]] = SkillMeta(
                    name=frontmatter["name"],
                    description=frontmatter["description"],
                    path=skill_md.parent.resolve(),
                )

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, str] | None:
        """解析 YAML frontmatter（简单实现，不依赖 PyYAML）。"""
        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return None
        result: dict[str, str] = {}
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and value:
                    result[key] = value
        return result if result else None

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """移除 SKILL.md 开头的 YAML frontmatter，返回正文。"""
        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return content
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1 :]).strip()
        return content
