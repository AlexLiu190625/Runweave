from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolagents import Tool

from runweave.skill.tools import LoadSkillTool, ReadSkillResourceTool, RunSkillScriptTool


@dataclass
class SkillMeta:
    """Skill metadata parsed from SKILL.md frontmatter."""

    name: str
    description: str
    path: Path  # Absolute path to the skill directory


class SkillLoader:
    """Scan the skills directory, parse SKILL.md files, and provide skill catalog and tools."""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir.resolve()
        self._skills: dict[str, SkillMeta] = {}
        self._scan()

    # -- Public methods ------------------------------------------------

    def get_catalog(self) -> str:
        """Generate a skill catalog string for injection into the system prompt."""
        if not self._skills:
            return ""
        lines = [
            "## Available Skills",
            "When a task matches one of the skills below, call load_skill(skill_name) to load detailed instructions before proceeding.",
            "",
        ]
        for meta in self._skills.values():
            lines.append(f"- **{meta.name}**: {meta.description}")
        return "\n".join(lines)

    def get_tools(self) -> list[Tool]:
        """Return skill-related Tool instances."""
        return [
            self.load_skill_tool,
            ReadSkillResourceTool(self),
            RunSkillScriptTool(self),
        ]

    @property
    def load_skill_tool(self) -> LoadSkillTool:
        """Return the LoadSkillTool singleton for tracking skill load history."""
        if not hasattr(self, "_load_skill_tool"):
            self._load_skill_tool = LoadSkillTool(self)
        return self._load_skill_tool

    def load_skill(self, name: str) -> str:
        """Read the body of the specified skill's SKILL.md (without frontmatter)."""
        meta = self._skills.get(name)
        if meta is None:
            return f"Error: skill '{name}' not found. Available skills: {', '.join(self._skills)}"
        content = (meta.path / "SKILL.md").read_text(encoding="utf-8")
        return self._strip_frontmatter(content)

    def read_resource(self, skill_name: str, path: str) -> str:
        """Read a file from the skill directory with path safety checks."""
        meta = self._skills.get(skill_name)
        if meta is None:
            return f"Error: skill '{skill_name}' not found."
        # Path safety: resolved path must stay within the skill directory
        target = (meta.path / path).resolve()
        if not target.is_relative_to(meta.path):
            return "Error: path traversal detected, access outside the skill directory is not allowed."
        if not target.is_file():
            return f"Error: file not found: {path}"
        return target.read_text(encoding="utf-8")

    def run_script(self, skill_name: str, script: str, args: str) -> str:
        """Execute a script from the skill's scripts/ directory with safety checks and timeout."""
        meta = self._skills.get(skill_name)
        if meta is None:
            return f"Error: skill '{skill_name}' not found."
        # Script must be inside the scripts/ subdirectory
        script_path = (meta.path / "scripts" / script).resolve()
        if not script_path.is_relative_to(meta.path / "scripts"):
            return "Error: script path traversal detected, only scripts/ directory files are allowed."
        if not script_path.is_file():
            return f"Error: script not found: scripts/{script}"
        # Build command
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
            return "Error: script execution timed out (30 second limit)."

    # -- Internal methods ----------------------------------------------

    def _scan(self) -> None:
        """Scan all subdirectories under skills_dir that contain a SKILL.md."""
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
        """Parse YAML frontmatter (simple implementation, no PyYAML dependency)."""
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
        """Remove the YAML frontmatter from the beginning of SKILL.md, returning the body."""
        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return content
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1 :]).strip()
        return content
