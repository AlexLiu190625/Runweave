from __future__ import annotations

from typing import TYPE_CHECKING

from smolagents import Tool

if TYPE_CHECKING:
    from runweave.skill.loader import SkillLoader


class LoadSkillTool(Tool):
    """Load the full instruction document for a specified skill."""

    name = "load_skill"
    description = (
        "Load detailed instructions for a specified skill. "
        "When your task matches an available skill, call this tool first "
        "to get the full instructions, then follow them."
    )
    inputs = {
        "skill_name": {
            "type": "string",
            "description": "Name of the skill to load (from the Available Skills list)",
        },
    }
    output_type = "string"

    def __init__(self, loader: SkillLoader) -> None:
        self.loader = loader
        # Tracks skills loaded during a single run; reset by get_loaded_and_reset().
        # Not thread-safe: assumes sequential Runtime.run() calls, which matches
        # smolagents' own single-threaded execution model.
        self._loaded: list[str] = []
        super().__init__()

    def forward(self, skill_name: str) -> str:
        self._loaded.append(skill_name)
        return self.loader.load_skill(skill_name)

    def get_loaded_and_reset(self) -> list[str]:
        """Return the deduplicated list of loaded skills and reset the record."""
        result = list(dict.fromkeys(self._loaded))
        self._loaded.clear()
        return result


class ReadSkillResourceTool(Tool):
    """Read a reference file from a skill directory."""

    name = "read_skill_resource"
    description = (
        "Read a file from a skill directory (references/, assets/, etc.). "
        "Use this when skill instructions reference additional files."
    )
    inputs = {
        "skill_name": {
            "type": "string",
            "description": "Skill name",
        },
        "path": {
            "type": "string",
            "description": "File path relative to the skill directory, e.g. references/FORMS.md",
        },
    }
    output_type = "string"

    def __init__(self, loader: SkillLoader) -> None:
        self.loader = loader
        super().__init__()

    def forward(self, skill_name: str, path: str) -> str:
        return self.loader.read_resource(skill_name, path)


class RunSkillScriptTool(Tool):
    """Execute a script from a skill's scripts/ directory."""

    name = "run_skill_script"
    description = (
        "Execute a script from a skill's scripts/ directory and return its output. "
        "Use this when skill instructions require running a script."
    )
    inputs = {
        "skill_name": {
            "type": "string",
            "description": "Skill name",
        },
        "script": {
            "type": "string",
            "description": "Script filename under scripts/, e.g. validate.py",
        },
        "args": {
            "type": "string",
            "description": "Command-line arguments for the script (can be empty string)",
        },
    }
    output_type = "string"

    def __init__(self, loader: SkillLoader) -> None:
        self.loader = loader
        super().__init__()

    def forward(self, skill_name: str, script: str, args: str) -> str:
        return self.loader.run_script(skill_name, script, args)
