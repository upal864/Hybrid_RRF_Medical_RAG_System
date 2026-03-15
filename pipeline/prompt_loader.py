"""
pipeline/prompt_loader.py

Loads prompts from prompts/prompts.yaml with in-memory caching.
Supports hot-reload in development by passing reload=True.
"""

import yaml
from pathlib import Path

_cache: dict | None = None
_PROMPT_FILE = Path("prompts/prompts.yaml")


def _load(reload: bool = False) -> dict:
    global _cache
    if _cache is None or reload:
        raw = _PROMPT_FILE.read_text(encoding="utf-8")
        _cache = yaml.safe_load(raw)
    return _cache


def get_prompt(name: str, reload: bool = False, **kwargs) -> dict:
    """
    Returns the system and formatted user message for the named prompt.
    Raises KeyError if the prompt name does not exist.
    """
    data = _load(reload=reload)
    try:
        prompt = data["prompts"][name]
    except KeyError:
        raise KeyError(f"Prompt '{name}' not found in {_PROMPT_FILE}. Available: {list(data['prompts'].keys())}")

    return {
        "system":  prompt["system"].strip(),
        "user":    prompt["user_template"].format(**kwargs).strip(),
        "version": data["version"],
    }


def get_version(reload: bool = False) -> str:
    return _load(reload=reload)["version"]
