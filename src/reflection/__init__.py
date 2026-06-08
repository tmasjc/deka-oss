"""LLM-powered three-phase reflection module (reasoning trace only).

Public API::

    from src.reflection import ReflectionAgent, ReflectionError

    agent = ReflectionAgent()          # reads OPENROUTER_API_KEY from env
    result = agent.reflect(state)      # returns ReflectionOutput
"""

from .agent import ReflectionAgent
from .errors import ReflectionError

__all__ = ["ReflectionAgent", "ReflectionError"]
