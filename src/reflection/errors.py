"""Exception hierarchy for the reflection module."""


class ReflectionError(RuntimeError):
    """Base exception for the reflection module."""


class LLMCallError(ReflectionError):
    """The LLM API call failed (network, auth, rate limit)."""


class PromptAssemblyError(ReflectionError):
    """Prompt template files cannot be loaded."""
