"""Exceptions for the span-extraction module."""

from __future__ import annotations


class ExtractionError(Exception):
    """Raised when the LLM response cannot be parsed into a span result."""


class CacheError(Exception):
    """Raised when the span cache cannot be read or written."""
