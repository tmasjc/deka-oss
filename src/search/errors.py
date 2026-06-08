"""Specific exceptions raised by the hybrid search module."""


class ConfigError(ValueError):
    """Raised when a SearchConfig is malformed or a config file is invalid."""


class EmbeddingServiceError(RuntimeError):
    """Raised when the BGE-M3 embedding service is unreachable or errors out."""


class MilvusSearchError(RuntimeError):
    """Raised when a Milvus search call fails or the collection is missing."""


class AdaptError(RuntimeError):
    """Raised when the adaptive default-config step cannot produce a usable config.

    The most common cause is all three retrieval paths returning zero hits
    in the Turn-0 probe — the query is genuinely uncovered by the corpus
    and the user should pick a different one.
    """
