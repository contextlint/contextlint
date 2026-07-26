"""contextlint — audit LLM prompt and context spend.

Finds recoverable tokens in prompt templates and request logs, and separates
what is provably safe to cut from what needs an eval before you ship it.

Runs entirely offline with no third-party dependencies.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
