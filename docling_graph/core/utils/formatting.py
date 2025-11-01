"""String formatting utilities for graph display."""

import re
from typing import Any


def format_property_value(value: Any, max_length: int = 80) -> str:
    """
    Format property value with smart truncation and list handling.

    Args:
        value: The value to format (can be str, list, dict, etc.)
        max_length: Maximum length before truncation

    Returns:
        Formatted string representation
    """
    # Handle lists - display as Python list notation
    if isinstance(value, list):
        return str(value)

    # Handle other types
    str_val = str(value)
    if len(str_val) <= max_length:
        return str_val

    return str_val[: max_length - 3] + "..."


def format_property_key(key: str) -> str:
    """
    Convert snake_case or camelCase to Title Case.

    Args:
        key: Property key to format

    Returns:
        Title-cased string
    """
    # Handle snake_case
    if "_" in key:
        return " ".join(word.capitalize() for word in key.split("_"))

    # Handle camelCase
    return re.sub(r"([A-Z])", r" \1", key).strip().title()


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to maximum length with suffix.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to add when truncating.

    Returns:
        Truncated string.

    Raises:
        ValueError: If max_length is less than suffix length.
    """
    if len(suffix) >= max_length:
        raise ValueError(
            f"max_length ({max_length}) must be greater than suffix length ({len(suffix)})"
        )

    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix
