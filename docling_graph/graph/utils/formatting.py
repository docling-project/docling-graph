"""String formatting utilities for graph display."""

from typing import Any
import re


def format_property_key(key: str) -> str:
    """Convert snake_case or camelCase to Title Case.

    Args:
        key: Property key to format.

    Returns:
        Formatted property key in Title Case.

    Examples:
        >>> format_property_key("user_name")
        'User Name'
        >>> format_property_key("userName")
        'User Name'
    """
    # Handle snake_case
    if '_' in key:
        return ' '.join(word.capitalize() for word in key.split('_'))

    # Handle camelCase
    return re.sub(r'([A-Z])', r' \1', key).strip().title()


def format_property_value(value: Any, max_length: int = 80) -> str:
    """Format property value with smart truncation.

    Args:
        value: Value to format.
        max_length: Maximum string length before truncation.

    Returns:
        Formatted and possibly truncated string.

    Examples:
        >>> format_property_value("Short text", 100)
        'Short text'
        >>> format_property_value("Very long text" * 10, 20)
        'Very long textVer...'
    """
    str_val = str(value)
    return truncate_string(str_val, max_length)


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
        raise ValueError(f"max_length ({max_length}) must be greater than suffix length ({len(suffix)})")

    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
