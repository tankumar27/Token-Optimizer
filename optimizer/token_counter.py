import re


def count_tokens(text: str) -> int:
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, int(len(pieces) * 1.15))


def count_message_tokens(messages: list[dict] | list[object]) -> int:
    total = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
        total += count_tokens(str(role)) + count_tokens(str(content)) + 4
    return total
