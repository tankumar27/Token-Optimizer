from optimizer.protect import protect_text, restore_text
from optimizer.pipeline import optimize_messages
from app.models import ChatMessage


def roundtrip(text: str) -> str:
    protected, regions = protect_text(text)
    return restore_text(protected, regions)


def test_protected_regions_roundtrip_byte_for_byte():
    samples = [
        "```python\nprint('ORD-900184')\n```",
        "Call `torch.softmax()` now.",
        "{\"id\":\"SUP-44891\",\"amount\":\"$749\"}",
        "```yaml\nid: BW-HIPAA-7741\n```",
        "<root><id>SUP-44891</id></root>",
        "Solve y = x^2 - x - 23",
        "Email ops@example.com and visit https://example.com",
        "Clause 4.2 says \"exact wording\" on March 14, 2025.",
    ]
    for sample in samples:
        assert roundtrip(sample) == sample


def test_optimizer_preserves_sensitive_content():
    text = "Email ops@example.com about ORD-900184 on 2026-06-01 for $10,000. Please please help help."
    result = optimize_messages([ChatMessage(role="user", content=text)], "safe", "gemini", "dry-run")
    optimized = result["optimized_messages"][0].content
    for value in ["ops@example.com", "ORD-900184", "2026-06-01", "$10,000"]:
        assert value in optimized
