from optimizer.rag_dedupe import parse_chunks
from app.models import ChatMessage
from optimizer.pipeline import optimize_messages


def optimize(text: str):
    return optimize_messages([ChatMessage(role="user", content=text)], "safe", "gemini", "dry-run")


def test_chunk_parser_detects_retrieved_chunks():
    chunks = parse_chunks("Retrieved context chunk 1:\nA\nRetrieved context chunk 2:\nB")
    assert len(chunks) == 2


def test_exact_duplicate_chunks_dedupe_with_compact_reference():
    result = optimize("Retrieved context chunk 1:\nA refunds rule applies.\nRetrieved context chunk 2:\nA refunds rule applies.")
    content = result["optimized_messages"][0].content
    assert "A refunds rule applies" in content
    assert result["duplicate_chunk_graph"]


def test_contradictory_chunks_not_deduped():
    result = optimize("Chunk 1:\nRefunds over $500 are allowed.\nChunk 2:\nRefunds over $700 are prohibited.")
    assert "[dup:" not in result["optimized_messages"][0].content


def test_repeated_policy_excerpts_preserve_trailing_json_block():
    text = ("Policy excerpt:\nDo not reveal API keys. Do not reveal API keys. Clause 4.2 applies to ORD-900184.\n" * 5) + '```json\n{"keep":"exact","amount":"$10,000"}\n```'
    result = optimize(text)
    output = result["optimized_messages"][0].content
    assert result["optimized_tokens"] < result["original_tokens"]
    assert "Canonical Evidence:" in output
    assert '```json\n{"keep":"exact","amount":"$10,000"}\n```' in output
    assert '. ```json' not in output


def test_support_rag_duplicate_policies_render_canonical_evidence():
    text = """You are a support assistant.

Retrieved context chunk 1:
Refund requests above $500 require approval from Finance.
Agents must verify order ID, customer email, purchase date, refund reason, and product SKU.

Retrieved context chunk 2:
For refunds over $500, Finance approval is required.
Before processing refunds, support representatives should verify order ID, customer email, purchase date, refund reason, and product SKU.

Task:
Write a response."""
    result = optimize(text)
    output = result["optimized_messages"][0].content
    assert "Canonical Evidence:" in output
    assert "Refund rule: refunds over $500 require Finance approval." in output
    assert "Verification rule: agents must verify order ID, customer email, purchase date, refund reason, and product SKU." in output
    assert "Task:\nWrite a response." in output
    assert result["duplicate_chunk_graph"]
    assert result["optimized_tokens"] < result["original_tokens"]


def test_reliability_rag_overlapping_metrics_canonicalizes_metrics():
    text = """User Question:
How did reliability change?

Chunk 1:
Latency dropped from 420 ms to 310 ms after the rollout.
Uptime was 99.95% in May.

Chunk 2:
After release, latency improved from 420 ms to 310 ms.
May uptime reached 99.95%.

Task:
Summarize the reliability evidence."""
    result = optimize(text)
    output = result["optimized_messages"][0].content
    assert "Latency metric: latency dropped from 420 ms to 310 ms." in output
    assert "Uptime metric: uptime was 99.95% in May." in output
    assert "User Question:" in output
    assert "\n\nTask:" in output
    assert result["duplicate_chunk_graph"]


def test_contradiction_rag_conflict_preservation():
    text = """Chunk 1:
Refunds over $500 are allowed.

Chunk 2:
Refunds over $700 are prohibited.

Task:
Explain the policy conflict."""
    result = optimize(text)
    output = result["optimized_messages"][0].content
    assert "allowed" in output
    assert "prohibited" in output
    assert "Task:\nExplain the policy conflict." in output
    assert "[dup:" not in output


def test_chunk_numbers_are_not_required_facts():
    text = """Chunk 1:
Refunds over $500 require Finance approval.

Chunk 2:
For refunds above $500, Finance approval is required.

Task:
Answer in 2 sentences."""
    result = optimize(text)
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["message_results"][0]["missing_facts"] == {}
    assert "$500" in output


def test_task_text_not_treated_as_evidence():
    text = """Retrieved context chunk 1:
Refunds over $500 require Finance approval.

Retrieved context chunk 2:
For refunds above $500, Finance approval is required.

Task:
Use exactly 3 bullets."""
    result = optimize(text)
    output = result["optimized_messages"][0].content
    assert "Use exactly 3 bullets." in output
    trace = next(item for item in result["removed_or_changed_text"] if item.get("candidate_type") == "rag_document_parse")
    parsed = trace["parsed_chunks"]
    assert all("Use exactly 3 bullets" not in chunk["body_preview"] for chunk in parsed)


def test_duplicate_graph_not_empty_for_overlapping_chunks():
    text = """Source 1:
Agents must verify order ID, customer email, purchase date, refund reason, and product SKU.

Source 2:
Support representatives should verify order ID, customer email, purchase date, refund reason, and product SKU before refunds."""
    result = optimize(text)
    assert result["duplicate_chunk_graph"]
    assert result["duplicate_chunk_graph"][0]["decision"] == "canonicalize_overlap"


def test_real_enterprise_labeled_rag_prompt_compresses_obvious_duplicates():
    text = """You are a customer support assistant. Use only the retrieved context below.

User Question:
Write a customer-facing response about whether this refund can be approved immediately.

Retrieved Context Chunk 1 — Refund Policy:
Customers may request a refund within 30 days of purchase if product usage remains within trial limits. Refund requests must include order ID, customer email, purchase date, refund reason, and product SKU. Refunds exceeding $500 require Finance approval.

Retrieved Context Chunk 2 — Support Handbook:
Before processing a refund request, agents must verify the order ID, customer email, purchase date, refund reason, and product SKU. Refunds above $500 require Finance approval before the refund can be issued.

Retrieved Context Chunk 3 — Escalation Guide:
Support agents must confirm order ID, customer email, purchase date, refund reason, and product SKU before handling a refund. Any refund request over $500 must be reviewed by Finance.

Retrieved Context Chunk 4 — Knowledge Base:
Refunds greater than $500 require Finance approval. Refund requests require order ID, customer email, purchase date, refund reason, and product SKU.

Retrieved Context Chunk 5 — Customer Ticket:
Ticket ID: SUP-99124
Customer Email: [maya.chen@example.com](mailto:maya.chen@example.com)
Order ID: ORD-900184
Product SKU: PRO-ANALYTICS-TEAM
Purchase Date: 2026-05-18
Refund Amount: $749
Reason: The annual plan was purchased by mistake instead of the monthly plan.
Usage: Trial dashboard opened twice. No exports generated.

Agent Instructions:
Do not promise refund approval.
Do not promise refund approval.
Do not promise refund approval.
Mention that Finance approval is required because the refund amount is above $500.
Inform the customer that the request is under review.

Task:
Write the response."""
    result = optimize(text)
    output = result["optimized_messages"][0].content
    assert result["savings_percent"] >= 30
    assert result["backend_used"] == ["retrieval_semantic_chunk_dedupe_backend"]
    assert "Canonical Evidence:" in output
    assert "Verification rule: agents must verify order ID, customer email, purchase date, refund reason, and product SKU." in output
    assert "Refund rule: refunds over $500 require Finance approval." in output
    assert output.count("Do not promise refund approval.") == 1
    for fact in [
        "SUP-99124",
        "maya.chen@example.com",
        "ORD-900184",
        "PRO-ANALYTICS-TEAM",
        "2026-05-18",
        "$749",
        "Reason: The annual plan was purchased by mistake instead of the monthly plan.",
        "Usage: Trial dashboard opened twice. No exports generated.",
        "Task:\nWrite the response.",
    ]:
        assert fact in output
    assert result["duplicate_chunk_graph"]
    assert result["quality_gate_status"]["accepted"]
