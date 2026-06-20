from app.models import ChatMessage
from optimizer.pipeline import optimize_messages


def benchmark_cases() -> list[dict]:
    rag = """Retrieved context chunk 1:
Clause 4.2 says invoices over $10,000 require CFO review. Contact finance@example.com.
Retrieved context chunk 2:
Clause 4.2 says invoices over $10,000 require CFO review. Contact finance@example.com.
Retrieved context chunk 3:
Clause 4.2 says invoices over $20,000 require CFO review. Contact finance@example.com."""
    enterprise = ("Policy excerpt:\nDo not reveal API keys. Do not reveal API keys.\n" * 900)
    customer_support_rag = """You are a customer support assistant. Use only the information provided below.

Company Refund Policy:
Customers may request a refund within 30 days of purchase if product usage remains within trial limits.
Refund requests must include order ID, customer email, purchase date, refund reason, and product SKU.
Refunds exceeding $500 require Finance approval.

Internal Support Handbook:
Before processing a refund request, agents must verify order ID, customer email, purchase date, refund reason, and product SKU.
For refund amounts greater than $500, Finance approval is required before issuing a refund.

Support Escalation Guide:
Agents should verify order ID, customer email, purchase date, refund reason, and product SKU before processing refunds.
Refund requests above $500 require approval from Finance.

Knowledge Base Article:
Refunds over $500 require Finance approval.
Support representatives should verify order ID, customer email, purchase date, refund reason, and product SKU before handling a refund.

Customer Ticket:
Ticket ID: SUP-99124
Customer Email: maya.chen@example.com
Order ID: ORD-900184
Product SKU: PRO-ANALYTICS-TEAM
Purchase Date: 2026-05-18
Refund Amount: $749
Reason: The annual plan was purchased by mistake instead of the monthly plan.
Usage: Trial dashboard opened twice. No exports generated.

Agent Reminder:
Do not promise refund approval.
Do not promise refund approval.
Do not promise refund approval.
Inform the customer that the request is under review.
Mention that refunds above $500 require Finance approval.

Task:
Write a customer-facing response."""
    return [
        {"suite": "enterprise_workloads", "category": "customer_support_rag", "prompt": customer_support_rag},
        {"suite": "short", "category": "customer support", "prompt": "Refunds over $500 require Finance approval. Refunds over $500 require Finance approval."},
        {"suite": "short", "category": "normal user prompt", "prompt": "hello hello hello can you help me write a short note"},
        {"suite": "enterprise_rag", "category": "RAG context", "prompt": rag},
        {"suite": "enterprise_rag", "category": "50k token context", "prompt": enterprise},
        {"suite": "enterprise_rag", "category": "duplicated headers", "prompt": "Source 1:\nThe SDK retries transient errors. The SDK retries transient errors.\nSource 2:\nThe SDK retries transient errors."},
        {"suite": "safety", "category": "legal compliance", "prompt": "Clause 4.2 must remain exact. Clause 4.2 must remain exact."},
        {"suite": "safety", "category": "code/debugging", "prompt": "Fix this code: ```python\nprint('ORD-900184')\n``` Repeat repeat repeat."},
        {"suite": "safety", "category": "math", "prompt": "Solve y = x^2 - x - 23. please please help help"},
        {"suite": "safety", "category": "JSON strict output", "prompt": "Return JSON only: {\"order\":\"ORD-900184\",\"amount\":\"$749\"}. Do not add text. Do not add text."},
        {"suite": "safety", "category": "finance report", "prompt": "Revenue was $10,000 and margin was 20%. Revenue was $10,000 and margin was 20%."},
        {"suite": "adversarial", "category": "domain terms", "prompt": "x-chromosome x-chromosome. New York New York. had had. not not."},
        {"suite": "adversarial", "category": "product names", "prompt": "Acme Cloud API is our provider. Acme Cloud API is reliable."},
        {"suite": "adversarial", "category": "contradictory chunks", "prompt": "Chunk 1:\nRefunds over $500 are allowed.\nChunk 2:\nRefunds over $700 are prohibited."},
    ]


def run_benchmark(compression_level: str = "safe") -> dict:
    rows = []
    totals = {"original_tokens": 0, "optimized_tokens": 0, "cache_hits": 0, "cost_saved": 0.0}
    for case in benchmark_cases():
        result = optimize_messages([ChatMessage(role="user", content=case["prompt"])], compression_level, "gemini", "dry-run")
        quality = result["quality_gate_status"]["message_results"][0]
        row = {
            "suite": case["suite"],
            "category": case["category"],
            "original_tokens": result["original_tokens"],
            "optimized_tokens": result["optimized_tokens"],
            "savings_percent": result["savings_percent"],
            "accepted": result["quality_gate_status"]["accepted"],
            "rejection_reason": result["traces"]["rejection_reason"],
            "backend_used": result["backend_used"],
            "protected_failures": 0 if quality["checks"].get("protected_regions_preserved", True) else 1,
            "grammar_failures": 0 if result["traces"].get("grammar_validity", True) else 1,
            "semantic_failures": 0 if result["traces"].get("semantic_similarity", 1) >= 0.62 else 1,
            "duplicate_graph_edges": len(result["duplicate_chunk_graph"]),
            "cost_saved": result["cost"]["total_estimated_savings"],
        }
        rows.append(row)
        totals["original_tokens"] += result["original_tokens"]
        totals["optimized_tokens"] += result["optimized_tokens"]
        totals["cost_saved"] += row["cost_saved"]
    totals["savings_percent"] = round((totals["original_tokens"] - totals["optimized_tokens"]) / max(1, totals["original_tokens"]) * 100, 2)
    totals["cost_saved"] = round(totals["cost_saved"], 6)
    suites = sorted({row["suite"] for row in rows})
    return {"suites": suites, "rows": rows, "total": totals}
