# AI Cost Optimization Middleware

Production-style MVP for a deterministic middleware layer between a company application and LLM providers, with Gemini Flash as the default live target and OpenAI as an optional adapter. It accepts OpenAI-style chat requests, detects the enterprise prompt type, chooses a safe optimization strategy, protects risky prompt regions, compiles repeated RAG/support context, routes requests, estimates cost, caches repeated requests, logs traces, and provides a validation dashboard.

## Why Companies Need It

LLM costs often rise because applications repeatedly send overlapping RAG chunks, long system prompts, duplicated policy text, chat history, agent traces, and support or compliance boilerplate. This middleware reduces that waste while preserving code, JSON, math, dates, IDs, money, URLs, emails, and exact quoted text.

## Architecture

Company App -> AI Cost Optimization Middleware -> Gemini Flash or OpenAI

The optimizer is deterministic. It does not use another LLM to blindly rewrite prompts. Optional local transformer packages may be used only for semantic validation, embeddings, entity detection, and safety checks.

The product goal is not universal English summarization. The goal is to reduce LLM spend safely across enterprise prompt patterns: customer support RAG, repeated policies, repeated instructions, duplicated retrieved context, chat history, tool traces, and repeated system templates.

## Quickstart

```bash
cd ai-cost-optimization-middleware
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_API_KEY="your key"
uvicorn app.main:app --reload
```

Dashboard: <http://127.0.0.1:8000/dashboard>

## Dry-Run Vs Live

Dry-run validates middleware mechanics only. It does not prove real model output quality. Live mode uses Gemini Flash when `GEMINI_API_KEY` is configured, or OpenAI when `OPENAI_API_KEY` is configured and the OpenAI provider is selected. If a key is missing or a provider call fails, the API falls back to dry-run and labels that clearly.

## Gemini Setup

Set `GEMINI_API_KEY` in your shell or deployment environment. Do not commit API keys. The default model is `gemini-1.5-flash`; override with `GEMINI_MODEL`.

## OpenAI Setup

Set `OPENAI_API_KEY` in your environment. The default model is `gpt-4o-mini`; override with `OPENAI_MODEL` or the dashboard model override.

## Local Transformer Validation

Set `ENABLE_LOCAL_TRANSFORMERS=1` to attempt local `sentence-transformers/all-MiniLM-L6-v2` and spaCy loading when those packages/models are installed. If unavailable, traces explicitly report deterministic lexical/rule fallback. No external API is called for validation.

## LLMLingua2 Second Layer

The middleware includes a second-stage `llm_lingua_backend` after the cheap layer. It treats the cheap-layer output as the safety baseline, then uses the true `llmlingua` package with `use_llmlingua2=True` for learned prompt-token compression when enabled.

Default behavior is deterministic and local unless `ENABLE_LLM_LINGUA=1` is set. Adjacent low-information filler such as repeated `please`/`kindly` can be compacted without loading the LLMLingua2 model.

To enable true local LLMLingua2 compression, install requirements and set:

```bash
ENABLE_LLM_LINGUA=1
LLM_LINGUA_BACKEND=llmlingua2
LLM_LINGUA2_MODEL=microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
LLM_LINGUA2_DEVICE=cpu
```

Gemini remains the final downstream LLM provider, not the preferred compression engine. A legacy `LLM_LINGUA_BACKEND=gemini` mode exists for experimentation, but production flow should be `cheap_layer -> LLMLingua2 -> validator -> Gemini`.

The LLMLingua2 candidate is accepted only if it is shorter and preserves protected placeholders, protected facts, state signatures, event signatures, risk/negation markers, entities, and semantic similarity. If any check fails, the layer falls back to the cheap-layer output and records the rejection reason in traces.

## Enterprise Optimization Architecture

The middleware runs:

raw messages -> prompt type detector -> strategy planner -> protected placeholder extraction -> specialized optimizer -> safety/quality gate -> provider/cache/router -> analytics trace.

Prompt types include `customer_support`, `rag_context`, `legal_compliance`, `policy_instruction`, `chat_history`, `agent_tool_trace`, `product_docs`, `finance_report`, and `general_prompt`.

Strategies include exact cache, semantic cache, RAG dedupe, customer support policy dedupe, repeated instruction cleanup, conservative semantic compression, routing, and no-change when optimization risk or overhead is not worth it.

Every accepted and rejected action is traceable with prompt type, chosen strategy, score, tokens saved, risk flags, and reason.

## Cheap Layer

The cheap layer is the first safety-first compression pass for obvious redundancy. It handles exact sentence duplicates, repeated instruction text, safe repeated list frames, number-word equivalents such as `eighty-three percent` vs `83%`, and conservative semantic duplicate candidates only when protected fact keys match.

It is not final deletion authority for risky language. Its validation gate preserves protected facts, entity identities, state signatures, negation/risk markers, and surface quality. If it cannot prove safety, it fails closed and leaves the prompt for the next layer.

Run its focused stress harness:

```bash
python -m app.cheap_layer_stress --cases 500 --seed 19
```

The audit trace includes original tokens, compressed tokens, surface changes, semantic removals, rejected removals, validation gate results, missing protected facts, missing state signatures, and final action.

## Customer Support RAG Optimizer

For support prompts assembled from policies, handbooks, escalation guides, knowledge base articles, ticket fields, reminders, and tasks, the middleware:

- collapses repeated verification requirements into one canonical rule
- collapses repeated Finance-approval requirements into one canonical rule
- collapses repeated reminders such as "Do not promise approval"
- preserves ticket fields, IDs, emails, SKUs, dates, and refund amounts exactly
- preserves the final task
- rejects output if protected facts disappear or tokens do not decrease

This is where the middleware should show meaningful savings on real enterprise workloads, typically much more than on tiny toy prompts.

## API Examples

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/optimize -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Refunds over $500 require Finance approval. Refunds over $500 require Finance approval.\"}],\"compression_level\":\"safe\",\"provider\":\"gemini\",\"mode\":\"dry-run\"}"
```

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"gemini-1.5-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"Please please summarize this.\"}],\"mode\":\"live\"}"
```

## RAG Context Compiler

The RAG compiler detects chunk headers such as `Retrieved context chunk 1:`, `Source A:`, `Chunk 3:`, and `Policy excerpt:`. It dedupes exact and near-duplicate chunks, removes repeated sentences inside chunks, preserves unique facts, avoids contradictory merges, and uses compact references like `[dup: chunk 1]`.

## Semantic Cache

The middleware supports exact response caching today and exposes conservative semantic-cache decision traces. Semantic reuse is rejected for time-sensitive prompts and whenever protected facts such as IDs, numbers, dates, or money differ. The MVP does not broadly reuse semantic cache entries unless a safe embedding index is added.

## Cost And Routing

The cost model estimates provider input/output cost before and after compression, prompt-compression savings, cache savings, routing savings, and total estimated savings. Rule-based routing selects cheaper or stronger models based on prompt length, code/math, regulated domains, strict JSON, reasoning signals, or user override.

## Validation Tools

- `POST /benchmark` runs short, enterprise workload, enterprise RAG, safety, and adversarial suites.
- `POST /evaluate-quality` compares original and optimized prompts with deterministic structural metrics.
- `POST /robustness-test` runs adversarial protection cases.
- `POST /company-pilot-sim` runs benchmark, quality evaluation, robustness, analytics, and readiness scoring.
- `POST /demo-report` generates `reports/demo_report.html`.

## Production Readiness Score

The score considers savings, estimated cost savings, quality failures, grammar failures, semantic failures, protected-region failures, robustness failures, cache hit rate, rejection rate, latency, and live-provider validation. Dry-run cannot score above 85.

Labels:

- `90-100`: `pilot_ready`
- `75-89`: `staging_only`
- `50-74`: `prototype_only`
- below `50`: `unsafe`

## Rollout Plan

1. Dry-run: validate mechanics and protected-region behavior.
2. Shadow mode: compare optimized and original requests without serving optimized responses.
3. Staging: use real Gemini Flash calls and inspect traces.
4. Limited traffic: start with safe compression and low-risk workflows.
5. Production: expand only after live quality metrics and rollback plans are in place.

## Failure Modes

The optimizer rejects compression when tokens do not decrease, protected regions change, sensitive facts disappear, JSON/code/math may be altered, compression is too aggressive, or backend confidence is low.
The grammar gate also rejects obvious broken output such as orphan verb starts, dangling conjunctions/prepositions, broken punctuation, double spaces, and fragments produced by unsafe span removal.

## What It Does Not Guarantee

Dry-run mode does not prove real model output quality. Heuristic similarity is not a semantic judge. The MVP is conservative by design: it may miss savings rather than corrupt a prompt. KV-cache reuse and a persistent vector index are still roadmap items.

## Roadmap

- Broader prompt-type-specific optimizers
- Persistent semantic embedding index
- KV-cache optimization
- vLLM/Ollama integration
