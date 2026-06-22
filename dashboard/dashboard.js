const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`${path} failed: ${response.status}`);
  return response.json();
}

function rows(tableId, html) {
  document.querySelector(`#${tableId} tbody`).innerHTML = html;
}

function showError(error) {
  const box = $("error");
  box.style.display = "block";
  box.textContent = error.message || String(error);
}

function money(value) {
  const number = Number(value || 0);
  if (number === 0) return "$0.000000";
  return `$${number.toFixed(6)}`;
}

function providerUsageText(usage) {
  if (!usage) return "none";
  const prompt = usage.prompt_tokens ?? "?";
  const output = usage.completion_tokens ?? "?";
  return `${prompt} in / ${output} out`;
}

async function refresh() {
  try {
    $("server-status").textContent = (await api("/health")).status;
    const data = await api("/analytics");
    $("total-requests").textContent = data.total_requests;
    $("original-tokens").textContent = data.original_tokens;
    $("optimized-tokens").textContent = data.optimized_tokens;
    $("tokens-saved").textContent = data.tokens_saved;
    $("savings-percent").textContent = `${data.savings_percent}%`;
    $("cost-saved").textContent = `$${data.estimated_cost_saved}`;
    $("cache-hit-rate").textContent = `${data.cache_hit_rate}%`;
    $("readiness-score").textContent = data.readiness_score;
    $("quality-failures").textContent = data.quality_failures;
    $("protected-failures").textContent = data.protected_failures;
    const traces = await api("/traces");
    rows("traces-table", traces.items.map(t => `<tr><td>${t.timestamp}</td><td>${t.request_id}</td><td>${t.savings_percent}%</td><td>${t.accepted}</td><td>${(t.backend_used || []).join(", ")}</td></tr>`).join(""));
  } catch (error) { showError(error); }
}

$("optimize").addEventListener("click", async () => {
  try {
    const payload = {
      messages: [
        { role: "system", content: $("system-prompt").value },
        { role: "user", content: $("user-prompt").value },
      ],
      compression_level: $("compression").value,
      provider: $("provider").value,
      mode: $("mode").value,
      model: $("model").value || null,
    };
    const live = payload.mode === "live" && payload.provider !== "dry-run";
    const raw = await api(live ? "/v1/chat/completions" : "/optimize", { method: "POST", body: JSON.stringify(payload) });
    const data = live ? raw.middleware : raw;
    const providerUsage = raw.usage || data.traces.provider_usage || null;
    const answer = live ? ((raw.choices || [])[0]?.message?.content || "") : "";
    $("original-cost").textContent = money(data.cost.estimated_original_total_with_output ?? data.cost.estimated_cost_before);
    $("optimized-cost").textContent = money(data.cost.estimated_optimized_total_with_output ?? data.cost.estimated_cost_after);
    $("cost-saved").textContent = money(data.cost.total_estimated_savings ?? data.cost.prompt_compression_savings);
    $("provider-usage").textContent = providerUsageText(providerUsage);
    $("optimize-summary").textContent = JSON.stringify({
      original_tokens: data.original_tokens,
      optimized_tokens: data.optimized_tokens,
      tokens_saved: Math.max(0, data.original_tokens - data.optimized_tokens),
      savings_percent: data.savings_percent,
      live_provider: raw.provider || data.traces.live_provider || payload.provider,
      model: raw.model || payload.model || data.route_decision?.selected_model,
      provider_usage: providerUsage,
      original_cost: data.cost.estimated_original_total_with_output ?? data.cost.estimated_cost_before,
      optimized_cost: data.cost.estimated_optimized_total_with_output ?? data.cost.estimated_cost_after,
      estimated_cost_saved: data.cost.total_estimated_savings ?? data.cost.prompt_compression_savings,
      pricing_basis: data.cost.pricing_basis,
      provider_price_key: data.cost.provider_price_key,
      backend_used: data.backend_used,
      prompt_analysis: data.traces.prompt_analysis,
      strategy_decision: data.traces.strategy_decision,
      semantic_optimizer_active: data.backend_used.includes("semantic_optimizer_backend"),
      llm_lingua_active: data.backend_used.includes("llm_lingua_backend"),
      enterprise_optimizer_active: data.backend_used.includes("enterprise_cost_optimizer"),
      enterprise_optimizer_traces: data.removed_or_changed_text.filter(item => item.backend === "enterprise_cost_optimizer"),
      llm_lingua_traces: data.removed_or_changed_text.filter(item => item.backend === "llm_lingua_backend"),
      semantic_optimizer_traces: data.removed_or_changed_text.filter(item => item.backend === "semantic_optimizer_backend"),
      protected_status: data.protected_region_status.status,
      quality_gate: data.quality_gate_status.accepted,
      grammar_status: data.grammar_status,
      semantic_status: data.semantic_status,
      removed_or_changed_text: data.removed_or_changed_text,
      safety_checks: data.quality_gate_status.message_results,
    }, null, 2);
    $("optimized-prompt").textContent = data.optimized_messages.map(m => `${m.role}: ${m.content}`).join("\n\n");
    $("provider-answer").textContent = live ? `Live provider answer:\n${answer}` : "Structural validation only. Switch Mode to live and Provider to Gemini Flash for a real Gemini call.";
    $("cache-output").textContent = JSON.stringify(data.traces.semantic_cache || { exact_hit: false, semantic_hit: false, rejection_reason: "not checked during optimize-only" }, null, 2);
    $("cost-output").textContent = JSON.stringify({
      cost: data.cost,
      route_decision: data.route_decision,
      provider_usage: providerUsage,
      provider: raw.provider || payload.provider,
      model: raw.model || payload.model || data.route_decision?.selected_model,
      mode: live ? "Live provider validation." : "Structural validation only.",
    }, null, 2);
    rows("duplicate-graph", data.duplicate_chunk_graph.map(g => `<tr><td>${g.canonical_chunk}</td><td>${g.duplicate_chunk}</td><td>${g.similarity}</td><td>${g.estimated_saved_tokens}</td><td>${g.contradiction_gate}</td><td>${g.decision}</td></tr>`).join(""));
    const candidates = data.removed_or_changed_text.flatMap(item => {
      const output = [];
      if (item.candidate_type) output.push(item);
      if (Array.isArray(item.surface_changes)) {
        item.surface_changes.forEach(change => output.push({
          accepted: true,
          candidate_type: change.type || "surface_change",
          span_text: change.removed_sentence || "",
          retained_span: change.kept_sentence || "",
          score: change.tokens_saved || "",
          reason: change.reason || item.reason || "",
          risk_flags: change.risk_flags || [],
        }));
      }
      if (Array.isArray(item.semantic_removed)) {
        item.semantic_removed.forEach(change => output.push({
          accepted: true,
          candidate_type: "semantic_removed",
          span_text: change.removed_sentence || "",
          retained_span: change.kept_sentence || "",
          score: change.combined_score || change.semantic_similarity || "",
          reason: change.reason || "",
          risk_flags: [],
        }));
      }
      if (Array.isArray(item.rejected_removals)) {
        item.rejected_removals.forEach(change => output.push({
          accepted: false,
          candidate_type: "rejected_removal",
          span_text: change.removed_sentence || "",
          retained_span: change.kept_sentence || "",
          score: change.combined_score || change.semantic_similarity || "",
          rejected_reason: change.rejected_reason || change.reason || "",
          risk_flags: [],
        }));
      }
      return output;
    });
    rows("candidate-table", candidates.map(c => `<tr><td>${c.accepted ? "accepted" : "rejected"}</td><td>${c.candidate_type}</td><td>${c.span_text || ""}</td><td>${c.retained_span || ""}</td><td>${c.score}</td><td>${c.rejected_reason || c.reason || ""}</td><td>${(c.risk_flags || []).join(", ")}</td></tr>`).join(""));
    const concepts = data.removed_or_changed_text.filter(item => item.backend === "concept_aggregation_backend");
    rows("concept-table", concepts.map(c => `<tr><td>${c.accepted ? "accepted" : "rejected"}</td><td>${c.cluster_theme || ""}</td><td>${(c.extracted_concepts || []).join(", ")}</td><td>${c.generated_aggregate_sentence || c.retained_span || ""}</td><td>${c.semantic_similarity || ""}</td><td>${c.tokens_saved || 0}</td><td>${c.rejected_reason || c.reason || ""}</td></tr>`).join(""));
    const information = data.removed_or_changed_text.filter(item => item.backend === "information_representation_backend");
    rows("information-table", information.map(c => `<tr><td>${c.accepted ? "accepted" : "rejected"}</td><td>${c.cluster_theme || ""}</td><td>${(c.information_units || []).length}</td><td>${c.generated_minimum_representation || c.retained_span || ""}</td><td>${c.information_recall || ""}</td><td>${c.tokens_saved || 0}</td><td>${c.rejected_reason || c.reason || ""}</td></tr>`).join(""));
    refresh();
  } catch (error) { showError(error); }
});

$("run-benchmark").addEventListener("click", async () => {
  try {
    const data = await api("/benchmark", { method: "POST", body: JSON.stringify({ compression_level: $("compression").value }) });
    rows("benchmark-table", data.rows.map(r => `<tr><td>${r.suite}</td><td>${r.category}</td><td>${r.original_tokens}</td><td>${r.optimized_tokens}</td><td>${r.savings_percent}%</td><td>${r.accepted}</td><td>$${r.cost_saved}</td><td>${r.backend_used.join(", ")}</td></tr>`).join(""));
  } catch (error) { showError(error); }
});

$("run-quality").addEventListener("click", async () => {
  try {
    const data = await api("/evaluate-quality", { method: "POST", body: JSON.stringify({ compression_level: $("compression").value, mode: $("mode").value }) });
    rows("quality-table", data.results.map(r => `<tr><td>${r.mode}</td><td>${r.provider}</td><td>${r.answer_similarity_heuristic}</td><td>${r.numeric_answer_preservation}</td><td>${r.latency_ms}ms</td></tr>`).join(""));
  } catch (error) { showError(error); }
});

$("run-robustness").addEventListener("click", async () => {
  try {
    const data = await api("/robustness-test", { method: "POST", body: JSON.stringify({ compression_level: $("compression").value }) });
    rows("robustness-table", data.rows.map(r => `<tr><td>${r.case}</td><td>${r.accepted}</td><td>${r.savings_percent}%</td><td>${r.protected_regions_preserved}</td><td>${r.failure_reason || ""}</td></tr>`).join(""));
  } catch (error) { showError(error); }
});

$("run-pilot").addEventListener("click", async () => {
  try {
    const data = await api("/company-pilot-sim", { method: "POST", body: JSON.stringify({ compression_level: $("compression").value, mode: $("mode").value }) });
    $("pilot-output").textContent = JSON.stringify(data.summary, null, 2);
    refresh();
  } catch (error) { showError(error); }
});

$("run-report").addEventListener("click", async () => {
  try {
    const data = await api("/demo-report", { method: "POST", body: JSON.stringify({ compression_level: $("compression").value, mode: $("mode").value }) });
    $("pilot-output").textContent = `${JSON.stringify(data.summary, null, 2)}\n\nReport: ${data.report_url}`;
  } catch (error) { showError(error); }
});

$("refresh").addEventListener("click", refresh);
refresh();
