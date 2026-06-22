import { Activity, BarChart3, Database, Gauge, Play, RefreshCw, ShieldCheck, Zap } from "lucide-react";
import React from "react";
import { useEffect, useMemo, useState } from "react";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const initialSystemPrompt = "You are a careful support assistant. Do not promise approval. Do not promise approval.";
const initialUserPrompt = `Retrieved context chunk 1:
Refunds over $500 require Finance approval. Contact finance@example.com.
Retrieved context chunk 2:
Refunds over $500 require Finance approval. Contact finance@example.com.
Please please kindly kindly answer for ORD-900184.`;

function money(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number === 0) return "$0.000000";
  return `$${number.toFixed(6)}`;
}

function usageText(usage) {
  if (!usage) return "none";
  return `${usage.prompt_tokens ?? "?"} in / ${usage.completion_tokens ?? "?"} out`;
}

function JsonBlock({ value }) {
  return <pre>{typeof value === "string" ? value : JSON.stringify(value, null, 2)}</pre>;
}

function StatCard({ icon: Icon, label, value }) {
  return (
    <article className="stat-card">
      <div className="stat-icon"><Icon size={17} /></div>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function Table({ columns, rows, empty = "No rows yet." }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr><td colSpan={columns.length}>{empty}</td></tr>
          ) : rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => <td key={column.key}>{column.render ? column.render(row) : row[column.key]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState("checking");
  const [analytics, setAnalytics] = useState({});
  const [traces, setTraces] = useState([]);
  const [systemPrompt, setSystemPrompt] = useState(initialSystemPrompt);
  const [userPrompt, setUserPrompt] = useState(initialUserPrompt);
  const [compression, setCompression] = useState("safe");
  const [provider, setProvider] = useState("gemini");
  const [mode, setMode] = useState("dry-run");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [benchmark, setBenchmark] = useState([]);
  const [quality, setQuality] = useState([]);
  const [robustness, setRobustness] = useState([]);
  const [pilot, setPilot] = useState(null);

  const liveResult = result?.raw;
  const middleware = result?.middleware;
  const providerUsage = liveResult?.usage || middleware?.traces?.provider_usage || null;
  const answer = liveResult?.choices?.[0]?.message?.content || "";
  const optimizedPrompt = middleware?.optimized_messages?.map((m) => `${m.role}: ${m.content}`).join("\n\n") || "";
  const removed = middleware?.removed_or_changed_text || [];

  const cards = useMemo(() => [
    { icon: Activity, label: "Server", value: health },
    { icon: BarChart3, label: "Total requests", value: analytics.total_requests ?? 0 },
    { icon: Database, label: "Original tokens", value: analytics.original_tokens ?? 0 },
    { icon: Database, label: "Optimized tokens", value: analytics.optimized_tokens ?? 0 },
    { icon: Zap, label: "Tokens saved", value: analytics.tokens_saved ?? 0 },
    { icon: Gauge, label: "Savings", value: `${analytics.savings_percent ?? 0}%` },
    { icon: BarChart3, label: "Cost saved", value: money(analytics.estimated_cost_saved) },
    { icon: Database, label: "Original cost", value: money(middleware?.cost?.estimated_original_total_with_output ?? middleware?.cost?.estimated_cost_before) },
    { icon: Database, label: "Optimized cost", value: money(middleware?.cost?.estimated_optimized_total_with_output ?? middleware?.cost?.estimated_cost_after) },
    { icon: Activity, label: "Provider usage", value: usageText(providerUsage) },
    { icon: ShieldCheck, label: "Readiness", value: analytics.readiness_score ?? 0 },
    { icon: ShieldCheck, label: "Protected failures", value: analytics.protected_failures ?? 0 },
  ], [analytics, health, middleware, providerUsage]);

  async function api(path, options = {}) {
    const response = await fetch(`${API_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) throw new Error(`${path} failed: ${response.status}`);
    return response.json();
  }

  async function refresh() {
    try {
      setError("");
      const [healthData, analyticsData, tracesData] = await Promise.all([
        api("/health"),
        api("/analytics"),
        api("/traces"),
      ]);
      setHealth(healthData.status);
      setAnalytics(analyticsData);
      setTraces(tracesData.items || []);
    } catch (err) {
      setError(err.message || String(err));
      setHealth("offline");
    }
  }

  async function optimize() {
    setLoading(true);
    setError("");
    try {
      const payload = {
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        compression_level: compression,
        provider,
        mode,
        model: model || null,
        temperature: 0,
      };
      const live = mode === "live" && provider !== "dry-run";
      const raw = await api(live ? "/v1/chat/completions" : "/optimize", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setResult({ raw, middleware: live ? raw.middleware : raw, live });
      await refresh();
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  async function runBenchmark() {
    const data = await api("/benchmark", { method: "POST", body: JSON.stringify({ compression_level: compression }) });
    setBenchmark(data.rows || []);
  }

  async function runQuality() {
    const data = await api("/evaluate-quality", { method: "POST", body: JSON.stringify({ compression_level: compression, mode }) });
    setQuality(data.results || []);
  }

  async function runRobustness() {
    const data = await api("/robustness-test", { method: "POST", body: JSON.stringify({ compression_level: compression }) });
    setRobustness(data.rows || []);
  }

  async function runPilot() {
    const data = await api("/company-pilot-sim", { method: "POST", body: JSON.stringify({ compression_level: compression, mode }) });
    setPilot(data.summary || data);
    await refresh();
  }

  useEffect(() => {
    refresh();
  }, []);

  const candidates = removed.flatMap((item) => {
    const output = [];
    if (item.candidate_type) output.push(item);
    if (Array.isArray(item.surface_changes)) {
      item.surface_changes.forEach((change) => output.push({
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
      item.semantic_removed.forEach((change) => output.push({
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
      item.rejected_removals.forEach((change) => output.push({
        accepted: false,
        candidate_type: "rejected_removal",
        span_text: change.removed_sentence || "",
        retained_span: change.kept_sentence || "",
        score: change.combined_score || change.semantic_similarity || "",
        reason: change.rejected_reason || change.reason || "",
        risk_flags: [],
      }));
    }
    return output;
  });

  return (
    <div>
      <header className="topbar">
        <div>
          <h1>AI Cost Optimization Middleware</h1>
          <p>Vercel dashboard for a Render-hosted FastAPI optimizer backend.</p>
          <p className="api-url">API: {API_URL}</p>
        </div>
        <button onClick={refresh} title="Refresh"><RefreshCw size={16} /> Refresh</button>
      </header>

      <main>
        <section className={mode === "live" ? "banner live" : "banner"}>
          {mode === "live" ? "Live provider validation. Gemini/OpenAI may be called through the Render backend." : "Dry-run validates middleware mechanics only. It does not prove real model output quality."}
        </section>

        <section className="cards">
          {cards.map((card) => <StatCard key={card.label} {...card} />)}
        </section>

        <section className="panel optimizer">
          <div className="panel-heading">
            <h2>Prompt Optimizer</h2>
            <button onClick={optimize} disabled={loading}><Play size={16} /> {loading ? "Running" : "Optimize"}</button>
          </div>
          <div className="form-grid">
            <label>System prompt<textarea value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} /></label>
            <label>User prompt<textarea value={userPrompt} onChange={(e) => setUserPrompt(e.target.value)} /></label>
            <label>Compression<select value={compression} onChange={(e) => setCompression(e.target.value)}><option>safe</option><option>balanced</option><option>aggressive</option></select></label>
            <label>Provider<select value={provider} onChange={(e) => setProvider(e.target.value)}><option value="dry-run">Dry-run</option><option value="gemini">Gemini Flash</option><option value="openai">OpenAI</option></select></label>
            <label>Mode<select value={mode} onChange={(e) => setMode(e.target.value)}><option value="dry-run">dry-run</option><option value="live">live</option></select></label>
            <label>Model override<input value={model} onChange={(e) => setModel(e.target.value)} placeholder="optional model" /></label>
          </div>
          <div className="split">
            <JsonBlock value={middleware ? {
              original_tokens: middleware.original_tokens,
              optimized_tokens: middleware.optimized_tokens,
              tokens_saved: Math.max(0, middleware.original_tokens - middleware.optimized_tokens),
              savings_percent: middleware.savings_percent,
              live_provider: liveResult?.provider || middleware.traces?.live_provider || provider,
              model: liveResult?.model || model || middleware.route_decision?.selected_model,
              provider_usage: providerUsage,
              cost: middleware.cost,
              backend_used: middleware.backend_used,
              protected_status: middleware.protected_region_status?.status,
              quality_gate: middleware.quality_gate_status?.accepted,
            } : "Run an optimization to see live metrics."} />
            <JsonBlock value={optimizedPrompt || "Optimized prompt will appear here."} />
          </div>
          <JsonBlock value={result?.live ? `Live provider answer:\n${answer}` : "Structural validation only. Switch Mode to live and Provider to Gemini Flash for a real Gemini call."} />
        </section>

        {error && <section className="panel error">{error}</section>}

        <section className="panel">
          <h2>Provider And Cost</h2>
          <JsonBlock value={middleware ? { cost: middleware.cost, route_decision: middleware.route_decision, provider_usage: providerUsage, provider: liveResult?.provider || provider, model: liveResult?.model || model } : {}} />
        </section>

        <section className="panel">
          <h2>Candidate Inspection</h2>
          <Table columns={[
            { key: "accepted", label: "Status", render: (r) => r.accepted ? "accepted" : "rejected" },
            { key: "candidate_type", label: "Type" },
            { key: "span_text", label: "Span" },
            { key: "retained_span", label: "Retained" },
            { key: "score", label: "Score" },
            { key: "reason", label: "Reason" },
            { key: "risk_flags", label: "Risk flags", render: (r) => (r.risk_flags || []).join(", ") },
          ]} rows={candidates} />
        </section>

        <section className="panel">
          <h2>Duplicate Chunk Graph</h2>
          <Table columns={[
            { key: "canonical_chunk", label: "Canonical" },
            { key: "duplicate_chunk", label: "Duplicate" },
            { key: "similarity", label: "Similarity" },
            { key: "estimated_saved_tokens", label: "Saved" },
            { key: "contradiction_gate", label: "Gate" },
            { key: "decision", label: "Decision" },
          ]} rows={middleware?.duplicate_chunk_graph || []} />
        </section>

        <section className="panel">
          <div className="panel-heading">
            <h2>Benchmark</h2>
            <button onClick={runBenchmark}>Run benchmark</button>
          </div>
          <Table columns={[
            { key: "suite", label: "Suite" },
            { key: "category", label: "Category" },
            { key: "original_tokens", label: "Original" },
            { key: "optimized_tokens", label: "Optimized" },
            { key: "savings_percent", label: "Savings", render: (r) => `${r.savings_percent}%` },
            { key: "accepted", label: "Accepted", render: (r) => String(r.accepted) },
            { key: "cost_saved", label: "Cost saved", render: (r) => money(r.cost_saved) },
          ]} rows={benchmark} />
        </section>

        <section className="panel">
          <div className="panel-heading">
            <h2>Quality Evaluation</h2>
            <button onClick={runQuality}>Run quality</button>
          </div>
          <Table columns={[
            { key: "mode", label: "Mode" },
            { key: "provider", label: "Provider" },
            { key: "answer_similarity_heuristic", label: "Similarity" },
            { key: "numeric_answer_preservation", label: "Numbers" },
            { key: "latency_ms", label: "Latency", render: (r) => `${r.latency_ms}ms` },
          ]} rows={quality} />
        </section>

        <section className="panel">
          <div className="panel-heading">
            <h2>Robustness</h2>
            <button onClick={runRobustness}>Run robustness</button>
          </div>
          <Table columns={[
            { key: "case", label: "Case" },
            { key: "accepted", label: "Accepted", render: (r) => String(r.accepted) },
            { key: "savings_percent", label: "Savings", render: (r) => `${r.savings_percent}%` },
            { key: "protected_regions_preserved", label: "Protected" },
            { key: "failure_reason", label: "Reason" },
          ]} rows={robustness} />
        </section>

        <section className="panel">
          <div className="panel-heading">
            <h2>Company Pilot</h2>
            <button onClick={runPilot}>Run pilot</button>
          </div>
          <JsonBlock value={pilot || "Pilot output will appear here."} />
        </section>

        <section className="panel">
          <h2>Analytics And Traces</h2>
          <Table columns={[
            { key: "timestamp", label: "Time" },
            { key: "request_id", label: "Request" },
            { key: "savings_percent", label: "Savings", render: (r) => `${r.savings_percent}%` },
            { key: "accepted", label: "Accepted", render: (r) => String(r.accepted) },
            { key: "backend_used", label: "Backends", render: (r) => (r.backend_used || []).join(", ") },
          ]} rows={traces} />
        </section>
      </main>
    </div>
  );
}
