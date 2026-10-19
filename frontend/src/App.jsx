import { useState } from "react";

const API = import.meta.env.VITE_API_URL ?? "";
const DOCS = "https://fastapi.tiangolo.com/";
const pageUrl = (page) => DOCS + page.replace(/index\.md$/, "").replace(/\.md$/, "/");

export default function App() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function ask(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setHistory((h) => [{ question, ...data }, ...h]);
      setQuestion("");
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <h1>DocPilot</h1>
        <p>Hybrid-retrieval RAG over the FastAPI documentation · BM25 + dense + cross-encoder rerank · LangGraph agent</p>
      </header>
      <form onSubmit={ask}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. How do I add CORS middleware?" />
        <button disabled={loading}>{loading ? "Thinking..." : "Ask"}</button>
      </form>
      {error && <p className="error">{error}</p>}
      {history.map((h, i) => (
        <article key={i}>
          <h3>{h.question}</h3>
          <p className="answer">{h.answer}</p>
          {h.citations.length > 0 && (
            <ol className="cites">
              {h.citations.map((c) => (
                <li key={c.chunk_id}>
                  <a href={pageUrl(c.page)} target="_blank" rel="noreferrer">{c.page}</a> · {c.section}
                </li>
              ))}
            </ol>
          )}
          <p className="meta">
            route: {h.route} · {h.cached ? "cache hit" : "cache miss"} · {Math.round(h.latency_ms)} ms
            {h.rewrites > 0 && ` · query rewritten ${h.rewrites}x`}
          </p>
        </article>
      ))}
    </main>
  );
}
