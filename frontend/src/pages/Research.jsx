import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { api, f3, figureUrl } from '../api'
export default function Research({ pending = [] }) {
  const [d, setD] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => { api.metrics().then(setD).catch((e) => setErr(e.message)) }, [])
  if (err) return <div className="alert">{err}</div>
  if (!d) return <p>Loading…</p>
  // Only trained models are shown; the zero-shot LLM is left out.
  const isLLM = (id) => String(id).startsWith('llm')
  const trained = d.leaderboard.filter((r) => !isLLM(r.model_id))
  const perLang = Object.entries(d.per_lang).filter(([m]) => !isLLM(m))
  const groups = (d.ablations || []).filter((g) => g.items.some((it) => !isLLM(it.spec)))
  const full = trained.filter((r) => r.scope === 'full')
  const chart = full.map((r) => ({ name: r.model_id, f1: Number(r.macro_f1?.toFixed(4)) }))
  const langs = [...new Set(perLang.flatMap(([, o]) => Object.keys(o)))].sort()
  return (
    <div>
      <section className="card">
        <h2>Leaderboard of trained models (940 held-out test reviews) · best: <code>{d.best_model}</code></h2>
        <ResponsiveContainer width="100%" height={Math.max(220, chart.length * 34)}>
          <BarChart data={chart} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" /><XAxis type="number" domain={[0, 1]} /><YAxis type="category" dataKey="name" width={170} /><Tooltip />

            <Bar dataKey="f1" fill="#4f46e5" />
          </BarChart>
        </ResponsiveContainer>
        <table>
          <thead><tr><th>Model</th><th>Scope</th><th>n</th><th>Acc</th><th>Macro-F1</th><th>95% CI</th><th>ECE</th><th>Params</th><th>ms/review</th></tr></thead>
          <tbody>{trained.map((r) => (
            <tr key={r.model_id}><td title={r.model_id}>{r.display}</td><td>{r.scope}</td><td>{r.n_test}</td><td>{f3(r.accuracy)}</td><td><b>{f3(r.macro_f1)}</b></td>
              <td>{r.ci_low != null ? `[${f3(r.ci_low)}, ${f3(r.ci_high)}]` : '—'}</td><td>{f3(r.ece)}</td>
              <td>{r.n_params ? Number(r.n_params).toLocaleString() : '—'}</td><td>{r.latency_ms ? r.latency_ms.toFixed(1) : '—'}</td></tr>
          ))}</tbody>
        </table>
      </section>
      {pending.length > 0 && (
        <section className="card">
          <h3>Training pending</h3>
          <ul>{pending.map((model) => <li key={model.id}>{model.display}</li>)}</ul>
        </section>
      )}
      <section className="card">
        <h3>Per-language macro-F1</h3>
        <table>
          <thead><tr><th>Model</th>{langs.map((l) => <th key={l}>{l}</th>)}</tr></thead>
          <tbody>{perLang.map(([m, o]) => (
            <tr key={m}><td>{m}</td>{langs.map((l) => <td key={l}>{f3(o[l])}</td>)}</tr>
          ))}</tbody>
        </table>
      </section>
      {(d.mcnemar || []).length > 0 && (
        <section className="card">
          <h3>McNemar significance tests</h3>
          <table><thead><tr><th>A</th><th>B</th><th>n</th><th>A✓B✗</th><th>A✗B✓</th><th>p-value</th><th>p&lt;0.05</th></tr></thead>
            <tbody>{(d.mcnemar || []).map((r, i) => (
              <tr key={i}><td>{r.model_a}</td><td>{r.model_b}</td><td>{r.n}</td><td>{r.a_right_b_wrong}</td><td>{r.a_wrong_b_right}</td><td>{Number(r.p_value).toExponential(2)}</td><td>{r['significant_at_0.05'] ? 'yes' : 'no'}</td></tr>
            ))}</tbody></table>
        </section>
      )}
      {groups.length > 0 && (
        <section className="card">
          <h3>Model comparisons</h3>
          {groups.map((g) => (
            <div key={g.group}><h4>{g.group}</h4>
              <table><tbody>{g.items.map((it) => <tr key={it.spec}><td>{it.label}</td><td><b>{f3(it.macro_f1)}</b></td></tr>)}</tbody></table>
            </div>
          ))}
        </section>
      )}
      <section className="card">
        <h3>Figures (also in backend/artifacts/figures)</h3>
        <div className="figs">{(d.figures || []).map((f) => (
          <figure key={f}><img src={figureUrl(f)} alt={f} /><figcaption>{f}</figcaption></figure>
        ))}</div>
      </section>
    </div>
  )
}
