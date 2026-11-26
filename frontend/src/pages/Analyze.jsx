import { useState } from 'react'
import { api, f3 } from '../api'
const SAMPLES = [

  { label: 'Hindi', text: 'यह फ़ोन बहुत अच्छा है, बैटरी भी लंबी चलती है।' },
  { label: 'Tamil', text: 'இந்த உணவகத்தின் சாப்பாடு மிகவும் சுவையாக இருந்தது.' },
  { label: 'Telugu', text: 'ఈ సినిమా చాలా బోరింగ్‌గా ఉంది.' },
  { label: 'Bengali', text: 'বইটি অসাধারণ, পড়তে খুব ভালো লাগলো।' },
  { label: 'Marathi', text: 'हा फोन खूप छान आहे, मला आवडला.' },
  { label: 'Hinglish', text: 'Product bilkul bekar hai, paisa barbaad ho gaya.' },
]
const tokenColor = (score, label) => {
  const supports = score > 0
  const positive = (label === 'positive') === supports
  const a = Math.min(0.85, Math.abs(score) * 4)
  return positive ? `rgba(22,163,74,${a})` : `rgba(220,38,38,${a})`
}
export default function Analyze() {
  const [text, setText] = useState(SAMPLES[0].text)
  const [res, setRes] = useState(null)
  const [cmp, setCmp] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const run = async () => {
    setLoading(true); setErr(''); setCmp(null)
    try { setRes(await api.predict({ text, explain: true })) }
    catch (e) { setErr(e.message) } finally { setLoading(false) }
  }
  const compare = async () => {
    setLoading(true); setErr('')
    try { setCmp(await api.compare({ text })) }
    catch (e) { setErr(e.message) } finally { setLoading(false) }
  }
  return (
    <div className="grid2">
      <section className="card">
        <h2>Single review</h2>
        <div className="chips">
          {SAMPLES.map((s) => <button key={s.label} className="chip" onClick={() => setText(s.text)}>{s.label}</button>)}
        </div>
        <textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste a review in any supported language…" />
        <p className="muted small">Language is detected automatically and the best multilingual model is used.</p>
        <div className="row">
          <button className="primary" disabled={loading || !text.trim()} onClick={run}>{loading ? 'Working…' : 'Analyze sentiment'}</button>
          <button disabled={loading || !text.trim()} onClick={compare}>Compare all models</button>
        </div>
        {err && <div className="alert">{err}</div>}
      </section>
      <section className="card">
        <h2>Result</h2>
        {!res && <p className="muted">Run an analysis to see the prediction.</p>}
        {res && (
          <>
            <div className={`badge ${res.label}`}>{res.label.toUpperCase()}</div>
            <p>Confidence <b>{(res.confidence * 100).toFixed(1)}%</b> · language <b>{res.lang_name}</b> · model <code>{res.model_id}</code> · {res.latency_ms.toFixed(1)} ms</p>
            <div className="bar"><div className="bar-pos" style={{ width: `${res.probs.positive * 100}%` }} /></div>
            <div className="row small"><span>negative {f3(res.probs.negative)}</span><span>positive {f3(res.probs.positive)}</span></div>
            {res.explanation && res.explanation.length > 0 && (
              <>
                <h3>Why? (word occlusion importance)</h3>
                <p className="tokens">
                  {res.explanation.map((t, i) => (
                    <span key={i} title={t.score.toFixed(3)} style={{ background: tokenColor(t.score, res.label) }}>{t.token} </span>
                  ))}
                </p>
                <p className="muted small">Green pushes towards positive, red towards negative. Darker means stronger.</p>
              </>
            )}
          </>
        )}
        {cmp && (
          <>
            <h3>All models on this review ({cmp.lang})</h3>
            <table>
              <thead><tr><th>Model</th><th>Trained on</th><th>Label</th><th>Conf.</th><th>ms</th><th>Test F1</th></tr></thead>
              <tbody>
                {cmp.results.map((r) => (
                  <tr key={r.model_id}>
                    <td>{r.display}</td>
                    <td>{Array.isArray(r.trained_on) ? r.trained_on.join(',') : r.trained_on || ''}</td>
                    <td>{r.error ? <span className="err">{r.error}</span> : <span className={`tag ${r.label}`}>{r.label}</span>}</td>
                    <td>{f3(r.confidence)}</td><td>{r.latency_ms ? r.latency_ms.toFixed(1) : '—'}</td><td>{f3(r.test_macro_f1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>
    </div>
  )
}
