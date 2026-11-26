import { useState } from 'react'
import { api } from '../api'
const SAMPLES = [
  { label: 'Hindi', text: 'फ़ोन का कैमरा शानदार है लेकिन बैटरी जल्दी खत्म हो जाती है और कीमत थोड़ी ज़्यादा है।' },
  { label: 'Hinglish', text: 'Delivery fast thi aur quality bhi achhi hai, but packaging was damaged.' },
  { label: 'Tamil', text: 'இந்த ஹெட்போன் ஒலி தரம் சிறப்பாக இருக்கிறது ஆனால் விலை அதிகம்' },
  { label: 'Telugu', text: 'ఈ ఫోన్ కెమెరా చాలా బాగుంది కానీ బ్యాటరీ త్వరగా అయిపోతుంది' },
  { label: 'Bengali', text: 'বইয়ের গল্প অসাধারণ কিন্তু ছাপার মান খারাপ' },
  { label: 'Marathi', text: 'हा फोन खूप छान आहे, कॅमेरा अप्रतिम आहे आणि बॅटरी दिवसभर टिकते.' },
]
export default function Aspects() {
  const [text, setText] = useState(SAMPLES[0].text)
  const [res, setRes] = useState(null)
  const [loading, setLoading] = useState('')
  const [err, setErr] = useState('')
  const run = async (mode) => {
    setLoading(mode); setErr('')
    try { setRes(await api.aspects({ text, mode })) }
    catch (e) { setErr(e.message) } finally { setLoading('') }
  }
  const r = res?.result
  const hasEnglish = r?.aspects?.some((a) => a.evidence_en)
  return (
    <div className="grid2">
      <section className="card">
        <h2>Aspect-based sentiment</h2>
        <p className="muted">Breaks a review into aspects such as price, quality, battery or delivery, each with its own sentiment and the words that show it. Language is detected automatically.</p>
        <div className="chips">
          {SAMPLES.map((s) => <button key={s.label} className="chip" onClick={() => setText(s.text)}>{s.label}</button>)}
        </div>
        <textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="row">
          <button className="primary" disabled={!!loading || !text.trim()} onClick={() => run('local')}>{loading === 'local' ? 'Extracting…' : 'Extract aspects'}</button>
        </div>
        {err && <div className="alert">{err}</div>}
      </section>
      <section className="card">
        <h2>Aspects</h2>
        {!r && <p className="muted">No result yet.</p>}
        {res?.note && <div className="alert">{res.note}</div>}
        {r && (
          <>
            <p>Overall: <span className={`tag ${r.overall}`}>{r.overall}</span>
              <span className="muted"> · {res.method === 'ai' ? `AI (${res.provider})` : `offline, model ${r.model_id}`}</span>
            </p>
            {r.translation_en && <p className="muted">EN: {r.translation_en}</p>}
            {(r.aspects || []).length === 0
              ? <p className="muted">No known aspect words found in this review. Try mentioning an aspect such as price, quality, battery or delivery.</p>
              : (
                <table>
                  <thead><tr><th>Aspect</th><th>Sentiment</th><th>Evidence</th>{hasEnglish && <th>English</th>}</tr></thead>
                  <tbody>{r.aspects.map((a, i) => (
                    <tr key={i}><td><b>{a.aspect}</b></td><td><span className={`tag ${a.sentiment}`}>{a.sentiment}</span></td><td>{a.evidence}</td>{hasEnglish && <td className="muted">{a.evidence_en}</td>}</tr>
                  ))}</tbody>
                </table>
              )}
          </>
        )}
      </section>
    </div>
  )
}
