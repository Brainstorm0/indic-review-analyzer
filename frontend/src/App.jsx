import { useEffect, useState } from 'react'
import { api } from './api'
import Analyze from './pages/Analyze'
import Aspects from './pages/Aspects'
import Research from './pages/Research'
import About from './pages/About'

const TABS = [
  ['analyze', 'Analyze a review'],
  ['aspects', 'Aspect-based (ABSA)'],
  ['research', 'Model comparison'],
  ['about', 'About'],
]
export default function App() {
  const [tab, setTab] = useState('analyze')
  const [models, setModels] = useState([])
  const [pending, setPending] = useState([])
  const [health, setHealth] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    api.health().then(setHealth).catch((e) => setErr(e.message))
    api.models().then((d) => {
      setModels(d.models)
      setPending(d.pending ?? [])
    }).catch((e) => setErr(e.message))
  }, [])
  return (
    <div className="app">
      <header>
        <div>
          <h1>Indic Review &amp; Sentiment Analyzer</h1>
          <p className="sub">Hindi · Tamil · Telugu · Bengali · Marathi · Hinglish — LR / Naive Bayes (Multinomial, Bernoulli) / BiLSTM / MuRIL</p>
        </div>
        <span className={`pill ${health ? 'ok' : 'bad'}`}>
          {health ? `API online · ${health.n_trained ?? health.n_models} trained${health.n_pending ? ` · ${health.n_pending} training pending` : ''}` : 'API offline'}
        </span>
      </header>
      <nav>
        {TABS.map(([k, label]) => (
          <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{label}</button>
        ))}
      </nav>
      {err && <div className="alert">Backend not reachable: {err}. Start the API using the README.</div>}
      <main>
        {tab === 'analyze' && <Analyze />}
        {tab === 'aspects' && <Aspects />}
        {tab === 'research' && <Research pending={pending} />}
        {tab === 'about' && <About />}
      </main>
    </div>
  )
}
