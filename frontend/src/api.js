
async function handle(res) {
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const d = data.detail
    throw new Error(typeof d === 'string' ? d : d ? JSON.stringify(d) : `HTTP ${res.status}`)
  }
  return data
}
const post = (url, body) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(handle)
export const api = {
  health: () => fetch('/api/health').then(handle),
  models: () => fetch('/api/models').then(handle),
  predict: (body) => post('/api/predict', body),
  compare: (body) => post('/api/compare', body),
  aspects: (body) => post('/api/aspects', body),
  metrics: () => fetch('/api/metrics').then(handle),
}
export const figureUrl = (name) => `/api/figures/${encodeURIComponent(name)}`
export const f3 = (v) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(3))
export const LANG_OPTIONS = [
  ['auto', 'Auto-detect'], ['hi', 'Hindi'], ['ta', 'Tamil'], ['te', 'Telugu'],
  ['bn', 'Bengali'], ['mr', 'Marathi'], ['hinglish', 'Hinglish'],
]
