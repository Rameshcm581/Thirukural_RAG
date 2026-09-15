import { useEffect, useState } from 'react'
import KuralCard from './components/KuralCard.jsx'
import './styles/App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [scenario, setScenario] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [hasSearched, setHasSearched] = useState(false)

  // Wake the backend on page load -- free hosts put it to sleep when idle, and
  // the first request after that can take a minute while the model loads.
  useEffect(() => {
    fetch(`${API_BASE}/health`).catch(() => {})
  }, [])

  async function handleSubmit(e) {
    e.preventDefault()
    if (!scenario.trim()) return

    setLoading(true)
    setError(null)
    setHasSearched(true)

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${res.status})`)
      }

      const data = await res.json()
      setResults(data.results)
    } catch (err) {
      setError(err.message)
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="hero-eyebrow">Thirukural</p>
        <h1 className="hero-question">What's on your mind?</h1>
        <p className="hero-sub">
          Describe a situation, in your own words. A 2,000-year-old couplet
          may already know exactly what you're going through.
        </p>

        <form onSubmit={handleSubmit} className="search-form">
          <textarea
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            placeholder="My coworker keeps taking credit for my work and I want to snap at them in the meeting…"
            rows={3}
          />
          <button type="submit" disabled={loading || !scenario.trim()}>
            {loading ? 'Searching…' : 'Find wisdom'}
          </button>
        </form>
      </header>

      <main className="results-area">
        {error && (
          <div className="error-panel">
            <p className="error-title">Something went wrong</p>
            <p>{error}</p>
          </div>
        )}

        {hasSearched && !loading && !error && results?.length === 0 && (
          <div className="empty-panel">
            <p>No kural quite matched that. Try describing the feeling underneath the situation instead.</p>
          </div>
        )}

        {results && results.length > 0 && (
          <div className="results-list">
            {results.map((kural, i) => (
              <KuralCard
                key={kural.kural_number}
                kural={kural}
                label={i === 0 ? 'Closest to your situation' : 'Also speaks to this'}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
