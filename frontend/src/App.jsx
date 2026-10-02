import { useEffect, useState } from 'react'
import KuralCard from './components/KuralCard.jsx'
import ChatView from './components/ChatView.jsx'
import './styles/App.css'

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:8000').replace(/\/+$/, '')

export default function App() {
  const [tab, setTab] = useState('search') // 'search' | 'chat'
  const [scenario, setScenario] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [hasSearched, setHasSearched] = useState(false)

  // Wake the backend on page load -- free hosts put it to sleep when idle
  useEffect(() => {
    fetch(`${API_BASE}/health`).catch(() => {})
  }, [])

  async function performSearch(text) {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    setLoading(true)
    setError(null)
    setResults(null)
    setHasSearched(true)

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: trimmed }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        let message = `Request failed (${res.status})`
        if (typeof body.detail === 'string') {
          message = body.detail
        } else if (Array.isArray(body.detail)) {
          message = body.detail.map((d) => d.msg || JSON.stringify(d)).join(', ')
        }
        throw new Error(message)
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

  function handleSubmit(e) {
    e.preventDefault()
    performSearch(scenario)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      performSearch(scenario)
    }
  }

  return (
    <div className="page">
      {/* Navigation Switcher */}
      <nav className="top-nav" aria-label="Mode selector">
        <button
          type="button"
          className={`nav-tab ${tab === 'search' ? 'active' : ''}`}
          onClick={() => setTab('search')}
        >
          🔍 Discover Wisdom
        </button>
        <button
          type="button"
          className={`nav-tab ${tab === 'chat' ? 'active' : ''}`}
          onClick={() => setTab('chat')}
        >
          💬 Counsel Dialogue
        </button>
      </nav>

      {tab === 'search' ? (
        <>
          <header className="hero">
            <p className="hero-eyebrow">Thirukural</p>
            <h1 className="hero-question">What's on your mind?</h1>
            <p className="hero-sub">
              Describe a situation in your own words. A 2,000-year-old couplet
              may already know exactly what you're going through.
            </p>

            <form onSubmit={handleSubmit} className="search-form">
              <textarea
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                onKeyDown={handleKeyDown}
                aria-label="Describe your modern life situation"
                placeholder="My coworker keeps taking credit for my work and I want to snap at them in the meeting…"
                rows={3}
              />
              <div className="form-actions">
                <button type="submit" disabled={loading || !scenario.trim()}>
                  {loading ? 'Searching…' : 'Find wisdom'}
                </button>
                <span className="submit-hint">Press Enter to search, Shift+Enter for new line</span>
              </div>
            </form>
          </header>

          <main className="results-area">
            {loading && (
              <div className="loading-panel">
                <div className="loading-spinner" />
                <p>Consulting the ancient wisdom for your situation…</p>
              </div>
            )}

            {error && (
              <div className="error-panel" role="alert">
                <p className="error-title">Something went wrong</p>
                <p>{error}</p>
              </div>
            )}

            {hasSearched && !loading && !error && results?.length === 0 && (
              <div className="empty-panel">
                <p>No kural quite matched that. Try describing the feeling underneath the situation instead.</p>
              </div>
            )}

            {!loading && results && results.length > 0 && (
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
        </>
      ) : (
        <main className="chat-main-area">
          <header className="hero hero-chat">
            <p className="hero-eyebrow">Thirukural Counseling</p>
            <h1 className="hero-question">Explore with Counsel</h1>
            <p className="hero-sub">
              Have a multi-turn conversation. Every exchange is stored in PostgreSQL and grounded in classical couplets.
            </p>
          </header>
          <ChatView apiBase={API_BASE} />
        </main>
      )}
    </div>
  )
}
