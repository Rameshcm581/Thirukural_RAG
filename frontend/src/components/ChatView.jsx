import { useEffect, useState, useRef } from 'react'
import KuralCard from './KuralCard.jsx'

export default function ChatView({ apiBase }) {
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState(() => {
    return localStorage.getItem('thirukural_session_id') || `session-${Date.now()}`
  })
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedKurals, setExpandedKurals] = useState({})
  const messagesEndRef = useRef(null)

  useEffect(() => {
    localStorage.setItem('thirukural_session_id', sessionId)
    loadHistory(sessionId)
    loadSessions()
  }, [sessionId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function loadSessions() {
    try {
      const res = await fetch(`${apiBase}/chat/sessions`)
      if (res.ok) {
        const data = await res.json()
        setSessions(data.sessions || [])
      }
    } catch {
      // Ignore background session list error
    }
  }

  async function loadHistory(sId) {
    try {
      const res = await fetch(`${apiBase}/chat/sessions/${sId}`)
      if (res.ok) {
        const data = await res.json()
        setMessages(data.messages || [])
      } else {
        setMessages([])
      }
    } catch {
      setMessages([])
    }
  }

  async function handleSend(e) {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || loading) return

    setInput('')
    setError(null)
    setLoading(true)

    // Optimistic user message
    const tempUserMsg = {
      id: Date.now(),
      role: 'user',
      content: trimmed,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, tempUserMsg])

    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: trimmed }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${res.status})`)
      }

      const data = await res.json()
      const assistantMsg = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.reply,
        kural_results: data.kural_results,
        mode: data.mode,
        created_at: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, assistantMsg])
      loadSessions()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend(e)
    }
  }

  function startNewChat() {
    const newId = `session-${Date.now()}`
    setSessionId(newId)
    setMessages([])
  }

  async function deleteSession(sId, e) {
    e.stopPropagation()
    try {
      await fetch(`${apiBase}/chat/sessions/${sId}`, { method: 'DELETE' })
      loadSessions()
      if (sId === sessionId) {
        startNewChat()
      }
    } catch {
      // Ignore delete errors
    }
  }

  function toggleKurals(msgId) {
    setExpandedKurals((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }))
  }

  const getModeBadge = (mode) => {
    switch (mode) {
      case 'clarifying_question':
        return '✦ Reflection Prompt'
      case 'multiple_options':
        return '✦ Multiple Perspectives'
      default:
        return '✦ Ethical Counsel'
    }
  }

  return (
    <div className="chat-container">
      {/* Session Management Bar */}
      <div className="chat-session-bar">
        <button type="button" className="btn-new-chat" onClick={startNewChat}>
          + New Dialogue
        </button>

        {sessions.length > 0 && (
          <div className="session-chips">
            {sessions.slice(0, 5).map((s) => (
              <div
                key={s.session_id}
                className={`session-chip ${s.session_id === sessionId ? 'active' : ''}`}
                onClick={() => setSessionId(s.session_id)}
              >
                <span className="session-title">{s.title || 'Conversation'}</span>
                <button
                  type="button"
                  className="session-delete"
                  onClick={(e) => deleteSession(s.session_id, e)}
                  title="Delete session"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Messages Thread */}
      <div className="chat-thread">
        {messages.length === 0 && (
          <div className="chat-welcome">
            <p className="chat-welcome-title">Dialogue with Ancient Wisdom</p>
            <p>
              Share your thoughts, challenges, or dilemmas. The counselor reflects on
              the Thirukural and converses with you across multiple turns.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={msg.id || idx} className={`chat-message ${msg.role}`}>
            <div className="chat-bubble">
              {msg.role === 'assistant' && msg.mode && (
                <div className="chat-mode-tag">{getModeBadge(msg.mode)}</div>
              )}
              <div className="chat-text">{msg.content}</div>

              {/* Collapsible Kural Citations */}
              {msg.role === 'assistant' && msg.kural_results && msg.kural_results.length > 0 && (
                <div className="chat-kurals-section">
                  <button
                    type="button"
                    className="btn-toggle-kurals"
                    onClick={() => toggleKurals(msg.id || idx)}
                  >
                    {expandedKurals[msg.id || idx]
                      ? `Hide Referenced Kurals (${msg.kural_results.length})`
                      : `View Referenced Kurals (${msg.kural_results.length}) 📜`}
                  </button>

                  {expandedKurals[msg.id || idx] && (
                    <div className="chat-kurals-list">
                      {msg.kural_results.map((k, kIdx) => (
                        <KuralCard
                          key={k.kural_number}
                          kural={k}
                          label={kIdx === 0 ? 'Primary Reference' : 'Related Wisdom'}
                        />
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <div className="chat-bubble loading">
              <span className="pulse-dot" />
              <span className="pulse-dot" />
              <span className="pulse-dot" />
            </div>
          </div>
        )}

        {error && (
          <div className="error-panel" role="alert">
            <p className="error-title">Dialogue Error</p>
            <p>{error}</p>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <form onSubmit={handleSend} className="chat-input-form">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          aria-label="Message Thirukural counselor"
          placeholder="Share what is happening or ask a follow-up question…"
          rows={2}
        />
        <div className="form-actions">
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? 'Thinking…' : 'Send'}
          </button>
          <span className="submit-hint">Enter to send, Shift+Enter for new line</span>
        </div>
      </form>
    </div>
  )
}
