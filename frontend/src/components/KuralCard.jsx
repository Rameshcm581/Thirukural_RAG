export default function KuralCard({ kural, label }) {
  const matchPct = kural.similarity_score ? Math.round(kural.similarity_score * 100) : null

  return (
    <article className="kural-card">
      <div className="kural-header">
        <span className="kural-label">{label}</span>
        {kural.matched_tag && (
          <span className="kural-badge" title={`Semantic similarity: ${matchPct}%`}>
            ✦ Matched scenario: "{kural.matched_tag}"
          </span>
        )}
      </div>

      <div className="kural-manuscript">
        <p className="kural-tamil">
          {kural.tamil_couplet[0]}
          <br />
          {kural.tamil_couplet[1]}
        </p>
        <p className="kural-translit">{kural.transliteration}</p>
      </div>

      <div className="kural-divider" aria-hidden="true" />

      <p className="kural-english">{kural.english_couplet}</p>

      {kural.english_explanation && (
        <div className="kural-explanation-block">
          <p className="kural-explanation-heading">Classical Meaning</p>
          <p className="kural-classical">{kural.english_explanation}</p>
        </div>
      )}

      {kural.explanation && (
        <div className="kural-explanation-block kural-wisdom-block">
          <p className="kural-explanation-heading">Why it applies to your situation</p>
          <p className="kural-explanation">{kural.explanation}</p>
        </div>
      )}

      <footer className="kural-meta">
        <span>Kural {kural.kural_number}</span>
        <span className="meta-separator">•</span>
        <span>{kural.book}</span>
        <span className="meta-separator">•</span>
        <span>Chapter: {kural.adikaram}</span>
      </footer>
    </article>
  )
}
