export default function KuralCard({ kural, label }) {
  return (
    <article className="kural-card">
      <p className="kural-label">{label}</p>

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

      {kural.explanation && (
        <p className="kural-explanation">{kural.explanation}</p>
      )}

      <footer className="kural-meta">
        Kural {kural.kural_number}, from the chapter on {kural.adikaram}
      </footer>
    </article>
  )
}
