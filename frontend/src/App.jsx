import { useEffect, useRef, useState } from 'react'

const initialModel = { available: false, prediction_enabled: false, reason: 'Checking model status…' }

export default function App() {
  const [model, setModel] = useState(initialModel)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [quality, setQuality] = useState(null)
  const [error, setError] = useState('')
  const [checking, setChecking] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch('/api/model/status')
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('API unavailable')))
      .then(setModel)
      .catch(() => setModel({ ...initialModel, reason: 'The API is not connected. Start the backend to use image quality checks.' }))
  }, [])

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  function selectFile(nextFile) {
    setError('')
    setQuality(null)
    setFile(nextFile || null)
    setPreview(nextFile ? URL.createObjectURL(nextFile) : '')
  }

  async function checkQuality(event) {
    event.preventDefault()
    if (!file) return
    setChecking(true)
    setError('')
    setQuality(null)
    const form = new FormData()
    form.append('image', file)
    try {
      const response = await fetch('/api/image/quality', { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not check this image.')
      setQuality(data)
    } catch (err) {
      setError(err.message || 'Could not reach the API. Check that the backend is running.')
    } finally {
      setChecking(false)
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="DermaVision home"><span className="brand-mark">D</span><span>Derma<span className="brand-light">Vision</span></span></a>
        <span className="research-label"><span className="status-dot" /> Research preview</span>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span /> SKIN IMAGE RESEARCH</div>
          <h1>A clearer view<br />starts with <em>care.</em></h1>
          <p className="hero-lede">A research interface exploring responsible image quality review and AI-assisted skin condition screening.</p>
          <div className="trust-row"><span className="shield">✳</span><span>Temporary quality check <b>·</b> The app does not retain your image</span></div>
        </div>
        <aside className="notice-card">
          <span className="notice-icon">i</span>
          <div><strong>Research tool, not a diagnosis</strong><p>DermaVision is experimental and cannot diagnose or guide treatment. If you have a health concern, speak with a qualified healthcare professional.</p></div>
        </aside>
      </section>

      <section className="workspace" aria-label="Image review workspace">
        <div className="section-heading"><div><span className="step">01</span><h2>Review an image</h2></div><span className="section-note">JPEG, PNG, WebP or BMP · up to 10 MB</span></div>
        <div className="review-grid">
          <div className="upload-panel">
            <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(event) => selectFile(event.target.files?.[0])} />
            {preview ? <div className="preview-frame"><img src={preview} alt="Selected skin image preview" /><button className="change-image" onClick={() => inputRef.current?.click()} type="button">Change image</button></div> : <button className="dropzone" type="button" onClick={() => inputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); selectFile(event.dataTransfer.files?.[0]) }}><span className="upload-icon">↑</span><strong>Choose a skin image</strong><span>or drag and drop it here</span><small>The app does not add your image to a dataset or save it.</small></button>}
            {file && <div className="file-row"><span className="file-icon">▧</span><span className="file-name">{file.name}<small>{(file.size / (1024 * 1024)).toFixed(2)} MB</small></span><button className="text-button" onClick={() => { selectFile(null); if (inputRef.current) inputRef.current.value = '' }} type="button">Remove</button></div>}
            <button className="primary-button" onClick={checkQuality} disabled={!file || checking}>{checking ? 'Checking image…' : 'Check image quality'} <span>→</span></button>
            <p className="privacy-note"><span>⌑</span> The app does not retain uploads. Server frameworks may buffer them temporarily during processing.</p>
          </div>

          <div className="results-panel">
            <div className="results-title"><div><span className="step">02</span><h2>Image review</h2></div><span className="pill">QUALITY CHECK</span></div>
            {error && <div className="error-box" role="alert">{error}</div>}
            {!quality && !error && <div className="empty-state"><span className="focus-icon">◎</span><strong>Your image review will appear here</strong><p>We’ll check basic image properties like dimensions, brightness and possible blur.</p></div>}
            {quality && <div className="quality-results"><div className="quality-summary"><span className="checkmark">✓</span><div><strong>Basic checks complete</strong><span>These checks do not assess skin health.</span></div></div><div className="metric-grid"><div><small>Dimensions</small><strong>{quality.width} × {quality.height}</strong></div><div><small>Colour</small><strong>{quality.channels}</strong></div><div><small>Format</small><strong>{quality.format}</strong></div><div><small>Brightness</small><strong>{quality.mean_brightness_0_to_1}</strong></div></div><div className="quality-warning"><strong>Image notes</strong>{quality.warnings.length ? <ul>{quality.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : <p>No basic image issues were detected. This does not mean the image is suitable for a diagnosis.</p>}</div></div>}
            <div className="model-status"><span className="status-icon">◷</span><div><strong>AI screening unavailable</strong><p>{model.reason}</p></div><span className="unavailable-tag">IN DEVELOPMENT</span></div>
          </div>
        </div>
      </section>

      <section className="how-section"><div className="section-heading"><div><span className="step">03</span><h2>Designed with care</h2></div><span className="section-note">An early research prototype</span></div><div className="care-grid"><article><span className="care-number">A</span><h3>Separate image types</h3><p>Clinical photographs and dermoscopic research images remain separate datasets and are not treated as interchangeable.</p></article><article><span className="care-number">B</span><h3>No unsupported predictions</h3><p>The current DermNet experiment is not suitable for screening. This app will not show a disease prediction until that changes.</p></article><article><span className="care-number">C</span><h3>People come first</h3><p>Automated image tools can be wrong. Seek a qualified clinician for concerns about your skin or changes you notice.</p></article></div></section>

      <footer><a className="brand footer-brand" href="#top"><span className="brand-mark">D</span><span>Derma<span className="brand-light">Vision</span></span></a><span>Research prototype · Not a medical device · No clinical validation</span><a href="https://www.aad.org/public" target="_blank" rel="noreferrer">Skin health resources ↗</a></footer>
    </main>
  )
}
