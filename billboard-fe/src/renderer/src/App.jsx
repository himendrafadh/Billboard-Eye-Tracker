import { useState, useEffect, useRef } from 'react'
import './assets/main.css'

const INTERVAL_SECONDS = 10 * 60

// ── komponen kecil ────────────────────────────────────────────────────────

function StatCard({ value, label, color }) {
  return (
    <div className="stat-card">
      <div className={`stat-val ${color}`}>{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function StatusDot({ state }) {
  return <span className={`dot dot-${state}`} />
}

// ── App utama ─────────────────────────────────────────────────────────────

export default function App() {
  const [status, setStatus]       = useState('idle')   // idle | loading | running | stopped
  const [sourceLabel, setSourceLabel] = useState('')
  const [imgSrc, setImgSrc]       = useState(null)
  const [stats, setStats]         = useState({
    active_people: 0,
    people_passing: 0,
    watching_now: 0,
    people_watching: 0,
    flush_in_seconds: INTERVAL_SECONDS
  })
  const [csvRows, setCsvRows]     = useState([])
  const imgRef = useRef(null)

  // ── terima pesan dari Python ─────────────────────────────────────────
  useEffect(() => {
    window.api.onMessage((msg) => {
      switch (msg.type) {
        case 'ready':
          setStatus('running')
          break
        case 'frame':
          setImgSrc('data:image/jpeg;base64,' + msg.data)
          break
        case 'stats':
          setStats(msg)
          break
        case 'csv_row':
          setCsvRows((prev) => [msg, ...prev].slice(0, 20))
          break
        case 'error':
          setStatus('stopped')
          break
        case 'done':
          setStatus('stopped')
          setImgSrc(null)
          break
      }
    })
    return () => window.api.offMessage()
  }, [])

  // ── handlers ─────────────────────────────────────────────────────────
  async function handleWebcam() {
    window.api.offMessage()
    window.api.onMessage(handleMsg)
    setStatus('loading')
    setSourceLabel('Webcam (0)')
    setCsvRows([])
    await window.api.startWebcam()
  }

  async function handleUpload() {
    window.api.offMessage()
    window.api.onMessage(handleMsg)
    const filePath = await window.api.startVideo()
    if (!filePath) return
    setStatus('loading')
    setSourceLabel(filePath.split('\\').pop())
    setCsvRows([])
  }

  async function handleStop() {
    await window.api.stopPipeline()
    setStatus('stopped')
    setImgSrc(null)
    setSourceLabel('')
  }

  // handler yang bisa di-reregister
  function handleMsg(msg) {
    switch (msg.type) {
      case 'ready':   setStatus('running'); break
      case 'frame':   setImgSrc('data:image/jpeg;base64,' + msg.data); break
      case 'stats':   setStats(msg); break
      case 'csv_row': setCsvRows((prev) => [msg, ...prev].slice(0, 20)); break
      case 'error':   setStatus('stopped'); break
      case 'done':    setStatus('stopped'); setImgSrc(null); break
    }
  }

  // ── flush countdown ──────────────────────────────────────────────────
  const remSec = stats.flush_in_seconds ?? INTERVAL_SECONDS
  const remMin = String(Math.floor(remSec / 60)).padStart(2, '0')
  const remS   = String(remSec % 60).padStart(2, '0')
  const barPct = Math.min(100, Math.round((remSec / INTERVAL_SECONDS) * 100))

  const isRunning = status === 'running' || status === 'loading'

  // ── status label ─────────────────────────────────────────────────────
  const statusLabel = {
    idle   : 'Tidak aktif',
    loading: 'Memulai...',
    running: 'Berjalan',
    stopped: 'Dihentikan'
  }[status]

  const dotState = {
    idle   : 'idle',
    loading: 'loading',
    running: 'running',
    stopped: 'stopped'
  }[status]

  return (
    <div className="app">

      {/* ── header ── */}
      <header className="header">
        <div className="logo">Billboard <span>Eye Tracker</span></div>
        <div className="btn-group">
          <button className="btn btn-webcam" onClick={handleWebcam} disabled={isRunning}>
            ▶ Live Webcam
          </button>
          <button className="btn btn-upload" onClick={handleUpload} disabled={isRunning}>
            ⬆ Upload Video
          </button>
          <button className="btn btn-stop" onClick={handleStop} disabled={!isRunning}>
            ■ Stop
          </button>
        </div>
      </header>

      {/* ── main ── */}
      <div className="main">

        {/* video panel */}
        <div className="video-panel">
          {imgSrc ? (
            <img ref={imgRef} src={imgSrc} alt="feed" className="video-feed" />
          ) : (
            <div className="placeholder">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="1.2">
                <rect x="2" y="7" width="20" height="14" rx="2"/>
                <path d="M16 3l-4 4-4-4"/>
              </svg>
              <span>Pilih mode Live Webcam atau Upload Video</span>
            </div>
          )}
        </div>

        {/* sidebar */}
        <aside className="sidebar">

          {/* status */}
          <section>
            <div className="section-title">Status</div>
            <div className="status-row">
              <StatusDot state={dotState} />
              <span>{statusLabel}</span>
            </div>
            {sourceLabel && (
              <div className="source-label">Sumber: {sourceLabel}</div>
            )}
          </section>

          {/* stats */}
          <section>
            <div className="section-title">Statistik interval ini</div>
            <div className="stat-grid">
              <StatCard value={stats.active_people}   label="Di frame"        color="cyan"   />
              <StatCard value={stats.people_passing}  label="Total lewat"     color="yellow" />
              <StatCard value={stats.watching_now}    label="Lihat sekarang"  color="green"  />
              <StatCard value={stats.people_watching} label="Total lihat"     color="green"  />
            </div>
          </section>

          {/* flush countdown */}
          <section>
            <div className="flush-header">
              <span className="section-title" style={{marginBottom:0}}>Flush CSV dalam</span>
              <span className="flush-time">{remMin}:{remS}</span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: barPct + '%' }} />
            </div>
          </section>

          {/* CSV log */}
          <section className="csv-section">
            <div className="section-title">Log CSV</div>
            <div className="csv-table-wrap">
              <table className="csv-table">
                <thead>
                  <tr>
                    <th>Waktu</th>
                    <th>Lewat</th>
                    <th>Lihat</th>
                  </tr>
                </thead>
                <tbody>
                  {csvRows.length === 0 ? (
                    <tr><td colSpan={3} className="empty">Belum ada data</td></tr>
                  ) : (
                    csvRows.map((row, i) => (
                      <tr key={i}>
                        <td>{row.timestamp?.slice(11) ?? '-'}</td>
                        <td>{row.people_passing}</td>
                        <td>{row.people_watching}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </section>

        </aside>
      </div>
    </div>
  )
}