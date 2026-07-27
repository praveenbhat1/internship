import { useEffect, useRef, useState } from 'react'
import SectionHead from './SectionHead.jsx'

const API = import.meta.env.VITE_API || 'http://127.0.0.1:8000'

const samples = [
  ['090001.jpg', '/assets/090001.jpg'],
  ['094166.jpg', '/assets/094166.jpg'],
  ['095832.jpg', '/assets/095832.jpg'],
]

export default function LiveDemo() {
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | done | error
  const [online, setOnline] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(() => setOnline(true))
      .catch(() => setOnline(false))
  }, [])

  async function analyzeBlob(blob, previewUrl) {
    setPreview(previewUrl)
    setResult(null)
    setStatus('loading')
    try {
      const fd = new FormData()
      fd.append('file', blob, 'upload.jpg')
      const r = await fetch(`${API}/predict`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error('bad response')
      setResult(await r.json())
      setStatus('done')
      setOnline(true)
    } catch {
      setStatus('error')
      setOnline(false)
    }
  }

  function onFile(file) {
    if (!file) return
    analyzeBlob(file, URL.createObjectURL(file))
  }

  async function onSample(src) {
    const blob = await (await fetch(src)).blob()
    analyzeBlob(blob, src)
  }

  const detected = result?.attrs.filter((a) => a.pred) || []
  const others = result?.attrs.filter((a) => !a.pred).slice(0, 4) || []

  return (
    <section id="demo" className="relative py-28 px-6 border-t border-line/60">
      <div className="max-w-6xl mx-auto">
        <SectionHead
          label="01 — Live demo"
          title="Try the model yourself."
          sub="Upload a cropped photo of one person (or pick a sample). The real trained model runs and returns exactly what demo.py shows — the attributes, where it looked, and its confidence."
        />

        {online === false && (
          <div className="reveal card p-5 mb-8 border-olive/40">
            <p className="text-sm text-cream mb-2">⚠︎ The model server isn't running.</p>
            <p className="text-sm text-muted">
              Start it once, then reload this page:
              <code className="block mt-2 font-mono text-olive2 text-xs bg-ink/60 rounded-lg p-3">
                cd internship/mvp &amp;&amp; python serve.py
              </code>
            </p>
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-8 items-start">
          {/* LEFT — input */}
          <div className="reveal">
            <div
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                onFile(e.dataTransfer.files?.[0])
              }}
              className="card p-4 cursor-pointer hover:border-olive/50 transition-colors relative overflow-hidden"
            >
              {preview ? (
                <img src={preview} alt="input" className="rounded-xl mx-auto max-h-[380px]" />
              ) : (
                <div className="h-[300px] flex flex-col items-center justify-center text-center gap-3 border-2 border-dashed border-line rounded-xl">
                  <div className="text-4xl">⬆</div>
                  <div className="text-cream font-medium">Drop an image or click to upload</div>
                  <div className="text-xs text-muted">a cropped photo of one person</div>
                </div>
              )}
              {status === 'loading' && (
                <div className="absolute inset-0 bg-ink/70 flex items-center justify-center">
                  <div className="text-olive2 font-mono text-sm animate-pulse">analyzing…</div>
                </div>
              )}
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])}
            />

            <div className="mt-4">
              <div className="text-xs text-muted mb-2">or try a sample:</div>
              <div className="flex gap-3">
                {samples.map(([name, src]) => (
                  <button
                    key={name}
                    onClick={() => onSample(src)}
                    className="rounded-xl overflow-hidden border-2 border-line hover:border-olive transition-all"
                  >
                    <img src={src} alt={name} className="h-20 w-auto" />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* RIGHT — output */}
          <div className="reveal">
            <div className="card p-6 min-h-[300px]">
              {!result && status !== 'loading' && (
                <div className="h-[260px] flex items-center justify-center text-muted text-sm text-center">
                  {online === false
                    ? 'Start the model server to run the demo.'
                    : 'Predictions will appear here.'}
                </div>
              )}
              {status === 'loading' && (
                <div className="h-[260px] flex items-center justify-center text-olive2 font-mono text-sm animate-pulse">
                  running the model…
                </div>
              )}
              {result && (
                <>
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-xs text-muted">model output</span>
                    <span className="text-xs px-2 py-1 rounded-full bg-olive/15 text-olive2">
                      viewpoint:{' '}
                      {Object.entries(result.viewpoint).sort((a, b) => b[1] - a[1])[0][0]}
                    </span>
                  </div>

                  <div className="space-y-2.5">
                    {[...detected, ...others].map((a) => (
                      <div key={a.name} className="flex items-center gap-3 text-sm">
                        <span className={`w-28 ${a.pred ? 'text-cream' : 'text-muted line-through'}`}>
                          {a.name}
                        </span>
                        <div className="flex-1 h-2 rounded-full bg-line overflow-hidden">
                          <div
                            className="h-full rounded-full transition-[width] duration-700 ease-out"
                            style={{
                              width: `${a.prob}%`,
                              background: a.pred
                                ? 'linear-gradient(90deg,#6f763a,#c8d17a)'
                                : '#2b2d1c',
                            }}
                          />
                        </div>
                        <span className="w-9 text-right font-mono text-xs text-muted">{a.prob}%</span>
                      </div>
                    ))}
                  </div>

                  <div className="mt-4 pt-4 border-t border-line text-sm">
                    {result.gender.reported ? (
                      <span className="text-olive2">
                        Gender: <b>{result.gender.label}</b> ({result.gender.conf}% confident)
                      </span>
                    ) : (
                      <span className="text-muted">
                        Gender: <b className="text-cream">not reported</b> — {result.gender.why}{' '}
                        (model abstains when unsure)
                      </span>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* stage images exactly like demo.py */}
        {result?.images && (
          <div className="reveal mt-8 space-y-5">
            {/* CMAA is a wide strip — give it the full width so the heatmaps are large */}
            {result.images.cmaa && (
              <div className="card p-5">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-mono text-olive2">
                    Step 2 — CMAA: where the model looks for each attribute
                  </span>
                  <span className="text-[0.65rem] font-mono text-muted hidden sm:block">
                    attention heatmap · detected attributes first
                  </span>
                </div>
                <img
                  src={result.images.cmaa}
                  alt="CMAA heatmaps"
                  className="rounded-lg w-full max-w-3xl mx-auto"
                />
              </div>
            )}

            {/* the two square-ish plots side by side */}
            <div className="grid md:grid-cols-2 gap-5">
              {[
                ['Step 1 — SigLIP feature', result.images.feature],
                ['Step 4 — DACG correlations', result.images.dacg],
              ].map(
                ([label, src]) =>
                  src && (
                    <div key={label} className="card p-5">
                      <div className="text-xs font-mono text-olive2 mb-3">{label}</div>
                      <img src={src} alt={label} className="rounded-lg mx-auto" />
                    </div>
                  ),
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
