import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_API || 'http://127.0.0.1:8000'

const links = [
  ['Home', 'top'],
  ['Live demo', 'demo'],
  ['How it works', 'how'],
  ['Results', 'results'],
  ['What we used', 'tech'],
  ['Future', 'future'],
]

export default function Nav() {
  const [scrolled, setScrolled] = useState(false)
  const [online, setOnline] = useState(null)
  const [active, setActive] = useState('top')
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 40)
      const h = document.documentElement
      const max = h.scrollHeight - h.clientHeight
      setProgress(max > 0 ? (h.scrollTop / max) * 100 : 0)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()

    // active-section highlighting
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id)
        })
      },
      { rootMargin: '-45% 0px -45% 0px' },
    )
    links.forEach(([, id]) => {
      const el = document.getElementById(id)
      if (el) obs.observe(el)
    })

    // Free HF Spaces sleep when idle; retry before showing the red "offline" dot.
    let cancelled = false
    ;(async () => {
      for (let i = 0; i < 15 && !cancelled; i++) {
        try {
          const r = await fetch(`${API}/health`, { cache: 'no-store' })
          if (r.ok) {
            if (!cancelled) setOnline(true)
            return
          }
        } catch {
          /* still waking */
        }
        await new Promise((s) => setTimeout(s, 5000))
      }
      if (!cancelled) setOnline(false)
    })()

    return () => {
      cancelled = true
      window.removeEventListener('scroll', onScroll)
      obs.disconnect()
    }
  }, [])

  return (
    <header
      className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${
        scrolled ? 'py-3 glass' : 'py-5 bg-transparent'
      }`}
    >
      {/* scroll progress bar */}
      <div
        className="absolute top-0 left-0 h-0.5 bg-olive transition-[width] duration-150"
        style={{ width: `${progress}%` }}
      />

      <div className="max-w-6xl mx-auto px-6 flex items-center justify-between">
        <a href="#top" className="flex items-center gap-2 font-mono text-sm font-semibold tracking-wider text-cream">
          <span>
            PAR<span className="text-olive">.</span>vision
          </span>
          <span
            title={online ? 'model online' : 'model offline'}
            className={`h-1.5 w-1.5 rounded-full ${
              online == null ? 'bg-muted' : online ? 'bg-olive shadow-[0_0_8px_#a8b14b]' : 'bg-red-500/70'
            }`}
          />
        </a>
        <nav className="hidden md:flex items-center gap-6 text-sm">
          {links.map(([label, id]) => (
            <a
              key={id}
              href={`#${id}`}
              className={`transition-colors ${active === id ? 'text-olive2' : 'text-muted hover:text-cream'}`}
            >
              {label}
            </a>
          ))}
        </nav>
        <a
          href="#demo"
          className="btn text-sm px-4 py-2 rounded-full bg-olive/15 border border-olive/40 text-cream hover:bg-olive/25"
        >
          Try it
        </a>
      </div>
    </header>
  )
}
