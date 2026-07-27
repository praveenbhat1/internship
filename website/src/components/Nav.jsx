import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_API || 'http://127.0.0.1:8000'

const links = [
  ['Home', 'top'],
  ['Live demo', 'demo'],
  ['How it works', 'how'],
  ['Results', 'results'],
  ['What we used', 'tech'],
]

export default function Nav() {
  const [scrolled, setScrolled] = useState(false)
  const [online, setOnline] = useState(null)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', onScroll)
    fetch(`${API}/health`)
      .then((r) => setOnline(r.ok))
      .catch(() => setOnline(false))
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${
        scrolled ? 'py-3 glass' : 'py-5 bg-transparent'
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 flex items-center justify-between">
        <a href="#top" className="flex items-center gap-2 font-mono text-sm font-semibold tracking-wider text-cream">
          PAR<span className="text-olive">.</span>vision
          <span
            title={online ? 'model online' : 'model offline'}
            className={`h-1.5 w-1.5 rounded-full ${
              online == null ? 'bg-muted' : online ? 'bg-olive shadow-[0_0_8px_#a8b14b]' : 'bg-red-500/70'
            }`}
          />
        </a>
        <nav className="hidden md:flex items-center gap-6 text-sm text-muted">
          {links.map(([label, id]) => (
            <a key={id} href={`#${id}`} className="hover:text-cream transition-colors">
              {label}
            </a>
          ))}
        </nav>
        <a
          href="#demo"
          className="text-sm px-4 py-2 rounded-full bg-olive/15 border border-olive/40 text-cream hover:bg-olive/25 transition-colors"
        >
          Try it
        </a>
      </div>
    </header>
  )
}
