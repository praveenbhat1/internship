import { useEffect, useState } from 'react'

const links = [
  ['Home', 'top'],
  ['Live demo', 'demo'],
  ['Results', 'results'],
  ['What we used', 'tech'],
]

export default function Nav() {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${
        scrolled ? 'py-3 glass' : 'py-5 bg-transparent'
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 flex items-center justify-between">
        <a href="#top" className="font-mono text-sm font-semibold tracking-wider text-cream">
          PAR<span className="text-olive">.</span>vision
        </a>
        <nav className="hidden md:flex items-center gap-7 text-sm text-muted">
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
