const tech = ['SigLIP-2', 'PyTorch', 'FastAPI', 'React', 'PA-100K', 'PETA']

export default function Footer() {
  return (
    <footer className="relative border-t border-line/60 px-6 pt-20 pb-10 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-40 [mask-image:radial-gradient(ellipse_60%_80%_at_50%_100%,#000_40%,transparent_100%)]" />
      <div className="relative max-w-6xl mx-auto">
        <div className="reveal text-center mb-14">
          <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-5 text-cream">
            Attribute recognition,
            <br />
            <span className="gradient-text">done responsibly.</span>
          </h2>
          <p className="text-muted max-w-xl mx-auto">
            Accurate in-domain, validated cross-dataset, honest about its limits — and you can run it
            yourself right here.
          </p>
        </div>

        <div className="flex flex-wrap justify-center gap-2 mb-14">
          {tech.map((t) => (
            <span
              key={t}
              className="text-xs font-mono px-3 py-1.5 rounded-full border border-line text-muted"
            >
              {t}
            </span>
          ))}
        </div>

        <div className="flex flex-col md:flex-row items-center justify-between gap-4 pt-8 border-t border-line/60 text-sm text-muted">
          <div className="font-mono">
            PAR<span className="text-olive">.</span>vision — 2026
          </div>
          <div className="flex items-center gap-6">
            <a href="mailto:praveenbhat46@gmail.com" className="hover:text-cream transition-colors">
              Email
            </a>
            <a href="#" className="hover:text-cream transition-colors">
              GitHub
            </a>
            <a href="#top" className="hover:text-cream transition-colors">
              Back to top ↑
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
