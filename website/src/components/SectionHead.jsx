export default function SectionHead({ label, title, sub }) {
  return (
    <div className="reveal max-w-3xl mb-12">
      <div className="flex items-center gap-3 mb-4">
        <span className="h-px w-8 bg-olive/70" />
        <span className="section-label">{label}</span>
      </div>
      <h2 className="text-3xl md:text-5xl font-bold tracking-tight leading-tight text-cream">
        {title}
      </h2>
      {sub && <p className="mt-5 text-muted text-base md:text-lg leading-relaxed">{sub}</p>}
    </div>
  )
}
