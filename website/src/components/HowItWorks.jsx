import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

const steps = [
  {
    n: '01',
    tag: 'SigLIP-2',
    title: 'See the image',
    body: 'A frozen SigLIP-2 backbone encodes the cropped photo into a rich 1024-dim feature vector — a compact numeric description of everything visible.',
  },
  {
    n: '02',
    tag: 'CMAA',
    title: 'Look per attribute',
    body: 'Cross-attribute attention produces a separate heatmap for every attribute, so the model looks in the right place for a hat vs. a backpack vs. a sleeve.',
  },
  {
    n: '03',
    tag: 'OCFR',
    title: 'Know the viewpoint',
    body: 'The orientation head decides front, side, or back — and reweights the features accordingly. On a back view it also switches gender off, since the face is hidden.',
  },
  {
    n: '04',
    tag: 'DACG + CCLoss',
    title: 'Predict, together',
    body: 'A correlation graph lets related attributes support each other, and calibrated thresholds turn scores into confident yes/no predictions — abstaining when unsure.',
  },
]

export default function HowItWorks() {
  const ref = useRef(null)
  const [active, setActive] = useState(0)

  useEffect(() => {
    const st = ScrollTrigger.create({
      trigger: ref.current,
      start: 'top top',
      end: 'bottom bottom',
      onUpdate: (self) => {
        const i = Math.min(steps.length - 1, Math.floor(self.progress * steps.length))
        setActive(i)
      },
    })
    return () => st.kill()
  }, [])

  return (
    <section id="how" ref={ref} style={{ height: `${steps.length * 100}vh` }} className="relative">
      <div className="sticky top-0 h-screen flex items-center overflow-hidden border-t border-line/60">
        <div className="absolute inset-0 grid-bg opacity-30" />
        {/* giant ghost number */}
        <div
          key={active}
          className="absolute right-4 md:right-16 bottom-0 font-extrabold text-olive/5 leading-none pointer-events-none select-none"
          style={{ fontSize: 'min(46vw, 640px)' }}
        >
          {steps[active].n}
        </div>

        <div className="relative max-w-6xl mx-auto px-6 w-full grid md:grid-cols-[auto,1fr] gap-10 items-center">
          {/* progress rail */}
          <div className="hidden md:flex flex-col gap-4">
            {steps.map((s, i) => (
              <div key={s.n} className="flex items-center gap-3">
                <div
                  className={`h-9 w-9 rounded-full border flex items-center justify-center font-mono text-xs transition-all ${
                    i === active
                      ? 'border-olive bg-olive text-ink'
                      : i < active
                        ? 'border-olive/50 text-olive2'
                        : 'border-line text-muted'
                  }`}
                >
                  {s.n}
                </div>
                <div
                  className={`h-px w-10 transition-colors ${i <= active ? 'bg-olive/60' : 'bg-line'}`}
                />
              </div>
            ))}
          </div>

          {/* active step */}
          <div>
            <div className="section-label mb-4">How it works · Step {steps[active].n}</div>
            <div key={active} className="animate-[fadein_0.5s_ease]">
              <div className="inline-block text-xs font-mono px-3 py-1 rounded-full border border-olive/40 text-olive2 mb-5">
                {steps[active].tag}
              </div>
              <h2 className="text-4xl md:text-6xl font-bold tracking-tight text-cream mb-5 leading-[1.05]">
                {steps[active].title}
              </h2>
              <p className="text-muted text-base md:text-lg leading-relaxed max-w-xl">
                {steps[active].body}
              </p>
            </div>

            <div className="mt-8 text-xs font-mono text-muted">
              {active + 1} / {steps.length} — keep scrolling
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
