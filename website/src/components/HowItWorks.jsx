import { useEffect, useRef, useState } from 'react'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

const steps = [
  {
    n: '01',
    tag: 'SigLIP-2',
    title: 'See the image',
    body: 'A frozen SigLIP-2 backbone encodes the cropped photo into a rich 1024-dim feature vector — a compact numeric description of everything visible.',
    img: '/assets/_feat.png',
    caption: 'The extracted visual feature',
  },
  {
    n: '02',
    tag: 'CMAA',
    title: 'Look per attribute',
    body: 'Cross-attribute attention produces a separate heatmap for every attribute, so the model looks in the right place for a hat vs. a backpack vs. a sleeve.',
    img: '/assets/_cmaa.png',
    caption: 'Attention heatmaps, one per attribute',
  },
  {
    n: '03',
    tag: 'OCFR',
    title: 'Know the viewpoint',
    body: 'The orientation head decides front, side, or back — and reweights the features accordingly. On a back view it switches gender off, since the face is hidden.',
    img: '/assets/090001.jpg',
    caption: 'Front / side / back changes what is visible',
  },
  {
    n: '04',
    tag: 'DACG + CCLoss',
    title: 'Predict, together',
    body: 'A correlation graph lets related attributes support each other, and calibrated thresholds turn scores into confident yes/no predictions — abstaining when unsure.',
    img: '/assets/_dacg.png',
    caption: 'Learned attribute correlations',
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

  const step = steps[active]

  return (
    <section id="how" ref={ref} style={{ height: `${steps.length * 100}vh` }} className="relative">
      <div className="sticky top-0 h-screen flex items-center overflow-hidden border-t border-line/60">
        <div className="absolute inset-0 grid-bg opacity-25" />

        <div className="relative max-w-6xl mx-auto px-6 w-full grid md:grid-cols-2 gap-10 md:gap-14 items-center">
          {/* LEFT — text + progress */}
          <div>
            <div className="flex items-center gap-3 mb-5">
              <span className="h-px w-8 bg-olive/70" />
              <span className="section-label">02 — How it works</span>
            </div>
            <div key={active} className="animate-[fadein_0.5s_ease]">
              <div className="inline-block text-xs font-mono px-3 py-1 rounded-full border border-olive/40 text-olive2 mb-5">
                {step.tag}
              </div>
              <h2 className="text-3xl md:text-5xl font-bold tracking-tight text-cream mb-5 leading-[1.06]">
                {step.title}
              </h2>
              <p className="text-muted text-base md:text-lg leading-relaxed max-w-lg">{step.body}</p>
            </div>

            {/* progress dots */}
            <div className="flex items-center gap-3 mt-9">
              {steps.map((s, i) => (
                <div key={s.n} className="flex items-center gap-3">
                  <div
                    className={`h-8 w-8 rounded-full border flex items-center justify-center font-mono text-[0.65rem] transition-all ${
                      i === active
                        ? 'border-olive bg-olive text-ink'
                        : i < active
                          ? 'border-olive/50 text-olive2'
                          : 'border-line text-muted'
                    }`}
                  >
                    {s.n}
                  </div>
                  {i < steps.length - 1 && (
                    <div className={`h-px w-6 transition-colors ${i < active ? 'bg-olive/60' : 'bg-line'}`} />
                  )}
                </div>
              ))}
            </div>
            <div className="mt-6 text-xs font-mono text-muted">
              {active + 1} / {steps.length} — keep scrolling
            </div>
          </div>

          {/* RIGHT — crossfading visual */}
          <div className="relative h-[300px] md:h-[440px] card overflow-hidden">
            {steps.map((s, i) => (
              <div
                key={s.n}
                className="absolute inset-0 flex flex-col items-center justify-center p-6 transition-opacity duration-500"
                style={{ opacity: i === active ? 1 : 0 }}
              >
                <img
                  src={s.img}
                  alt={s.title}
                  className="max-h-[76%] max-w-full rounded-lg object-contain"
                />
                <div className="mt-4 text-xs font-mono text-muted text-center">{s.caption}</div>
              </div>
            ))}
            {/* step counter badge */}
            <div className="absolute top-4 right-4 font-mono text-xs text-olive2/70">{step.n}/04</div>
          </div>
        </div>
      </div>
    </section>
  )
}
