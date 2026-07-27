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

const boundaries = steps.slice(1).map((_, i) => i + 1) // internal step boundaries: 1,2,3

export default function HowItWorks() {
  const ref = useRef(null)
  const textRef = useRef(null)
  const barRef = useRef(null)
  const layerRefs = useRef([])
  const activeRef = useRef(0)
  const [active, setActive] = useState(0)

  useEffect(() => {
    const st = ScrollTrigger.create({
      trigger: ref.current,
      start: 'top top',
      end: 'bottom bottom',
      onUpdate: (self) => {
        const p = self.progress * steps.length // 0..steps.length
        let i = Math.min(steps.length - 1, Math.floor(p))
        if (i < 0) i = 0

        // continuous progress bar
        if (barRef.current) barRef.current.style.width = `${self.progress * 100}%`

        // text dips to near-invisible exactly at each boundary, so the content
        // swap is hidden and the transition reads as one smooth motion
        const d = Math.min(...boundaries.map((b) => Math.abs(p - b)))
        const fade = Math.max(0.12, Math.min(1, d * 4))
        if (textRef.current) {
          textRef.current.style.opacity = fade
          textRef.current.style.transform = `translateY(${(1 - fade) * 10}px)`
        }

        // crossfade visuals + gentle parallax on the active one
        const frac = p - Math.floor(p)
        layerRefs.current.forEach((el, k) => {
          if (!el) return
          el.style.opacity = k === i ? 1 : 0
          if (k === i) el.style.transform = `translateY(${(frac - 0.5) * -14}px) scale(1.01)`
        })

        if (i !== activeRef.current) {
          activeRef.current = i
          setActive(i)
        }
      },
    })
    return () => st.kill()
  }, [])

  const step = steps[active]

  return (
    <section id="how" ref={ref} style={{ height: `${steps.length * 100}vh` }} className="relative">
      <div className="sticky top-0 h-screen flex items-center overflow-hidden border-t border-line/60">
        <div className="absolute inset-0 grid-bg opacity-25" />

        {/* giant background step number — left watermark behind the text */}
        <div
          key={active}
          aria-hidden
          className="absolute left-[-2vw] md:left-[2vw] top-1/2 -translate-y-1/2 font-extrabold leading-none pointer-events-none select-none text-cream/[0.04] animate-[fadein_0.6s_ease]"
          style={{ fontSize: 'min(42vw, 540px)' }}
        >
          {step.n}
        </div>

        <div className="relative max-w-6xl mx-auto px-6 w-full grid md:grid-cols-2 gap-10 md:gap-14 items-center">
          {/* LEFT — text + progress */}
          <div>
            <div className="flex items-center gap-3 mb-5">
              <span className="h-px w-8 bg-olive/70" />
              <span className="section-label">02 — How it works</span>
            </div>

            <div ref={textRef} className="transition-none will-change-[opacity,transform]">
              <div className="inline-block text-xs font-mono px-3 py-1 rounded-full border border-olive/40 text-olive2 mb-5">
                {step.tag}
              </div>
              <h2 className="text-3xl md:text-5xl font-bold tracking-tight text-cream mb-5 leading-[1.06]">
                {step.title}
              </h2>
              <p className="text-muted text-base md:text-lg leading-relaxed max-w-lg">{step.body}</p>
            </div>

            {/* discrete dots */}
            <div className="flex items-center gap-3 mt-9">
              {steps.map((s, i) => (
                <div key={s.n} className="flex items-center gap-3">
                  <div
                    className={`h-8 w-8 rounded-full border flex items-center justify-center font-mono text-[0.65rem] transition-all duration-300 ${
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

            {/* continuous progress bar */}
            <div className="mt-6 h-1 w-full max-w-xs rounded-full bg-line overflow-hidden">
              <div ref={barRef} className="h-full bg-olive" style={{ width: '0%' }} />
            </div>
          </div>

          {/* RIGHT — crossfading visual */}
          <div className="relative h-[300px] md:h-[440px] card overflow-hidden">
            {steps.map((s, i) => (
              <div
                key={s.n}
                ref={(el) => (layerRefs.current[i] = el)}
                className="absolute inset-0 flex flex-col items-center justify-center p-6 will-change-[opacity,transform]"
                style={{
                  opacity: i === 0 ? 1 : 0,
                  transition: 'opacity 0.6s cubic-bezier(0.4,0,0.2,1)',
                }}
              >
                <img
                  src={s.img}
                  alt={s.title}
                  className="max-h-[76%] max-w-full rounded-lg object-contain"
                />
                <div className="mt-4 text-xs font-mono text-muted text-center">{s.caption}</div>
              </div>
            ))}
            <div className="absolute top-4 right-4 font-mono text-xs text-olive2/70">{step.n}/04</div>
          </div>
        </div>
      </div>
    </section>
  )
}
