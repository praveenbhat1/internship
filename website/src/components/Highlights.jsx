import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import SectionHead from './SectionHead.jsx'

const stats = [
  ['90.8', 'mA — in-domain (PA-100K)'],
  ['77.8', 'mA — cross-dataset (PETA)'],
  ['23', 'attributes'],
]

export default function Highlights() {
  const ref = useRef(null)
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.utils.toArray('.counter').forEach((el) => {
        const end = parseFloat(el.dataset.end)
        const dec = end % 1 === 0 ? 0 : 1
        const obj = { v: 0 }
        gsap.to(obj, {
          v: end,
          duration: 1.6,
          ease: 'power2.out',
          scrollTrigger: { trigger: el, start: 'top 90%' },
          onUpdate: () => (el.textContent = obj.v.toFixed(dec)),
        })
      })
    }, ref)
    return () => ctx.revert()
  }, [])

  return (
    <section id="results" ref={ref} className="relative py-28 px-6 border-t border-line/60">
      <div className="max-w-6xl mx-auto">
        <SectionHead
          label="03 — Results"
          title="Accurate in-domain, and it generalizes."
          sub="The headline numbers. The model was trained on PA-100K and then tested — with no retraining — on PETA, a completely different dataset, to prove it learned general features."
        />

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          {stats.map(([n, l]) => (
            <div key={l} className="reveal card p-7 text-center">
              <div className="counter text-5xl font-bold gradient-text" data-end={n}>
                0
              </div>
              <div className="text-xs text-muted mt-2 uppercase tracking-widest">{l}</div>
            </div>
          ))}
        </div>

        <div className="reveal card p-6 md:p-8 overflow-x-auto">
          <div className="text-sm font-semibold text-cream mb-4">
            Cross-dataset validation — PA-100K → PETA (14,437 images, zero retraining)
          </div>
          <table className="w-full text-sm min-w-[480px]">
            <thead>
              <tr className="text-muted text-left border-b border-line">
                <th className="py-2 pr-4 font-medium"></th>
                <th className="py-2 px-3 font-medium">mA</th>
                <th className="py-2 px-3 font-medium">Accuracy</th>
                <th className="py-2 px-3 font-medium">Precision</th>
                <th className="py-2 px-3 font-medium">F1</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line/50">
                <td className="py-2.5 pr-4 text-muted">In-domain (PA-100K)</td>
                <td className="py-2.5 px-3">91.1</td>
                <td className="py-2.5 px-3">72.7</td>
                <td className="py-2.5 px-3">76.4</td>
                <td className="py-2.5 px-3">83.4</td>
              </tr>
              <tr className="text-cream font-semibold">
                <td className="py-2.5 pr-4">Cross-domain (PETA)</td>
                <td className="py-2.5 px-3 gradient-text">77.8</td>
                <td className="py-2.5 px-3">57.0</td>
                <td className="py-2.5 px-3">61.3</td>
                <td className="py-2.5 px-3">72.3</td>
              </tr>
            </tbody>
          </table>
          <p className="text-xs text-muted mt-4">
            The ~13-point drop is the expected domain gap — but 77.8 mA is well above chance, evidence
            the model learned general pedestrian features, not dataset quirks.
          </p>
        </div>
      </div>
    </section>
  )
}
