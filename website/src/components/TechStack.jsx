import SectionHead from './SectionHead.jsx'

const stack = [
  ['SigLIP-2', 'A strong pretrained vision-language backbone, used frozen as the feature extractor.'],
  ['OCFR', 'Orientation-conditioned refinement — reasons differently for front / side / back views.'],
  ['CMAA', 'Cross-attribute attention — a separate heatmap per attribute (the maps in the demo).'],
  ['DACG', 'Attribute-correlation graph — related attributes support each other.'],
  ['CCLoss', 'Correlation-consistency loss + mutual exclusion (one viewpoint, one sleeve length).'],
  ['Responsible AI', 'Gender is only reported when the face is visible and ≥85% confident — otherwise it abstains.'],
]

const datasets = [
  ['PA-100K', 'Training + in-domain test — 100k people, 23 attributes.'],
  ['PETA', 'Cross-dataset test — a different distribution, no retraining.'],
]

export default function TechStack() {
  return (
    <section id="tech" className="relative py-28 px-6 border-t border-line/60">
      <div className="max-w-6xl mx-auto">
        <SectionHead
          label="03 — What we used"
          title="A frozen backbone, with light modules on top."
          sub="Rather than fine-tuning a giant model, the design keeps SigLIP frozen and adds small, interpretable modules — each earning its place."
        />

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
          {stack.map(([name, desc], i) => (
            <div key={name} className="reveal card p-6">
              <div className="font-mono text-xs text-olive mb-1">0{i + 1}</div>
              <div className="font-semibold text-cream mb-2">{name}</div>
              <p className="text-[13px] leading-relaxed text-muted">{desc}</p>
            </div>
          ))}
        </div>

        <div className="grid sm:grid-cols-2 gap-4">
          {datasets.map(([name, desc]) => (
            <div key={name} className="reveal card p-6 flex gap-4">
              <div className="mt-1 h-2.5 w-2.5 rounded-full bg-olive shrink-0 shadow-[0_0_12px_#a8b14b]" />
              <div>
                <div className="font-semibold text-cream mb-1">{name}</div>
                <p className="text-sm text-muted leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
