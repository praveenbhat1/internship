import SectionHead from './SectionHead.jsx'

const items = [
  {
    tag: 'Real-world video',
    title: 'From crops to CCTV',
    body: 'Add a person detector (YOLO) and tracking in front of the model, so it reads attributes directly from live camera footage — not just pre-cropped images.',
  },
  {
    tag: 'Retrieval',
    title: 'Search by attribute',
    body: 'Once every person is described, allow queries like "red top + backpack, facing front" — the person-retrieval use case that makes PAR genuinely useful.',
  },
  {
    tag: 'Deployment',
    title: 'Real-time on the edge',
    body: 'Distil or quantise the model so it runs fast on-device, and aggregate predictions across frames for stable, low-latency results.',
  },
  {
    tag: 'Model',
    title: 'Broader & fairer',
    body: 'Add more attributes, sharpen fine-grained textures (plaid, small accessories), and keep auditing for dataset bias — extending the responsible-AI design.',
  },
]

export default function FutureWork() {
  return (
    <section id="future" className="relative py-28 px-6 border-t border-line/60">
      <div className="max-w-6xl mx-auto">
        <SectionHead
          label="05 — Future work"
          title="Where this goes next."
          sub="The model is the hard part — and it's done. These are the steps that turn it from a validated research model into a deployable system."
        />
        <div className="grid sm:grid-cols-2 gap-4">
          {items.map((it) => (
            <div key={it.title} className="reveal card p-7">
              <div className="text-olive2 font-mono text-xs tracking-widest uppercase mb-3">
                {it.tag}
              </div>
              <div className="text-lg font-semibold text-cream mb-2">{it.title}</div>
              <p className="text-sm text-muted leading-relaxed">{it.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
