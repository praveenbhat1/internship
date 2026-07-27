import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { gsap } from 'gsap'

const OLIVE = 0xa8b14b
const OLIVE_LT = 0xd7e08a

// build one low-poly wireframe limb/part
function part(geo, x, y, z, rot = [0, 0, 0]) {
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({ color: OLIVE, wireframe: true, transparent: true, opacity: 0.55 }),
  )
  mesh.position.set(x, y, z)
  mesh.rotation.set(...rot)
  return mesh
}

export default function Hero() {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const contentRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.set(0, 0.2, 6.2)

    // ---- the person (stylized low-poly wireframe) ----
    const person = new THREE.Group()
    person.add(part(new THREE.SphereGeometry(0.34, 10, 8), 0, 2.05, 0)) // head
    person.add(part(new THREE.CapsuleGeometry(0.44, 0.95, 4, 10), 0, 1.1, 0)) // torso
    person.add(part(new THREE.CapsuleGeometry(0.14, 0.85, 3, 7), 0.62, 1.15, 0, [0, 0, 0.28])) // R arm
    person.add(part(new THREE.CapsuleGeometry(0.14, 0.85, 3, 7), -0.62, 1.15, 0, [0, 0, -0.28])) // L arm
    person.add(part(new THREE.CapsuleGeometry(0.17, 1.0, 3, 7), 0.22, -0.15, 0)) // R leg
    person.add(part(new THREE.CapsuleGeometry(0.17, 1.0, 3, 7), -0.22, -0.15, 0)) // L leg

    // ---- glowing attribute nodes (what the model detects) ----
    const nodeSpots = [
      [0, 2.05, 0.34], // head  -> Hat
      [0, 1.35, 0.46], // chest -> Sleeve
      [0.7, 1.0, 0.2], // side  -> Bag
      [0, 0.2, 0.42], // hips  -> Trousers
      [0.22, -1.15, 0.2], // foot -> Boots
    ]
    const nodes = []
    const lineMat = new THREE.LineBasicMaterial({ color: OLIVE, transparent: true, opacity: 0.35 })
    nodeSpots.forEach(([x, y, z]) => {
      const n = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.075, 0),
        new THREE.MeshBasicMaterial({ color: OLIVE_LT }),
      )
      n.position.set(x, y, z)
      person.add(n)
      nodes.push(n)
      // thin connector from body centerline to the node
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, y, 0),
        new THREE.Vector3(x, y, z),
      ])
      person.add(new THREE.LineSegments(g, lineMat))
    })

    person.position.y = -0.55
    scene.add(person)

    // ---- sweeping scan ring (detection sweep) ----
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.15, 0.012, 8, 60),
      new THREE.MeshBasicMaterial({
        color: OLIVE_LT,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
      }),
    )
    ring.rotation.x = Math.PI / 2
    scene.add(ring)

    // faint particle dust
    const pts = new Float32Array(180 * 3)
    for (let i = 0; i < pts.length; i++) pts[i] = (Math.random() - 0.5) * 9
    const pg = new THREE.BufferGeometry()
    pg.setAttribute('position', new THREE.BufferAttribute(pts, 3))
    const dust = new THREE.Points(
      pg,
      new THREE.PointsMaterial({ color: OLIVE, size: 0.03, transparent: true, opacity: 0.5 }),
    )
    scene.add(dust)

    let mx = 0
    const onMove = (e) => (mx = e.clientX / window.innerWidth - 0.5)
    window.addEventListener('mousemove', onMove)

    const resize = () => {
      const w = wrap.clientWidth
      const h = wrap.clientHeight
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(wrap)

    let frame
    const clock = new THREE.Clock()
    const animate = () => {
      const t = clock.getElapsedTime()
      person.rotation.y = t * 0.4 + mx * 0.8
      dust.rotation.y = t * 0.05
      // scan ring sweeps up and down the body
      const sweep = (Math.sin(t * 0.9) * 0.5 + 0.5) * 3.1 - 1.7
      ring.position.y = sweep
      ring.material.opacity = 0.35 + 0.45 * (0.5 + 0.5 * Math.sin(t * 0.9 + 1.5))
      // pulse the detected-attribute nodes
      nodes.forEach((n, i) => {
        const s = 1 + 0.35 * Math.sin(t * 2.4 + i)
        n.scale.setScalar(s)
      })
      renderer.render(scene, camera)
      frame = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('mousemove', onMove)
      ro.disconnect()
      renderer.dispose()
      scene.traverse((o) => {
        o.geometry?.dispose?.()
        o.material?.dispose?.()
      })
    }
  }, [])

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap
        .timeline({ defaults: { ease: 'power3.out' } })
        .from('.hero-line', { y: 40, autoAlpha: 0, duration: 1, stagger: 0.12, delay: 0.2 })
        .from('.hero-sub', { y: 24, autoAlpha: 0, duration: 0.9 }, '-=0.5')
        .from('.hero-cta', { y: 20, autoAlpha: 0, duration: 0.6, stagger: 0.08 }, '-=0.3')
    }, contentRef)
    return () => ctx.revert()
  }, [])

  return (
    <section
      id="top"
      className="relative min-h-screen flex items-center overflow-hidden pt-32 pb-20 md:pt-28"
    >
      <div className="absolute inset-0 grid-bg opacity-60" />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-ink" />

      <div className="relative max-w-6xl mx-auto px-6 w-full grid md:grid-cols-2 gap-10 items-center">
        {/* LEFT — copy */}
        <div ref={contentRef} className="text-center md:text-left">
          <div className="hero-sub section-label mb-5 text-[0.68rem] md:text-xs">
            Computer Vision · Internship Project 2026
          </div>
          <h1 className="text-[2.6rem] leading-[1.02] sm:text-5xl lg:text-[3.75rem] font-extrabold tracking-[-0.02em] text-cream">
            <span className="hero-line block">Pedestrian</span>
            <span className="hero-line block">Attribute</span>
            <span className="hero-line block gradient-text">Recognition</span>
          </h1>
          <p className="hero-sub mt-6 text-[0.95rem] md:text-base text-muted max-w-md mx-auto md:mx-0 leading-relaxed">
            Upload a photo of a person and the model reads 23 attributes — clothing, accessories,
            and viewpoint — showing exactly where it looks and how confident it is, then abstaining
            when it isn't sure.
          </p>

          {/* on-topic attribute chips */}
          <div className="hero-sub mt-6 flex flex-wrap gap-2 justify-center md:justify-start max-w-md mx-auto md:mx-0">
            {['Gender', 'Hat', 'Backpack', 'Glasses', 'Sleeve', 'Trousers', '+17 more'].map((t) => (
              <span
                key={t}
                className="text-xs font-mono px-2.5 py-1 rounded-full border border-line text-muted"
              >
                {t}
              </span>
            ))}
          </div>

          <div className="mt-8 flex flex-wrap items-center justify-center md:justify-start gap-3.5">
            <a
              href="#demo"
              className="hero-cta px-6 py-3 rounded-full bg-olive text-ink font-semibold hover:brightness-110 transition glow"
            >
              Try the live demo ↓
            </a>
            <a
              href="#results"
              className="hero-cta px-6 py-3 rounded-full border border-line text-muted hover:text-cream hover:border-olive/50 transition"
            >
              See results
            </a>
          </div>

          {/* stats strip */}
          <div className="hero-sub mt-10 grid grid-cols-3 gap-3 max-w-md mx-auto md:mx-0 border-t border-line pt-6">
            {[
              ['23', 'attributes'],
              ['~91%', 'mean accuracy'],
              ['77.8', 'cross-dataset mA'],
            ].map(([n, l]) => (
              <div key={l} className="text-center md:text-left">
                <div className="text-2xl font-bold gradient-text leading-none">{n}</div>
                <div className="text-[0.68rem] text-muted mt-1.5 uppercase tracking-wider">{l}</div>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT — 3D person being scanned */}
        <div ref={wrapRef} className="relative h-[380px] md:h-[560px]">
          <div className="absolute inset-0 [background:radial-gradient(circle_at_50%_45%,rgba(168,177,75,0.15),transparent_60%)]" />
          <canvas ref={canvasRef} className="relative w-full h-full" />
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 text-muted text-xs font-mono animate-pulse">
        scroll
      </div>
    </section>
  )
}
