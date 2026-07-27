import { useEffect } from 'react'
import Lenis from 'lenis'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

import Nav from './components/Nav.jsx'
import Hero from './components/Hero.jsx'
import LiveDemo from './components/LiveDemo.jsx'
import HowItWorks from './components/HowItWorks.jsx'
import Highlights from './components/Highlights.jsx'
import TechStack from './components/TechStack.jsx'
import FutureWork from './components/FutureWork.jsx'
import Footer from './components/Footer.jsx'

gsap.registerPlugin(ScrollTrigger)

export default function App() {
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.15,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    })
    lenis.on('scroll', ScrollTrigger.update)
    const raf = (time) => lenis.raf(time * 1000)
    gsap.ticker.add(raf)
    gsap.ticker.lagSmoothing(0)

    const ctx = gsap.context(() => {
      gsap.utils.toArray('.reveal').forEach((el) => {
        gsap.fromTo(
          el,
          { y: 44, autoAlpha: 0 },
          {
            y: 0,
            autoAlpha: 1,
            duration: 0.9,
            ease: 'power3.out',
            scrollTrigger: { trigger: el, start: 'top 88%' },
          },
        )
      })
    })

    // Recalculate ScrollTrigger positions whenever the page height changes
    // (images loading, live-demo results appearing). Without this, pinned/
    // progress triggers like "How it works" read stale offsets and skip steps.
    let rt
    const ro = new ResizeObserver(() => {
      clearTimeout(rt)
      rt = setTimeout(() => ScrollTrigger.refresh(), 200)
    })
    ro.observe(document.body)
    window.addEventListener('load', () => ScrollTrigger.refresh())

    return () => {
      gsap.ticker.remove(raf)
      lenis.destroy()
      ctx.revert()
      ro.disconnect()
      clearTimeout(rt)
    }
  }, [])

  return (
    <>
      <Nav />
      <main>
        <Hero />
        <LiveDemo />
        <HowItWorks />
        <Highlights />
        <TechStack />
        <FutureWork />
      </main>
      <Footer />
    </>
  )
}
