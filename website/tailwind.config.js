/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0b0c07', // near-black, faint olive tint
        panel: '#14150d',
        line: '#2b2d1c',
        olive: '#a8b14b', // primary accent
        olive2: '#c8d17a', // light olive
        olived: '#6f763a', // deep olive
        cream: '#f3f2e9', // off-white text
        muted: '#9a9b83',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
