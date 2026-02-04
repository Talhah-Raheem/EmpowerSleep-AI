/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // EmpowerSleep brand colors (derived from logo)
        empower: {
          50: '#f0f5f7',
          100: '#e1eaee',
          200: '#c3d5dd',
          300: '#9ab8c5',
          400: '#7a9dac',
          500: '#5b7a8a',  // Logo primary color
          600: '#4a6573',
          700: '#3d525e',
          800: '#344550',
          900: '#2d3b44',
        },
        // Accent teal for CTAs
        accent: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
        },
      },
      fontFamily: {
        sans: ['Open Sans', 'system-ui', 'sans-serif'],
        heading: ['Poppins', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
