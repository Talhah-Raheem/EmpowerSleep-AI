/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Custom colors for EmpowerSleep branding
        primary: {
          50: '#f0f5ff',
          100: '#e0ebff',
          500: '#4f6bed',
          600: '#3d5bd9',
          700: '#2d4ac4',
        },
        sleep: {
          light: '#e8eeff',
          DEFAULT: '#6366f1',
          dark: '#312e81',
        },
      },
    },
  },
  plugins: [],
}
