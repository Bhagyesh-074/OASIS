/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        oasis: {
          bg: "#0B0F19",
          card: "#131B2E",
          border: "#1E293B",
          primary: "#38BDF8",
          accent: "#818CF8",
          success: "#34D399",
          warning: "#FBBF24",
          danger: "#F87171",
        }
      }
    },
  },
  plugins: [],
}
