/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#18202f",
        muted: "#667085",
        panel: "#ffffff",
        line: "#d9dee8",
        accent: "#12655f",
      },
    },
  },
  plugins: [],
};
