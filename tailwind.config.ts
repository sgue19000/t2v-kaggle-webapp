import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#07070c",
        panel: "#111118",
        line: "#232333",
        accent: "#7c5cff",
        mint: "#3ee0b2",
      },
      boxShadow: {
        glow: "0 0 40px rgba(124, 92, 255, 0.25)",
      },
    },
  },
  plugins: [],
};

export default config;
