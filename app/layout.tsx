import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lumen Clip — Open Text-to-Video",
  description:
    "Generate short videos from text using an open-source model on a free Kaggle GPU.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: "#07070c",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="bg-ink text-zinc-100 antialiased">{children}</body>
    </html>
  );
}
