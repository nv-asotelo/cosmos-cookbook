import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "nvidia/cosmos-predict1-5b Model by NVIDIA | NVIDIA NIM",
  description: "World foundation model for generating future video from image or video conditioning."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="nv-dark">
      <body>{children}</body>
    </html>
  );
}
