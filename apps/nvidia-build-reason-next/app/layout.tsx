import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "cosmos-reason2-8b Model by NVIDIA | NVIDIA NIM",
  description:
    "Vision language model that excels in understanding the physical world using structured reasoning on videos or images."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="nv-dark">
      <body>{children}</body>
    </html>
  );
}
