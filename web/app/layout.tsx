import type { Metadata, Viewport } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Guitar Coach AR",
  description: "Record a take, get graded against the exercise, replay with an AR overlay.",
};
export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="top">
          <Link href="/" className="brand">🎸 Guitar Coach AR</Link>
          <Link href="/practice/1">Live vision</Link>
          <Link href="/sessions">Past takes</Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
