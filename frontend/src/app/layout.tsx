import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "F1 Oracle Lab",
  description: "AI-Powered Formula 1 Race Prediction Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
