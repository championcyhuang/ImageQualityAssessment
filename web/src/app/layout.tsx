import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "图像质量评估",
  description: "Image Quality Assessment Web Admin",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
