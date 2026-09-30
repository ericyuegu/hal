import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'HAL',
  description: 'Queue for a direct Slippi set against a HAL policy.',
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <link
          rel="preload"
          href="/fonts/schibsted-grotesk-latin-wght.woff2"
          as="font"
          type="font/woff2"
          crossOrigin=""
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
