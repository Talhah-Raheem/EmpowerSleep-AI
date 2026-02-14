import type { Metadata } from 'next'
import { Open_Sans, Poppins } from 'next/font/google'
import { ThemeProvider } from 'next-themes'
import './globals.css'

const openSans = Open_Sans({
  subsets: ['latin'],
  variable: '--font-open-sans',
  display: 'swap',
})

const poppins = Poppins({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-poppins',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'EmpowerSleep - Sleep Education Assistant',
  description: 'Sleep care, simplified. Get answers to your sleep questions with AI-powered education grounded in expert content.',
  icons: {
    icon: '/favicon.svg',
    apple: '/empower_sleep_logo.jpeg',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${openSans.variable} ${poppins.variable} font-sans`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
