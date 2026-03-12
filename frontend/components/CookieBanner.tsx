'use client'

import { useState, useEffect } from 'react'
import { initPostHog } from '@/lib/analytics'

/**
 * Cookie consent banner shown to first-time visitors.
 * - Accept  → saves 'accepted' to localStorage, initializes PostHog, hides banner
 * - Decline → saves 'declined' to localStorage, hides banner (PostHog never loads)
 * - Returns null if the user has already made a choice (returning visitor case
 *   is handled by AnalyticsProvider in providers.tsx).
 */
export function CookieBanner() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // Only show if no decision has been stored yet
    if (!localStorage.getItem('analytics_consent')) {
      setVisible(true)
    }
  }, [])

  if (!visible) return null

  const handleAccept = () => {
    localStorage.setItem('analytics_consent', 'accepted')
    initPostHog()
    setVisible(false)
  }

  const handleDecline = () => {
    localStorage.setItem('analytics_consent', 'declined')
    setVisible(false)
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 flex justify-center animate-fade-in">
      <div className="w-full max-w-2xl bg-white/90 dark:bg-black/70 backdrop-blur-md border border-white/30 dark:border-white/10 rounded-2xl shadow-lg px-5 py-4 flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <p className="text-sm text-empower-700 dark:text-empower-200 flex-1">
          We use analytics to understand how people use EmpowerSleep and improve the experience. No personal data is sold or shared.{' '}
          <a
            href="https://www.empowersleep.com/privacy-policy"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-empower-600 dark:hover:text-empower-300"
          >
            Privacy Policy
          </a>
        </p>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={handleDecline}
            className="px-4 py-2 text-sm text-empower-500 dark:text-empower-400 hover:text-empower-700 dark:hover:text-empower-200 transition-colors"
          >
            Decline
          </button>
          <button
            onClick={handleAccept}
            className="px-4 py-2 text-sm bg-empower-500 hover:bg-empower-600 text-white rounded-full font-medium transition-colors shadow-sm"
          >
            Accept
          </button>
        </div>
      </div>
    </div>
  )
}
