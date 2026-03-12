'use client'

import { useEffect } from 'react'
import { initPostHog } from '@/lib/analytics'

/**
 * Checks localStorage on mount and initializes PostHog if the user
 * previously accepted analytics. This covers returning visitors —
 * the CookieBanner handles first-time visitors.
 */
export function AnalyticsProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    if (localStorage.getItem('analytics_consent') === 'accepted') {
      initPostHog()
    }
  }, [])

  return <>{children}</>
}
