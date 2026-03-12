import posthog from 'posthog-js'

let initialized = false

/**
 * Initialize PostHog. Only runs client-side and only once.
 * Call this after the user gives consent.
 */
export function initPostHog() {
  if (typeof window === 'undefined' || initialized) return
  const key = process.env.NEXT_PUBLIC_POSTHOG_KEY
  if (!key) return

  posthog.init(key, {
    api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://us.i.posthog.com',
    capture_pageview: true,
    persistence: 'localStorage',
  })
  initialized = true
}

/**
 * Track a custom event. Silently does nothing if PostHog isn't initialized
 * (i.e. user declined or hasn't consented yet).
 */
export function trackEvent(event: string, properties?: Record<string, unknown>) {
  if (!initialized) return
  posthog.capture(event, properties)
}
