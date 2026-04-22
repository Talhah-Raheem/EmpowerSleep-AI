/** @type {import('next').NextConfig} */
const isDev = process.env.NODE_ENV === 'development';

const nextConfig = {
  reactStrictMode: true,

  // Remove the x-powered-by: Next.js header to avoid fingerprinting
  poweredByHeader: false,

  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          // Prevent clickjacking — disallow embedding in iframes
          { key: 'X-Frame-Options', value: 'DENY' },

          // Prevent MIME-type sniffing
          { key: 'X-Content-Type-Options', value: 'nosniff' },

          // Force HTTPS for 1 year on all subdomains
          { key: 'Strict-Transport-Security', value: 'max-age=31536000; includeSubDomains' },

          // Only send the origin (no path) in the Referer header
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },

          // Restrict browser API access
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },

          // Content Security Policy
          // Notes:
          //   script-src 'unsafe-inline' — required by Next.js App Router hydration
          //   script-src 'unsafe-eval'   — required by webpack dev server (eval source maps); dev only
          //   style-src  'unsafe-inline' — required by Tailwind CSS inline styles
          //   img-src    blob: data:     — required for file upload previews & lightbox
          //   font-src   fonts.gstatic   — Google Fonts file delivery
          //   connect-src covers both the Railway API URL and localhost dev server
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              isDev ? "script-src 'self' 'unsafe-inline' 'unsafe-eval'" : "script-src 'self' 'unsafe-inline'",
              "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
              "font-src 'self' https://fonts.gstatic.com",
              "img-src 'self' data: blob:",
              "connect-src 'self' http://localhost:8000 https://api.empowersleep.ai https://us.i.posthog.com https://app.posthog.com",
              "frame-ancestors 'none'",
            ].join('; '),
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
