interface EmpowerLogoProps {
  className?: string;
  animate?: boolean;
}

/**
 * Inline SVG of the EmpowerSleep logo (top chevron + bottom oval).
 * No white background. Inherits text color via currentColor.
 *
 * Set animate={true} for the breathing + glow effect used in the loader.
 */
export function EmpowerLogo({ className = 'h-10 w-10', animate = false }: EmpowerLogoProps) {
  return (
    <svg
      viewBox="0 0 100 120"
      className={`${className} ${animate ? 'overflow-visible' : ''}`}
      aria-hidden="true"
    >
      <path
        className={animate ? 'logo-top-breathe' : undefined}
        d="M50 5 C36 5 12 30 12 46 C12 54 22 54 34 48 C42 43 47 37 50 32 C53 37 58 43 66 48 C78 54 88 54 88 46 C88 30 64 5 50 5Z"
        fill="currentColor"
      />
      <path
        className={animate ? 'logo-bottom-breathe' : undefined}
        d="M50 55 C30 55 16 72 16 86 C16 102 30 115 50 115 C70 115 84 102 84 86 C84 72 70 55 50 55Z"
        fill="currentColor"
      />
    </svg>
  );
}
