'use client';

import { useEffect, useRef } from 'react';

interface Star {
  x: number;
  y: number;
  size: number;
  duration: number;
  delay: number;
}

function generateStars(count: number): Star[] {
  return Array.from({ length: count }, () => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 2.5 + 0.5, // 0.5–3px
    duration: Math.random() * 3 + 2,  // 2–5s
    delay: Math.random() * 5,          // 0–5s offset
  }));
}

const STARS = generateStars(120);

export function StarField() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {STARS.map((star, i) => (
        <span
          key={i}
          className="star"
          style={{
            left: `${star.x}%`,
            top: `${star.y}%`,
            width: `${star.size}px`,
            height: `${star.size}px`,
            ['--duration' as string]: `${star.duration}s`,
            ['--delay' as string]: `${star.delay}s`,
          }}
        />
      ))}
    </div>
  );
}
