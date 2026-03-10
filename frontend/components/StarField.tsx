'use client';

import { useEffect, useRef, useState } from 'react';

interface Star {
  x: number;
  y: number;
  size: number;
  duration: number;
  delay: number;
}

interface ShootingStar {
  x: number;
  y: number;
  key: number;
}

function generateStars(count: number): Star[] {
  return Array.from({ length: count }, () => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 2.5 + 0.5,
    duration: Math.random() * 3 + 2,
    delay: Math.random() * 5,
  }));
}

const STARS = generateStars(120);

export function StarField() {
  const [shootingStar, setShootingStar] = useState<ShootingStar | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    let outerTimer: ReturnType<typeof setTimeout>;
    let innerTimer: ReturnType<typeof setTimeout>;

    const schedule = () => {
      const delay = 12000 + Math.random() * 8000;
      outerTimer = setTimeout(() => {
        if (!mountedRef.current) return;
        setShootingStar({
          x: Math.random() * 60,
          y: Math.random() * 40,
          key: Date.now(),
        });
        innerTimer = setTimeout(() => {
          if (!mountedRef.current) return;
          setShootingStar(null);
          schedule();
        }, 1000);
      }, delay);
    };

    schedule();
    return () => {
      mountedRef.current = false;
      clearTimeout(outerTimer);
      clearTimeout(innerTimer);
    };
  }, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Twinkling stars */}
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

      {/* Shooting star */}
      {shootingStar && (
        <span
          key={shootingStar.key}
          className="shooting-star"
          style={{
            left: `${shootingStar.x}%`,
            top: `${shootingStar.y}%`,
          }}
        />
      )}
    </div>
  );
}
