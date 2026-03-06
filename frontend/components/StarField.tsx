'use client';

import { useEffect, useState } from 'react';

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

  useEffect(() => {
    const schedule = () => {
      const delay = 12000 + Math.random() * 8000; // every 12–20s
      return setTimeout(() => {
        setShootingStar({
          x: Math.random() * 60,  // start in left 60% of screen
          y: Math.random() * 40,  // start in top 40% of screen
          key: Date.now(),
        });
        setTimeout(() => {
          setShootingStar(null);
          schedule();
        }, 1000);
      }, delay);
    };

    const timer = schedule();
    return () => clearTimeout(timer);
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
