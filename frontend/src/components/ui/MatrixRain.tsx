"use client";

import { useMemo } from "react";

export function MatrixRain() {
  // Generate columns with random properties - memoized so it doesn't regenerate
  const columns = useMemo(() => {
    return Array.from({ length: 30 }, (_, i) => ({
      id: i,
      left: `${(i / 30) * 100 + Math.random() * 3}%`,
      delay: Math.random() * 20,
      duration: 15 + Math.random() * 20,
      opacity: 0.03 + Math.random() * 0.06,
      char: getRandomChar(),
    }));
  }, []);

  return (
    <div className="matrix-container" aria-hidden="true">
      {columns.map((col) => (
        <div
          key={col.id}
          className="matrix-column"
          style={{
            left: col.left,
            animationDelay: `${col.delay}s`,
            animationDuration: `${col.duration}s`,
            opacity: col.opacity,
          }}
        >
          {col.char}
        </div>
      ))}
    </div>
  );
}

function getRandomChar(): string {
  const chars = "01アイウエオカキクケコサシスセソタチツテト";
  return chars[Math.floor(Math.random() * chars.length)];
}
