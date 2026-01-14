"use client";

import { useMounted } from "@/lib/utils";

// Pre-generated static column data to avoid Math.random() during render
// Values are pseudo-random but deterministic for SSR consistency
const COLUMNS = [
  { id: 0, left: "1.2%", delay: 3, duration: 18, opacity: 0.04, char: "0" },
  { id: 1, left: "4.5%", delay: 8, duration: 25, opacity: 0.06, char: "1" },
  { id: 2, left: "7.8%", delay: 12, duration: 20, opacity: 0.05, char: "ア" },
  { id: 3, left: "11.1%", delay: 1, duration: 32, opacity: 0.03, char: "イ" },
  { id: 4, left: "14.4%", delay: 15, duration: 22, opacity: 0.07, char: "0" },
  { id: 5, left: "17.7%", delay: 6, duration: 28, opacity: 0.04, char: "ウ" },
  { id: 6, left: "21.0%", delay: 19, duration: 17, opacity: 0.08, char: "1" },
  { id: 7, left: "24.3%", delay: 4, duration: 30, opacity: 0.05, char: "エ" },
  { id: 8, left: "27.6%", delay: 11, duration: 24, opacity: 0.06, char: "オ" },
  { id: 9, left: "30.9%", delay: 17, duration: 19, opacity: 0.04, char: "0" },
  { id: 10, left: "34.2%", delay: 2, duration: 33, opacity: 0.07, char: "カ" },
  { id: 11, left: "37.5%", delay: 9, duration: 21, opacity: 0.05, char: "1" },
  { id: 12, left: "40.8%", delay: 14, duration: 27, opacity: 0.03, char: "キ" },
  { id: 13, left: "44.1%", delay: 7, duration: 16, opacity: 0.06, char: "ク" },
  { id: 14, left: "47.4%", delay: 18, duration: 29, opacity: 0.04, char: "0" },
  { id: 15, left: "50.7%", delay: 5, duration: 23, opacity: 0.08, char: "ケ" },
  { id: 16, left: "54.0%", delay: 13, duration: 31, opacity: 0.05, char: "1" },
  { id: 17, left: "57.3%", delay: 10, duration: 18, opacity: 0.07, char: "コ" },
  { id: 18, left: "60.6%", delay: 16, duration: 26, opacity: 0.04, char: "サ" },
  { id: 19, left: "63.9%", delay: 3, duration: 20, opacity: 0.06, char: "0" },
  { id: 20, left: "67.2%", delay: 12, duration: 34, opacity: 0.03, char: "シ" },
  { id: 21, left: "70.5%", delay: 8, duration: 22, opacity: 0.05, char: "1" },
  { id: 22, left: "73.8%", delay: 19, duration: 28, opacity: 0.07, char: "ス" },
  { id: 23, left: "77.1%", delay: 1, duration: 17, opacity: 0.04, char: "セ" },
  { id: 24, left: "80.4%", delay: 15, duration: 25, opacity: 0.06, char: "0" },
  { id: 25, left: "83.7%", delay: 6, duration: 30, opacity: 0.08, char: "ソ" },
  { id: 26, left: "87.0%", delay: 11, duration: 19, opacity: 0.05, char: "1" },
  { id: 27, left: "90.3%", delay: 17, duration: 32, opacity: 0.04, char: "タ" },
  { id: 28, left: "93.6%", delay: 4, duration: 24, opacity: 0.07, char: "チ" },
  { id: 29, left: "96.9%", delay: 14, duration: 21, opacity: 0.05, char: "ツ" },
];

export function MatrixRain() {
  const mounted = useMounted();

  if (!mounted) return null;

  return (
    <div className="matrix-container" aria-hidden="true">
      {COLUMNS.map((col) => (
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
