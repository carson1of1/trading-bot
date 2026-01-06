"use client";

import { useEffect, useState, useRef } from "react";

interface CountUpProps {
  end: number;
  start?: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  separator?: string;
  className?: string;
}

export function CountUp({
  end,
  start = 0,
  duration = 1.5,
  decimals = 0,
  prefix = "",
  suffix = "",
  separator = ",",
  className = "",
}: CountUpProps) {
  const [count, setCount] = useState(start);
  const countRef = useRef(start);
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    countRef.current = start;
    startTimeRef.current = null;

    const animate = (timestamp: number) => {
      if (!startTimeRef.current) {
        startTimeRef.current = timestamp;
      }

      const progress = Math.min(
        (timestamp - startTimeRef.current) / (duration * 1000),
        1
      );

      // Easing function - easeOutExpo for satisfying deceleration
      const easeOutExpo = 1 - Math.pow(2, -10 * progress);
      const currentValue = start + (end - start) * easeOutExpo;

      countRef.current = currentValue;
      setCount(currentValue);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        setCount(end);
      }
    };

    const animationFrame = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationFrame);
  }, [end, start, duration]);

  const formatNumber = (num: number): string => {
    const fixed = num.toFixed(decimals);
    const parts = fixed.split(".");
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, separator);
    return parts.join(".");
  };

  return (
    <span className={`number-animate ${className}`}>
      {prefix}
      {formatNumber(count)}
      {suffix}
    </span>
  );
}
