"use client";

export function AuroraBackground() {
  return (
    <div className="aurora-container" aria-hidden="true">
      {/* Multiple aurora layers for depth */}
      <div className="aurora aurora-1" />
      <div className="aurora aurora-2" />
      <div className="aurora aurora-3" />
    </div>
  );
}
