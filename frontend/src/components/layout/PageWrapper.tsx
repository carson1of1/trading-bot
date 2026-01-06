"use client";

import { Dock } from "./Dock";

interface PageWrapperProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
}

export function PageWrapper({ children, title, subtitle }: PageWrapperProps) {
  return (
    <div className="min-h-screen pb-24 page-transition">
      {/* Page header */}
      {title && (
        <header className="px-6 pt-8 pb-6">
          <div className="max-w-7xl mx-auto">
            <h1 className="text-2xl font-semibold text-white animate-fade-in">
              {title}
            </h1>
            {subtitle && (
              <p className="mt-1 text-sm text-text-secondary animate-fade-in stagger-1">
                {subtitle}
              </p>
            )}
          </div>
        </header>
      )}

      {/* Page content */}
      <main className="px-6">
        <div className="max-w-7xl mx-auto">{children}</div>
      </main>

      {/* Fixed dock navigation */}
      <Dock />
    </div>
  );
}
