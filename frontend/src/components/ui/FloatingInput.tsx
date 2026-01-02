"use client";

import { InputHTMLAttributes, forwardRef } from "react";

interface FloatingInputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
}

export const FloatingInput = forwardRef<HTMLInputElement, FloatingInputProps>(
  ({ label, className = "", id, ...props }, ref) => {
    const inputId = id || label.toLowerCase().replace(/\s+/g, "-");

    return (
      <div className="floating-input-wrapper">
        <input
          ref={ref}
          id={inputId}
          className={`floating-input ${className}`}
          placeholder=" "
          {...props}
        />
        <label htmlFor={inputId} className="floating-label">
          {label}
        </label>
      </div>
    );
  }
);

FloatingInput.displayName = "FloatingInput";
