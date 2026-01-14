import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { useSyncExternalStore } from "react"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Empty subscribe function - we never need to re-subscribe
const emptySubscribe = () => () => {}

/**
 * Hook to detect if component has mounted on the client
 * Uses useSyncExternalStore for a clean SSR-safe implementation
 * Returns false during SSR, true after hydration
 */
export function useMounted() {
  return useSyncExternalStore(
    emptySubscribe,
    () => true,  // Client value
    () => false  // Server value
  )
}
