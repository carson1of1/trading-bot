"use client";

import { useState, useEffect, useCallback, useRef } from "react";

interface UsePollingOptions<T> {
  fetcher: () => Promise<T>;
  interval?: number;
  enabled?: boolean;
  onError?: (error: Error) => void;
  deps?: unknown[]; // Additional dependencies that trigger refetch
}

interface UsePollingResult<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  refetch: () => Promise<void>;
}

export function usePolling<T>({
  fetcher,
  interval = 5000,
  enabled = true,
  onError,
  deps = [],
}: UsePollingOptions<T>): UsePollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const result = await fetcher();
      setData(result);
      setError(null);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      onError?.(error);
    } finally {
      setIsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onError, ...deps]);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    // Initial fetch
    fetchData();

    // Set up polling
    intervalRef.current = setInterval(fetchData, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, interval, fetchData]);

  return { data, error, isLoading, refetch: fetchData };
}
