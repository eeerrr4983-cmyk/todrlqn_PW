/**
 * Performance Optimization: API Response Caching
 * 
 * Implements in-memory caching for API responses to reduce
 * redundant API calls and improve performance.
 */

interface CacheEntry<T> {
  data: T
  timestamp: number
  expiresAt: number
}

interface CacheOptions {
  ttl?: number // Time to live in milliseconds (default: 5 minutes)
  maxSize?: number // Maximum number of entries (default: 100)
}

class APICache {
  private cache: Map<string, CacheEntry<any>>
  private maxSize: number
  private defaultTTL: number

  constructor(options: CacheOptions = {}) {
    this.cache = new Map()
    this.maxSize = options.maxSize || 100
    this.defaultTTL = options.ttl || 5 * 60 * 1000 // 5 minutes
  }

  /**
   * Generate cache key from URL and options
   */
  private generateKey(url: string, options?: any): string {
    const optionsStr = options ? JSON.stringify(options) : ''
    return `${url}:${optionsStr}`
  }

  /**
   * Get cached data if available and not expired
   */
  get<T>(url: string, options?: any): T | null {
    const key = this.generateKey(url, options)
    const entry = this.cache.get(key)

    if (!entry) {
      return null
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key)
      return null
    }

    return entry.data as T
  }

  /**
   * Set cache entry with custom TTL
   */
  set<T>(url: string, data: T, options?: any, customTTL?: number): void {
    const key = this.generateKey(url, options)
    const ttl = customTTL || this.defaultTTL

    // Check cache size limit
    if (this.cache.size >= this.maxSize) {
      // Remove oldest entry
      const firstKey = this.cache.keys().next().value
      if (firstKey) {
        this.cache.delete(firstKey)
      }
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      expiresAt: Date.now() + ttl,
    })
  }

  /**
   * Check if cache has valid entry
   */
  has(url: string, options?: any): boolean {
    return this.get(url, options) !== null
  }

  /**
   * Invalidate specific cache entry
   */
  invalidate(url: string, options?: any): void {
    const key = this.generateKey(url, options)
    this.cache.delete(key)
  }

  /**
   * Invalidate all cache entries matching pattern
   */
  invalidatePattern(pattern: RegExp): void {
    for (const key of this.cache.keys()) {
      if (pattern.test(key)) {
        this.cache.delete(key)
      }
    }
  }

  /**
   * Clear all cache
   */
  clear(): void {
    this.cache.clear()
  }

  /**
   * Get cache statistics
   */
  getStats() {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      entries: Array.from(this.cache.keys()),
    }
  }

  /**
   * Clean up expired entries
   */
  cleanup(): void {
    const now = Date.now()
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key)
      }
    }
  }
}

// Global cache instances for different purposes
export const analysisCache = new APICache({ ttl: 10 * 60 * 1000, maxSize: 50 }) // 10 minutes
export const projectCache = new APICache({ ttl: 30 * 60 * 1000, maxSize: 30 }) // 30 minutes
export const universityCache = new APICache({ ttl: 60 * 60 * 1000, maxSize: 20 }) // 1 hour

/**
 * Wrapper function for cached fetch
 * 
 * Usage:
 * const data = await cachedFetch('/api/analyze', { body: ... }, analysisCache, 600000)
 */
export async function cachedFetch<T>(
  url: string,
  options?: RequestInit,
  cache: APICache = analysisCache,
  ttl?: number
): Promise<T> {
  // Check cache first
  const cached = cache.get<T>(url, options?.body)
  if (cached) {
    console.log(`[Cache HIT] ${url}`)
    return cached
  }

  console.log(`[Cache MISS] ${url}`)

  // Fetch from API
  const response = await fetch(url, options)
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }

  const data = await response.json() as T

  // Store in cache
  cache.set(url, data, options?.body, ttl)

  return data
}

/**
 * Setup automatic cache cleanup interval
 */
export function startCacheCleanup(intervalMs: number = 60000) {
  setInterval(() => {
    analysisCache.cleanup()
    projectCache.cleanup()
    universityCache.cleanup()
  }, intervalMs)
}

// Start cleanup on module load (runs every minute)
if (typeof window !== 'undefined') {
  startCacheCleanup()
}
