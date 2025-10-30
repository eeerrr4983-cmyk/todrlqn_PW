/**
 * Performance Monitoring Utilities
 * 
 * Track and measure application performance metrics
 */

export interface PerformanceMetrics {
  name: string
  duration: number
  timestamp: number
}

class PerformanceMonitor {
  private metrics: PerformanceMetrics[] = []
  private marks: Map<string, number> = new Map()

  /**
   * Start measuring a performance metric
   */
  start(name: string): void {
    this.marks.set(name, performance.now())
  }

  /**
   * End measuring and record the metric
   */
  end(name: string): number | null {
    const startTime = this.marks.get(name)
    if (!startTime) {
      console.warn(`[Performance] No start mark found for "${name}"`)
      return null
    }

    const duration = performance.now() - startTime
    this.metrics.push({
      name,
      duration,
      timestamp: Date.now(),
    })

    this.marks.delete(name)

    if (process.env.NODE_ENV === 'development') {
      console.log(`[Performance] ${name}: ${duration.toFixed(2)}ms`)
    }

    return duration
  }

  /**
   * Measure a function execution time
   */
  async measure<T>(name: string, fn: () => T | Promise<T>): Promise<T> {
    this.start(name)
    try {
      const result = await fn()
      this.end(name)
      return result
    } catch (error) {
      this.end(name)
      throw error
    }
  }

  /**
   * Get all recorded metrics
   */
  getMetrics(): PerformanceMetrics[] {
    return [...this.metrics]
  }

  /**
   * Get metrics by name
   */
  getMetricsByName(name: string): PerformanceMetrics[] {
    return this.metrics.filter(m => m.name === name)
  }

  /**
   * Get average duration for a metric
   */
  getAverageDuration(name: string): number {
    const filtered = this.getMetricsByName(name)
    if (filtered.length === 0) return 0

    const total = filtered.reduce((sum, m) => sum + m.duration, 0)
    return total / filtered.length
  }

  /**
   * Clear all metrics
   */
  clear(): void {
    this.metrics = []
    this.marks.clear()
  }

  /**
   * Export metrics as JSON
   */
  export(): string {
    return JSON.stringify(this.metrics, null, 2)
  }
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor()

/**
 * Web Vitals Monitoring
 */
export interface WebVitalsMetric {
  name: 'CLS' | 'FID' | 'FCP' | 'LCP' | 'TTFB' | 'INP'
  value: number
  rating: 'good' | 'needs-improvement' | 'poor'
  delta: number
  id: string
}

export function reportWebVitals(metric: WebVitalsMetric) {
  if (process.env.NODE_ENV === 'development') {
    console.log(`[Web Vitals] ${metric.name}:`, {
      value: metric.value,
      rating: metric.rating,
      delta: metric.delta,
    })
  }

  // Send to analytics
  if (typeof window !== 'undefined' && (window as any).gtag) {
    (window as any).gtag('event', metric.name, {
      value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
      event_category: 'Web Vitals',
      event_label: metric.id,
      non_interaction: true,
    })
  }
}

/**
 * Resource timing analysis
 */
export function analyzeResourceTiming() {
  if (typeof window === 'undefined' || !window.performance) {
    return null
  }

  const entries = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
  
  const analysis = {
    total: entries.length,
    byType: {} as Record<string, number>,
    slow: [] as Array<{ name: string; duration: number; type: string }>,
  }

  entries.forEach(entry => {
    // Count by type
    const type = entry.initiatorType
    analysis.byType[type] = (analysis.byType[type] || 0) + 1

    // Track slow resources (>500ms)
    if (entry.duration > 500) {
      analysis.slow.push({
        name: entry.name,
        duration: entry.duration,
        type: entry.initiatorType,
      })
    }
  })

  return analysis
}

/**
 * Memory usage monitoring
 */
export function getMemoryUsage() {
  if (typeof window === 'undefined') return null

  const memory = (performance as any).memory
  if (!memory) return null

  return {
    usedJSHeapSize: memory.usedJSHeapSize,
    totalJSHeapSize: memory.totalJSHeapSize,
    jsHeapSizeLimit: memory.jsHeapSizeLimit,
    usagePercent: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100,
  }
}

/**
 * Bundle size tracker
 */
export function trackBundleSize() {
  if (typeof window === 'undefined') return

  const scripts = document.querySelectorAll('script[src]')
  const styles = document.querySelectorAll('link[rel="stylesheet"]')

  let totalSize = 0
  const resources: Array<{ url: string; size: number; type: string }> = []

  const entries = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
  
  entries.forEach(entry => {
    if (entry.initiatorType === 'script' || entry.initiatorType === 'link') {
      const size = entry.transferSize || entry.encodedBodySize || 0
      totalSize += size

      resources.push({
        url: entry.name,
        size,
        type: entry.initiatorType,
      })
    }
  })

  return {
    totalSize,
    totalSizeMB: (totalSize / 1024 / 1024).toFixed(2),
    scripts: resources.filter(r => r.type === 'script'),
    styles: resources.filter(r => r.type === 'link'),
  }
}

/**
 * First Contentful Paint (FCP)
 */
export function getFCP(): number | null {
  if (typeof window === 'undefined') return null

  const entries = performance.getEntriesByType('paint')
  const fcpEntry = entries.find(e => e.name === 'first-contentful-paint')
  
  return fcpEntry ? fcpEntry.startTime : null
}

/**
 * Largest Contentful Paint (LCP)
 */
export function getLCP(): number | null {
  if (typeof window === 'undefined') return null

  const entries = performance.getEntriesByType('largest-contentful-paint')
  const lcpEntry = entries[entries.length - 1]
  
  return lcpEntry ? (lcpEntry as any).startTime : null
}

/**
 * Time to First Byte (TTFB)
 */
export function getTTFB(): number | null {
  if (typeof window === 'undefined') return null

  const navigationEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
  
  return navigationEntry ? navigationEntry.responseStart - navigationEntry.requestStart : null
}

/**
 * Comprehensive performance report
 */
export function generatePerformanceReport() {
  if (typeof window === 'undefined') return null

  return {
    timestamp: new Date().toISOString(),
    metrics: {
      fcp: getFCP(),
      lcp: getLCP(),
      ttfb: getTTFB(),
    },
    resources: analyzeResourceTiming(),
    memory: getMemoryUsage(),
    bundle: trackBundleSize(),
    custom: performanceMonitor.getMetrics(),
  }
}

/**
 * Log performance report to console (dev only)
 */
export function logPerformanceReport() {
  if (process.env.NODE_ENV !== 'development') return

  const report = generatePerformanceReport()
  if (!report) return

  console.group('📊 Performance Report')
  console.log('Timestamp:', report.timestamp)
  
  if (report.metrics.fcp) {
    console.log(`FCP: ${report.metrics.fcp.toFixed(2)}ms`)
  }
  if (report.metrics.lcp) {
    console.log(`LCP: ${report.metrics.lcp.toFixed(2)}ms`)
  }
  if (report.metrics.ttfb) {
    console.log(`TTFB: ${report.metrics.ttfb.toFixed(2)}ms`)
  }

  if (report.memory) {
    console.log(`Memory: ${report.memory.usagePercent.toFixed(1)}%`)
  }

  if (report.bundle) {
    console.log(`Bundle Size: ${report.bundle.totalSizeMB}MB`)
  }

  if (report.resources?.slow && report.resources.slow.length > 0) {
    console.warn('Slow Resources:', report.resources.slow)
  }

  console.groupEnd()
}
