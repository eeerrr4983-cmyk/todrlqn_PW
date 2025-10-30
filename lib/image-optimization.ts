/**
 * Performance Optimization: Image Optimization Utilities
 * 
 * Provides utilities for optimizing image loading and display
 */

/**
 * Compress image file before upload
 * Reduces file size while maintaining quality
 */
export async function compressImage(
  file: File,
  options: {
    maxWidth?: number
    maxHeight?: number
    quality?: number
  } = {}
): Promise<File> {
  const {
    maxWidth = 1920,
    maxHeight = 1080,
    quality = 0.85,
  } = options

  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    
    reader.onload = (e) => {
      const img = new Image()
      
      img.onload = () => {
        // Calculate new dimensions
        let width = img.width
        let height = img.height
        
        if (width > maxWidth || height > maxHeight) {
          const ratio = Math.min(maxWidth / width, maxHeight / height)
          width *= ratio
          height *= ratio
        }
        
        // Create canvas and compress
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        
        const ctx = canvas.getContext('2d')
        if (!ctx) {
          reject(new Error('Failed to get canvas context'))
          return
        }
        
        ctx.drawImage(img, 0, 0, width, height)
        
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error('Failed to compress image'))
              return
            }
            
            const compressedFile = new File([blob], file.name, {
              type: file.type,
              lastModified: Date.now(),
            })
            
            resolve(compressedFile)
          },
          file.type,
          quality
        )
      }
      
      img.onerror = () => reject(new Error('Failed to load image'))
      img.src = e.target?.result as string
    }
    
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

/**
 * Lazy load images with Intersection Observer
 */
export function lazyLoadImage(img: HTMLImageElement, options?: IntersectionObserverInit): void {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const target = entry.target as HTMLImageElement
          const src = target.dataset.src
          
          if (src) {
            target.src = src
            target.removeAttribute('data-src')
            observer.unobserve(target)
          }
        }
      })
    },
    {
      rootMargin: '50px', // Start loading 50px before entering viewport
      ...options,
    }
  )
  
  observer.observe(img)
}

/**
 * Generate blur placeholder for images
 */
export function generateBlurDataURL(width: number = 10, height: number = 10): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  
  const ctx = canvas.getContext('2d')
  if (!ctx) return ''
  
  // Create gradient blur effect
  const gradient = ctx.createLinearGradient(0, 0, width, height)
  gradient.addColorStop(0, '#f3f4f6')
  gradient.addColorStop(1, '#e5e7eb')
  
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, width, height)
  
  return canvas.toDataURL()
}

/**
 * Preload critical images
 */
export function preloadImage(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve()
    img.onerror = reject
    img.src = src
  })
}

/**
 * Preload multiple images
 */
export async function preloadImages(urls: string[]): Promise<void[]> {
  return Promise.all(urls.map(preloadImage))
}

/**
 * Get optimal image dimensions based on device
 */
export function getOptimalDimensions(
  originalWidth: number,
  originalHeight: number
): { width: number; height: number } {
  const devicePixelRatio = typeof window !== 'undefined' ? window.devicePixelRatio : 1
  const screenWidth = typeof window !== 'undefined' ? window.innerWidth : 1920
  
  // Target width based on screen size and pixel ratio
  const targetWidth = Math.min(screenWidth * devicePixelRatio, originalWidth)
  const ratio = targetWidth / originalWidth
  
  return {
    width: Math.round(targetWidth),
    height: Math.round(originalHeight * ratio),
  }
}

/**
 * Convert File to base64 data URL
 */
export function fileToDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

/**
 * Estimate image file size reduction
 */
export function estimateSizeReduction(
  originalSize: number,
  quality: number = 0.85
): number {
  // Rough estimation: quality 0.85 typically reduces size by 40-60%
  const reductionFactor = 1 - (quality * 0.5)
  return Math.round(originalSize * reductionFactor)
}
