'use client'

/**
 * Performance Optimization: Lazy Loading Components
 * 
 * Components that are not immediately needed can be lazy-loaded
 * to reduce initial bundle size and improve page load time.
 */

import dynamic from 'next/dynamic'
import { ComponentType } from 'react'
import { Loader2 } from 'lucide-react'

// Loading fallback component
const LoadingFallback = () => (
  <div className="flex items-center justify-center p-8">
    <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
  </div>
)

// Lazy load heavy components that are not critical for initial render

/**
 * AI Killer Detector - Only loaded when user clicks the detection button
 */
export const LazyAIKillerDetector = dynamic(
  () => import('@/components/ai-killer-detector').then(mod => mod.AIKillerDetector),
  {
    loading: LoadingFallback,
    ssr: false, // Not needed for SSR
  }
)

/**
 * University Predictor - Only loaded when user opens the predictor modal
 */
export const LazyUniversityPredictor = dynamic(
  () => import('@/components/university-predictor').then(mod => mod.UniversityPredictor),
  {
    loading: LoadingFallback,
    ssr: false,
  }
)

/**
 * Project Recommender - Only loaded when user clicks recommendations
 */
export const LazyProjectRecommender = dynamic(
  () => import('@/components/project-recommender').then(mod => mod.ProjectRecommender),
  {
    loading: LoadingFallback,
    ssr: false,
  }
)

/**
 * Teacher Communication Helper - Modal, can be lazy loaded
 */
export const LazyTeacherCommunicationHelper = dynamic(
  () => import('@/components/teacher-communication-helper').then(mod => mod.TeacherCommunicationHelper),
  {
    loading: LoadingFallback,
    ssr: false,
  }
)

/**
 * AI Mentoring - Chatbot interface, lazy load
 */
export const LazyAIMentoring = dynamic(
  () => import('@/components/ai-mentoring').then(mod => mod.AIMentoring),
  {
    loading: LoadingFallback,
    ssr: false,
  }
)

/**
 * Storage Manager - Admin/utility component, lazy load
 */
export const LazyStorageManager = dynamic(
  () => import('@/components/storage-manager').then(mod => mod.StorageManager),
  {
    loading: LoadingFallback,
    ssr: false,
  }
)

/**
 * Notification Center - Can be lazy loaded as it's not immediately visible
 */
export const LazyNotificationCenter = dynamic(
  () => import('@/components/notification-center').then(mod => mod.NotificationCenter),
  {
    loading: () => null, // No loading indicator needed for notifications
    ssr: false,
  }
)

/**
 * Utility: Create a lazy-loaded component with custom options
 */
export function createLazyComponent<T extends ComponentType<any>>(
  importFn: () => Promise<{ default: T } | T>,
  options?: {
    loading?: () => JSX.Element | null
    ssr?: boolean
  }
) {
  return dynamic(
    () => importFn().then(mod => ('default' in mod ? mod.default : mod)),
    {
      loading: options?.loading || LoadingFallback,
      ssr: options?.ssr ?? false,
    }
  )
}
