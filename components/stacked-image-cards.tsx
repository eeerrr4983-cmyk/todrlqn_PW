"use client"

import { motion, AnimatePresence } from "framer-motion"
import { X } from "lucide-react"
import { useState } from "react"
import Image from "next/image"

interface StackedImageCardsProps {
  imageUrls: string[]
  onRemove?: (index: number) => void
  readonly?: boolean
}

/**
 * Stacked Card UI for multiple image uploads (UX-09)
 * - Images overlap with 3D effect
 * - Click on back cards to bring to front
 * - Smooth animations with Framer Motion
 * - No left/right arrows (removed as per requirement)
 */
export function StackedImageCards({ imageUrls, onRemove, readonly = false }: StackedImageCardsProps) {
  const [activeIndex, setActiveIndex] = useState(0)
  // 🔴 FIX: 처음부터 펼쳐진 상태로 시작
  const [isExpanded, setIsExpanded] = useState(true)

  if (imageUrls.length === 0) return null

  // Single image - no stacking needed
  if (imageUrls.length === 1) {
    return (
      <div className="relative w-full max-w-sm mx-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative aspect-[4/3] rounded-2xl overflow-hidden shadow-lg"
        >
          <Image
            src={imageUrls[0]}
            alt="Uploaded image"
            fill
            className="object-cover"
          />
          {!readonly && onRemove && (
            <button
              onClick={() => onRemove(0)}
              className="absolute top-2 right-2 w-7 h-7 bg-black/60 hover:bg-black/80 rounded-full flex items-center justify-center transition-colors z-10"
            >
              <X className="w-4 h-4 text-white" />
            </button>
          )}
        </motion.div>
        <p className="text-center text-xs text-gray-600 mt-2">1개 이미지</p>
      </div>
    )
  }

  // 🔴 FIX: 클릭하면 선택 + 자동 접힘
  const handleCardClick = (index: number) => {
    if (index !== activeIndex) {
      setActiveIndex(index)
      // 선택 후 자동으로 접힘
      setTimeout(() => setIsExpanded(false), 100)
    }
  }

  // Multiple images - stacked card UI with magic spread/collapse
  return (
    <div className="relative w-full max-w-sm mx-auto">
      <div className="relative h-[280px] flex items-center justify-center">
        <AnimatePresence mode="popLayout">
          {imageUrls.map((url, index) => {
            const offset = index - activeIndex
            const absOffset = Math.abs(offset)
            const isActive = index === activeIndex
            
            // 🔴 FIX: 간격 좁히기 (80→55, 20→12)
            const spreadAngle = isExpanded ? (index - imageUrls.length / 2 + 0.5) * 8 : 0
            const spreadX = isExpanded ? (index - imageUrls.length / 2 + 0.5) * 55 : offset * (isActive ? 0 : offset > 0 ? 15 : -15)
            const spreadY = isExpanded ? Math.abs(index - imageUrls.length / 2) * 12 : absOffset * 10
            const spreadScale = isExpanded ? 0.92 : (isActive ? 1 : 0.85 - (absOffset * 0.05))
            const spreadOpacity = isExpanded ? 1 : (isActive ? 1 : 0.6 - (absOffset * 0.2))
            const spreadRotate = isExpanded ? spreadAngle : offset * -3

            // Only show 3 cards max when collapsed (active + 2 in background)
            if (!isExpanded && absOffset > 2) return null

            return (
              <motion.div
                key={`${url}-${index}`}
                initial={{ 
                  scale: 0.8, 
                  opacity: 0,
                  rotateY: 0,
                }}
                animate={{
                  scale: spreadScale,
                  opacity: spreadOpacity,
                  x: spreadX,
                  y: spreadY,
                  z: isExpanded ? -index * 5 : -absOffset * 20,
                  rotateY: spreadRotate,
                  rotateX: isExpanded ? 0 : absOffset * 2,
                  rotateZ: isExpanded ? spreadAngle * 0.5 : 0,
                  zIndex: isExpanded ? imageUrls.length - Math.abs(index - imageUrls.length / 2) : imageUrls.length - absOffset,
                }}
                exit={{ 
                  scale: 0.8, 
                  opacity: 0,
                  transition: { duration: 0.2 }
                }}
                transition={{
                  type: "spring",
                  stiffness: isExpanded ? 200 : 300,
                  damping: isExpanded ? 25 : 30,
                  mass: 0.8,
                }}
                style={{
                  position: "absolute",
                  transformStyle: "preserve-3d",
                  perspective: 1000,
                }}
                onClick={() => handleCardClick(index)}
                className={`aspect-[4/3] w-full max-w-[320px] rounded-2xl overflow-hidden shadow-xl ${
                  !isActive || isExpanded ? "cursor-pointer hover:scale-[0.95] transition-transform" : ""
                }`}
              >
                <div className="relative w-full h-full">
                  <Image
                    src={url}
                    alt={`Uploaded image ${index + 1}`}
                    fill
                    className="object-cover"
                  />
                  
                  {/* Card number badge */}
                  <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm text-white text-xs font-semibold px-2 py-1 rounded-full z-10">
                    {index + 1}/{imageUrls.length}
                  </div>

                  {/* Remove button - only on active card */}
                  {!readonly && onRemove && isActive && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onRemove(index)
                        // Adjust active index if needed
                        if (activeIndex >= imageUrls.length - 1) {
                          setActiveIndex(Math.max(0, activeIndex - 1))
                        }
                      }}
                      className="absolute top-2 right-2 w-8 h-8 bg-red-500/90 hover:bg-red-600 rounded-full flex items-center justify-center transition-colors z-10 shadow-md"
                    >
                      <X className="w-4 h-4 text-white" />
                    </button>
                  )}

                  {/* Tap to view indicator on back cards */}
                  {!isActive && (
                    <div className="absolute inset-0 bg-black/20 backdrop-blur-[1px] flex items-center justify-center">
                      <div className="bg-white/90 backdrop-blur-sm px-3 py-1.5 rounded-full text-xs font-medium text-gray-700">
                        탭하여 보기
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>

      {/* 🔴 FIX: 펼치기 버튼 제거, 네비게이션 도트만 유지 */}
      
      {/* Navigation dots */}
      <div className="flex items-center justify-center gap-1.5 mt-4">
        {imageUrls.map((_, index) => (
          <button
            key={index}
            onClick={() => setActiveIndex(index)}
            className={`h-1.5 rounded-full transition-all ${
              index === activeIndex
                ? "w-6 bg-gray-900"
                : "w-1.5 bg-gray-300 hover:bg-gray-400"
            }`}
            aria-label={`이미지 ${index + 1}로 이동`}
          />
        ))}
      </div>

      <p className="text-center text-xs text-gray-600 mt-2">
        {imageUrls.length}개 이미지 • 원하는 카드를 선택하세요
      </p>
    </div>
  )
}
