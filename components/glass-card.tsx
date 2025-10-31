"use client"

import type React from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface GlassCardProps {
  children: React.ReactNode
  className?: string
  hover?: boolean
  glow?: boolean
}

export function GlassCard({ children, className, hover = true, glow = false }: GlassCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.34, 1.56, 0.64, 1] }}
      whileHover={hover ? { scale: 1.01, y: -2 } : {}}
      className={cn(
        "liquid-glass relative overflow-hidden",
        hover && "cursor-pointer",
        glow && "animate-soft-glow",
        className,
      )}
      style={{
        backdropFilter: 'blur(40px) saturate(150%)',
        WebkitBackdropFilter: 'blur(40px) saturate(150%)',
        backgroundColor: 'rgba(255, 255, 255, 0.88)',
        border: '1px solid rgba(255, 255, 255, 0.3)',
        boxShadow: glow
          ? '0 8px 32px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.98), inset 0 0 24px rgba(255, 255, 255, 0.5)'
          : '0 4px 16px rgba(0, 0, 0, 0.06), 0 1px 4px rgba(0, 0, 0, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.95), inset 0 0 20px rgba(255, 255, 255, 0.4)',
      }}
    >
      {children}
    </motion.div>
  )
}
