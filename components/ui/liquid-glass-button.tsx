"use client"

import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import { motion } from 'framer-motion'

import { cn } from '@/lib/utils'

const liquidGlassButtonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none relative overflow-hidden",
  {
    variants: {
      variant: {
        default: 'liquid-glass-button-default text-white',
        destructive: 'liquid-glass-button-destructive text-white',
        outline: 'liquid-glass-button-outline',
        secondary: 'liquid-glass-button-secondary',
        ghost: 'liquid-glass-button-ghost',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-9 px-4 py-2 has-[>svg]:px-3 rounded-xl',
        sm: 'h-8 rounded-lg gap-1.5 px-3 has-[>svg]:px-2.5',
        lg: 'h-10 rounded-2xl px-6 has-[>svg]:px-4',
        icon: 'size-9 rounded-xl',
        'icon-sm': 'size-8 rounded-lg',
        'icon-lg': 'size-10 rounded-2xl',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
)

function LiquidGlassButton({
  className,
  variant,
  size,
  asChild = false,
  children,
  ...props
}: React.ComponentProps<'button'> &
  VariantProps<typeof liquidGlassButtonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : 'button'

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
    >
      <Comp
        data-slot="button"
        className={cn(liquidGlassButtonVariants({ variant, size, className }))}
        {...props}
      >
        <span className="liquid-glass-button-content relative z-10">
          {children}
        </span>
        <span className="liquid-glass-button-shimmer absolute inset-0 z-0" />
      </Comp>
    </motion.div>
  )
}

export { LiquidGlassButton, liquidGlassButtonVariants }
