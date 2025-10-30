"use client"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { Home, Compass, Sparkles, LogOut, UserCircle } from "lucide-react"
import { useEffect, useState } from "react"
import { useAuth } from "@/lib/auth-context"

export function Navigation() {
  const pathname = usePathname()
  const router = useRouter()
  const [hasResults, setHasResults] = useState(false)
  const { user, logout } = useAuth()
  const [showProfileIcon, setShowProfileIcon] = useState(true)
  const [isModalOpen, setIsModalOpen] = useState(false)

  useEffect(() => {
    const checkResults = () => {
      if (typeof window !== "undefined") {
        const currentAnalysis = sessionStorage.getItem("current_analysis")
        const isAnalyzing = sessionStorage.getItem("is_analyzing") === "true"
        
        // 🔴 PHASE 10: 완전 재설계된 분석 탭 아이콘 로직
        // 세션 기반 로직: 앱 종료 시 아이콘 숨김, 업로드 후에만 표시
        // sessionStorage는 탭/창 닫으면 자동 삭제됨
        
        // sessionStorage에 'has_uploaded' 플래그가 있는지 확인
        const hasUploadedInSession = sessionStorage.getItem("has_uploaded_in_session") === "true"
        
        // Show analysis icon ONLY if:
        // 1. User has uploaded in THIS session (has_uploaded_in_session=true), OR
        // 2. Currently analyzing (is_analyzing=true with current_analysis)
        const shouldShowAnalysisIcon = hasUploadedInSession || (!!currentAnalysis && isAnalyzing)
        setHasResults(shouldShowAnalysisIcon)
        
        console.log(`[Navigation PHASE 10] Analysis icon: ${shouldShowAnalysisIcon} (uploaded: ${hasUploadedInSession}, analyzing: ${isAnalyzing})`)
      }
    }

    // Initial check
    checkResults()
    
    // Custom event listener for immediate updates (with detail support)
    const handleAnalysisChange = (e: Event) => {
      const customEvent = e as CustomEvent
      if (customEvent.detail && typeof customEvent.detail.hasResults === 'boolean') {
        setHasResults(customEvent.detail.hasResults)
      } else {
        checkResults()
      }
    }
    window.addEventListener('analysisStateChange', handleAnalysisChange)
    
    // Check on pathname change with slight delay to avoid flickering (UX-03)
    const timeoutId = setTimeout(checkResults, 50)
    
    return () => {
      window.removeEventListener('analysisStateChange', handleAnalysisChange)
      clearTimeout(timeoutId)
    }
  }, [pathname])

  useEffect(() => {
    // 🔴 CRITICAL FIX: 프로필 아이콘은 메인홈+탐색탭에서 항상 표시 (모든 조건 제거)
    const isHomePage = pathname === "/"
    const isExplorePage = pathname === "/explore"
    
    // 프로필은 홈 또는 탐색 페이지에서 무조건 표시 (hasResults, isModalOpen 무관)
    const shouldShowProfile = isHomePage || isExplorePage
    setShowProfileIcon(shouldShowProfile)
    
    console.log(`[Navigation CRITICAL FIX] Profile: ${shouldShowProfile} (path: ${pathname})`)
  }, [pathname])
  
  useEffect(() => {
    // Listen for modal state changes
    const handleModalChange = (e: Event) => {
      const customEvent = e as CustomEvent
      setIsModalOpen(customEvent.detail?.isModalOpen || false)
    }
    window.addEventListener('modalStateChange', handleModalChange)
    return () => window.removeEventListener('modalStateChange', handleModalChange)
  }, [])

  const navItems = [
    { href: "/", label: "홈", icon: Home, isHome: true },
    ...(hasResults
      ? [
          {
            href: "/",
            label: "분석",
            icon: Sparkles,
            isAnalysis: true,
          },
        ]
      : []),
    {
      href: "/explore",
      label: "탐색",
      icon: Compass,
    },
  ]

  const getActiveState = (item: any) => {
    // For analysis button, active when on home page with results
    if (item.isAnalysis) {
      return pathname === "/" && hasResults
    }
    // For home button, only active when on home and NO results showing
    if (item.isHome) {
      return pathname === "/" && !hasResults
    }
    // For other buttons (like explore)
    return pathname === item.href
  }

  const handleLogout = () => {
    if (confirm("로그아웃 하시겠습니까?")) {
      logout()
      router.push("/")
    }
  }

  const handleGuestProfileClick = () => {
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("openAuthModal"))
    }
  }

  const handleNavClick = (e: React.MouseEvent, item: any) => {
    e.preventDefault()
    
    // FIX: Direct navigation without flickering third tab
    // Immediately update active state to prevent ghost highlighting
    
    // 🔴 CRITICAL FIX: Home button - 상태 업데이트 먼저, 네비게이션 나중에
    if (item.isHome) {
      // Clear all analysis-related session storage
      if (typeof window !== 'undefined') {
        sessionStorage.removeItem('is_analyzing')
        sessionStorage.removeItem('current_analysis')
        sessionStorage.removeItem('analysis_phase')
        sessionStorage.removeItem('ocr_progress')
        
        // Check if user has uploaded in this session
        const hasUploaded = sessionStorage.getItem('has_uploaded_in_session') === 'true'
        
        // 🔴 CRITICAL FIX: 즉시 상태 업데이트
        setHasResults(hasUploaded)
        
        // Dispatch event
        window.dispatchEvent(new CustomEvent('analysisStateChange', {
          detail: { hasResults: hasUploaded }
        }))
        
        console.log(`[Navigation CRITICAL FIX] 홈 버튼: hasResults=${hasUploaded}, pathname=${pathname}`)
      }
      
      // 🔴 CRITICAL FIX: 항상 네비게이션 실행 (조건 없이)
      if (pathname !== '/') {
        console.log('[Navigation CRITICAL FIX] 홈으로 이동 실행')
        router.push('/')
      } else {
        console.log('[Navigation CRITICAL FIX] 이미 홈 - 스크롤만')
        window.scrollTo({ top: 0, behavior: 'smooth' })
      }
      return
    }
    
    // Analysis button: scroll to results
    if (item.isAnalysis) {
      if (pathname === '/') {
        window.scrollTo({ top: 0, behavior: 'smooth' })
      } else {
        router.push('/')
      }
      return
    }
    
    // Other buttons: direct navigation only to target
    if (pathname !== item.href) {
      router.push(item.href)
    }
  }

  return (
    <>
      {/* 🔴 CRITICAL FIX: 프로필 아이콘 - 홈과 탐색 페이지에서 무조건 표시 */}
      {user?.isGuest && showProfileIcon && (
        <div className="fixed top-4 right-4 z-50">
          <motion.button
            onClick={handleGuestProfileClick}
            initial={{ opacity: 0, scale: 0.8, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            className="flex items-center gap-2 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-full px-3 py-1.5 shadow-sm hover:shadow-md transition-all cursor-pointer"
          >
            <UserCircle className="w-4 h-4 text-gray-600" />
            <span className="text-xs font-medium text-gray-700">비회원</span>
          </motion.button>
        </div>
      )}

      {/* 🔴 PHASE 11: Mobile-optimized navigation with touch targets */}
      <nav className="fixed bottom-0 left-0 right-0 z-50 pb-safe pointer-events-none">
        <div className="backdrop-blur-xl bg-white/98 border-t border-gray-200 shadow-[0_-2px_16px_rgba(0,0,0,0.04)] px-3 py-1.5 safe-area-inset-bottom pointer-events-auto fixed-bottom-nav">
          <div className="flex items-center justify-around max-w-md mx-auto gap-1.5">
            <AnimatePresence mode="popLayout">
              {navItems.map((item) => {
                const isActive = getActiveState(item)
                const Icon = item.icon

                return (
                  <motion.div
                    key={`${item.label}-${item.isAnalysis ? 'analysis' : item.isHome ? 'home' : 'other'}`}
                    initial={item.isAnalysis ? { scale: 0, opacity: 0 } : false}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    transition={{
                      type: "spring",
                      stiffness: 500,
                      damping: 30,
                      mass: 0.8,
                    }}
                    className="flex-1"
                  >
                    {/* 🔴 PHASE 11: Touch-optimized button with larger tap target */}
                    <button onClick={(e) => handleNavClick(e, item)} className="block w-full min-h-[48px]">
                      <motion.div
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.95 }}
                        className={cn(
                          "relative px-3 py-2 rounded-xl transition-all duration-200 flex flex-col items-center gap-0.5 min-h-[44px] justify-center",
                          isActive
                            ? "bg-gray-900 text-white shadow-md"
                            : "text-gray-600 hover:text-gray-900 hover:bg-gray-100",
                        )}
                      >
                        <Icon className="w-5 h-5" />
                        <span className="text-[10px] font-medium">{item.label}</span>
                      </motion.div>
                    </button>
                  </motion.div>
                )
              })}

              {user && !user.isGuest && (
                <motion.div
                  key="logout"
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{
                    type: "spring",
                    stiffness: 500,
                    damping: 30,
                    mass: 0.8,
                  }}
                  className="flex-1"
                >
                  <button onClick={handleLogout} className="block w-full">
                    <motion.div
                      whileHover={{ scale: 1.03 }}
                      whileTap={{ scale: 0.97 }}
                      className="relative px-3 py-1.5 rounded-xl transition-all duration-200 flex flex-col items-center gap-0.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                    >
                      <LogOut className="w-4 h-4" />
                      <span className="text-[10px] font-medium">로그아웃</span>
                    </motion.div>
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </nav>
    </>
  )
}
