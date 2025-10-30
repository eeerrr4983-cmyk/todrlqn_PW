"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { useRouter } from "next/navigation"
import { Navigation } from "@/components/navigation"
import { LiquidBackground } from "@/components/liquid-background"
import { GlassCard } from "@/components/glass-card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Upload,
  FileText,
  Sparkles,
  CheckCircle2,
  Loader2,
  Share2,
  Eye,
  AlertCircle,
  X,
  MessageSquare,
  Download,
  Shield,
  TrendingUp,
  Compass,
  Lightbulb,
  User,
  History,
  ChevronRight,
  ChevronDown,
} from "lucide-react"
import { StorageManager } from "@/components/storage-manager"
import type { AnalysisResult } from "@/lib/types"
// 🔴 PHASE 2: Enhanced OCR with detailed progress tracking
import { extractTextFromImage } from "@/lib/ocr-enhanced"
import { useAuth } from "@/lib/auth-context"
import { AuthGuard } from "@/components/auth-guard"
import { StackedImageCards } from "@/components/stacked-image-cards"
import { getUserStudentId, getUserDisplayName, initUserSession } from "@/lib/user-session"
import { compressImage } from "@/lib/image-optimization"
// Performance: Lazy load heavy components
import {
  LazyTeacherCommunicationHelper,
  LazyAIKillerDetector,
  LazyUniversityPredictor,
  LazyProjectRecommender,
  LazyNotificationCenter,
} from "@/lib/lazy-components"

type Phase = "idle" | "uploading" | "ocr" | "analyzing" | "analysisComplete" | "complete"
type ResultTab = "strengths" | "improvements"

const PROGRESS_TIPS = [
  "구체적 활동 내용이 생기부의 핵심이에요.",
  "진로 연계성이 대학 평가의 중요 요소예요.",
  "수상은 의미있는 것만 선별하세요.",
  "지속적인 봉사가 진정성을 보여줘요.",
  "전공 관련 독서가 학업 열정을 드러내요.",
]

const PROGRESS_MESSAGES = {
  uploading: ["파일을 업로드하는 중이에요."],
  ocr: [
    "AI가 생기부를 읽어보고 있어요.",
    "텍스트를 정밀하게 추출하는 중이에요.",
    "생기부 내용을 분석하고 있어요.",
    "문서 구조를 파악하는 중이에요.",
  ],
  analyzing: [
    "AI가 정밀하게 탐지하는 중이에요.",
    "오류를 검사하고 있어요.",
    "강점과 보완점을 찾고 있어요.",
    "종합 평가를 계산하는 중이에요.",
  ],
}

// LB-10: Using global user session management
// getUserStudentId() is now imported from @/lib/user-session

// Helper function to generate AI-based titles for analysis history  
const generateAnalysisTitle = (analysis: AnalysisResult, index: number): string => {
  // LB-10: Use global getUserDisplayName for consistent naming
  const displayName = getUserDisplayName()
  // 🔴 FIX: Student → 학생 변환
  return displayName.replace(/Student(\d+)/g, '학생$1')
}

// 🔴 NEW: AI가 생기부 내용을 기반으로 학과/분야 키워드 추정
const generateAIKeyword = (analysis: AnalysisResult): string => {
  // 진로방향이 있으면 우선 사용
  if (analysis.careerDirection && analysis.careerDirection.trim()) {
    return analysis.careerDirection
  }
  
  // 생기부 내용 기반 키워드 추정
  const originalText = analysis.originalText?.toLowerCase() || ''
  
  // 이공계열 키워드
  if (originalText.includes('공학') || originalText.includes('과학') || originalText.includes('수학') || originalText.includes('물리') || originalText.includes('화학')) {
    if (originalText.includes('컴퓨터') || originalText.includes('소프트웨어') || originalText.includes('프로그래밍') || originalText.includes('코딩')) {
      return '컴퓨터공학과'
    }
    if (originalText.includes('기계') || originalText.includes('로봇')) {
      return '기계공학과'
    }
    if (originalText.includes('전기') || originalText.includes('전자')) {
      return '전자공학과'
    }
    return '이공계열'
  }
  
  // 의학계열
  if (originalText.includes('의학') || originalText.includes('간호') || originalText.includes('보건') || originalText.includes('병원') || originalText.includes('의사')) {
    if (originalText.includes('간호')) {
      return '간호학과'
    }
    return '의예과'
  }
  
  // 경영/경제
  if (originalText.includes('경영') || originalText.includes('경제') || originalText.includes('사업') || originalText.includes('마케팅')) {
    return '경영학과'
  }
  
  // 인문계열
  if (originalText.includes('문학') || originalText.includes('역사') || originalText.includes('철학') || originalText.includes('언어')) {
    return '인문계열'
  }
  
  // 예체능
  if (originalText.includes('미술') || originalText.includes('음악') || originalText.includes('체육') || originalText.includes('디자인')) {
    return '예체능계열'
  }
  
  // 기본값
  return '종합계열'
}

// Helper function for smart time display
const getSmartTimeDisplay = (uploadDate: Date): string => {
  const now = new Date()
  const diffMs = now.getTime() - uploadDate.getTime()
  const diffMinutes = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffHours / 24)
  const diffWeeks = Math.floor(diffDays / 7)
  const diffMonths = Math.floor(diffDays / 30)

  if (diffMinutes < 60) {
    return "오늘"
  } else if (diffHours < 24) {
    return "오늘"
  } else if (diffDays === 1) {
    return "어제"
  } else if (diffDays === 2) {
    return "2일전"
  } else if (diffDays === 3) {
    return "3일전"
  } else if (diffDays < 7) {
    return `${diffDays}일전`
  } else if (diffWeeks === 1) {
    return "일주일전"
  } else if (diffWeeks === 2) {
    return "2주전"
  } else if (diffMonths === 1) {
    return "한달전"
  } else if (diffMonths === 2) {
    return "두달전"
  } else if (diffMonths < 12) {
    return `${diffMonths}달전`
  } else {
    return `${uploadDate.getMonth() + 1}/${uploadDate.getDate()}`
  }
}

export default function HomePage() {
  const { user } = useAuth()
  const router = useRouter()

  const [phase, setPhase] = useState<Phase>("idle")
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [uploadedImageUrls, setUploadedImageUrls] = useState<string[]>([])
  const [careerDirection, setCareerDirection] = useState("")
  const [careerDirectionExpanded, setCareerDirectionExpanded] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [showShareDialog, setShowShareDialog] = useState(false)
  const [shareData, setShareData] = useState({ studentId: "", name: "", agreedToTerms: false, isPrivate: false })
  const [ocrProgress, setOcrProgress] = useState(0)
  const [showTeacherHelper, setShowTeacherHelper] = useState(false)
  const [selectedError, setSelectedError] = useState<any>(null)
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([])
  const [showAIKiller, setShowAIKiller] = useState(false)
  const [showUniversityPredictor, setShowUniversityPredictor] = useState(false)
  const [showProjectRecommender, setShowProjectRecommender] = useState(false)
  
  // Notify navigation when modals open/close
  useEffect(() => {
    const anyModalOpen = showAIKiller || showUniversityPredictor || showProjectRecommender || showTeacherHelper
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('modalStateChange', {
        detail: { isModalOpen: anyModalOpen }
      }))
    }
  }, [showAIKiller, showUniversityPredictor, showProjectRecommender, showTeacherHelper])
  const [progressMessage, setProgressMessage] = useState("")
  const [currentTip, setCurrentTip] = useState("")
  const [resultTab, setResultTab] = useState<ResultTab>("strengths")
  // LB-10: Initialize user session using global function
  const [userSessionId] = useState(() => {
    if (typeof window !== "undefined") {
      const sessionId = initUserSession()
      const storedStudentId = sessionStorage.getItem("student_id")
      const storedName = sessionStorage.getItem("student_name")
      if (storedStudentId && storedName) {
        setShareData((prev) => ({ ...prev, studentId: storedStudentId, name: storedName }))
      }
      return sessionId
    }
    return "user-" + Date.now()
  })

  const fileInputRef = useRef<HTMLInputElement>(null)
  const [showAnalysisComplete, setShowAnalysisComplete] = useState(false)
  const [hasShownCompletion, setHasShownCompletion] = useState(false)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)

  useEffect(() => {
    // CRITICAL FIX: Load history on mount AND keep it updated
    const loadHistory = () => {
      const history = StorageManager.getAnalysisHistory()
      // Sort by upload date (latest first) and take top 3
      const sorted = history.sort((a, b) => 
        new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime()
      )
      setAnalysisHistory(sorted.slice(0, 3))
    }
    
    loadHistory()
    
    // Also check for existing analysis state
    const checkAnalysisState = () => {
      if (typeof window !== "undefined") {
        const currentAnalysis = sessionStorage.getItem("current_analysis")
        const isAnalyzing = sessionStorage.getItem("is_analyzing") === "true"
        if (currentAnalysis && isAnalyzing) {
          setAnalysisResult(JSON.parse(currentAnalysis))
          setPhase("complete")
        }
      }
    }
    checkAnalysisState()
    
    // Listen for analysis state changes AND history updates
    const handleStateChange = () => {
      checkAnalysisState()
      loadHistory() // Reload history when state changes
    }
    window.addEventListener("analysisStateChange", handleStateChange)
    
    // Also listen for storage changes (for cross-tab sync)
    window.addEventListener("storage", loadHistory)
    
    return () => {
      window.removeEventListener("analysisStateChange", handleStateChange)
      window.removeEventListener("storage", loadHistory)
    }
  }, [userSessionId])

  useEffect(() => {
    // Select tip only once at the start of analysis
    if ((phase === "ocr" || phase === "analyzing") && !currentTip) {
      const randomTip = PROGRESS_TIPS[Math.floor(Math.random() * PROGRESS_TIPS.length)]
      setCurrentTip(randomTip)
    }
  }, [phase, currentTip])

  useEffect(() => {
    // Set initial message based on phase, but don't auto-rotate
    // Messages will be updated by actual progress in handleAnalyze
    if (phase === "ocr") {
      setProgressMessage(PROGRESS_MESSAGES.ocr[0])
    } else if (phase === "analyzing") {
      setProgressMessage(PROGRESS_MESSAGES.analyzing[0])
    }
  }, [phase])

  useEffect(() => {
    const savedAnalysis = sessionStorage.getItem("current_analysis")
    const isAnalyzing = sessionStorage.getItem("is_analyzing") === "true"

    if (savedAnalysis && !analysisResult && isAnalyzing) {
      setAnalysisResult(JSON.parse(savedAnalysis))
      setPhase("complete")
    } else if (!isAnalyzing) {
      setPhase("idle")
      setAnalysisResult(null)
    }
  }, [])

  useEffect(() => {
    if (analysisResult) {
      sessionStorage.setItem("current_analysis", JSON.stringify(analysisResult))
    }
  }, [analysisResult])

  useEffect(() => {
    if (typeof window !== "undefined") {
      if (
        phase === "uploading" ||
        phase === "ocr" ||
        phase === "analyzing" ||
        phase === "analysisComplete" ||
        phase === "complete"
      ) {
        sessionStorage.setItem("is_analyzing", "true")
      } else {
        sessionStorage.removeItem("is_analyzing")
      }
    }
  }, [phase])

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      
      // Performance: Compress images before upload
      console.log("[Upload] Compressing images...")
      const compressedFiles = await Promise.all(
        files.map(async (file) => {
          if (file.type.startsWith('image/')) {
            try {
              const compressed = await compressImage(file, {
                maxWidth: 1920,
                maxHeight: 1080,
                quality: 0.85,
              })
              console.log(`[Upload] ${file.name}: ${(file.size / 1024).toFixed(1)}KB → ${(compressed.size / 1024).toFixed(1)}KB`)
              return compressed
            } catch (error) {
              console.warn(`[Upload] Failed to compress ${file.name}, using original`)
              return file
            }
          }
          return file
        })
      )
      
      setUploadedFiles(compressedFiles)

      const urls = compressedFiles.map((file) => URL.createObjectURL(file))
      setUploadedImageUrls(urls)

      startAnalysis(compressedFiles)
    }
  }

  const handleFileSelectClick = () => {
    fileInputRef.current?.click()
  }

  const startAnalysis = async (files: File[]) => {
    console.log("[Page] === 분석 시작 ===")
    console.log(`[Page] 📁 파일 개수: ${files.length}`)
    
    setHasShownCompletion(false)

    if (typeof window !== "undefined") {
      sessionStorage.setItem("is_analyzing", "true")
      // 🔴 PHASE 10: 이미지 업로드 시 세션 플래그 설정
      sessionStorage.setItem("has_uploaded_in_session", "true")
      // Dispatch event to show analysis icon in navigation
      window.dispatchEvent(new CustomEvent("analysisStateChange", {
        detail: { hasResults: true }
      }))
      console.log("[Page PHASE 10] 📤 업로드 완료 - 분석 아이콘 활성화")
    }

    // 🔴 FIX: 진행도 0%부터 시작
    setOcrProgress(0)
    setPhase("uploading")
    setProgressMessage("파일을 업로드하는 중이에요.")
    
    // 🔴 FIX: 업로드 단계 정확히 1.5초 딜레이 후 체크
    await new Promise((resolve) => setTimeout(resolve, 1500))

    setPhase("ocr")
    setProgressMessage(PROGRESS_MESSAGES.ocr[0])
    const extractedTexts: string[] = []

    const totalFiles = files.length

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      console.log(`[Page] 📄 OCR 처리 중 (${i + 1}/${totalFiles}): ${file.name}`)

      const fileProgressStart = (i / totalFiles) * 100
      const fileProgressEnd = ((i + 1) / totalFiles) * 100

      try {
        // Use real OCR progress tracking
        const text = await extractTextFromImage(file, (progress) => {
          // Map OCR progress (0-100) to current file's progress range
          const fileProgress = fileProgressStart + (progress.progress / 100) * (fileProgressEnd - fileProgressStart)
          setOcrProgress(Math.min(99, fileProgress))
          
          // Update message from OCR service
          if (progress.status) {
            setProgressMessage(progress.status)
          }
        })
        extractedTexts.push(text)
        console.log(`[Page] ✅ OCR 완료 (${i + 1}/${totalFiles}): ${text.length} 글자`)
      } catch (error) {
        console.error(`[Page] ❌ OCR 오류 (파일 ${i + 1}):`, error)
        extractedTexts.push("")
      }
    }

    console.log(`[Page] 📊 전체 추출된 텍스트: ${extractedTexts.filter(t => t.length > 0).length}/${totalFiles} 성공`)

    // Check if all OCR failed
    const validTexts = extractedTexts.filter(t => t.trim().length > 0)
    if (validTexts.length === 0) {
      console.error("[Page] ❌ 모든 OCR이 실패함")
      setPhase("idle")
      setProgressMessage("")
      alert("모든 이미지에서 텍스트를 추출할 수 없었습니다.\n\n• 이미지가 흐릿하거나 해상도가 낮은지 확인해주세요\n• 텍스트가 명확하게 보이는 이미지를 사용해주세요\n• 이미지 파일 크기를 줄여서 다시 시도해주세요")
      return
    }

    setOcrProgress(100)
    await new Promise((resolve) => setTimeout(resolve, 200))

    setPhase("analyzing")
    setProgressMessage("AI가 생기부를 정밀하게 분석하는 중...")
    console.log("[Page] 🤖 Gemini 분석 시작...")

    // Import Gemini service for real analysis
    const { analyzeSaenggibu } = await import("@/lib/gemini-service")
    
    let analysisResult: AnalysisResult
    
    try {
      const combinedText = extractedTexts.filter(t => t.trim().length > 0).join("\n\n")
      console.log(`[Page] 📝 결합된 텍스트 길이: ${combinedText.length} 글자`)
      
      const analysisStart = Date.now()
      
      const baseAnalysis = await analyzeSaenggibu(combinedText, careerDirection, (progress) => {
        if (progress < 30) {
          setProgressMessage("AI가 생기부를 정밀하게 읽는 중...")
        } else if (progress < 60) {
          setProgressMessage("금지/주의 사항을 탐지하는 중...")
        } else if (progress < 90) {
          setProgressMessage("강점과 개선점을 분석하는 중...")
        } else {
          setProgressMessage("종합 평가를 완성하는 중...")
        }
      })
      
      console.log(`[Page] ✅ Gemini 분석 완료 (${Date.now() - analysisStart}ms)`)
      console.log(`[Page] 📊 점수: ${baseAnalysis.overallScore}, 오류: ${baseAnalysis.errors.length}개`)
      
      const analysisTimestamp = new Date().toISOString()
      
      analysisResult = {
        ...baseAnalysis,
        id: baseAnalysis.id || analysisStart.toString(),
        studentName: baseAnalysis.studentName || "",
        careerDirection: baseAnalysis.careerDirection || careerDirection || "미지정",
        uploadDate: baseAnalysis.uploadDate || analysisTimestamp,
        analyzedAt: baseAnalysis.analyzedAt || analysisTimestamp,
        files: files.map((f) => f.name),
        isPrivate: true,
        likes: typeof baseAnalysis.likes === "number" ? baseAnalysis.likes : 0,
        saves: typeof baseAnalysis.saves === "number" ? baseAnalysis.saves : 0,
        comments: Array.isArray(baseAnalysis.comments) ? baseAnalysis.comments : [],
        userId: baseAnalysis.userId || userSessionId,
        originalText:
          baseAnalysis.originalText && baseAnalysis.originalText.trim().length > 0
            ? baseAnalysis.originalText
            : combinedText,
      }
      
    } catch (error) {
      console.error("[Page] ❌❌❌ Analysis Error ❌❌❌")
      console.error(error)
      
      const errorMessage = error instanceof Error ? error.message : "알 수 없는 오류"
      console.error(`[Page] 에러 메시지: ${errorMessage}`)
      
      setPhase("idle")
      setProgressMessage("")
      
      // More detailed error message for user
      let userErrorMessage = "분석 중 오류가 발생했습니다.\n\n"
      
      if (errorMessage.includes("timeout") || errorMessage.includes("타임아웃")) {
        userErrorMessage += "⏱️ AI 분석 시간이 초과되었습니다.\n\n• 텍스트가 너무 길 수 있습니다\n• 네트워크가 느릴 수 있습니다\n• 잠시 후 다시 시도해주세요"
      } else if (errorMessage.includes("API") || errorMessage.includes("fetch")) {
        userErrorMessage += "🔌 서버 연결에 문제가 있습니다.\n\n• 네트워크 연결을 확인해주세요\n• 잠시 후 다시 시도해주세요"
      } else if (errorMessage.includes("JSON") || errorMessage.includes("파싱")) {
        userErrorMessage += "🤖 AI 응답 처리 중 오류가 발생했습니다.\n\n• AI 서비스가 일시적으로 불안정할 수 있습니다\n• 잠시 후 다시 시도해주세요"
      } else {
        userErrorMessage += `에러: ${errorMessage}\n\n다시 시도해주세요.`
      }
      
      alert(userErrorMessage)
      return
    }

    // Show completion popup ONCE
    if (!hasShownCompletion) {
      setPhase("analysisComplete")
      setShowAnalysisComplete(true)
      setHasShownCompletion(true)
      
      // Wait for popup to display, then smoothly hide it
      await new Promise((resolve) => setTimeout(resolve, 1800))
      setShowAnalysisComplete(false)
      
      // Small delay to allow exit animation to complete
      await new Promise((resolve) => setTimeout(resolve, 400))
    }

    setAnalysisResult(analysisResult)
    setPhase("complete")
    
    // CRITICAL: Set analysis state immediately for navigation
    if (typeof window !== "undefined") {
      sessionStorage.setItem("current_analysis", JSON.stringify(analysisResult))
      sessionStorage.setItem("is_analyzing", "true")
      window.dispatchEvent(new CustomEvent("analysisStateChange"))
    }
    
    // Save to history immediately (even without sharing)
    StorageManager.saveToHistory(analysisResult)
    
    // Reload history to show in "나의 최근 활동" (sorted by date)
    const updatedHistory = StorageManager.getAnalysisHistory()
      .sort((a, b) => new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime())
      .slice(0, 3)
    setAnalysisHistory(updatedHistory)
    
    // Notify navigation to show analysis icon
    window.dispatchEvent(new CustomEvent("analysisStateChange", {
      detail: { hasResults: true }
    }))
  }

  const handleShareClick = () => {
    setShowShareDialog(true)
  }

  const handleShareConfirm = () => {
    if (!shareData.studentId || !shareData.name || !shareData.agreedToTerms) {
      alert("모든 정보를 입력하고 동의해주세요.")
      return
    }

    sessionStorage.setItem("student_id", shareData.studentId)
    sessionStorage.setItem("student_name", shareData.name)

    if (analysisResult) {
      const sharedResult = {
        ...analysisResult,
        studentName: shareData.name,
        studentId: shareData.studentId,
        isPrivate: shareData.isPrivate,
      }

      StorageManager.saveAnalysis(sharedResult)
      setShowShareDialog(false)

      if (!shareData.isPrivate) {
        setTimeout(() => {
          router.push("/explore")
        }, 500)
      } else {
        alert("비공개로 저장되었습니다.")
      }
    }
  }

  const handleTeacherHelp = (error: any) => {
    setSelectedError(error)
    setShowTeacherHelper(true)
  }

  const handleExportPDF = () => {
    if (!analysisResult) return

    const report = `생기부 AI 분석 결과
===================

종합 평가: ${analysisResult.overallScore}점

${analysisResult.studentProfile ? `학생 프로필: ${analysisResult.studentProfile}` : ""}

${
  analysisResult.careerDirection &&
  `진로 방향: ${analysisResult.careerDirection}
진로 적합성: ${analysisResult.careerAlignment?.percentage}% (${analysisResult.careerAlignment?.summary})
`
}

강점:
${analysisResult.strengths.map((s, i) => `${i + 1}. ${s}`).join("\n")}

보완 사항:
${analysisResult.improvements.map((s, i) => `${i + 1}. ${s}`).join("\n")}

오류 ${analysisResult.errors.length}건:
${analysisResult.errors.map((e, i) => `${i + 1}. [${e.type}] ${e.content}\n   사유: ${e.reason}`).join("\n")}

개선 제안:
${analysisResult.suggestions.map((s, i) => `${i + 1}. ${s}`).join("\n")}
    `.trim()

    const blob = new Blob([report], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    const now = new Date()
    const month = String(now.getMonth() + 1).padStart(2, '0')
    const day = String(now.getDate()).padStart(2, '0')
    const hour = String(now.getHours()).padStart(2, '0')
    const minute = String(now.getMinutes()).padStart(2, '0')
    a.download = `생기부분석결과_${month}${day}_${hour}${minute}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const resetAnalysis = () => {
    setPhase("idle")
    setUploadedFiles([])
    uploadedImageUrls.forEach((url) => URL.revokeObjectURL(url))
    setUploadedImageUrls([])
    setAnalysisResult(null)
    setOcrProgress(0)
    setProgressMessage("")
    setCurrentTip("")
    setCareerDirection("")
    sessionStorage.removeItem("current_analysis")
    sessionStorage.removeItem("is_analyzing")
    
    // Dispatch event to update navigation state
    window.dispatchEvent(new CustomEvent("analysisStateChange", {
      detail: { hasResults: false }
    }))
    
    // Force navigation to home (ensuring bottom navigation also updates)
    window.scrollTo({ top: 0, behavior: "smooth" })
  }

  const isFixedScreen =
    phase === "idle" ||
    phase === "uploading" ||
    phase === "ocr" ||
    phase === "analyzing" ||
    phase === "analysisComplete"

  return (
    <AuthGuard>
      {/* 🔴 PHASE 11: Mobile-optimized layout with iOS Safari fixes */}
      <div className={`relative h-screen w-screen bg-gray-50 ios-fix-vh ${isFixedScreen ? "overflow-hidden" : ""}`}>
        <LiquidBackground />
        <Navigation />
        {user && !user.isGuest && <LazyNotificationCenter />}

        <div
          className={`relative z-10 h-full px-4 pt-3 pb-20 mobile-compact mobile-scroll ${
            isFixedScreen ? "overflow-hidden flex flex-col items-center justify-center" : "overflow-y-auto"
          }`}
        >
          {phase === "idle" && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center mb-3"
            >
              <h1 className="text-2xl font-bold tracking-tight text-gray-900 mb-0.5" style={{ fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif' }}>사상고 생기부AI</h1>
              <p className="text-xs text-gray-500 font-normal">학생 생활기록부 AI작성 탐지기</p>
            </motion.div>
          )}

          <div className={`max-w-lg mx-auto w-full ${isFixedScreen ? "" : ""}`}>
            <AnimatePresence mode="wait">
              {phase === "idle" && (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.96 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-3"
                >
                  {/* 🔴 SMOOTH ANIMATION: 진로방향 박스 - 부드러운 열기/닫기 */}
                  <motion.div
                    initial={false}
                    animate={{ 
                      height: careerDirectionExpanded ? "120px" : "36px",
                      width: careerDirectionExpanded ? "100%" : "auto",
                      opacity: 1
                    }}
                    transition={{ 
                      duration: 0.35, 
                      ease: [0.25, 0.1, 0.25, 1], // 더 부드러운 cubic-bezier
                      height: { duration: 0.35 },
                      width: { duration: 0.3 }
                    }}
                    className="overflow-hidden mx-auto"
                    style={{ 
                      maxWidth: careerDirectionExpanded ? "100%" : "280px",
                      willChange: "height, width"
                    }}
                  >
                    <GlassCard className="p-2 space-y-1 shadow-md border-2 border-blue-100/50">
                      <button
                        onClick={() => setCareerDirectionExpanded(!careerDirectionExpanded)}
                        className="w-full flex items-center justify-between gap-2 hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 rounded-lg px-2 py-1 transition-all duration-300"
                      >
                        <div className="flex items-center gap-1.5">
                          <Compass className="w-3.5 h-3.5 text-blue-600" />
                          <h3 className="text-[11px] font-bold text-gray-900">진로방향</h3>
                          {careerDirection && !careerDirectionExpanded && (
                            <span className="text-[9px] text-blue-600 font-semibold truncate max-w-[140px] bg-blue-50 px-2 py-0.5 rounded-full">
                              {careerDirection}
                            </span>
                          )}
                        </div>
                        <motion.div
                          animate={{ rotate: careerDirectionExpanded ? 180 : 0 }}
                          transition={{ duration: 0.3, ease: "easeInOut" }}
                        >
                          <ChevronDown className="w-3.5 h-3.5 text-blue-600" />
                        </motion.div>
                      </button>
                      
                      <AnimatePresence mode="wait">
                        {careerDirectionExpanded && (
                          <motion.div
                            initial={{ opacity: 0, y: -8, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: -8, scale: 0.95 }}
                            transition={{ 
                              duration: 0.25,
                              ease: [0.25, 0.1, 0.25, 1]
                            }}
                            className="space-y-1 pt-1"
                          >
                            <p className="text-[9px] text-gray-600 leading-relaxed">나의 진학목표를 입력시 더 정밀한 분석을 받아볼수있어요.</p>
                            <Input
                              placeholder="희망학과를 입력해주세요"
                              value={careerDirection}
                              onChange={(e) => setCareerDirection(e.target.value)}
                              className="h-7 text-xs placeholder:text-gray-400 placeholder:opacity-50"
                            />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </GlassCard>
                  </motion.div>

                  <GlassCard className="w-full p-5 text-center space-y-3.5" glow>
                    <motion.div
                      animate={{
                        scale: [1, 1.05, 1],
                        rotate: [0, 5, -5, 0],
                      }}
                      transition={{
                        duration: 4,
                        repeat: Number.POSITIVE_INFINITY,
                        ease: "easeInOut",
                      }}
                      className="w-16 h-16 rounded-full bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center mx-auto shadow-md"
                    >
                      <Upload className="w-7 h-7 text-gray-700" />
                    </motion.div>
                    <div className="space-y-1">
                      <h3 className="text-lg font-bold text-gray-900">생기부 분석 시작</h3>
                      <p className="text-xs text-gray-500">이미지를 업로드하여 AI 분석을 시작하세요</p>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept="image/*,.pdf"
                      className="hidden"
                      onChange={handleFileUpload}
                    />
                    <Button
                      size="lg"
                      className="w-full max-w-sm text-sm h-11 rounded-full bg-gray-900 hover:bg-gray-800 text-white shadow-md hover:shadow-xl transition-all font-semibold"
                      onClick={handleFileSelectClick}
                    >
                      <FileText className="w-4 h-4 mr-2" />
                      사진 선택
                    </Button>
                  </GlassCard>

                  {analysisHistory.length > 0 && (
                    <GlassCard className="p-2.5 space-y-1.5">
                      <div className="flex items-center gap-1.5">
                        <History className="w-3.5 h-3.5 text-gray-700" />
                        <h3 className="text-xs font-semibold text-gray-900">나의 최근 활동</h3>
                      </div>
                      <div className="space-y-1.5">
                        {analysisHistory.map((analysis, index) => {
                          const uploadDate = new Date(analysis.uploadDate)
                          const timeDisplay = getSmartTimeDisplay(uploadDate)
                          const title = generateAnalysisTitle(analysis, index)

                          return (
                            <div
                              key={analysis.id}
                              className="w-full p-2.5 bg-gray-50/80 rounded-lg border border-gray-200/50 transition-all hover:border-gray-300/60"
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs font-semibold text-gray-900 truncate">{title}</span>
                                    <span className="text-[10px] text-gray-500 whitespace-nowrap">{timeDisplay}</span>
                                  </div>
                                  {/* 🔴 FIX: 종합점수 대신 AI 키워드 표시 */}
                                  <div className="flex items-center gap-1.5">
                                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-gradient-to-r from-blue-100 to-purple-100 text-blue-800 border border-blue-200">
                                      {generateAIKeyword(analysis)}
                                    </span>
                                  </div>
                                </div>
                                <Button
                                  size="sm"
                                  onClick={() => {
                                    setAnalysisResult(analysis)
                                    setPhase("complete")
                                    if (typeof window !== "undefined") {
                                      sessionStorage.setItem("current_analysis", JSON.stringify(analysis))
                                      sessionStorage.setItem("is_analyzing", "true")
                                    }
                                    window.scrollTo({ top: 0, behavior: "smooth" })
                                  }}
                                  className="h-7 px-2.5 text-[11px] bg-gray-900 hover:bg-gray-800 text-white rounded-md flex items-center gap-1 whitespace-nowrap flex-shrink-0"
                                >
                                  바로가기
                                  <ChevronRight className="w-3 h-3" />
                                </Button>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </GlassCard>
                  )}
                </motion.div>
              )}

              {(phase === "uploading" || phase === "ocr" || phase === "analyzing" || phase === "analysisComplete") && (
                <motion.div
                  key="process"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-2"
                >
                  {(phase === "ocr" || phase === "analyzing") && (
                    <>
                      {progressMessage && (
                        <GlassCard className="p-2 text-center">
                          <p className="text-xs font-medium text-gray-700">{progressMessage}</p>
                        </GlassCard>
                      )}

                      <GlassCard className="p-2 space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-gray-700">
                            {phase === "ocr" ? "텍스트 추출 중..." : "AI 분석 중..."}
                          </span>
                          {phase === "ocr" && (
                            <span className="text-xs font-bold text-gray-900">{Math.floor(ocrProgress)}%</span>
                          )}
                        </div>
                        {phase === "ocr" && (
                          <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                            <motion.div
                              className="h-full bg-gradient-to-r from-gray-800 via-gray-600 to-gray-800 rounded-full shadow-lg relative overflow-hidden"
                              style={{ width: `${ocrProgress}%` }}
                              transition={{ duration: 0.2, ease: "easeOut" }}
                            >
                              {/* 🔴 PHASE 5: Enhanced loading shimmer - seamless infinite loop */}
                              <motion.div
                                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent"
                                animate={{
                                  x: ["-100%", "200%"],
                                }}
                                transition={{
                                  duration: 1.5,
                                  repeat: Infinity,
                                  ease: "linear",
                                  repeatType: "loop",
                                  repeatDelay: 0, // 🔴 No gap between loops
                                }}
                                style={{
                                  willChange: "transform",
                                }}
                              />
                              {/* 🔴 PHASE 5: Secondary shimmer for depth */}
                              <motion.div
                                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                                animate={{
                                  x: ["-100%", "200%"],
                                }}
                                transition={{
                                  duration: 2.2,
                                  repeat: Infinity,
                                  ease: "linear",
                                  repeatType: "loop",
                                  repeatDelay: 0,
                                }}
                                style={{
                                  willChange: "transform",
                                }}
                              />
                            </motion.div>
                          </div>
                        )}
                        {phase === "analyzing" && (
                          <div className="flex justify-center py-4">
                            {/* 🔴 UX REDESIGN: AI Scanning Animation - 스캐닝 느낌의 AI 분석 애니메이션 */}
                            <div className="relative w-32 h-32">
                              {/* 중앙 스캔 코어 - Brain Icon */}
                              <motion.div
                                className="absolute inset-0 flex items-center justify-center"
                                animate={{
                                  scale: [1, 1.05, 1],
                                }}
                                transition={{
                                  duration: 2,
                                  repeat: Infinity,
                                  ease: "easeInOut",
                                }}
                              >
                                <div className="relative w-14 h-14 bg-gradient-to-br from-blue-600 via-blue-500 to-cyan-500 rounded-2xl shadow-2xl flex items-center justify-center">
                                  {/* Brain/AI Icon SVG */}
                                  <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                  </svg>
                                </div>
                              </motion.div>
                              
                              {/* Horizontal Scanning Lines - 상하로 이동하는 스캔선들 */}
                              {[0, 1, 2, 3, 4].map((i) => (
                                <motion.div
                                  key={`scan-${i}`}
                                  className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-blue-400 to-transparent"
                                  style={{
                                    filter: 'blur(1px)',
                                  }}
                                  animate={{
                                    top: ['0%', '100%'],
                                    opacity: [0, 1, 1, 0],
                                  }}
                                  transition={{
                                    duration: 2,
                                    repeat: Infinity,
                                    ease: "linear",
                                    delay: i * 0.4,
                                  }}
                                />
                              ))}

                              
                              {/* Rotating Radar Sweep - 회전하는 레이더 스윕 */}
                              <motion.div
                                className="absolute inset-0"
                                animate={{
                                  rotate: [0, 360],
                                }}
                                transition={{
                                  duration: 3,
                                  repeat: Infinity,
                                  ease: "linear",
                                }}
                              >
                                <div 
                                  className="absolute top-1/2 left-1/2 w-full h-0.5 origin-left"
                                  style={{
                                    background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.8) 0%, rgba(59, 130, 246, 0) 100%)',
                                    transform: 'translateY(-50%)',
                                    filter: 'blur(2px)',
                                  }}
                                />
                              </motion.div>
                              
                              {/* Data Points - 스캔된 데이터 포인트들 */}
                              {[0, 1, 2, 3, 4, 5].map((i) => {
                                const angle = (i * Math.PI * 2) / 6;
                                const radius = 45;
                                return (
                                  <motion.div
                                    key={`data-${i}`}
                                    className="absolute w-2 h-2 bg-cyan-400 rounded-full shadow-lg"
                                    style={{
                                      left: '50%',
                                      top: '50%',
                                      marginLeft: Math.cos(angle) * radius - 4,
                                      marginTop: Math.sin(angle) * radius - 4,
                                    }}
                                    animate={{
                                      scale: [0.8, 1.3, 0.8],
                                      opacity: [0.4, 1, 0.4],
                                    }}
                                    transition={{
                                      duration: 2,
                                      repeat: Infinity,
                                      ease: "easeInOut",
                                      delay: i * 0.2,
                                    }}
                                  />
                                );
                              })}

                              
                              {/* Scanning Rings - 확산되는 스캔 링들 */}
                              {[0, 1, 2].map((ring) => (
                                <motion.div
                                  key={`ring-${ring}`}
                                  className="absolute inset-0"
                                  animate={{
                                    scale: [0.5, 1.5],
                                    opacity: [0.8, 0],
                                  }}
                                  transition={{
                                    duration: 2.5,
                                    repeat: Infinity,
                                    ease: "easeOut",
                                    delay: ring * 0.8,
                                  }}
                                >
                                  <div className="absolute inset-0 border border-blue-400 rounded-full" />
                                </motion.div>
                              ))}
                              
                              {/* Static Grid Ring - 정적 그리드 링 */}
                              <div className="absolute inset-0 border border-dashed border-gray-300 rounded-full opacity-30" />
                              <div className="absolute inset-0 border border-dashed border-gray-300 rounded-full opacity-20" style={{ transform: 'scale(0.65)' }} />
                            </div>
                          </div>
                        )}
                      </GlassCard>

                      <div className="grid grid-cols-3 gap-1.5">
                        <ProcessCard 
                          icon={Upload} 
                          title="업로드" 
                          active={phase === "uploading"} 
                          complete={phase !== "uploading"} 
                        />
                        <ProcessCard
                          icon={Sparkles}
                          title="AI 분석"
                          active={phase === "ocr" || phase === "analyzing"}
                          complete={phase === "analysisComplete"}
                        />
                        <ProcessCard 
                          icon={CheckCircle2} 
                          title="완료" 
                          active={false} 
                          complete={phase === "analysisComplete"} 
                        />
                      </div>

                      {uploadedImageUrls.length > 0 && (
                        <GlassCard className="p-4">
                          {/* UX-09: Stacked Card UI with 3D effects and tap-to-view */}
                          <StackedImageCards 
                            imageUrls={uploadedImageUrls} 
                            readonly={true}
                          />
                        </GlassCard>
                      )}

                      {currentTip && (
                        <div className="text-center px-4 pt-2">
                          <p className="text-[10px] text-gray-500 italic">💡 {currentTip}</p>
                        </div>
                      )}
                    </>
                  )}

                  <AnimatePresence mode="wait">
                    {phase === "analysisComplete" && showAnalysisComplete && (
                      <motion.div
                        key="completion-popup"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8, transition: { duration: 0.2 } }}
                        transition={{
                          type: "spring",
                          stiffness: 300,
                          damping: 25,
                        }}
                        className="flex items-center justify-center"
                      >
                        <GlassCard className="p-10 text-center" glow>
                          <motion.div
                            initial={{ scale: 0.8 }}
                            animate={{
                              scale: [0.8, 1.1, 1],
                            }}
                            transition={{
                              duration: 0.6,
                              ease: "easeOut",
                              times: [0, 0.6, 1],
                            }}
                            className="w-28 h-28 rounded-full bg-gradient-to-br from-green-100 to-green-200 flex items-center justify-center mx-auto mb-4 shadow-lg"
                          >
                            <CheckCircle2 className="w-14 h-14 text-green-700" />
                          </motion.div>
                          <h3 className="text-4xl font-bold text-gray-900 mb-2">분석 완료!</h3>
                          <p className="text-lg text-gray-600">결과를 확인하세요.</p>
                        </GlassCard>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}

              {phase === "complete" && analysisResult && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="space-y-2.5 pb-4"
                >
                  <GlassCard className="p-3.5 text-center rounded-2xl" glow>
                    <div className="space-y-2">
                      <div className="text-4xl font-bold text-black">{analysisResult.overallScore}점</div>
                      <p className="text-xs text-gray-600">종합 평가 (상위 12%)</p>

                      {analysisResult.studentProfile && (
                        <div className="pt-1.5 border-t border-gray-200/50">
                          <div className="flex items-center justify-center gap-1.5 mb-0.5">
                            <User className="w-3.5 h-3.5 text-blue-600" />
                            <p className="text-xs font-semibold text-gray-700">학생의 전문성은?</p>
                          </div>
                          <p className="text-xs text-gray-600">{analysisResult.studentProfile}.</p>
                        </div>
                      )}
                    </div>
                  </GlassCard>

                  {analysisResult.careerAlignment && (
                    <GlassCard className="p-3 space-y-2 bg-gradient-to-br from-blue-50/80 to-purple-50/80 rounded-2xl">
                      <div className="flex items-center gap-2">
                        <Compass className="w-4.5 h-4.5 text-blue-600" />
                        <h3 className="text-sm font-semibold text-gray-900">진로 적합성</h3>
                      </div>
                      <div className="text-center py-1.5">
                        <div className="text-3xl font-bold text-blue-600">
                          {analysisResult.careerAlignment.percentage}%
                        </div>
                        <p className="text-xs text-gray-600 mt-0.5">{analysisResult.careerAlignment.summary}</p>
                      </div>
                    </GlassCard>
                  )}

                  <GlassCard className="p-3 space-y-2 rounded-2xl">
                    <div className="flex gap-2 border-b border-gray-200">
                      <button
                        onClick={() => setResultTab("strengths")}
                        className={`flex-1 pb-1.5 text-sm font-semibold transition-colors ${
                          resultTab === "strengths"
                            ? "text-green-700 border-b-2 border-green-600"
                            : "text-gray-400 hover:text-gray-600"
                        }`}
                      >
                        강점
                      </button>
                      <button
                        onClick={() => setResultTab("improvements")}
                        className={`flex-1 pb-1.5 text-sm font-semibold transition-colors ${
                          resultTab === "improvements"
                            ? "text-orange-700 border-b-2 border-orange-600"
                            : "text-gray-400 hover:text-gray-600"
                        }`}
                      >
                        보완
                      </button>
                    </div>
                    <ul className="space-y-1.5">
                      {resultTab === "strengths"
                        ? analysisResult.strengths.map((strength, idx) => (
                            <li key={idx} className="text-xs text-gray-700 leading-relaxed flex gap-2">
                              <span className="text-green-600 flex-shrink-0">•</span>
                              <span>{strength}.</span>
                            </li>
                          ))
                        : analysisResult.improvements.map((improvement, idx) => (
                            <li key={idx} className="text-xs text-gray-700 leading-relaxed flex gap-2">
                              <span className="text-orange-600 flex-shrink-0">•</span>
                              <span>{improvement}.</span>
                            </li>
                          ))}
                    </ul>
                  </GlassCard>

                  {analysisResult.errors.length > 0 && (
                    <GlassCard className="p-3 space-y-2 rounded-2xl">
                      <h3 className="text-sm font-semibold text-black flex items-center gap-1.5">
                        <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
                        오류 {analysisResult.errors.length}건
                      </h3>
                      <div className="space-y-1.5">
                        {analysisResult.errors.map((error, idx) => (
                          <div
                            key={idx}
                            className={`p-2.5 rounded-xl border space-y-1.5 ${
                              error.type === "금지"
                                ? "bg-red-50/60 border-red-200/60"
                                : "bg-orange-50/60 border-orange-200/60"
                            }`}
                          >
                            <div className="flex items-start gap-2">
                              <span
                                className={`px-1.5 py-0.5 rounded-full text-[9px] font-bold flex-shrink-0 ${
                                  error.type === "금지" ? "bg-red-600 text-white" : "bg-orange-500 text-white"
                                }`}
                              >
                                {error.type}
                              </span>
                              <div className="flex-1 space-y-0.5 min-w-0">
                                <p className="text-xs font-medium text-gray-900">"{error.content}"</p>
                                <p className="text-[10px] text-gray-600 leading-relaxed">{error.reason}.</p>
                                {error.suggestion && (
                                  <div className="mt-1.5 p-1.5 bg-green-50/80 rounded-lg border border-green-200/60">
                                    <p className="text-[10px] text-green-800 font-medium">✓ 올바른 표현:</p>
                                    <p className="text-[10px] text-green-700 mt-0.5">{error.suggestion}.</p>
                                  </div>
                                )}
                                <button
                                  onClick={() => handleTeacherHelp(error)}
                                  className="mt-1 text-[10px] text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1"
                                >
                                  <MessageSquare className="w-3 h-3" />
                                  선생님께 요청
                                </button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </GlassCard>
                  )}

                  {analysisResult.suggestions.length > 0 && (
                    <GlassCard className="p-3 space-y-1.5 rounded-2xl">
                      <h3 className="text-sm font-semibold text-black flex items-center gap-1.5">
                        <Sparkles className="w-4 h-4 text-blue-600 flex-shrink-0" />
                        개선 제안
                      </h3>
                      <ul className="space-y-1">
                        {analysisResult.suggestions.map((suggestion, idx) => (
                          <li key={idx} className="text-xs text-gray-700 leading-relaxed">
                            • {suggestion}.
                          </li>
                        ))}
                      </ul>
                    </GlassCard>
                  )}

                  <Button
                    size="sm"
                    className="w-full rounded-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white h-11 text-sm font-medium shadow-lg"
                    onClick={() => setShowProjectRecommender(true)}
                  >
                    <Lightbulb className="w-4 h-4 mr-1.5" />
                    나에게 딱맞는 자율과제 추천받기!
                  </Button>

                  <Button
                    size="sm"
                    variant="outline"
                    className="w-full rounded-full border-2 border-blue-200 hover:bg-blue-50 h-9 text-sm bg-white font-medium text-blue-700"
                    onClick={() => setShowUniversityPredictor(true)}
                  >
                    <TrendingUp className="w-4 h-4 mr-1.5" />
                    대학 예측
                  </Button>

                  <Button
                    size="sm"
                    variant="outline"
                    className="w-full rounded-full border-2 border-purple-200 hover:bg-purple-50 h-9 text-sm bg-white font-medium text-purple-700"
                    onClick={() => setShowAIKiller(true)}
                  >
                    <Shield className="w-4 h-4 mr-1.5" />
                    AI작성 탐지
                  </Button>

                  <div className="flex flex-col gap-2 pt-1">
                    <Button
                      size="lg"
                      className="w-full rounded-full bg-black hover:bg-gray-900 text-white h-11 text-sm font-medium shadow-lg"
                      onClick={() => router.push("/results")}
                    >
                      <Eye className="w-4 h-4 mr-1.5" />
                      상세 결과
                    </Button>
                    <div className="grid grid-cols-3 gap-1.5 mt-1">
                      <Button
                        size="lg"
                        variant="outline"
                        className="rounded-full border-2 border-gray-200 hover:bg-gray-50 h-9 text-sm font-medium flex items-center justify-center gap-1 bg-transparent"
                        onClick={handleShareClick}
                      >
                        <Share2 className="w-3.5 h-3.5 flex-shrink-0" />
                        <span className="text-xs">공유</span>
                      </Button>
                      <Button
                        size="lg"
                        variant="outline"
                        className="rounded-full border-2 border-gray-200 hover:bg-gray-50 h-9 text-sm font-medium flex items-center justify-center gap-1 bg-transparent"
                        onClick={handleExportPDF}
                      >
                        <Download className="w-3.5 h-3.5 flex-shrink-0" />
                        <span className="text-xs">다운로드</span>
                      </Button>
                      <Button
                        size="lg"
                        variant="outline"
                        className="rounded-full border-2 border-gray-200 hover:bg-gray-50 h-9 text-sm font-medium flex items-center justify-center gap-1 bg-transparent"
                        onClick={resetAnalysis}
                      >
                        <X className="w-3.5 h-3.5 flex-shrink-0" />
                        <span className="text-xs">종료</span>
                      </Button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {showShareDialog && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                className="w-full max-w-sm"
              >
                <GlassCard className="p-4 space-y-3 rounded-2xl">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-black">공유 설정</h3>
                    <button
                      onClick={() => setShowShareDialog(false)}
                      className="p-1 hover:bg-gray-100 rounded-full transition-colors"
                    >
                      <X className="w-5 h-5 text-gray-500" />
                    </button>
                  </div>

                  <div className="space-y-2">
                    <div>
                      <label className="text-xs font-medium text-gray-700 mb-1 block">학번 (4자리 숫자)</label>
                      <Input
                        value={shareData.studentId}
                        onChange={(e) => {
                          const value = e.target.value
                          // Only allow 4-digit numbers
                          if (value === '' || (/^\d{1,4}$/.test(value))) {
                            setShareData({ ...shareData, studentId: value })
                          }
                        }}
                        maxLength={4}
                        placeholder=""
                        className="h-9 text-sm"
                      />
                    </div>

                    <div>
                      <label className="text-xs font-medium text-gray-700 mb-1 block">이름</label>
                      <Input
                        value={shareData.name}
                        onChange={(e) => setShareData({ ...shareData, name: e.target.value })}
                        placeholder=""
                        className="h-9 text-sm"
                      />
                    </div>

                    <div className="p-2 bg-blue-50 rounded-lg border border-blue-200">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <p className="text-xs font-medium text-gray-800">비공개로 저장</p>
                          <p className="text-[10px] text-gray-600 mt-0.5">탐색 페이지에 공유 안 함</p>
                        </div>
                        <Switch
                          checked={shareData.isPrivate}
                          onCheckedChange={(checked) => setShareData({ ...shareData, isPrivate: checked })}
                        />
                      </div>
                    </div>

                    <div className="p-2 bg-amber-50 rounded-lg border border-amber-200">
                      <div className="flex items-start gap-2">
                        <input
                          type="checkbox"
                          checked={shareData.agreedToTerms}
                          onChange={(e) => setShareData({ ...shareData, agreedToTerms: e.target.checked })}
                          className="mt-0.5"
                        />
                        <div className="flex-1">
                          <p className="text-xs text-gray-800 leading-relaxed">
                            민감한 정보는 자동으로
                            <br />
                            암호화 처리되어 게시됩니다.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-2 pt-1">
                    <Button
                      onClick={() => setShowShareDialog(false)}
                      variant="outline"
                      className="flex-1 rounded-full border-2 border-gray-200 hover:bg-gray-50 h-10 text-sm font-medium"
                    >
                      취소
                    </Button>
                    <Button
                      onClick={handleShareConfirm}
                      className="flex-1 rounded-full bg-black hover:bg-gray-900 text-white h-10 text-sm font-medium"
                    >
                      {shareData.isPrivate ? "비공개 저장" : "공유하기"}
                    </Button>
                  </div>
                </GlassCard>
              </motion.div>
            </div>
          )}

          {showTeacherHelper && selectedError && (
            <LazyTeacherCommunicationHelper
              error={selectedError}
              onClose={() => {
                setShowTeacherHelper(false)
                setSelectedError(null)
              }}
            />
          )}

          {showAIKiller && analysisResult && (
            <LazyAIKillerDetector analysisResult={analysisResult} onClose={() => setShowAIKiller(false)} />
          )}

          {showUniversityPredictor && analysisResult && (
            <LazyUniversityPredictor analysisResult={analysisResult} onClose={() => setShowUniversityPredictor(false)} />
          )}

          {showProjectRecommender && analysisResult && (
            <LazyProjectRecommender
              analysisResult={analysisResult}
              careerDirection={careerDirection}
              onClose={() => setShowProjectRecommender(false)}
            />
          )}
        </div>
      </div>
    </AuthGuard>
  )
}

interface ProcessCardProps {
  icon: React.ElementType
  title: string
  active: boolean
  complete: boolean
}

function ProcessCard({ icon: Icon, title, active, complete }: ProcessCardProps) {
  return (
    <GlassCard
      className={`p-3 text-center space-y-1.5 transition-all duration-500 ${
        active ? "ring-2 ring-black shadow-lg scale-105" : ""
      } ${complete ? "bg-gray-50/30" : ""}`}
      hover={false}
    >
      <div className="flex justify-center">
        {active && !complete ? (
          <motion.div
            animate={{ rotate: [0, 360] }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: "linear",
            }}
            style={{
              willChange: "transform",
              transformOrigin: "center center",
            }}
          >
            <Loader2 className="w-5 h-5 text-black" />
          </motion.div>
        ) : complete ? (
          <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.5 }}>
            <CheckCircle2 className="w-5 h-5 text-black" />
          </motion.div>
        ) : (
          <Icon className="w-5 h-5 text-gray-300" />
        )}
      </div>
      <h4 className="font-medium text-xs text-black">{title}</h4>
    </GlassCard>
  )
}
