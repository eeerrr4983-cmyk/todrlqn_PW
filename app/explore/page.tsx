"use client"

import { useState, useEffect, useRef } from "react"
import { motion } from "framer-motion"
import { Navigation } from "@/components/navigation"
import { LiquidBackground } from "@/components/liquid-background"
import { GlassCard } from "@/components/glass-card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
  FileText,
  Heart,
  Search,
  Clock,
  Sparkles,
  TrendingUp,
  Bookmark,
  MessageCircle,
  Crown,
  Send,
  Lock,
  CheckCircle2,
  AlertCircle,
  X,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Users,
  Filter,
} from "lucide-react"
import { StorageManager } from "@/components/storage-manager"
import type { AnalysisResult, Comment, Reply } from "@/lib/types"
import { useRouter } from "next/navigation"
import { AIMentoring } from "@/components/ai-mentoring"
import { AuthModal } from "@/components/auth-modal"
import { useAuth } from "@/lib/auth-context"
import { getUserDisplayName, getUserStudentId } from "@/lib/user-session"

type SortOption = "recent" | "popular"
type TabOption = "all" | "saved"
type CategoryFilter = "all" | "science" | "humanities" // 🔴 PHASE 9: 이과/문과 필터

// 🔴 PHASE 7: AI 키워드 생성 함수 (app/page.tsx와 동일)
const generateAIKeyword = (analysis: AnalysisResult): string => {
  // 🔴 FIX: 진로방향이 있고 "미지정"이 아닌 경우에만 사용
  if (analysis.careerDirection && analysis.careerDirection.trim() && analysis.careerDirection !== '미지정') {
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

// 🔴 PHASE 7: 2줄 AI 요약 생성 (민감정보 필터링 포함)
const generateAISummary = (analysis: AnalysisResult): string => {
  const strengths = analysis.strengths || []
  const improvements = analysis.improvements || []
  
  // 민감정보 필터링 함수
  const filterSensitiveInfo = (text: string): string => {
    return text
      .replace(/[가-힣]{2,4}(중학교|고등학교|대학교|초등학교)/g, 'XX학교')
      .replace(/[가-힣]{2,4}시\s?[가-힣]{2,4}구/g, 'XX시 XX구')
      .replace(/\d{3}-\d{4}-\d{4}/g, 'XXX-XXXX-XXXX')
      .replace(/[가-힣]{3,4}\s?선생님/g, 'XX 선생님')
      .replace(/19\d{2}년|20\d{2}년/g, 'XXXX년')
  }
  
  // 🔴 UX FIX: 주요 강점 1개 선택 (더 짧게 제한하여 "..." 방지)
  const mainStrength = strengths.length > 0 
    ? strengths.reduce((longest, current) => 
        current.length > longest.length ? current : longest, strengths[0]
      ).substring(0, 50) // 50자로 단축 (기존 80자)
    : '다양한 활동 경험'
  
  // 🔴 UX FIX: 주요 개선점 1개 선택 (더 짧게 제한)
  const mainImprovement = improvements.length > 0
    ? improvements[0].substring(0, 50) // 50자로 단축 (기존 80자)
    : '추가 발전 가능'
  
  // 2줄 요약 생성 (민감정보 필터링 적용)
  const line1 = filterSensitiveInfo(`강점: ${mainStrength}`)
  const line2 = filterSensitiveInfo(`개선: ${mainImprovement}`)
  
  return `${line1}... ${line2}...`
}

// Smart timestamp formatter: 몇시간전/어제/MM.DD format
const formatCommentTimestamp = (dateString: string): string => {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMinutes = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffHours / 24)

  if (diffMinutes < 1) {
    return "방금 전"
  } else if (diffMinutes < 60) {
    return `${diffMinutes}분 전`
  } else if (diffHours < 24) {
    return `${diffHours}시간 전`
  } else if (diffDays === 1) {
    return "어제"
  } else {
    // Format as MM.DD
    const month = (date.getMonth() + 1).toString().padStart(2, '0')
    const day = date.getDate().toString().padStart(2, '0')
    return `${month}.${day}`
  }
}
type DetailTabOption = "strengths" | "improvements"

export default function ExplorePage() {
  const { user } = useAuth()
  const [searchQuery, setSearchQuery] = useState("")
  const [sortBy, setSortBy] = useState<SortOption>("recent")
  const [tab, setTab] = useState<TabOption>("all")
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([])
  const [trendingAnalyses, setTrendingAnalyses] = useState<AnalysisResult[]>([])
  const [recommendedAnalyses, setRecommendedAnalyses] = useState<AnalysisResult[]>([])
  const [interaction, setInteraction] = useState({ likedAgents: new Set<string>(), savedAgents: new Set<string>(), likedComments: new Set<string>() })
  const [detailTab, setDetailTab] = useState<DetailTabOption>("strengths")
  const [isRefreshing, setIsRefreshing] = useState(false)
  const router = useRouter()
  const [showAIMentoring, setShowAIMentoring] = useState(false)
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>("all") // 🔴 PHASE 9
  const [expandedSubKeywords, setExpandedSubKeywords] = useState(false) // 🔴 PHASE 9
  const [isCategoryFilterExpanded, setIsCategoryFilterExpanded] = useState(false) // 🔴 PHASE 12: Collapsible category filter
  const [detailedKeywordFilter, setDetailedKeywordFilter] = useState<string | null>(null) // 🔴 UX: Detailed keyword filter

  const isGuest = user?.isGuest || false

  useEffect(() => {
    // Load interaction data on client-side only
    if (typeof window !== "undefined") {
      setInteraction(StorageManager.getInteraction())
    }
    loadAnalyses()
    
    // Listen for profile click from navigation
    const handleOpenAuthModal = () => {
      if (isGuest) {
        setShowAuthModal(true)
      }
    }
    window.addEventListener('openAuthModal', handleOpenAuthModal)
    return () => window.removeEventListener('openAuthModal', handleOpenAuthModal)
  }, [isGuest])

  useEffect(() => {
    setRecommendedAnalyses(StorageManager.getPersonalizedRecommendations(searchQuery))
  }, [searchQuery])

  const loadAnalyses = () => {
    const allAnalyses = StorageManager.getPublicAnalyses()
    setAnalyses(allAnalyses)
    setTrendingAnalyses(StorageManager.getTrendingAnalyses())
  }

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await new Promise((resolve) => setTimeout(resolve, 600))
    loadAnalyses()
    setIsRefreshing(false)
  }

  const handleMentoringClick = () => {
    if (isGuest) {
      setShowAuthModal(true)
    } else {
      setShowAIMentoring(true)
    }
  }

  // 🔴 PHASE 9: 카테고리 분류 함수
  const getCategoryFromKeyword = (keyword: string): CategoryFilter => {
    const scienceKeywords = ['컴퓨터공학과', '기계공학과', '전자공학과', '이공계열', '의예과', '간호학과']
    const humanitiesKeywords = ['경영학과', '인문계열', '예체능계열']
    
    if (scienceKeywords.includes(keyword)) return 'science'
    if (humanitiesKeywords.includes(keyword)) return 'humanities'
    return 'all'
  }

  const filteredAnalyses = analyses
    .filter((analysis) => {
      if (tab === "saved" && !interaction.savedAgents.has(analysis.id)) return false
      
      // 🔴 PHASE 9: 카테고리 필터링
      if (categoryFilter !== "all") {
        const keyword = generateAIKeyword(analysis)
        const category = getCategoryFromKeyword(keyword)
        if (category !== categoryFilter && category !== 'all') return false
      }
      
      if (searchQuery) {
        const query = searchQuery.toLowerCase().trim()
        const studentTitle = `${analysis.studentId || ""}${analysis.studentName}`.toLowerCase()
        const strengthsText = analysis.strengths.join(" ").toLowerCase()
        const improvementsText = analysis.improvements.join(" ").toLowerCase()

        return (
          studentTitle.includes(query) ||
          analysis.studentName.toLowerCase().includes(query) ||
          (analysis.studentId && analysis.studentId.toLowerCase().includes(query)) ||
          strengthsText.includes(query) ||
          improvementsText.includes(query)
        )
      }
      return true
    })
    .sort((a, b) => {
      if (sortBy === "recent") {
        return new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime()
      } else {
        return b.likes - a.likes
      }
    })

  const topAnalysis = sortBy === "popular" && filteredAnalyses.length > 0 ? filteredAnalyses[0] : null

  const toggleLike = (analysisId: string) => {
    const newInteraction = { ...interaction }
    const analysis = analyses.find((a) => a.id === analysisId)
    if (!analysis) return

    if (newInteraction.likedAgents.has(analysisId)) {
      newInteraction.likedAgents.delete(analysisId)
      StorageManager.updateAnalysis(analysisId, { likes: Math.max(0, analysis.likes - 1) })
    } else {
      newInteraction.likedAgents.add(analysisId)
      StorageManager.updateAnalysis(analysisId, { likes: analysis.likes + 1 })
    }
    setInteraction(newInteraction)
    StorageManager.saveInteraction(newInteraction)
    loadAnalyses()
  }

  const toggleSave = (analysisId: string) => {
    const newInteraction = { ...interaction }
    const analysis = analyses.find((a) => a.id === analysisId)
    if (!analysis) return

    if (newInteraction.savedAgents.has(analysisId)) {
      newInteraction.savedAgents.delete(analysisId)
      StorageManager.updateAnalysis(analysisId, { saves: Math.max(0, analysis.saves - 1) })
    } else {
      newInteraction.savedAgents.add(analysisId)
      StorageManager.updateAnalysis(analysisId, { saves: analysis.saves + 1 })
    }
    setInteraction(newInteraction)
    StorageManager.saveInteraction(newInteraction)
    loadAnalyses()
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-gray-50">
      <LiquidBackground />
      <Navigation />

      <div className="relative z-10 h-full overflow-y-auto px-4 pb-20 pt-4">
        <div className="max-w-4xl mx-auto space-y-2.5">
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center space-y-0.5 mb-1.5"
          >
            <h1 className="text-2xl font-semibold tracking-tight text-gray-900">탐색</h1>
            <p className="text-xs text-gray-600">새로운 생기부를 만나보세요.</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <GlassCard className="p-2.5 space-y-2">
              <div className="flex gap-1.5">
                <Button
                  onClick={handleMentoringClick}
                  className="flex-1 rounded-lg bg-gray-50 hover:bg-gray-100 text-gray-600 hover:text-gray-800 h-7 text-xs font-medium border border-gray-200 transition-all"
                >
                  <Users className="w-3 h-3 mr-1.5" />
                  AI 멘토링 매칭
                </Button>
              </div>

              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="생기부 검색"
                    className="pl-9 h-9 bg-white border border-gray-200 focus:border-gray-900 focus:ring-1 focus:ring-gray-900 rounded-lg text-sm placeholder:text-gray-400"
                  />
                </div>
              </div>

              {searchQuery && recommendedAnalyses.length > 0 && (
                <div className="space-y-1">
                  <p className="text-xs text-gray-500 font-medium">맞춤 추천</p>
                  <div className="flex flex-wrap gap-1.5">
                    {/* 🔴 PHASE 7: 점수 → AI 키워드 변경 */}
                    {recommendedAnalyses.slice(0, 3).map((analysis) => (
                      <Badge
                        key={analysis.id}
                        variant="outline"
                        className="cursor-pointer hover:bg-black hover:text-white transition-colors text-xs"
                        onClick={() => setSearchQuery(analysis.studentName)}
                      >
                        {generateAIKeyword(analysis)}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {trendingAnalyses.length > 0 && (
                <div className="space-y-1">
                  <p className="text-xs text-gray-500 font-medium">인기 급상승</p>
                  <div className="flex flex-wrap gap-1.5">
                    {trendingAnalyses.slice(0, 3).map((analysis) => {
                      const studentTitle = `${analysis.studentId || "1405"}${analysis.studentName.replace(/\*/g, "정후")}`
                      return (
                        <Badge
                          key={analysis.id}
                          variant="outline"
                          className="cursor-pointer hover:bg-black hover:text-white transition-colors text-xs"
                          onClick={() => {
                            const card = document.querySelector(`[data-analysis-id="${analysis.id}"]`)
                            if (card) {
                              card.scrollIntoView({ behavior: "smooth", block: "center" })
                              setTimeout(() => {
                                const detailButton = card.querySelector(
                                  "button[data-detail-button]",
                                ) as HTMLButtonElement
                                detailButton?.click()
                              }, 500)
                            }
                          }}
                        >
                          <TrendingUp className="w-3 h-3 mr-1" />
                          {studentTitle}
                        </Badge>
                      )
                    })}
                  </div>
                </div>
              )}

              <div className="flex flex-wrap gap-2 items-center">
                <div className="flex gap-1.5">
                  <Button
                    variant={tab === "all" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setTab("all")}
                    className={`rounded-full text-xs h-7 px-3 font-medium ${
                      tab === "all"
                        ? "bg-gray-900 text-white hover:bg-gray-800"
                        : "border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                    }`}
                  >
                    전체
                  </Button>
                  <Button
                    variant={tab === "saved" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setTab("saved")}
                    className={`rounded-full text-xs h-7 px-3 font-medium ${
                      tab === "saved"
                        ? "bg-gray-900 text-white hover:bg-gray-800"
                        : "border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                    }`}
                  >
                    <Bookmark className="w-3 h-3 mr-1" />
                    저장됨
                  </Button>
                </div>

                <div className="h-4 w-px bg-gray-200" />

                {/* 🔴 PHASE 12: 카테고리 필터 - 접을 수 있는 아이콘 형태 */}
                {!isCategoryFilterExpanded ? (
                  <motion.button
                    onClick={() => setIsCategoryFilterExpanded(true)}
                    whileHover={{ scale: 1.08 }}
                    whileTap={{ scale: 0.92 }}
                    className={`p-2 rounded-full transition-all relative ${
                      categoryFilter !== "all" 
                        ? "bg-blue-100 hover:bg-blue-200" 
                        : "hover:bg-gray-100"
                    }`}
                    title={categoryFilter !== "all" ? `${categoryFilter === 'science' ? '이과' : '문과'} 필터 활성` : "카테고리 필터"}
                  >
                    <Filter className={`w-3.5 h-3.5 transition-colors ${
                      categoryFilter !== "all" ? "text-blue-600" : "text-gray-600"
                    }`} />
                    {categoryFilter !== "all" && (
                      <motion.span
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-blue-500 rounded-full border-2 border-white"
                      />
                    )}
                  </motion.button>
                ) : (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex gap-1.5 items-center"
                  >
                    <Button
                      variant={categoryFilter === "all" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setCategoryFilter("all")}
                      className={`rounded-full text-xs h-7 px-3 font-medium ${
                        categoryFilter === "all"
                          ? "bg-gray-900 text-white hover:bg-gray-800"
                          : "border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                      }`}
                    >
                      모두
                    </Button>
                    <Button
                      variant={categoryFilter === "science" ? "default" : "outline"}
                      size="sm"
                      onClick={() => {
                        setCategoryFilter("science")
                        setExpandedSubKeywords(!expandedSubKeywords)
                      }}
                      className={`rounded-full text-xs h-7 px-3 font-medium ${
                        categoryFilter === "science"
                          ? "bg-blue-600 text-white hover:bg-blue-700"
                          : "border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                      }`}
                    >
                      이과
                    </Button>
                    <Button
                      variant={categoryFilter === "humanities" ? "default" : "outline"}
                      size="sm"
                      onClick={() => {
                        setCategoryFilter("humanities")
                        setExpandedSubKeywords(!expandedSubKeywords)
                      }}
                      className={`rounded-full text-xs h-7 px-3 font-medium ${
                        categoryFilter === "humanities"
                          ? "bg-purple-600 text-white hover:bg-purple-700"
                          : "border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                      }`}
                    >
                      문과
                    </Button>
                    <motion.button
                      onClick={() => setIsCategoryFilterExpanded(false)}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      className="p-0.5 hover:bg-gray-100 rounded-full transition-colors"
                      title="접기"
                    >
                      <X className="w-3 h-3 text-gray-500" />
                    </motion.button>
                  </motion.div>
                )}

                <div className="h-4 w-px bg-gray-200" />

                <div className="flex gap-1.5">
                  <Button
                    variant={sortBy === "recent" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSortBy("recent")}
                    className={`rounded-full text-xs h-7 px-3 font-medium ${
                      sortBy === "recent"
                        ? "bg-gray-900 text-white hover:bg-gray-800"
                        : "border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                    }`}
                  >
                    <Clock className="w-3 h-3 mr-1" />
                    최신순
                  </Button>
                  <Button
                    variant={sortBy === "popular" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSortBy("popular")}
                    className={`rounded-full text-xs h-7 px-3 font-medium ${
                      sortBy === "popular"
                        ? "bg-gray-900 text-white hover:bg-gray-800"
                        : "border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                    }`}
                  >
                    <TrendingUp className="w-3 h-3 mr-1" />
                    인기순
                  </Button>
                </div>

                <div className="h-4 w-px bg-gray-200" />

                <motion.button
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                  animate={{ rotate: isRefreshing ? 360 : 0 }}
                  transition={{ duration: 0.6, ease: "easeInOut" }}
                  className="p-1.5 hover:bg-gray-100 rounded-full transition-colors disabled:opacity-50"
                  title="새로고침"
                >
                  <RefreshCw className="w-3.5 h-3.5 text-gray-600" />
                </motion.button>
              </div>

              {/* 🔴 PHASE 12: 확장 가능한 하위 키워드 - 메인 필터가 펼쳐진 경우에만 표시 */}
              {categoryFilter !== "all" && expandedSubKeywords && isCategoryFilterExpanded && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-1"
                >
                  <p className="text-xs text-gray-500 font-medium">
                    {categoryFilter === "science" ? "이과 세부 분야" : "문과 세부 분야"}
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {categoryFilter === "science" ? (
                      <>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("컴퓨터공학")}
                          className="cursor-pointer hover:bg-blue-100 hover:border-blue-400 transition-all text-xs"
                        >
                          컴퓨터공학과
                        </Badge>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("기계공학")}
                          className="cursor-pointer hover:bg-blue-100 hover:border-blue-400 transition-all text-xs"
                        >
                          기계공학과
                        </Badge>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("전자공학")}
                          className="cursor-pointer hover:bg-blue-100 hover:border-blue-400 transition-all text-xs"
                        >
                          전자공학과
                        </Badge>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("의예과")}
                          className="cursor-pointer hover:bg-blue-100 hover:border-blue-400 transition-all text-xs"
                        >
                          의예과
                        </Badge>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("간호학과")}
                          className="cursor-pointer hover:bg-blue-100 hover:border-blue-400 transition-all text-xs"
                        >
                          간호학과
                        </Badge>
                      </>
                    ) : (
                      <>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("경영학과")}
                          className="cursor-pointer hover:bg-purple-100 hover:border-purple-400 transition-all text-xs"
                        >
                          경영학과
                        </Badge>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("경제")}
                          className="cursor-pointer hover:bg-purple-100 hover:border-purple-400 transition-all text-xs"
                        >
                          경제학과
                        </Badge>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("인문계열")}
                          className="cursor-pointer hover:bg-purple-100 hover:border-purple-400 transition-all text-xs"
                        >
                          인문계열
                        </Badge>
                        <Badge 
                          variant="outline" 
                          onClick={() => setSearchQuery("예체능")}
                          className="cursor-pointer hover:bg-purple-100 hover:border-purple-400 transition-all text-xs"
                        >
                          예체능계열
                        </Badge>
                      </>
                    )}
                  </div>
                </motion.div>
              )}
            </GlassCard>
          </motion.div>

          {analyses.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <GlassCard className="p-8 text-center space-y-3">
                <motion.div
                  animate={{
                    scale: [1, 1.05, 1],
                    rotate: [0, 3, -3, 0],
                  }}
                  transition={{
                    duration: 3,
                    repeat: Number.POSITIVE_INFINITY,
                    ease: "easeInOut",
                  }}
                  className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mx-auto"
                >
                  <Sparkles className="w-6 h-6 text-gray-400" />
                </motion.div>

                <div className="space-y-1">
                  <h3 className="text-base font-semibold text-gray-900">생기부를 탐지하세요.</h3>
                  <p className="text-xs text-gray-600">나만의 생기부를 AI에게 맡겨보세요.</p>
                </div>

                <Button
                  onClick={() => (window.location.href = "/")}
                  className="rounded-full bg-gray-900 hover:bg-gray-800 text-white px-6 h-9 text-sm mt-2 font-medium"
                >
                  <Sparkles className="w-4 h-4 mr-1.5" />
                  지금 시작
                </Button>
              </GlassCard>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="space-y-2.5"
            >
              <div className="flex items-center justify-between px-1">
                <h2 className="text-base font-semibold text-black">
                  {tab === "saved" ? "저장된 분석" : "모든 분석"} ({filteredAnalyses.length})
                </h2>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                {filteredAnalyses.map((analysis, index) => (
                  <motion.div
                    key={analysis.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: 0.3 + index * 0.03 }}
                  >
                    <AnalysisCard
                      analysis={analysis}
                      liked={interaction.likedAgents.has(analysis.id)}
                      saved={interaction.savedAgents.has(analysis.id)}
                      isTopRanked={topAnalysis?.id === analysis.id}
                      onToggleLike={toggleLike}
                      onToggleSave={toggleSave}
                      onReload={loadAnalyses}
                      detailTab={detailTab}
                      setDetailTab={setDetailTab}
                    />
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {showAuthModal && (
        <AuthModal
          onClose={() => setShowAuthModal(false)}
          onGuestContinue={() => {
            setShowAuthModal(false)
          }}
        />
      )}

      {showAIMentoring && !isGuest && <AIMentoring onClose={() => setShowAIMentoring(false)} />}
    </div>
  )
}

function AnalysisCard({
  analysis,
  liked,
  saved,
  isTopRanked,
  onToggleLike,
  onToggleSave,
  onReload,
  detailTab,
  setDetailTab,
}: {
  analysis: AnalysisResult
  liked: boolean
  saved: boolean
  isTopRanked: boolean
  onToggleLike: (id: string) => void
  onToggleSave: (id: string) => void
  onReload: () => void
  detailTab: DetailTabOption
  setDetailTab: (tab: DetailTabOption) => void
}) {
  const [showComments, setShowComments] = useState(false)
  const [unifiedInput, setUnifiedInput] = useState("")
  const [replyContext, setReplyContext] = useState<{
    type: "comment" | "reply" | "nested-reply"
    commentId?: string
    replyId?: string
    userName?: string
  } | null>(null)
  const [showDetail, setShowDetail] = useState(false)
  // 🔴 CRITICAL FIX: Auto-expand comments with 2+ replies
  const [expandedReplies, setExpandedReplies] = useState<Set<string>>(() => {
    const autoExpand = new Set<string>()
    const comments = analysis.comments || []
    comments.forEach((comment) => {
      const directReplies = (comment.replies || []).filter((r: any) => !r.parentReplyId)
      // Auto-expand comments with 2 or more direct replies
      if (directReplies.length >= 2) {
        autoExpand.add(comment.id)
      }
    })
    return autoExpand
  })
  const commentSectionRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const lastCommentRef = useRef<HTMLDivElement>(null)

  const comments = analysis.comments || []
  const likes = analysis.likes || 0
  const saves = analysis.saves || 0

  // Calculate total comment count (comments + all replies)
  const getTotalCommentCount = () => {
    let total = comments.length
    comments.forEach((comment) => {
      total += (comment.replies || []).length
    })
    return total
  }

  const totalCommentCount = getTotalCommentCount()

  // LB-10: Use global user session management (removed duplicate function)

  const handleSubmit = () => {
    if (!unifiedInput.trim()) return

    if (!replyContext || replyContext.type === "comment") {
      // Add new comment
      // LB-10: Using global getUserDisplayName and getUserStudentId
      const newComment: Comment = {
        id: Date.now().toString(),
        userId: getUserStudentId(),
        userName: getUserDisplayName(),
        content: unifiedInput,
        createdAt: new Date().toISOString(),
        replies: [],
        likes: 0,
      }

      const updatedComments = [...comments, newComment]
      StorageManager.updateAnalysis(analysis.id, { comments: updatedComments })
      setUnifiedInput("")
      setReplyContext(null)
      onReload()

      setTimeout(() => {
        if (commentSectionRef.current) {
          commentSectionRef.current.scrollTop = commentSectionRef.current.scrollHeight
        }
      }, 150) // 🔴 CRITICAL FIX: 더 긴 딜레이로 확실한 렌더링
    } else if (replyContext.type === "reply" && replyContext.commentId) {
      // Add reply to comment (LB-10: using global session)
      const newReply: Reply = {
        id: Date.now().toString(),
        userId: getUserStudentId(),
        userName: getUserDisplayName(),
        content: unifiedInput,
        createdAt: new Date().toISOString(),
        likes: 0,
      }

      const updatedComments = comments.map((comment) => {
        if (comment.id === replyContext.commentId) {
          return { ...comment, replies: [...(comment.replies || []), newReply] }
        }
        return comment
      })

      StorageManager.updateAnalysis(analysis.id, { comments: updatedComments })
      setUnifiedInput("")

      const newExpanded = new Set(expandedReplies)
      newExpanded.add(replyContext.commentId)
      setExpandedReplies(newExpanded)
      setReplyContext(null)
      onReload()

      setTimeout(() => {
        if (commentSectionRef.current) {
          commentSectionRef.current.scrollTop = commentSectionRef.current.scrollHeight
        }
      }, 150) // 🔴 CRITICAL FIX: 더 긴 딜레이로 확실한 렌더링
    } else if (replyContext.type === "nested-reply" && replyContext.commentId && replyContext.replyId) {
      // Nested reply (LB-10: using global session)
      const newReply: Reply = {
        id: Date.now().toString(),
        userId: getUserStudentId(),
        userName: getUserDisplayName(),
        content: unifiedInput,
        createdAt: new Date().toISOString(),
        likes: 0,
        parentReplyId: replyContext.replyId,
      }

      const updatedComments = comments.map((comment) => {
        if (comment.id === replyContext.commentId) {
          return { ...comment, replies: [...(comment.replies || []), newReply] }
        }
        return comment
      })

      StorageManager.updateAnalysis(analysis.id, { comments: updatedComments })
      setUnifiedInput("")

      const newExpanded = new Set(expandedReplies)
      newExpanded.add(replyContext.commentId)
      setExpandedReplies(newExpanded)
      setReplyContext(null)
      onReload()

      setTimeout(() => {
        if (commentSectionRef.current) {
          commentSectionRef.current.scrollTop = commentSectionRef.current.scrollHeight
        }
      }, 150) // 🔴 CRITICAL FIX: 더 긴 딜레이로 확실한 렌더링
    }
  }

  const handleReplyClick = (commentId: string, userName: string, replyId?: string) => {
    if (replyId) {
      setReplyContext({ type: "nested-reply", commentId, replyId, userName })
    } else {
      setReplyContext({ type: "reply", commentId, userName })
    }
    inputRef.current?.focus()
  }

  const cancelReply = () => {
    setReplyContext(null)
    setUnifiedInput("")
  }

  const toggleReplies = (commentId: string) => {
    const newExpanded = new Set(expandedReplies)
    if (newExpanded.has(commentId)) {
      newExpanded.delete(commentId)
    } else {
      newExpanded.add(commentId)
    }
    setExpandedReplies(newExpanded)
  }

  const studentTitle = `${analysis.studentId || "1405"}${analysis.studentName.replace(/\*/g, "정후")}`

  useEffect(() => {
    if (showComments && inputRef.current) {
      inputRef.current.focus()
    }
  }, [showComments])

  return (
    <>
      <GlassCard
        className="p-3.5 space-y-2.5 h-full hover:shadow-lg transition-all rounded-2xl"
        hover
        data-analysis-id={analysis.id}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-xl bg-black/5 flex items-center justify-center relative">
              <FileText className="w-4.5 h-4.5 text-black" />
              {isTopRanked && (
                <motion.div
                  initial={{ scale: 0, rotate: -45 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ type: "spring", bounce: 0.6 }}
                  className="absolute -top-1.5 -right-1.5"
                >
                  <Crown className="w-4.5 h-4.5 text-yellow-500 fill-yellow-500" />
                </motion.div>
              )}
              {analysis.isPrivate && (
                <div className="absolute -bottom-0.5 -right-0.5 bg-white rounded-full p-0.5">
                  <Lock className="w-2.5 h-2.5 text-gray-600" />
                </div>
              )}
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-sm text-black line-clamp-1">{studentTitle}</h3>
              <p className="text-[10px] text-gray-400">{new Date(analysis.uploadDate).toLocaleDateString("ko-KR")}</p>
            </div>
          </div>
          {/* 🔴 PHASE 7: 점수 → AI 키워드 변경 */}
          <div className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold bg-gradient-to-r from-blue-100 to-purple-100 text-blue-900 border border-blue-200 shadow-sm">
            {generateAIKeyword(analysis)}
          </div>
        </div>

        {/* 🔴 PHASE 7: 2줄 AI 요약 추가 (민감정보 필터링) */}
        <div className="bg-gray-50/80 rounded-lg px-2.5 py-2 border border-gray-100">
          <p className="text-[11px] text-gray-700 leading-relaxed line-clamp-2">
            {generateAISummary(analysis)}
          </p>
        </div>

        {/* 🔴 PHASE 7: 강점/보완 개수 표시 간소화 */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
            <span className="text-xs text-gray-600">강점 {analysis.strengths.length}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5 text-orange-600" />
            <span className="text-xs text-gray-600">보완 {analysis.improvements.length}</span>
          </div>
        </div>

        <div className="flex items-center justify-between pt-1.5 border-t border-gray-100">
          <div className="flex items-center gap-2">
            <button
              onClick={() => onToggleLike(analysis.id)}
              className="flex items-center gap-1 text-xs text-gray-500 hover:text-black transition-colors"
            >
              <Heart className={`w-3.5 h-3.5 ${liked ? "fill-black text-black" : ""}`} />
              <span className="font-medium">{likes}</span>
            </button>

            <button
              onClick={() => onToggleSave(analysis.id)}
              className="flex items-center gap-1 text-xs text-gray-500 hover:text-black transition-colors"
            >
              <Bookmark className={`w-3.5 h-3.5 ${saved ? "fill-black text-black" : ""}`} />
            </button>

            <button
              onClick={() => setShowComments(!showComments)}
              className="flex items-center gap-1 text-xs text-gray-500 hover:text-black transition-colors"
            >
              <MessageCircle className="w-3.5 h-3.5" />
              <span className="font-medium">{totalCommentCount}</span>
            </button>
          </div>

          <Button
            size="sm"
            onClick={() => setShowDetail(true)}
            data-detail-button
            className="rounded-full bg-black hover:bg-gray-800 text-white px-4 h-7 text-[11px] font-medium"
          >
            상세보기
          </Button>
        </div>

        {showComments && (
          <div className="space-y-2 pt-2 border-t border-gray-100">
            <div ref={commentSectionRef} className="space-y-1.5 max-h-36 overflow-y-auto">
              {comments
                .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
                .map((comment) => {
                  const directReplies = (comment.replies || []).filter((r) => !r.parentReplyId)

                  return (
                    <div key={comment.id} className="space-y-1">
                      {/* 🔴 PHASE 8: 댄글 UI 개선 */}
                      <div className="bg-gray-50 rounded-lg p-2 space-y-0.5">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold text-black">{comment.userName}</span>
                          <span className="text-xs text-gray-400">
                            {formatCommentTimestamp(comment.createdAt)}
                          </span>
                        </div>
                        <p className="text-sm text-gray-700">{comment.content}</p>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleReplyClick(comment.id, comment.userName)}
                            className="text-xs text-gray-500 hover:text-black transition-colors"
                          >
                            답글
                          </button>
                          {directReplies.length > 0 && (
                            <button
                              onClick={() => toggleReplies(comment.id)}
                              className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 transition-colors"
                            >
                              {expandedReplies.has(comment.id) ? (
                                <ChevronUp className="w-3 h-3" />
                              ) : (
                                <ChevronDown className="w-3 h-3" />
                              )}
                              답글 {directReplies.length}개
                            </button>
                          )}
                        </div>
                      </div>

                      {expandedReplies.has(comment.id) && directReplies.length > 0 && (
                        <div className="ml-4 space-y-1">
                          {/* LB-09: Replies sorted by newest first (최신순) */}
                          {directReplies
                            .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
                            .map((reply) => {
                            // Find nested replies to this reply (also sorted newest first)
                            const nestedReplies = (comment.replies || [])
                              .filter((r) => r.parentReplyId === reply.id)
                              .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())

                            return (
                              <div key={reply.id} className="space-y-1">
                                {/* 🔴 PHASE 8: "{comment.userName}님의 답글" 라벨 추가 */}
                                <div className="bg-blue-50 rounded-lg p-2 space-y-0.5 border border-blue-100">
                                  <div className="flex items-start justify-between gap-2">
                                    <div className="flex-1">
                                      <div className="flex items-center gap-1.5 mb-0.5">
                                        <span className="text-xs font-semibold text-black">{reply.userName}</span>
                                        <span className="text-[10px] text-blue-600 font-medium">
                                          {comment.userName}님의 답글
                                        </span>
                                      </div>
                                      <p className="text-sm text-gray-700">{reply.content}</p>
                                    </div>
                                    <span className="text-xs text-gray-400 flex-shrink-0">
                                      {formatCommentTimestamp(reply.createdAt)}
                                    </span>
                                  </div>
                                  <button
                                    onClick={() => handleReplyClick(comment.id, reply.userName, reply.id)}
                                    className="text-xs text-gray-500 hover:text-black transition-colors"
                                  >
                                    답글
                                  </button>
                                </div>

                                {nestedReplies.length > 0 && (
                                  <div className="ml-4 space-y-1">
                                    {nestedReplies.map((nestedReply) => (
                                      <div
                                        key={nestedReply.id}
                                        className="bg-purple-50 rounded-lg p-2 space-y-0.5 border border-purple-100"
                                      >
                                        {/* 🔴 PHASE 8: 중첩 답글에도 "{reply.userName}님의 답글" 라벨 */}
                                        <div className="flex items-start justify-between gap-2">
                                          <div className="flex-1">
                                            <div className="flex items-center gap-1.5 mb-0.5">
                                              <span className="text-xs font-semibold text-black">
                                                {nestedReply.userName}
                                              </span>
                                              <span className="text-[10px] text-purple-600 font-medium">
                                                {reply.userName}님의 답글
                                              </span>
                                            </div>
                                            <p className="text-sm text-gray-700">{nestedReply.content}</p>
                                          </div>
                                          <span className="text-xs text-gray-400 flex-shrink-0">
                                            {formatCommentTimestamp(nestedReply.createdAt)}
                                          </span>
                                        </div>
                                        <button
                                          onClick={() =>
                                            handleReplyClick(comment.id, nestedReply.userName, nestedReply.id)
                                          }
                                          className="text-xs text-gray-500 hover:text-black transition-colors"
                                        >
                                          답글
                                        </button>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  )
                })}
            </div>

            {/* 🔴 PHASE 8: 답글 작성 UI 개선 */}
            <div className="space-y-1.5 pt-1">
              {replyContext && (
                <motion.div
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-center justify-between bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg px-2 py-1.5"
                >
                  <span className="text-xs text-blue-700 font-medium">
                    <span className="font-bold">{replyContext.userName}</span>님에게 답글 작성 중
                  </span>
                  <button onClick={cancelReply} className="text-blue-600 hover:text-blue-800 transition-colors">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </motion.div>
              )}
              <div className="flex gap-1.5">
                <input
                  ref={inputRef}
                  type="text"
                  value={unifiedInput}
                  onChange={(e) => setUnifiedInput(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && handleSubmit()}
                  placeholder={replyContext ? "답글 입력" : "댓글 입력"}
                  className="flex-1 text-sm bg-gray-50 border border-gray-200 rounded-lg px-2 py-1 focus:outline-none focus:ring-2 focus:ring-black/10 min-w-0"
                />
                <Button
                  size="sm"
                  onClick={handleSubmit}
                  className="rounded-lg bg-black text-white px-2 h-7 flex-shrink-0"
                >
                  <Send className="w-3 h-3" />
                </Button>
              </div>
            </div>
          </div>
        )}
      </GlassCard>

      {showDetail && (
        <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 backdrop-blur-sm p-4 pt-8 overflow-hidden">
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            className="w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col"
          >
            <GlassCard className="flex flex-col max-h-full rounded-2xl">
              <div className="flex items-center justify-between p-3 border-b border-gray-200/50 rounded-t-2xl bg-white/95 backdrop-blur-sm flex-shrink-0">
                <h3 className="text-lg font-bold text-black">분석 상세</h3>
                <button
                  onClick={() => {
                    setShowDetail(false)
                  }}
                  className="p-1.5 hover:bg-gray-100 rounded-full transition-colors flex-shrink-0"
                >
                  <X className="w-5 h-5 text-gray-500" />
                </button>
              </div>

              <div className="overflow-y-auto p-3 space-y-2 flex-1">
                <GlassCard className="p-3 text-center rounded-2xl">
                  <div className="text-4xl font-bold text-black mb-1">{analysis.overallScore}점</div>
                  <p className="text-xs text-gray-600">종합 평가</p>
                </GlassCard>

                <GlassCard className="p-2.5 space-y-1.5 rounded-2xl">
                  <div className="flex gap-2 border-b border-gray-200">
                    <button
                      onClick={() => setDetailTab("strengths")}
                      className={`flex-1 pb-1.5 text-sm font-semibold transition-colors ${
                        detailTab === "strengths"
                          ? "text-green-700 border-b-2 border-green-600"
                          : "text-gray-400 hover:text-gray-600"
                      }`}
                    >
                      강점
                    </button>
                    <button
                      onClick={() => setDetailTab("improvements")}
                      className={`flex-1 pb-1.5 text-sm font-semibold transition-colors ${
                        detailTab === "improvements"
                          ? "text-orange-700 border-b-2 border-orange-600"
                          : "text-gray-400 hover:text-gray-600"
                      }`}
                    >
                      보완
                    </button>
                  </div>
                  <ul className="space-y-1">
                    {detailTab === "strengths"
                      ? analysis.strengths.map((strength, idx) => (
                          <li key={idx} className="text-xs text-gray-700 leading-relaxed flex gap-2">
                            <span className="text-green-600 flex-shrink-0">•</span>
                            <span>{strength}.</span>
                          </li>
                        ))
                      : analysis.improvements.map((improvement, idx) => (
                          <li key={idx} className="text-xs text-gray-700 leading-relaxed flex gap-2">
                            <span className="text-orange-600 flex-shrink-0">•</span>
                            <span>{improvement}.</span>
                          </li>
                        ))}
                  </ul>
                </GlassCard>

                <Button
                  onClick={() => setShowDetail(false)}
                  className="w-full rounded-full bg-black hover:bg-gray-900 text-white h-10 text-sm font-medium"
                >
                  닫기
                </Button>
              </div>
            </GlassCard>
          </motion.div>
        </div>
      )}
    </>
  )
}
