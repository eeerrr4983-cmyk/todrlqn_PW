"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { useRouter } from "next/navigation"
import { Navigation } from "@/components/navigation"
import { LiquidBackground } from "@/components/liquid-background"
import { GlassCard } from "@/components/glass-card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import {
  CheckCircle2,
  AlertCircle,
  Download,
  Share2,
  ArrowLeft,
  FileText,
  Sparkles,
  TrendingUp,
  Shield,
  Zap,
  AlertTriangle,
  Copy,
  MessageSquare,
} from "lucide-react"
import type { AIKillerResult, RiskLevel } from "@/lib/types"

export default function ResultsPage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<"overview" | "errors" | "suggestions" | "ai-killer" | "teacher-script">(
    "overview",
  )
  const [showAIKiller, setShowAIKiller] = useState(false)
  const [aiKillerResult, setAIKillerResult] = useState<AIKillerResult | null>(null)
  const [isAnalyzingAI, setIsAnalyzingAI] = useState(false)
  const [teacherScript, setTeacherScript] = useState("")
  const [copiedScript, setCopiedScript] = useState(false)

  // Mock data - in real app, this would come from props or state management
  const mockResult = {
    overallScore: 88,
    strengths: [
      "AI 및 데이터 분석 관련 탐구 활동이 구체적이고 심층적임.",
      "수학 세특에서 문제 해결 과정과 사고력이 명확히 드러남.",
      "창의적 체험활동에서 리더십과 협업 역량이 우수함.",
    ],
    improvements: [
      "진로 희망 대비 전공 적합성을 보완할 추가 활동 필요.",
      "3학년 1학기 세특에서 심화 탐구 내용 보강 권장.",
      "교과 간 연계성을 강화하여 일관된 서사 구축 필요.",
    ],
    errors: [
      {
        type: "금지" as const,
        content: "○○대학교 AI 캠프 참여",
        reason: "대학명 직접 명시 금지 (교육부 훈령 제530호)",
        page: 1,
        suggestion: "대학 주관 프로그램 → 교육기관 주관 프로그램으로 수정",
      },
      {
        type: "주의" as const,
        content: "매사에 성실하고 적극적이며 앞으로가 기대됨",
        reason: "모호한 칭찬 표현 (구체적 근거 부족)",
        page: 2,
        suggestion: "구체적인 활동 사례와 관찰 근거를 포함하여 재작성",
      },
      {
        type: "금지" as const,
        content: "TOEIC 900점 취득",
        reason: "공인어학시험 점수 기재 금지",
        page: 3,
        suggestion: "영어 의사소통 능력 향상을 위한 자기주도적 학습 수행으로 수정",
      },
    ],
    suggestions: [
      "수학 세특: '데이터 분석' 키워드를 활용한 심화 탐구 추가 권장.",
      "과학 세특: AI 윤리 관련 탐구로 진로 연계성 강화.",
      "동아리 활동: 구체적인 역할과 성과를 명확히 기술.",
      "독서 활동: 전공 관련 도서 추가로 진로 탐색 깊이 강화.",
    ],
  }

  const runAIKillerAnalysis = async () => {
    setIsAnalyzingAI(true)
    setShowAIKiller(true)

    // Simulate AI detection analysis
    await new Promise((resolve) => setTimeout(resolve, 2500))

    const mockAIKillerResult: AIKillerResult = {
      overallAIProbability: 34,
      detectedSections: [
        {
          content:
            "학생은 수학적 사고력과 문제 해결 능력이 뛰어나며, 복잡한 개념을 이해하고 적용하는 데 탁월한 역량을 보였습니다.",
          aiProbability: 78,
          lineNumber: 12,
          reason: "과도하게 형식적이고 일반적인 표현, 구체적 사례 부족",
          humanWritingIndicators: [],
          aiWritingIndicators: ["형식적 문장 구조", "추상적 표현", "구체성 부족"],
        },
        {
          content:
            "데이터 분석 프로젝트에서 Python을 활용하여 실제 데이터를 수집하고, 이를 시각화하여 의미 있는 인사이트를 도출함.",
          aiProbability: 15,
          lineNumber: 24,
          reason: "구체적 도구와 과정 명시, 자연스러운 서술",
          humanWritingIndicators: ["구체적 도구 명시", "실제 활동 기술", "자연스러운 흐름"],
          aiWritingIndicators: [],
        },
        {
          content:
            "협업 과정에서 팀원들과의 원활한 소통을 통해 프로젝트를 성공적으로 완수하였으며, 리더십을 발휘하여 팀을 이끌었습니다.",
          aiProbability: 65,
          lineNumber: 35,
          reason: "일반적인 칭찬 표현, 구체적 에피소드 부재",
          humanWritingIndicators: [],
          aiWritingIndicators: ["추상적 칭찬", "구체성 부족", "형식적 표현"],
        },
      ],
      riskAssessment: "주의",
      recommendations: [
        "AI 확률이 높은 섹션(78%, 65%)을 구체적인 관찰 사례로 재작성 권장",
        "학생의 실제 행동과 발언을 직접 인용하여 진정성 강화",
        "추상적 표현 대신 구체적인 수치, 날짜, 활동 내용 포함",
        "선생님의 개인적 관찰과 평가를 자연스럽게 서술",
      ],
    }

    setAIKillerResult(mockAIKillerResult)
    setIsAnalyzingAI(false)
  }

  const generateTeacherScript = () => {
    const errors = mockResult.errors
    const improvements = mockResult.improvements

    let script = "선생님, 생기부 AI 분석 결과를 확인했는데 몇 가지 수정이 필요한 부분이 있어서 말씀드립니다.\n\n"

    if (errors.length > 0) {
      script += "【오류 수정 요청】\n"
      errors.forEach((error, idx) => {
        script += `${idx + 1}. "${error.content}"\n`
        script += `   → 문제: ${error.reason}\n`
        if (error.type === "금지") {
          script += `   → 이 부분은 교육부 규정상 기재가 금지되어 있어서, 삭제하거나 다른 표현으로 수정해주시면 감사하겠습니다.\n\n`
        } else {
          script += `   → 이 부분을 좀 더 구체적으로 수정해주시면 좋을 것 같습니다.\n\n`
        }
      })
    }

    if (improvements.length > 0) {
      script += "【보완 제안】\n"
      improvements.forEach((improvement, idx) => {
        script += `${idx + 1}. ${improvement}\n`
      })
      script += "\n"
    }

    script += "바쁘신 중에 번거롭게 해드려 죄송합니다. 시간 되실 때 확인해주시면 감사하겠습니다.\n감사합니다."

    setTeacherScript(script)
  }

  const copyTeacherScript = () => {
    if (!teacherScript) {
      generateTeacherScript()
    }
    navigator.clipboard.writeText(teacherScript || "")
    setCopiedScript(true)
    setTimeout(() => setCopiedScript(false), 2000)
  }

  const getRiskColor = (risk: RiskLevel) => {
    switch (risk) {
      case "안전":
        return "text-green-600 bg-green-50 border-green-200"
      case "주의":
        return "text-yellow-600 bg-yellow-50 border-yellow-200"
      case "위험":
        return "text-orange-600 bg-orange-50 border-orange-200"
      case "매우위험":
        return "text-red-600 bg-red-50 border-red-200"
    }
  }

  const getRiskIcon = (risk: RiskLevel) => {
    switch (risk) {
      case "안전":
        return <Shield className="w-5 h-5" />
      case "주의":
        return <AlertTriangle className="w-5 h-5" />
      case "위험":
        return <AlertCircle className="w-5 h-5" />
      case "매우위험":
        return <Zap className="w-5 h-5" />
    }
  }

  useEffect(() => {
    if (typeof window !== "undefined") {
      sessionStorage.setItem("is_analyzing", "true")
    }

    return () => {
      if (typeof window !== "undefined") {
        const currentAnalysis = sessionStorage.getItem("current_analysis")
        if (currentAnalysis) {
          sessionStorage.setItem("is_analyzing", "true")
        }
      }
    }
  }, [])

  const handleShare = async () => {
    const resultUrl = window.location.href
    const shareText = `생기부 AI 분석 결과\n\n종합 평가: ${mockResult.overallScore}점\n\n강점:\n${mockResult.strengths.map((s, i) => `${i + 1}. ${s}`).join("\n")}\n\n보완 필요:\n${mockResult.improvements.map((i, idx) => `${idx + 1}. ${i}`).join("\n")}`

    if (navigator.share) {
      try {
        await navigator.share({
          title: "생기부 AI 분석 결과 - Huntfire",
          text: shareText,
          url: resultUrl,
        })
      } catch (error) {
        console.log("[v0] Share cancelled or failed:", error)
      }
    } else {
      // Fallback: copy link to clipboard
      try {
        await navigator.clipboard.writeText(`${shareText}\n\n결과 보기: ${resultUrl}`)
        alert("결과 링크가 클립보드에 복사되었습니다!")
      } catch (err) {
        console.error("[v0] Failed to copy:", err)
      }
    }
  }

  const handleExportText = () => {
    const report = `생기부 AI 분석 결과
===================

종합 평가: ${mockResult.overallScore}점

강점:
${mockResult.strengths.map((s, i) => `${i + 1}. ${s}`).join("\n")}

보완 사항:
${mockResult.improvements.map((s, i) => `${i + 1}. ${s}`).join("\n")}

오류 ${mockResult.errors.length}건:
${mockResult.errors.map((e, i) => `${i + 1}. [${e.type}] ${e.content}\n   사유: ${e.reason}`).join("\n")}

개선 제안:
${mockResult.suggestions.map((s, i) => `${i + 1}. ${s}`).join("\n")}
    `.trim()

    const blob = new Blob([report], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    const now = new Date()
    const month = now.getMonth() + 1
    const day = now.getDate()
    a.download = `생기부AI분석결과_${month}/${day}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-white">
      <LiquidBackground />
      <Navigation />

      <div className="relative z-10 h-full overflow-y-auto px-4 sm:px-6 pb-20">
        <div className="max-w-6xl mx-auto pt-6 sm:pt-8 space-y-4">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-3"
          >
            <Button
              variant="ghost"
              onClick={() => {
                // 🔴 UX FIX: Go back to main home page
                router.push('/')
                // Scroll to top
                setTimeout(() => {
                  window.scrollTo({ top: 0, behavior: 'smooth' })
                }, 100)
              }}
              className="rounded-full hover:bg-gray-100 h-9 px-4 text-sm font-medium shadow-sm border border-gray-200"
            >
              <ArrowLeft className="w-4 h-4 mr-1.5" />
              메인으로 돌아가기
            </Button>

            <div className="text-center space-y-2">
              <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-black">분석 결과</h1>
              <p className="text-sm sm:text-base text-gray-600 max-w-2xl mx-auto">
                AI가 분석한 생기부 종합 평가 및 개선 제안입니다.
              </p>
            </div>
          </motion.div>

          {/* Overall Score */}
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <GlassCard className="p-8 sm:p-10 text-center" glow>
              <div className="space-y-3">
                <div className="text-5xl sm:text-6xl font-bold text-black">{mockResult.overallScore}점</div>
                <p className="text-base sm:text-lg text-gray-600">종합 평가 점수 (상위 12%)</p>
                <div className="flex items-center justify-center gap-4 sm:gap-6 pt-4">
                  <div className="flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-green-600" />
                    <span className="text-xs sm:text-sm text-gray-600">강점 {mockResult.strengths.length}개</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <AlertCircle className="w-4 h-4 text-orange-600" />
                    <span className="text-xs sm:text-sm text-gray-600">보완 {mockResult.improvements.length}개</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <FileText className="w-4 h-4 text-red-600" />
                    <span className="text-xs sm:text-sm text-gray-600">오류 {mockResult.errors.length}개</span>
                  </div>
                </div>
              </div>
            </GlassCard>
          </motion.div>

          {/* Tabs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <GlassCard className="p-2">
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                <Button
                  variant={activeTab === "overview" ? "default" : "ghost"}
                  onClick={() => setActiveTab("overview")}
                  className={`rounded-xl text-xs sm:text-sm ${
                    activeTab === "overview" ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-50"
                  }`}
                >
                  <TrendingUp className="w-4 h-4 mr-1.5" />
                  종합
                </Button>
                <Button
                  variant={activeTab === "errors" ? "default" : "ghost"}
                  onClick={() => setActiveTab("errors")}
                  className={`rounded-xl text-xs sm:text-sm ${
                    activeTab === "errors" ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-50"
                  }`}
                >
                  <AlertCircle className="w-4 h-4 mr-1.5" />
                  오류
                </Button>
                <Button
                  variant={activeTab === "suggestions" ? "default" : "ghost"}
                  onClick={() => setActiveTab("suggestions")}
                  className={`rounded-xl text-xs sm:text-sm ${
                    activeTab === "suggestions" ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-50"
                  }`}
                >
                  <Sparkles className="w-4 h-4 mr-1.5" />
                  제안
                </Button>
                <Button
                  variant={activeTab === "ai-killer" ? "default" : "ghost"}
                  onClick={() => {
                    setActiveTab("ai-killer")
                    if (!aiKillerResult) runAIKillerAnalysis()
                  }}
                  className={`rounded-xl text-xs sm:text-sm ${
                    activeTab === "ai-killer" ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-50"
                  }`}
                >
                  <Shield className="w-4 h-4 mr-1.5" />
                  AI 킬러
                </Button>
                <Button
                  variant={activeTab === "teacher-script" ? "default" : "ghost"}
                  onClick={() => {
                    setActiveTab("teacher-script")
                    if (!teacherScript) generateTeacherScript()
                  }}
                  className={`rounded-xl text-xs sm:text-sm ${
                    activeTab === "teacher-script" ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-50"
                  }`}
                >
                  <MessageSquare className="w-4 h-4 mr-1.5" />
                  선생님께
                </Button>
              </div>
            </GlassCard>
          </motion.div>

          {/* Content */}
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="space-y-4"
          >
            {activeTab === "overview" && (
              <div className="grid md:grid-cols-2 gap-4">
                <GlassCard className="p-6 space-y-4">
                  <h3 className="text-2xl font-bold text-black flex items-center gap-2">
                    <CheckCircle2 className="w-6 h-6 text-green-600" />
                    주요 강점
                  </h3>
                  <ul className="space-y-3">
                    {mockResult.strengths.map((strength, idx) => (
                      <motion.li
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="text-base text-gray-700 leading-relaxed pl-6 relative before:content-['•'] before:absolute before:left-0 before:text-green-600 before:font-bold"
                      >
                        {strength}
                      </motion.li>
                    ))}
                  </ul>
                </GlassCard>

                <GlassCard className="p-6 space-y-4">
                  <h3 className="text-2xl font-bold text-black flex items-center gap-2">
                    <AlertCircle className="w-6 h-6 text-orange-600" />
                    보완 필요
                  </h3>
                  <ul className="space-y-3">
                    {mockResult.improvements.map((improvement, idx) => (
                      <motion.li
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="text-base text-gray-700 leading-relaxed pl-6 relative before:content-['•'] before:absolute before:left-0 before:text-orange-600 before:font-bold"
                      >
                        {improvement}
                      </motion.li>
                    ))}
                  </ul>
                </GlassCard>
              </div>
            )}

            {activeTab === "errors" && (
              <div className="space-y-4">
                {mockResult.errors.map((error, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.1 }}
                  >
                    <GlassCard className="p-6">
                      <div className="flex items-start gap-4">
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-bold ${
                            error.type === "금지" ? "bg-red-600 text-white" : "bg-orange-500 text-white"
                          }`}
                        >
                          {error.type}
                        </span>
                        <div className="flex-1 space-y-3">
                          <div>
                            <p className="text-lg font-bold text-gray-900">"{error.content}"</p>
                            <p className="text-sm text-gray-600 mt-1">{error.reason}</p>
                            <p className="text-xs text-gray-500 mt-1">페이지 {error.page}</p>
                          </div>
                          {error.suggestion && (
                            <div className="bg-green-50 border border-green-200 rounded-xl p-4">
                              <p className="text-xs font-semibold text-green-800 mb-1 flex items-center gap-1">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                                수정 제안
                              </p>
                              <p className="text-sm text-green-700">{error.suggestion}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </GlassCard>
                  </motion.div>
                ))}
              </div>
            )}

            {activeTab === "suggestions" && (
              <GlassCard className="p-6 space-y-4">
                <h3 className="text-2xl font-bold text-black flex items-center gap-2">
                  <Sparkles className="w-6 h-6 text-black" />
                  AI 개선 제안
                </h3>
                <ul className="space-y-4">
                  {mockResult.suggestions.map((suggestion, idx) => (
                    <motion.li
                      key={idx}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.1 }}
                      className="bg-gray-50 rounded-xl p-4 text-base text-gray-700 leading-relaxed"
                    >
                      {suggestion}
                    </motion.li>
                  ))}
                </ul>
              </GlassCard>
            )}

            {activeTab === "ai-killer" && (
              <div className="space-y-4">
                {isAnalyzingAI ? (
                  <GlassCard className="p-12 text-center space-y-4">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                      className="w-16 h-16 rounded-full bg-black/5 flex items-center justify-center mx-auto"
                    >
                      <Shield className="w-8 h-8 text-black" />
                    </motion.div>
                    <div>
                      <h3 className="text-xl font-bold text-black">AI 작성 여부 탐지 중...</h3>
                      <p className="text-sm text-gray-600 mt-2">생기부 텍스트를 심층 분석하고 있습니다</p>
                    </div>
                  </GlassCard>
                ) : aiKillerResult ? (
                  <>
                    {/* Overall AI Probability */}
                    <GlassCard className="p-8 text-center" glow>
                      <div className="space-y-4">
                        <div className="flex items-center justify-center gap-3">
                          {getRiskIcon(aiKillerResult.riskAssessment)}
                          <h3 className="text-2xl font-bold text-black">AI 작성 확률</h3>
                        </div>
                        <div className="text-6xl font-bold text-black">{aiKillerResult.overallAIProbability}%</div>
                        <div
                          className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border-2 ${getRiskColor(
                            aiKillerResult.riskAssessment,
                          )}`}
                        >
                          <span className="font-bold text-sm">{aiKillerResult.riskAssessment}</span>
                        </div>
                        <p className="text-sm text-gray-600 max-w-2xl mx-auto">
                          {aiKillerResult.overallAIProbability < 30
                            ? "대부분 인간이 작성한 것으로 판단됩니다. 자연스러운 서술이 돋보입니다."
                            : aiKillerResult.overallAIProbability < 60
                              ? "일부 섹션에서 AI 작성 가능성이 감지되었습니다. 구체성을 보완하면 좋습니다."
                              : "다수 섹션에서 AI 작성 패턴이 발견되었습니다. 재작성을 권장합니다."}
                        </p>
                      </div>
                    </GlassCard>

                    {/* Detected Sections */}
                    <div className="space-y-4">
                      <h3 className="text-xl font-bold text-black px-2">탐지된 섹션 분석</h3>
                      {aiKillerResult.detectedSections.map((section, idx) => (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.1 }}
                        >
                          <GlassCard className="p-6 space-y-4">
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <span className="text-xs font-bold text-gray-500">라인 {section.lineNumber}</span>
                                  <span
                                    className={`px-2 py-1 rounded-full text-xs font-bold ${
                                      section.aiProbability > 70
                                        ? "bg-red-100 text-red-700"
                                        : section.aiProbability > 40
                                          ? "bg-yellow-100 text-yellow-700"
                                          : "bg-green-100 text-green-700"
                                    }`}
                                  >
                                    AI {section.aiProbability}%
                                  </span>
                                </div>
                                <p className="text-sm text-gray-900 leading-relaxed mb-3 bg-gray-50 p-3 rounded-lg">
                                  "{section.content}"
                                </p>
                                <p className="text-xs text-gray-600 mb-3">
                                  <strong>판단 근거:</strong> {section.reason}
                                </p>
                              </div>
                            </div>

                            <div className="grid sm:grid-cols-2 gap-3">
                              {section.humanWritingIndicators.length > 0 && (
                                <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                                  <h4 className="text-xs font-bold text-green-800 mb-2 flex items-center gap-1">
                                    <CheckCircle2 className="w-3.5 h-3.5" />
                                    인간 작성 지표
                                  </h4>
                                  <ul className="space-y-1">
                                    {section.humanWritingIndicators.map((indicator, i) => (
                                      <li key={i} className="text-xs text-green-700">
                                        • {indicator}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}

                              {section.aiWritingIndicators.length > 0 && (
                                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                                  <h4 className="text-xs font-bold text-red-800 mb-2 flex items-center gap-1">
                                    <AlertCircle className="w-3.5 h-3.5" />
                                    AI 작성 지표
                                  </h4>
                                  <ul className="space-y-1">
                                    {section.aiWritingIndicators.map((indicator, i) => (
                                      <li key={i} className="text-xs text-red-700">
                                        • {indicator}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          </GlassCard>
                        </motion.div>
                      ))}
                    </div>

                    {/* Recommendations */}
                    <GlassCard className="p-8 space-y-4">
                      <h3 className="text-2xl font-bold text-black flex items-center gap-2">
                        <Sparkles className="w-6 h-6" />
                        개선 권장사항
                      </h3>
                      <ul className="space-y-3">
                        {aiKillerResult.recommendations.map((rec, idx) => (
                          <motion.li
                            key={idx}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-900"
                          >
                            💡 {rec}
                          </motion.li>
                        ))}
                      </ul>
                    </GlassCard>
                  </>
                ) : null}
              </div>
            )}

            {activeTab === "teacher-script" && (
              <div className="space-y-4">
                <GlassCard className="p-6 space-y-4" glow>
                  <div className="space-y-2">
                    <h3 className="text-2xl font-bold text-black flex items-center gap-2">
                      <MessageSquare className="w-6 h-6" />
                      선생님께 드릴 말씀
                    </h3>
                    <p className="text-sm text-gray-600">
                      AI가 탐지한 오류와 개선사항을 바탕으로 선생님께 드릴 수정 요청 스크립트를 자동으로 생성했습니다.
                      복사하여 사용하세요.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Textarea
                      value={teacherScript}
                      onChange={(e) => setTeacherScript(e.target.value)}
                      className="min-h-[400px] text-sm leading-relaxed font-normal resize-none"
                      placeholder="스크립트를 생성하는 중..."
                    />

                    <div className="flex gap-3 justify-center">
                      <Button
                        onClick={copyTeacherScript}
                        className="rounded-full bg-black hover:bg-gray-900 text-white h-12 text-sm font-medium px-8"
                      >
                        <Copy className="w-4 h-4 mr-2" />
                        {copiedScript ? "복사됨!" : "복사하기"}
                      </Button>
                      <Button
                        onClick={generateTeacherScript}
                        variant="outline"
                        className="rounded-full border-2 border-gray-200 hover:bg-gray-50 h-12 text-sm font-medium bg-transparent px-8"
                      >
                        <Sparkles className="w-4 h-4 mr-2" />
                        다시 생성
                      </Button>
                    </div>
                  </div>

                  <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                    <h4 className="text-sm font-bold text-blue-900 mb-2 flex items-center gap-1.5">
                      <AlertCircle className="w-4 h-4" />
                      사용 팁
                    </h4>
                    <ul className="space-y-1 text-xs text-blue-800">
                      <li>• 스크립트를 그대로 사용하거나, 상황에 맞게 수정하여 사용하세요</li>
                      <li>• 선생님께 직접 말씀드리기 어려운 경우, 메시지나 이메일로 전달할 수 있습니다</li>
                      <li>• 공손하고 정중한 표현으로 작성되어 있으니 안심하고 사용하세요</li>
                    </ul>
                  </div>
                </GlassCard>
              </div>
            )}
          </motion.div>

          {/* Action Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="pb-4"
          >
            <GlassCard className="p-4">
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button
                  size="lg"
                  className="rounded-full bg-black hover:bg-gray-900 text-white px-10 h-12 font-medium"
                  onClick={handleExportText}
                >
                  <Download className="w-5 h-5 mr-2" />
                  다운로드
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  className="rounded-full border-2 border-gray-200 hover:bg-gray-50 px-10 h-12 bg-white font-medium"
                  onClick={handleShare}
                >
                  <Share2 className="w-5 h-5 mr-2" />
                  공유하기
                </Button>
              </div>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
