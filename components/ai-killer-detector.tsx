"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { GlassCard } from "@/components/glass-card"
import { Button } from "@/components/ui/button"
import { X, Shield, AlertTriangle, CheckCircle2, Loader2, Sparkles } from "lucide-react"
import type { AnalysisResult, AIKillerResult, RiskLevel } from "@/lib/types"

interface AIKillerDetectorProps {
  analysisResult: AnalysisResult
  onClose: () => void
}

export function AIKillerDetector({ analysisResult, onClose }: AIKillerDetectorProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(true)
  const [result, setResult] = useState<AIKillerResult | null>(null)

  useEffect(() => {
    document.body.style.overflow = "hidden"
    return () => {
      document.body.style.overflow = "unset"
    }
  }, [])

  useEffect(() => {
    const runDetection = async () => {
      setIsAnalyzing(true)
      
      try {
        // Import the AI detection function
        const { detectAIWriting } = await import('@/lib/gemini-service')
        
        // Use original OCR text, NOT the AI-generated analysis
        const textToAnalyze = (analysisResult as any).originalText || 
          '원본 텍스트를 찾을 수 없습니다. OCR 추출된 생기부 원본이 필요합니다.'
        
        if (!textToAnalyze || textToAnalyze.includes('원본 텍스트를 찾을 수 없습니다')) {
          throw new Error('OCR 원본 텍스트가 없습니다')
        }
        
        const detectionResult = await detectAIWriting(textToAnalyze)
        setResult(detectionResult)
      } catch (error) {
        console.error('[AI Killer Detection Error]', error)
        // Show error message
        setResult({
          overallAIProbability: 0,
          riskAssessment: "안전",
          detectedSections: [],
          recommendations: [
            'AI 작성 탐지를 완료할 수 없습니다.',
            '원본 생기부 텍스트가 필요합니다.',
            error instanceof Error ? error.message : '알 수 없는 오류'
          ]
        })
      } finally {
        setIsAnalyzing(false)
      }
    }
    
    runDetection()
  }, [])

  const getRiskColor = (risk: RiskLevel) => {
    switch (risk) {
      case "안전":
        return "text-green-600 bg-green-50/80 border-green-200/70"
      case "주의":
        return "text-yellow-600 bg-yellow-50/80 border-yellow-200/70"
      case "위험":
        return "text-orange-600 bg-orange-50/80 border-orange-200/70"
      case "매우위험":
        return "text-red-600 bg-red-50/80 border-red-200/70"
    }
  }

  const getRiskIcon = (risk: RiskLevel) => {
    switch (risk) {
      case "안전":
        return <CheckCircle2 className="w-5 h-5" />
      case "주의":
      case "위험":
      case "매우위험":
        return <AlertTriangle className="w-5 h-5" />
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 backdrop-blur-sm p-4 pt-8 overflow-hidden">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-2xl max-h-[90vh] flex flex-col"
      >
        <GlassCard className="flex flex-col max-h-full rounded-2xl overflow-hidden">
          <div className="flex items-center justify-between p-4 pb-3 border-b border-gray-200/50 flex-shrink-0 bg-white/95 backdrop-blur-sm">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-purple-600" />
              <h3 className="text-sm font-bold text-black">AI 작성 탐지</h3>
            </div>
            <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded-full transition-colors">
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 pt-3">
            <p className="text-xs text-gray-600 mb-3">선생님의 AI 사용 여부 분석.</p>

            {isAnalyzing ? (
              <div className="py-1.5 text-center space-y-1.5">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                  className="w-9 h-9 mx-auto"
                >
                  <Loader2 className="w-9 h-9 text-purple-600" />
                </motion.div>
                <div className="space-y-0.5">
                  <h4 className="text-sm font-semibold text-gray-900">분석 중...</h4>
                  <p className="text-xs text-gray-600">정밀 분석 진행 중.</p>
                </div>
              </div>
            ) : result ? (
              <div className="space-y-3 pb-3">
                <GlassCard className="p-4 space-y-3 rounded-2xl">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-gray-900">AI 작성 가능성</h4>
                    <div className="text-3xl font-bold text-purple-600">{result.overallAIProbability}%</div>
                  </div>
                  <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${result.overallAIProbability}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className={`h-full ${
                        result.overallAIProbability < 30
                          ? "bg-green-500"
                          : result.overallAIProbability < 60
                            ? "bg-yellow-500"
                            : "bg-red-500"
                      }`}
                    />
                  </div>
                  <p className="text-xs text-gray-600">
                    {result.overallAIProbability < 30
                      ? "인간 작성으로 판단."
                      : result.overallAIProbability < 60
                        ? "일부 AI 사용 가능성."
                        : "AI 사용 가능성 높음."}
                  </p>
                </GlassCard>

                <div className={`p-4 rounded-xl border-2 ${getRiskColor(result.riskAssessment)}`}>
                  <div className="flex items-center gap-2 mb-0.5">
                    {getRiskIcon(result.riskAssessment)}
                    <h4 className="font-semibold text-sm">위험도: {result.riskAssessment}</h4>
                  </div>
                  <p className="text-xs">
                    {result.riskAssessment === "안전" ? "향후 불이익 가능성 매우 낮음." : "일부 검토 필요."}
                  </p>
                </div>

                <div className="space-y-3">
                  <h4 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-purple-600" />
                    탐지된 섹션 ({result.detectedSections?.length || 0}개)
                  </h4>
                  {(result.detectedSections || []).map((section, idx) => (
                    <GlassCard key={idx} className="p-4 space-y-3 rounded-2xl">
                      <div className="flex items-start justify-between">
                        <span className="text-xs text-gray-500">라인 {section.lineNumber}</span>
                        <span
                          className={`text-xs font-bold ${
                            section.aiProbability < 30
                              ? "text-green-600"
                              : section.aiProbability < 60
                                ? "text-yellow-600"
                                : "text-red-600"
                          }`}
                        >
                          AI {section.aiProbability}%
                        </span>
                      </div>
                      <p className="text-sm text-gray-900 font-medium">"{section.content}"</p>
                      <p className="text-xs text-gray-600">{section.reason}.</p>

                      {(section.humanWritingIndicators?.length || 0) > 0 && (
                        <div className="mt-1">
                          <p className="text-xs font-medium text-green-700 mb-0.5">✓ 인간 작성 지표:</p>
                          <div className="flex flex-wrap gap-1">
                            {(section.humanWritingIndicators || []).map((indicator, i) => (
                              <span
                                key={i}
                                className="text-[10px] px-2 py-0.5 bg-green-100/80 text-green-700 rounded-full"
                              >
                                {indicator}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {(section.aiWritingIndicators?.length || 0) > 0 && (
                        <div className="mt-1">
                          <p className="text-xs font-medium text-orange-700 mb-0.5">⚠ AI 작성 지표:</p>
                          <div className="flex flex-wrap gap-1">
                            {(section.aiWritingIndicators || []).map((indicator, i) => (
                              <span
                                key={i}
                                className="text-[10px] px-2 py-0.5 bg-orange-100/80 text-orange-700 rounded-full"
                              >
                                {indicator}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </GlassCard>
                  ))}
                </div>

                <GlassCard className="p-4 space-y-1 rounded-2xl">
                  <h4 className="text-sm font-semibold text-gray-900">종합 의견</h4>
                  <ul className="space-y-0.5">
                    {(result.recommendations || []).map((rec, idx) => (
                      <li key={idx} className="text-xs text-gray-700 leading-relaxed">
                        • {rec}
                      </li>
                    ))}
                  </ul>
                </GlassCard>

                <Button
                  onClick={onClose}
                  className="w-full rounded-full bg-black hover:bg-gray-900 text-white h-10 text-sm font-medium"
                >
                  확인
                </Button>
              </div>
            ) : null}
          </div>
        </GlassCard>
      </motion.div>
    </div>
  )
}
