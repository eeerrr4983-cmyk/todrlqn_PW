import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"
import { getModelForTask, globalCostTracker } from "@/lib/ai-model-router"

const GEMINI_API_KEY = process.env.GEMINI_API_KEY

export const maxDuration = 60

interface GeminiResponse {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string
      }>
    }
  }>
}

interface UniversityTierRecommendation {
  tier: string
  universities: string[]
  probability: "도전" | "적정" | "안정"
}

interface UniversityPrediction {
  nationalPercentile: number
  academicLevel: string
  reachableUniversities: UniversityTierRecommendation[]
  strengthAnalysis: string
  improvementNeeded: string
  recommendations: string[]
}

const KOREAN_UNIVERSITY_TIERS = `
### 한국 대학 계층 구조 (2025학년도 기준)

**최상위권 (상위 1-5%)**
- 서울대학교, 연세대학교, 고려대학교, KAIST, 포항공대

**상위권 (상위 6-15%)**
- 성균관대, 한양대, 서강대, 중앙대, 경희대, 한국외대, 서울시립대

**중상위권 (상위 16-30%)**
- 건국대, 동국대, 홍익대, 숙명여대, 국민대, 숭실대, 세종대

**중위권 (상위 31-50%)**
- 단국대, 광운대, 명지대, 상명대, 가천대, 아주대, 인하대

**지방 거점 국립대 (상위 20-35%)**
- 부산대, 경북대, 전남대, 전북대, 충남대, 충북대, 강원대
`

function createUniversityPrompt(analysisResult: any, careerDirection: string): string {
  return `당신은 한국 대학 입학 전문가입니다. 학생의 생기부 분석 결과를 바탕으로 지원 가능한 대학을 정밀하게 예측하세요.

${KOREAN_UNIVERSITY_TIERS}

### 학생 정보:
**진로 방향**: ${careerDirection || "미지정"}
**생기부 종합 점수**: ${analysisResult.overallScore}/100
**학생 프로필**: ${analysisResult.studentProfile || "정보 없음"}
**주요 강점**: ${analysisResult.strengths?.slice(0, 3).join(" • ") || "정보 없음"}
**개선 필요 사항**: ${analysisResult.improvements?.slice(0, 3).join(" • ") || "정보 없음"}

### 분석 요청:
위 정보를 바탕으로 학생이 지원 가능한 대학을 예측하고, 다음 JSON 형식으로 응답하세요:

\`\`\`json
{
  "nationalPercentile": 1-100 사이의 백분위 (낮을수록 상위권),
  "academicLevel": "최상위권", "상위권", "중상위권", "중위권", "중하위권" 중 하나,
  "reachableUniversities": [
    {
      "tier": "계층명 (예: 최상위권, 상위권 대학 등)",
      "universities": ["대학명1", "대학명2", "대학명3"],
      "probability": "도전", "적정", "안정" 중 하나
    }
  ],
  "strengthAnalysis": "학생의 강점을 진로와 연계하여 2-3문장으로 분석",
  "improvementNeeded": "보완이 필요한 부분을 구체적으로 2-3문장으로 설명",
  "recommendations": [
    "실행 가능한 구체적 조언 1",
    "실행 가능한 구체적 조언 2",
    "실행 가능한 구체적 조언 3"
  ]
}
\`\`\`

**중요 지침**:
- nationalPercentile은 점수를 기반으로 정확히 계산 (90점 이상: 1-10%, 80-89점: 11-25%, 70-79점: 26-40%, 60-69점: 41-60%)
- reachableUniversities는 최소 2개, 최대 4개 계층 포함
- 각 계층마다 2-3개 대학 추천
- 진로 방향과 대학의 특성을 고려한 현실적 추천
- 모든 텍스트는 한국어로 명확하고 구체적으로 작성
- JSON 형식을 정확히 준수`
}

function extractJsonBlock(text: string): string | null {
  const codeBlockMatch = text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/)
  if (codeBlockMatch) {
    return codeBlockMatch[1]
  }

  const jsonMatch = text.match(/\{[\s\S]*\}/)
  if (jsonMatch) {
    return jsonMatch[0]
  }

  return null
}

function buildFallbackPrediction(analysisResult: any, careerDirection: string): UniversityPrediction {
  const score = analysisResult.overallScore || 75
  let percentile: number
  
  if (score >= 90) percentile = Math.max(1, Math.round((100 - score) * 2))
  else if (score >= 80) percentile = Math.round(10 + (90 - score))
  else if (score >= 70) percentile = Math.round(20 + (80 - score) * 2)
  else if (score >= 60) percentile = Math.round(40 + (70 - score) * 2)
  else percentile = Math.round(60 + (60 - score) * 1.5)

  let academicLevel: string
  if (percentile <= 5) academicLevel = "최상위권"
  else if (percentile <= 15) academicLevel = "상위권"
  else if (percentile <= 30) academicLevel = "중상위권"
  else if (percentile <= 50) academicLevel = "중위권"
  else academicLevel = "중하위권"

  const reachableUniversities: UniversityTierRecommendation[] = []

  if (percentile <= 15) {
    reachableUniversities.push({
      tier: "최상위권 (SKY/특수대학)",
      universities: ["서울대학교", "연세대학교", "고려대학교"],
      probability: percentile <= 5 ? "적정" : "도전"
    })
  }

  if (percentile <= 25) {
    reachableUniversities.push({
      tier: "상위권 대학",
      universities: ["성균관대학교", "한양대학교", "서강대학교"],
      probability: percentile <= 15 ? "적정" : "도전"
    })
  }

  if (percentile <= 40) {
    reachableUniversities.push({
      tier: "중상위권 대학",
      universities: ["건국대학교", "동국대학교", "홍익대학교"],
      probability: percentile <= 30 ? "적정" : "도전"
    })
  }

  if (percentile <= 60) {
    reachableUniversities.push({
      tier: "중위권 대학",
      universities: ["단국대학교", "광운대학교", "가천대학교"],
      probability: percentile <= 50 ? "적정" : "안정"
    })
  }

  return {
    nationalPercentile: percentile,
    academicLevel,
    reachableUniversities: reachableUniversities.slice(0, 3),
    strengthAnalysis: `${careerDirection || "진로 미지정"} 방향에서 ${analysisResult.strengths?.[0] || "전반적으로 우수한 역량"}을 보유하고 있습니다.`,
    improvementNeeded: `${analysisResult.improvements?.[0] || "지속적인 개선"}이 필요합니다.`,
    recommendations: [
      "학교 생활기록부의 강점을 지속적으로 강화하세요.",
      "진로 적합성을 높일 수 있는 심화 활동을 추가하세요.",
      "구체적인 활동 사례와 성과를 기록해보세요."
    ]
  }
}

export async function POST(request: NextRequest) {
  try {
    if (!GEMINI_API_KEY) {
      console.error("[University] Missing GEMINI_API_KEY")
      return NextResponse.json({ error: "서버에 Gemini API 키가 설정되지 않았습니다." }, { status: 500 })
    }

    const { analysisResult, careerDirection } = await request.json()

    if (!analysisResult) {
      return NextResponse.json({ error: "분석 결과가 필요합니다." }, { status: 400 })
    }

    const prompt = createUniversityPrompt(analysisResult, careerDirection || "")

    // 🧠 하이브리드 AI: 대학 예측은 복잡한 작업 (한국 대학 계층 이해 필요)
    const selectedModel = getModelForTask({ 
      type: 'university',
      requiresDeepReasoning: true
    })
    globalCostTracker.trackRequest(selectedModel)
    console.log(`[University] 🚀 ${selectedModel.name} 사용`)

    const response = await fetch(`${selectedModel.endpoint}?key=${GEMINI_API_KEY}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt,
              },
            ],
          },
        ],
        generationConfig: {
          temperature: 0.3,
          topK: 32,
          topP: 0.9,
          maxOutputTokens: 4096,
        },
      }),
    })

    if (!response.ok) {
      console.error("[University] Gemini API error", response.status)
      return NextResponse.json(
        { result: buildFallbackPrediction(analysisResult, careerDirection) },
        { status: 200 }
      )
    }

    const payload = (await response.json()) as GeminiResponse
    const generatedText = payload.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || ""

    if (!generatedText) {
      return NextResponse.json(
        { result: buildFallbackPrediction(analysisResult, careerDirection) },
        { status: 200 }
      )
    }

    const jsonBlock = extractJsonBlock(generatedText)

    if (!jsonBlock) {
      return NextResponse.json(
        { result: buildFallbackPrediction(analysisResult, careerDirection) },
        { status: 200 }
      )
    }

    try {
      const parsed = JSON.parse(jsonBlock) as UniversityPrediction
      return NextResponse.json({ result: parsed })
    } catch (parseError) {
      console.error("[University] JSON parse error", parseError)
      return NextResponse.json(
        { result: buildFallbackPrediction(analysisResult, careerDirection) },
        { status: 200 }
      )
    }
  } catch (error) {
    console.error("[University] Unexpected error", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "대학 예측 중 오류가 발생했습니다." },
      { status: 500 }
    )
  }
}
