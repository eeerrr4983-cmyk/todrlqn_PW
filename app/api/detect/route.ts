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

interface DetectedSection {
  section: string
  probability: number
  indicators: string[]
}

interface AIKillerResult {
  overallAIProbability: number
  riskAssessment: "안전" | "주의" | "위험" | "매우위험"
  detectedSections: DetectedSection[]
  recommendations: string[]
  humanIndicators?: string[]
  aiIndicators?: string[]
}

function createDetectionPrompt(text: string): string {
  return `당신은 AI 작성 탐지 전문가입니다. 제공된 생기부 텍스트가 AI로 작성되었는지 정밀하게 분석하세요.

### 분석 대상 텍스트:
\`\`\`
${text}
\`\`\`

### AI 작성 의심 지표:
1. **과도한 완벽성**: 문법적으로 너무 완벽하고 자연스러운 흐름 부재
2. **일반적/추상적 표현**: 구체적 사례 없이 "~에 기여했다", "~를 통해 성장했다" 등 반복
3. **비현실적 완벽 서술**: 실제 학생이 쓰기 어려운 고급 어휘나 복잡한 문장 구조
4. **감정 표현 부재**: 학생 개인의 감정이나 시행착오 없이 결과만 나열
5. **일관된 톤**: 모든 섹션이 동일한 문체와 어조로 작성됨
6. **세부 정보 부족**: "다양한", "여러", "많은" 등 모호한 표현 과다

### 인간 작성 지표:
1. **자연스러운 실수**: 약간의 문법 오류나 어색한 표현
2. **구체적 사례**: "~과목 시간에", "~선생님과", "~친구들과" 등 구체적 맥락
3. **개인적 감정**: "어려웠지만", "처음에는 힘들었으나", "놀랍게도" 등 감정 표현
4. **시행착오 언급**: 실패나 어려움을 극복한 과정 서술
5. **일관성 없는 문체**: 섹션마다 약간씩 다른 톤과 표현

### 분석 요청:
위 지표를 바탕으로 텍스트를 분석하고, 다음 JSON 형식으로 응답하세요:

\`\`\`json
{
  "overallAIProbability": 0-100 사이의 숫자 (AI 작성 가능성 백분율),
  "riskAssessment": "안전", "주의", "위험", "매우위험" 중 하나,
  "detectedSections": [
    {
      "section": "의심되는 섹션 또는 문장 일부 인용",
      "probability": 해당 섹션의 AI 작성 가능성 0-100,
      "indicators": ["발견된 AI 작성 지표 1", "발견된 AI 작성 지표 2"]
    }
  ],
  "humanIndicators": [
    "인간 작성으로 보이는 근거 1",
    "인간 작성으로 보이는 근거 2",
    "인간 작성으로 보이는 근거 3"
  ],
  "aiIndicators": [
    "AI 작성으로 의심되는 근거 1",
    "AI 작성으로 의심되는 근거 2",
    "AI 작성으로 의심되는 근거 3"
  ],
  "recommendations": [
    "개선 권장사항 1 (더 자연스럽게 만들기 위한 조언)",
    "개선 권장사항 2",
    "개선 권장사항 3"
  ]
}
\`\`\`

**중요 지침**:
- overallAIProbability는 신중하게 판단 (30% 이하: 안전, 31-50%: 주의, 51-75%: 위험, 76% 이상: 매우위험)
- detectedSections는 실제로 의심되는 부분만 포함 (없으면 빈 배열)
- humanIndicators와 aiIndicators 모두 제시
- recommendations는 실행 가능하고 구체적으로 작성
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

function assessRiskLevel(probability: number): "안전" | "주의" | "위험" | "매우위험" {
  if (probability <= 30) return "안전"
  if (probability <= 50) return "주의"
  if (probability <= 75) return "위험"
  return "매우위험"
}

function buildFallbackDetection(text: string): AIKillerResult {
  // 간단한 휴리스틱 기반 탐지
  const textLength = text.length
  const sentenceCount = text.split(/[.!?]/).filter(s => s.trim().length > 0).length
  const avgSentenceLength = textLength / Math.max(sentenceCount, 1)

  // 의심 키워드 카운트
  const perfectKeywords = ["성장했다", "기여했다", "함양했다", "발전시켰다", "향상시켰다"]
  const vagueKeywords = ["다양한", "여러", "많은", "풍부한", "폭넓은"]
  
  let suspicionScore = 0
  
  perfectKeywords.forEach(keyword => {
    const count = (text.match(new RegExp(keyword, "g")) || []).length
    suspicionScore += count * 10
  })
  
  vagueKeywords.forEach(keyword => {
    const count = (text.match(new RegExp(keyword, "g")) || []).length
    suspicionScore += count * 5
  })

  // 평균 문장 길이가 너무 일정하면 의심
  if (avgSentenceLength > 80 && avgSentenceLength < 120) {
    suspicionScore += 15
  }

  const probability = Math.min(100, Math.max(0, suspicionScore))
  const riskAssessment = assessRiskLevel(probability)

  const detectedSections: DetectedSection[] = []
  
  if (probability > 30) {
    // 의심 섹션 추출 (첫 100자)
    const firstSection = text.substring(0, 100) + "..."
    detectedSections.push({
      section: firstSection,
      probability: probability,
      indicators: [
        "반복적인 완벽한 표현 사용",
        "구체적 사례 부족",
        "일반적이고 추상적인 서술"
      ]
    })
  }

  return {
    overallAIProbability: probability,
    riskAssessment,
    detectedSections,
    humanIndicators: [
      "일부 구체적인 활동 내용 포함",
      "자연스러운 문장 흐름 존재",
      "개인적인 표현 일부 발견"
    ],
    aiIndicators: probability > 30 ? [
      "과도하게 완벽한 문법 구조",
      "일반적이고 추상적인 표현 반복",
      "구체적 사례와 맥락 부족"
    ] : [],
    recommendations: [
      "더 구체적인 활동 사례와 맥락을 추가하세요.",
      "개인적인 감정이나 시행착오 과정을 포함하세요.",
      "다양한 문체와 표현을 사용하여 자연스러움을 높이세요."
    ]
  }
}

export async function POST(request: NextRequest) {
  try {
    if (!GEMINI_API_KEY) {
      console.error("[Detect] Missing GEMINI_API_KEY")
      return NextResponse.json({ error: "서버에 Gemini API 키가 설정되지 않았습니다." }, { status: 500 })
    }

    const { text } = await request.json()

    if (!text || typeof text !== "string" || text.trim().length === 0) {
      return NextResponse.json({ error: "분석할 텍스트가 필요합니다." }, { status: 400 })
    }

    const prompt = createDetectionPrompt(text)

    // 🧠 하이브리드 AI: AI 탐지는 텍스트 길이에 따라 모델 선택
    const selectedModel = getModelForTask({ 
      type: 'detect', 
      textLength: text.length 
    })
    globalCostTracker.trackRequest(selectedModel)
    console.log(`[Detect] 🚀 ${selectedModel.name} 사용 (텍스트: ${text.length}자)`)

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
          temperature: selectedModel.temperature,
          topK: 20,
          topP: 0.85,
          maxOutputTokens: selectedModel.maxTokens,
        },
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("[Detect] Gemini API error", response.status, errorText)
      return NextResponse.json(
        { error: `AI 작성 탐지 API 호출 실패 (${response.status})`, details: errorText },
        { status: 502 }
      )
    }

    const payload = (await response.json()) as GeminiResponse
    const generatedText = payload.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || ""

    if (!generatedText) {
      console.error("[Detect] Empty response from Gemini")
      return NextResponse.json(
        { error: "AI가 응답하지 않았습니다. 다시 시도해주세요." },
        { status: 502 }
      )
    }

    const jsonBlock = extractJsonBlock(generatedText)

    if (!jsonBlock) {
      console.error("[Detect] Failed to extract JSON from response:", generatedText.substring(0, 200))
      return NextResponse.json(
        { error: "AI 응답 형식 오류. AI가 예상치 못한 형식으로 응답했습니다.", raw: generatedText.substring(0, 200) },
        { status: 502 }
      )
    }

    try {
      const parsed = JSON.parse(jsonBlock) as Partial<AIKillerResult>
      
      // Ensure riskAssessment is properly set
      const probability = parsed.overallAIProbability || 0
      const result: AIKillerResult = {
        overallAIProbability: probability,
        riskAssessment: parsed.riskAssessment || assessRiskLevel(probability),
        detectedSections: parsed.detectedSections || [],
        humanIndicators: parsed.humanIndicators || [],
        aiIndicators: parsed.aiIndicators || [],
        recommendations: parsed.recommendations || [
          "더 구체적인 활동 사례를 추가하세요.",
          "자연스러운 표현을 사용하세요."
        ]
      }
      
      return NextResponse.json({ result })
    } catch (parseError) {
      console.error("[Detect] JSON parse error", parseError)
      return NextResponse.json(
        { error: "AI 응답 JSON 파싱 실패. AI 응답 형식이 올바르지 않습니다.", raw: jsonBlock?.substring(0, 200) },
        { status: 502 }
      )
    }
  } catch (error) {
    console.error("[Detect] Unexpected error", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "AI 작성 탐지 중 오류가 발생했습니다." },
      { status: 500 }
    )
  }
}
