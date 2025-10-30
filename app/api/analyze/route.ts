import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"
import { randomUUID } from "crypto"
import { getModelForTask, globalCostTracker } from "@/lib/ai-model-router"

const GEMINI_API_KEY = process.env.GEMINI_API_KEY

export const maxDuration = 180 // 🔴 PHASE 1: Increased to 180 seconds for better reliability

interface GeminiCandidatePart {
  text?: string
}

interface GeminiCandidate {
  content?: {
    parts?: GeminiCandidatePart[]
  }
}

interface GeminiResponse {
  candidates?: GeminiCandidate[]
  error?: { message?: string }
}

interface GeminiAnalysis {
  overallScore?: number
  studentProfile?: string
  careerAlignment?: {
    percentage?: number
    summary?: string
    strengths?: string[]
    improvements?: string[]
  }
  errors?: Array<{
    type?: string
    content?: string
    reason?: string
    page?: number
    suggestion?: string
  }>
  strengths?: string[]
  improvements?: string[]
  suggestions?: string[]
}

interface AnalysisError {
  type: "금지" | "주의"
  content: string
  reason: string
  page: number
  suggestion?: string
}

interface NormalizedAnalysisResult {
  id: string
  overallScore: number
  studentProfile: string
  careerDirection: string
  careerAlignment?: {
    percentage: number
    summary: string
    strengths: string[]
    improvements: string[]
  }
  errors: AnalysisError[]
  strengths: string[]
  improvements: string[]
  suggestions: string[]
  originalText: string
  analyzedAt: string
}

type AnalysisRequestBody = {
  text?: string
  careerDirection?: string
}

const GUIDELINE_2025 = `
### 2025 학교생활기록부 세부능력 및 특기사항 전수검토 가이드라인

당신은 2025학년도 교육부 훈령 제530호를 완벽히 숙지한 생기부 전문 검토 AI입니다.

#### 절대 금지 사항 (발견 시 즉시 지적):
1. **대학명**: 서울대, 연세대, 고려대, KAIST 등 모든 대학/전문대/대학원 명칭 및 변형 표현
2. **기관명**: 교육관련기관 외 모든 기관/단체/회사/재단/협회/학회/연구소 (예외: 교육부, 시도교육청만 허용)
3. **상호명**: 학원, 출판사, 기업, 앱, 플랫폼 (예: △△학원, EBS, 메가스터디)
4. **강사/인물명**: 특정인 실명 (예: ○○ 강사, △△ 교수)
5. **공인어학시험**: TOEIC, TOEFL, HSK, JLPT, IELTS, OPIc 등 모든 시험명 및 점수/급수
6. **인증시험**: 컴퓨터활용능력, 한자급수, 바리스타 등 모든 인증 및 급수
7. **모의고사 성적**: 등급, 백분위, 표준점수 등 모든 정량 지표
8. **대회 용어**: 교외대회 참가/수상, '대회' 단어 사용 자체 금지
9. **교외상/교외대회**: 학교 밖 모든 시상, 대회, 경연, 올림피아드
10. **온라인 플랫폼**: K-MOOC, MOOC, Coursera, edX, 유튜브 강의
11. **특허/논문/출판**: 특허출원, 논문게재, 도서출간, 저작권 등록
12. **사교육 유발 요소**: 장학금, 교외 멘토링, 캠프/워크숍, 해외연수

#### 주의 사항 (개선 권고):
1. **정량성적 오기재**: 점수, 등급, 석차 등 숫자 지표 (성적 미산출 과목 제외)
2. **단순 나열**: "~했다", "~참여함" 반복만으로 구체적 관찰 근거 없음
3. **모호한 칭찬**: "성실하고 적극적", "열심히 함" 등 근거 부족 표현
4. **미래 전망 추측**: "~할 것으로 예상", "~에 적합할 것" 등 추정성 문구
5. **교과 무관 내용**: 해당 교과 성취기준과 무관한 일반적 태도 서술
6. **인용문 내부 금지**: 도서 인용 시 출판사명, 저자명 포함 금지
`

function createAnalysisPrompt(text: string, careerDirection: string): string {
  return `${GUIDELINE_2025}

#### 분석 대상:
**학생 진로 방향**: ${careerDirection || "미지정"}

**생기부 텍스트**:
\`\`\`
${text}
\`\`\`

#### 분석 요청사항:
위 가이드라인을 철저히 적용하여 생기부 텍스트를 정밀 분석하고, 다음 JSON 형식으로 응답하세요:

\`\`\`json
{
  "overallScore": 0-100 사이의 점수 (금지사항 발견 시 대폭 감점),
  "studentProfile": "학생의 전문성과 특징을 한 문장으로 요약 (진로 연계)",
  "careerAlignment": {
    "percentage": 0-100,
    "summary": "진로 방향과의 연계성 평가 (2-3문장)"
  },
  "errors": [
    {
      "type": "금지" 또는 "주의",
      "content": "문제가 되는 원문을 정확히 인용",
      "reason": "위반 사유를 구체적으로 설명 (가이드라인 항목 번호 명시)",
      "page": 페이지 번호 (알 수 없으면 1),
      "suggestion": "구체적이고 실행 가능한 수정 방안 제시"
    }
  ],
  "strengths": [
    "발견된 강점을 구체적으로 설명 (진로 연계, 활동의 심화성, 지속성 등)",
    "최소 3개, 각각 1-2문장으로 상세 작성"
  ],
  "improvements": [
    "개선이 필요한 부분을 구체적으로 설명",
    "최소 3개, 각각 1-2문장으로 상세 작성"
  ],
  "suggestions": [
    "실행 가능한 구체적 개선 제안 (예: 진로 연계 활동 추가, 독서 심화 등)",
    "최소 3개"
  ]
}
\`\`\`

**중요**:
- errors 배열은 실제로 발견된 금지/주의 사항만 포함 (최대 5개까지만)
- 금지사항 1개 발견 시 overallScore에서 -15점, 주의사항 1개당 -5점
- 모든 항목은 한국어로 명확하고 간결하게 작성 (각 항목 50자 이내)
- errors의 reason은 핵심만 30자 이내로 간단히 작성
- JSON 형식을 정확히 준수 (중괄호, 따옴표, 쉼표 확인)
- 응답은 반드시 완전한 JSON으로 마무리할 것
- **절대 금지**: JSON 응답에 "Gemini", "Gemini는", "2.5 Pro" 등의 AI 모델 관련 언급 포함 금지
- **절대 금지**: JSON 응답에 개인정보 보호, 면책 조항, 경고 문구 포함 금지`
}

function extractGeneratedText(payload: GeminiResponse): string {
  return payload.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? ""
}

function extractJsonBlock(text: string): string | null {
  // 🔴 CRITICAL FIX: Robust JSON extraction with truncation repair
  const codeBlockMatch = text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/)
  if (codeBlockMatch) {
    return repairTruncatedJson(codeBlockMatch[1])
  }

  const jsonMatch = text.match(/\{[\s\S]*\}/)
  if (jsonMatch) {
    return repairTruncatedJson(jsonMatch[0])
  }

  return null
}

// 🔴 CRITICAL FIX: Remove Gemini contamination text
function removeGeminiContamination(text: string): string {
  // Remove Gemini disclaimers and warnings that contaminate JSON
  const contaminationPatterns = [
    /Gemini는.*?(?=\n|$)/g,
    /Gemini.*?(?=\n|$)/g,
    /2\.5 Pro.*?(?=\n|$)/g,
    /인물 등에 관한 정보 제공 시 실수를 할 수 있으니.*?(?=\n|$)/g,
    /다시 한번 확인하세요.*?(?=\n|$)/g,
    /개인 정보 보호.*?(?=\n|$)/g,
  ]
  
  let cleaned = text
  for (const pattern of contaminationPatterns) {
    cleaned = cleaned.replace(pattern, '')
  }
  
  return cleaned
}

// 🔴 CRITICAL FIX: Repair truncated JSON from Gemini
function repairTruncatedJson(jsonStr: string): string {
  try {
    // Try to parse as-is first
    JSON.parse(jsonStr)
    return jsonStr
  } catch (error) {
    console.log('[Analyze] 🔧 Repairing truncated JSON...')
    
    // Step 1: Remove Gemini contamination
    let repaired = removeGeminiContamination(jsonStr.trim())
    console.log('[Analyze] 🧹 Removed Gemini contamination')
    
    // Step 2: Handle incomplete errors array
    if (repaired.includes('"errors": [') && !repaired.includes(']')) {
      // Find the last complete error object
      const errorsStartIndex = repaired.indexOf('"errors": [')
      const afterErrors = repaired.substring(errorsStartIndex + '"errors": ['.length)
      
      // Find all opening braces after errors start
      const openBraces = (afterErrors.match(/\{/g) || []).length
      const closeBraces = (afterErrors.match(/\}/g) || []).length
      
      if (openBraces > closeBraces) {
        // We're inside an incomplete error object - remove it
        const lastCompleteObjectIndex = repaired.lastIndexOf('},')
        if (lastCompleteObjectIndex > errorsStartIndex) {
          // Keep up to the last complete object
          repaired = repaired.substring(0, lastCompleteObjectIndex + 1)
        } else {
          // No complete objects, empty the array
          repaired = repaired.substring(0, errorsStartIndex + '"errors": ['.length)
        }
      }
      
      // Close errors array and main object
      repaired += '\n  ]\n}'
    } else if (!repaired.endsWith('}')) {
      // Close main object
      repaired += '\n}'
    }
    
    // Step 3: Validate and clean up incomplete strings in content/reason fields
    try {
      const testParse = JSON.parse(repaired)
      if (testParse.errors && Array.isArray(testParse.errors)) {
        // Filter out errors with contaminated or incomplete text
        testParse.errors = testParse.errors.filter((err: any) => {
          const hasContent = err.content && err.content.length > 0
          const hasReason = err.reason && err.reason.length > 0
          const notContaminated = !err.content?.includes('Gemini') && !err.reason?.includes('Gemini')
          return hasContent && hasReason && notContaminated
        })
        repaired = JSON.stringify(testParse)
      }
    } catch {
      // If still invalid, try one more time
      console.log('[Analyze] 🔧 Final repair attempt...')
    }
    
    console.log('[Analyze] 🔧 Repaired JSON length:', repaired.length)
    return repaired
  }
}

function toNumber(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return fallback
}

function toArrayOfStrings(value: unknown, minFallback?: string[]): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "string" ? item.trim() : ""))
      .filter((item) => item.length > 0)
  }
  return minFallback ?? []
}

function normalizeErrors(value: GeminiAnalysis["errors"]): AnalysisError[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .map((item) => ({
      type: typeof item?.type === "string" && item.type.includes("금지") ? "금지" : "주의",
      content: typeof item?.content === "string" ? item.content.trim() : "",
      reason: typeof item?.reason === "string" ? item.reason.trim() : "",
      page:
        typeof item?.page === "number" && Number.isFinite(item.page) && item.page > 0
          ? Math.floor(item.page)
          : 1,
      suggestion: typeof item?.suggestion === "string" ? item.suggestion.trim() : undefined,
    }))
    .filter((item) => item.content.length > 0 || item.reason.length > 0)
}

function normalizeAnalysis(
  raw: GeminiAnalysis,
  careerDirection: string,
  originalText: string,
): NormalizedAnalysisResult {
  const overallScore = Math.max(0, Math.min(100, toNumber(raw.overallScore, 75)))

  const careerAlignment = raw.careerAlignment
    ? {
        percentage: Math.max(0, Math.min(100, toNumber(raw.careerAlignment.percentage, 50))),
        summary: typeof raw.careerAlignment.summary === "string" ? raw.careerAlignment.summary.trim() : "",
        strengths: toArrayOfStrings(raw.careerAlignment.strengths),
        improvements: toArrayOfStrings(raw.careerAlignment.improvements),
      }
    : undefined

  const normalized: NormalizedAnalysisResult = {
    id: randomUUID(),
    overallScore,
    studentProfile: typeof raw.studentProfile === "string" ? raw.studentProfile.trim() : "",
    careerDirection: careerDirection || "미지정",
    careerAlignment,
    errors: normalizeErrors(raw.errors),
    strengths: toArrayOfStrings(raw.strengths, []),
    improvements: toArrayOfStrings(raw.improvements, []),
    suggestions: toArrayOfStrings(raw.suggestions, []),
    originalText,
    analyzedAt: new Date().toISOString(),
  }

  // No fallback values - if AI doesn't provide data, show empty arrays
  // This ensures we show real AI analysis only
  return normalized
}

export async function POST(request: NextRequest) {
  console.log("[Analyze] === 분석 API 호출 시작 ===")
  
  try {
    if (!GEMINI_API_KEY) {
      console.error("[Analyze] ❌ Missing GEMINI_API_KEY environment variable")
      return NextResponse.json(
        { error: "서버에 Gemini API 키가 설정되지 않았습니다." },
        { status: 500 },
      )
    }

    const { text, careerDirection }: AnalysisRequestBody = await request.json()
    console.log(`[Analyze] 📄 받은 텍스트 길이: ${text?.length || 0} 글자`)
    console.log(`[Analyze] 🎯 진로 방향: ${careerDirection || "미지정"}`)

    if (!text || typeof text !== "string" || text.trim().length === 0) {
      console.error("[Analyze] ❌ 텍스트가 비어있음")
      return NextResponse.json(
        { error: "분석할 생기부 텍스트가 필요합니다." },
        { status: 400 },
      )
    }

    const prompt = createAnalysisPrompt(text, careerDirection ?? "")
    console.log(`[Analyze] 📝 프롬프트 생성 완료 (${prompt.length} 글자)`)
    
    // 🧠 하이브리드 AI: 생기부 분석은 복잡한 작업이므로 Gemini 2.0 Flash-Exp 사용
    const selectedModel = getModelForTask({ 
      type: 'analyze', 
      textLength: text.length,
      requiresDeepReasoning: true
    })
    globalCostTracker.trackRequest(selectedModel)
    
    console.log(`[Analyze] 🚀 ${selectedModel.name} API 호출 중...`)

    // 🔴 PHASE 1: Extended timeout with retry capability
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 180000) // 180 second timeout

    let response: Response
    try {
      response = await fetch(`${selectedModel.endpoint}?key=${GEMINI_API_KEY}`, {
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
          topK: 32,
          topP: 0.95,
          maxOutputTokens: selectedModel.maxTokens,
        },
        safetySettings: [
          {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_HATE_SPEECH",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold: "BLOCK_NONE",
          },
          {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold: "BLOCK_NONE",
          },
        ],
      }),
      signal: controller.signal,
    })
    clearTimeout(timeoutId)
    } catch (fetchError) {
      clearTimeout(timeoutId)
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        console.error("[Analyze] ⏱️ Gemini API 타임아웃")
        return NextResponse.json(
          { error: "AI 분석 시간이 초과되었습니다. 텍스트가 너무 길거나 서버가 혼잡합니다. 다시 시도해주세요." },
          { status: 504 },
        )
      }
      console.error("[Analyze] ❌ Fetch 오류:", fetchError)
      throw fetchError
    }

    console.log(`[Analyze] ✅ Gemini API 응답 받음 (상태: ${response.status})`)

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`[Analyze] ❌ Gemini API 오류 (${response.status}):`, errorText)
      return NextResponse.json(
        { error: `Gemini API 호출에 실패했습니다 (${response.status}). 잠시 후 다시 시도해주세요.`, details: errorText },
        { status: 502 },
      )
    }

    const payload = (await response.json()) as GeminiResponse
    console.log("[Analyze] 🔍 Gemini 응답 JSON 파싱 완료")
    
    const generatedText = extractGeneratedText(payload)
    console.log(`[Analyze] 📜 생성된 텍스트 길이: ${generatedText.length} 글자`)

    if (!generatedText) {
      console.error("[Analyze] ❌ Gemini 응답에 텍스트가 없습니다:", JSON.stringify(payload, null, 2))
      return NextResponse.json(
        { error: "AI 응답에서 결과를 찾을 수 없습니다. AI 서비스가 응답하지 않았습니다." },
        { status: 502 },
      )
    }

    const jsonBlock = extractJsonBlock(generatedText)
    console.log("[Analyze] 🔧 JSON 블록 추출 시도...")

    if (!jsonBlock) {
      console.error("[Analyze] ❌ JSON 블록 찾기 실패. 원본 응답:")
      console.error(generatedText.substring(0, 500))
      return NextResponse.json(
        { error: "AI 응답을 JSON으로 파싱할 수 없습니다. AI가 예상치 못한 형식으로 응답했습니다.", raw: generatedText.substring(0, 500) },
        { status: 502 },
      )
    }
    
    console.log("[Analyze] ✅ JSON 블록 추출 성공")

    let parsed: GeminiAnalysis

    try {
      parsed = JSON.parse(jsonBlock) as GeminiAnalysis
      console.log("[Analyze] ✅ JSON 파싱 성공")
      console.log(`[Analyze] 📊 점수: ${parsed.overallScore}, 오류: ${parsed.errors?.length || 0}개, 강점: ${parsed.strengths?.length || 0}개`)
    } catch (error) {
      console.error("[Analyze] ❌ JSON 파싱 오류:", error)
      console.error("[Analyze] 실패한 JSON:", jsonBlock.substring(0, 500))
      return NextResponse.json(
        { error: "AI 응답을 JSON으로 파싱하는 중 오류가 발생했습니다. AI 응답 형식이 올바르지 않습니다.", raw: jsonBlock.substring(0, 500) },
        { status: 502 },
      )
    }

    const normalized = normalizeAnalysis(parsed, careerDirection ?? "", text)
    console.log(`[Analyze] ✅ 정규화 완료 (ID: ${normalized.id})`)
    console.log("[Analyze] === 분석 성공적으로 완료 ===")

    return NextResponse.json({ result: normalized, raw: generatedText })
  } catch (error) {
    console.error("[Analyze] ❌❌❌ 예상치 못한 오류 발생 ❌❌❌")
    console.error(error)
    
    const errorMessage = error instanceof Error ? error.message : "알 수 없는 오류"
    const errorStack = error instanceof Error ? error.stack : undefined
    
    console.error("[Analyze] 오류 메시지:", errorMessage)
    if (errorStack) {
      console.error("[Analyze] 스택 트레이스:", errorStack)
    }
    
    return NextResponse.json(
      {
        error: `생기부 분석 중 오류가 발생했습니다: ${errorMessage}`,
        details: errorStack?.split('\n').slice(0, 3).join('\n'),
      },
      { status: 500 },
    )
  }
}
