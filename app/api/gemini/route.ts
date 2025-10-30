import { type NextRequest, NextResponse } from "next/server"
import { getModelForTask, globalCostTracker } from "@/lib/ai-model-router"

const GEMINI_API_KEY = process.env.GEMINI_API_KEY

export async function POST(request: NextRequest) {
  try {
    const { prompt, task } = await request.json()

    if (!GEMINI_API_KEY) {
      console.error("[v0] Missing GEMINI_API_KEY environment variable")
      return NextResponse.json({ error: "서버에 Gemini API 키가 설정되지 않았습니다." }, { status: 500 })
    }

    // 🧠 하이브리드 AI: 일반 작업은 task 타입에 따라 모델 선택
    const selectedModel = getModelForTask({ 
      type: task || 'general',
      textLength: prompt?.length || 0
    })
    globalCostTracker.trackRequest(selectedModel)
    console.log(`[Gemini] 🚀 ${selectedModel.name} 사용 (task: ${task})`)

    const response = await fetch(
      `${selectedModel.endpoint}?key=${GEMINI_API_KEY}`,
      {
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
            topK: 40,
            topP: 0.95,
            maxOutputTokens: task === "name" ? 50 : task === "description" ? 200 : selectedModel.maxTokens,
          },
        }),
      },
    )

    if (!response.ok) {
      const errorText = await response.text()
      console.error("[v0] Gemini API error:", errorText)
      throw new Error(`Gemini API error: ${response.status}`)
    }

    const data = await response.json()
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || ""

    return NextResponse.json({ text })
  } catch (error) {
    console.error("[v0] Gemini route error:", error)
    return NextResponse.json({ error: "Gemini API 호출 실패" }, { status: 500 })
  }
}
