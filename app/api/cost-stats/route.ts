import { NextResponse } from "next/server"
import { globalCostTracker } from "@/lib/ai-model-router"

/**
 * 비용 절감 통계 조회 API
 * GET /api/cost-stats
 */
export async function GET() {
  try {
    const stats = globalCostTracker.getStats()
    
    return NextResponse.json({
      success: true,
      stats,
      message: "하이브리드 AI 시스템으로 비용 최적화 중"
    })
  } catch (error) {
    console.error("[Cost Stats] Error:", error)
    return NextResponse.json(
      { error: "비용 통계 조회 중 오류가 발생했습니다." },
      { status: 500 }
    )
  }
}

/**
 * 비용 통계 초기화 API
 * POST /api/cost-stats/reset
 */
export async function POST() {
  try {
    globalCostTracker.reset()
    
    return NextResponse.json({
      success: true,
      message: "비용 통계가 초기화되었습니다."
    })
  } catch (error) {
    console.error("[Cost Stats] Reset error:", error)
    return NextResponse.json(
      { error: "비용 통계 초기화 중 오류가 발생했습니다." },
      { status: 500 }
    )
  }
}
