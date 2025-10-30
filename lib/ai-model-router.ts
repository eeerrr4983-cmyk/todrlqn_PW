/**
 * 🧠 Hybrid AI Model Router
 * 
 * 작업 복잡도에 따라 최적의 AI 모델을 자동 선택하여 비용 효율성 극대화
 * 
 * 모델 계층:
 * - Gemini 2.0 Flash-Exp: 복잡한 분석, 고급 추론 (높은 비용, 최고 성능)
 * - Gemini 1.5 Flash-8B: 단순 작업, 빠른 응답 (낮은 비용, 적절한 성능)
 */

export type TaskComplexity = 'simple' | 'medium' | 'complex'

export interface AITask {
  type: 'analyze' | 'detect' | 'university' | 'projects' | 'general' | 'name' | 'description'
  complexity?: TaskComplexity
  textLength?: number
  requiresDeepReasoning?: boolean
}

export interface ModelConfig {
  name: string
  endpoint: string
  costMultiplier: number // 1.0 = baseline, 0.5 = half cost, 2.0 = double cost
  maxTokens: number
  temperature: number
  description: string
}

/**
 * 사용 가능한 AI 모델들
 */
export const AI_MODELS = {
  // 🚀 메인 모델 - gemini-2.5-flash-lite (고급 작업 전용)
  'gemini-2.5-flash-lite': {
    name: 'Gemini 2.5 Flash-Lite',
    endpoint: 'https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent',
    costMultiplier: 2.0,
    maxTokens: 8192,
    temperature: 0.4,
    description: '최신 메인 모델, 복잡한 분석/고급 추론/멀티모달 작업에 최적'
  } as ModelConfig,
  
  // ⚡ 서브 모델 - gemini-2.0-flash-lite (경량 작업 전용)
  'gemini-2.0-flash-lite': {
    name: 'Gemini 2.0 Flash-Lite',
    endpoint: 'https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-lite:generateContent',
    costMultiplier: 0.5,
    maxTokens: 4096,
    temperature: 0.7,
    description: '경량 서브 모델, 단순 작업 및 빠른 응답으로 비용 절감'
  } as ModelConfig,
  
  // 🎯 백업 모델 - gemini-1.5-flash (중간 작업)
  'gemini-1.5-flash': {
    name: 'Gemini 1.5 Flash',
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent',
    costMultiplier: 1.0,
    maxTokens: 8192,
    temperature: 0.5,
    description: '백업 모델, 대부분의 작업에 적합'
  } as ModelConfig,
} as const

/**
 * 작업 복잡도 자동 분석
 */
export function analyzeTaskComplexity(task: AITask): TaskComplexity {
  // 명시적으로 지정된 복잡도가 있으면 우선 사용
  if (task.complexity) {
    return task.complexity
  }

  // 작업 타입에 따른 복잡도 판단
  switch (task.type) {
    case 'analyze':
      // 생기부 분석은 항상 복잡한 작업
      // - 2025 교육부 지침 완벽 이해 필요
      // - 다층적 분석 (강점, 개선점, 오류 탐지)
      // - 진로 연계성 평가
      return 'complex'
    
    case 'detect':
      // AI 작성 탐지는 중간 복잡도
      // - 패턴 분석 필요
      // - 다양한 지표 종합 판단
      // - 텍스트 길이에 따라 다름
      if (task.textLength && task.textLength > 2000) {
        return 'complex'
      }
      return 'medium'
    
    case 'university':
      // 대학 예측은 복잡한 작업
      // - 한국 대학 계층 구조 이해 필요
      // - 백분위 계산 및 매칭
      // - 다층적 추천 시스템
      return 'complex'
    
    case 'projects':
      // 프로젝트 추천은 중간 복잡도
      // - 진로 연계성 분석
      // - 창의적 아이디어 생성
      return 'medium'
    
    case 'name':
    case 'description':
      // 이름 생성, 간단한 설명은 단순 작업
      // - 짧은 텍스트 생성
      // - 고급 추론 불필요
      return 'simple'
    
    case 'general':
    default:
      // 일반 작업은 텍스트 길이로 판단
      if (task.textLength) {
        if (task.textLength > 3000) return 'complex'
        if (task.textLength > 1000) return 'medium'
      }
      return 'simple'
  }
}

/**
 * 복잡도에 따른 최적 모델 선택
 */
export function selectOptimalModel(complexity: TaskComplexity): ModelConfig {
  switch (complexity) {
    case 'complex':
      console.log('[AI Router] 🚀 복잡한 작업 → Gemini 2.5 Flash-Lite 메인 모델 사용')
      return AI_MODELS['gemini-2.5-flash-lite']
    
    case 'medium':
      console.log('[AI Router] 🎯 중간 작업 → Gemini 1.5 Flash 백업 모델 사용')
      return AI_MODELS['gemini-1.5-flash']
    
    case 'simple':
      console.log('[AI Router] ⚡ 단순 작업 → Gemini 2.0 Flash-Lite 서브 모델 사용 (50% 비용 절감)')
      return AI_MODELS['gemini-2.0-flash-lite']
  }
}

/**
 * 작업에 대한 최적 모델 자동 선택
 */
export function getModelForTask(task: AITask): ModelConfig {
  const complexity = analyzeTaskComplexity(task)
  const model = selectOptimalModel(complexity)
  
  console.log(`[AI Router] 📊 작업 분석:`, {
    type: task.type,
    complexity,
    model: model.name,
    costMultiplier: model.costMultiplier,
    savings: `${((1 - model.costMultiplier / 2.0) * 100).toFixed(0)}%`
  })
  
  return model
}

/**
 * 비용 절감 통계 추적
 */
export class CostTracker {
  private totalRequests = 0
  private totalCostMultiplier = 0
  
  trackRequest(model: ModelConfig) {
    this.totalRequests++
    this.totalCostMultiplier += model.costMultiplier
  }
  
  getStats() {
    const avgCostMultiplier = this.totalRequests > 0 
      ? this.totalCostMultiplier / this.totalRequests 
      : 0
    
    const savingsPercent = ((2.0 - avgCostMultiplier) / 2.0) * 100
    
    return {
      totalRequests: this.totalRequests,
      averageCostMultiplier: avgCostMultiplier.toFixed(2),
      estimatedSavings: `${savingsPercent.toFixed(1)}%`,
      details: `${this.totalRequests}건의 요청으로 평균 ${savingsPercent.toFixed(1)}% 비용 절감`
    }
  }
  
  reset() {
    this.totalRequests = 0
    this.totalCostMultiplier = 0
  }
}

// 전역 비용 추적기
export const globalCostTracker = new CostTracker()

/**
 * 사용 예시:
 * 
 * // 복잡한 생기부 분석
 * const model1 = getModelForTask({ type: 'analyze' })
 * // → Gemini 2.0 Flash-Exp (최고 성능)
 * 
 * // 단순 이름 생성
 * const model2 = getModelForTask({ type: 'name' })
 * // → Gemini 1.5 Flash-8B (50% 비용 절감)
 * 
 * // AI 작성 탐지 (텍스트 길이에 따라)
 * const model3 = getModelForTask({ type: 'detect', textLength: 3000 })
 * // → Gemini 2.0 Flash-Exp (긴 텍스트)
 * 
 * const model4 = getModelForTask({ type: 'detect', textLength: 500 })
 * // → Gemini 1.5 Flash (짧은 텍스트)
 */
