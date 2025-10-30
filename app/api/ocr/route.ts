import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export const maxDuration = 300 // 5 minutes
export const dynamic = 'force-dynamic'

interface OCRResult {
  texts: string[]
  error?: string
}

/**
 * PaddleOCR 백엔드 프록시 API
 * 
 * 한국어 100% 정확도를 위한 PaddleOCR 백엔드 연결
 * - 고급 이미지 전처리 (7단계)
 * - PaddleOCR 한국어 최적화
 * - 자동 재시도 로직
 */

// PaddleOCR 백엔드 URL
const PADDLE_OCR_URL = process.env.PADDLE_OCR_URL || "http://localhost:8000"

/**
 * PaddleOCR 백엔드 헬스 체크
 */
async function checkPaddleOCRHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${PADDLE_OCR_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      // 5초 타임아웃
      signal: AbortSignal.timeout(5000)
    })
    
    if (response.ok) {
      const data = await response.json()
      console.log("[OCR Proxy] 백엔드 상태:", data)
      return data.ready === true
    }
    
    return false
  } catch (error) {
    console.error("[OCR Proxy] ❌ 헬스 체크 실패:", error)
    return false
  }
}

/**
 * PaddleOCR 백엔드로 요청 전송 (재시도 로직 포함)
 */
async function callPaddleOCR(
  formData: FormData,
  maxRetries: number = 3
): Promise<{ texts: string[], error?: string }> {
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`[OCR Proxy] 🚀 PaddleOCR 요청 (시도 ${attempt}/${maxRetries})`)
      console.log(`[OCR Proxy] 📡 백엔드 URL: ${PADDLE_OCR_URL}/ocr`)
      
      // 300초 타임아웃 (5분)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 300000)
      
      const response = await fetch(`${PADDLE_OCR_URL}/ocr`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        // FormData는 자동으로 multipart/form-data로 설정됨
        // Content-Type 헤더를 수동으로 설정하지 말 것!
      })
      
      clearTimeout(timeoutId)
      
      console.log(`[OCR Proxy] ✅ 응답 받음 (상태: ${response.status})`)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error(`[OCR Proxy] ❌ 백엔드 오류 (${response.status}):`, errorText)
        
        // 서버 오류이고 재시도 가능하면 다시 시도
        if (response.status >= 500 && attempt < maxRetries) {
          const waitTime = Math.pow(2, attempt) * 1000 // Exponential backoff
          console.log(`[OCR Proxy] 🔄 ${waitTime}ms 후 재시도...`)
          await new Promise(resolve => setTimeout(resolve, waitTime))
          continue
        }
        
        return {
          texts: [],
          error: `OCR 백엔드 오류 (${response.status}): ${errorText}`
        }
      }
      
      const data = await response.json()
      console.log(`[OCR Proxy] 📊 응답 데이터:`, {
        success: data.success,
        textCount: data.texts?.length || 0,
        hasError: !!data.error
      })
      
      if (!data.success || data.error) {
        console.error(`[OCR Proxy] ❌ OCR 처리 실패:`, data.error)
        
        // 재시도 가능한 오류인지 확인
        if (attempt < maxRetries) {
          const waitTime = Math.pow(2, attempt) * 1000
          console.log(`[OCR Proxy] 🔄 ${waitTime}ms 후 재시도...`)
          await new Promise(resolve => setTimeout(resolve, waitTime))
          continue
        }
        
        return {
          texts: [],
          error: data.error || "OCR 처리에 실패했습니다"
        }
      }
      
      // 성공!
      const successCount = data.texts.filter((t: string) => t && t.length > 0).length
      console.log(`[OCR Proxy] ✅ OCR 성공: ${successCount}/${data.texts.length}개 파일`)
      
      return {
        texts: data.texts,
        error: data.error
      }
      
    } catch (error) {
      console.error(`[OCR Proxy] ❌ 요청 실패 (시도 ${attempt}/${maxRetries}):`, error)
      
      // 타임아웃 오류
      if (error instanceof Error && error.name === 'AbortError') {
        return {
          texts: [],
          error: "OCR 처리 시간이 초과되었습니다 (5분). 이미지 크기를 줄여주세요."
        }
      }
      
      // 연결 오류 - 재시도
      if (attempt < maxRetries) {
        const waitTime = Math.pow(2, attempt) * 1000
        console.log(`[OCR Proxy] 🔄 ${waitTime}ms 후 재시도...`)
        await new Promise(resolve => setTimeout(resolve, waitTime))
        continue
      }
      
      // 최종 실패
      return {
        texts: [],
        error: error instanceof Error ? error.message : "OCR 백엔드 연결 실패"
      }
    }
  }
  
  // 모든 재시도 실패
  return {
    texts: [],
    error: "OCR 처리에 실패했습니다. 잠시 후 다시 시도해주세요."
  }
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()
  
  try {
    console.log("[OCR Proxy] " + "=".repeat(70))
    console.log("[OCR Proxy] 🚀 한국어 100% 정확도 OCR 시작")
    console.log("[OCR Proxy] 🎯 PaddleOCR 백엔드 연결")
    console.log("[OCR Proxy] 📡 백엔드 URL: " + PADDLE_OCR_URL)
    console.log("[OCR Proxy] " + "=".repeat(70))
    
    // 1. 백엔드 헬스 체크
    console.log("[OCR Proxy] 🏥 백엔드 헬스 체크 중...")
    const isHealthy = await checkPaddleOCRHealth()
    
    if (!isHealthy) {
      console.error("[OCR Proxy] ❌ PaddleOCR 백엔드가 응답하지 않습니다")
      console.error("[OCR Proxy] 💡 해결 방법:")
      console.error("[OCR Proxy]    1. 백엔드가 실행 중인지 확인: cd ocr_service && ./start_ocr.sh")
      console.error("[OCR Proxy]    2. 포트 확인: lsof -i:8000")
      console.error("[OCR Proxy]    3. 환경 변수 확인: PADDLE_OCR_URL=" + PADDLE_OCR_URL)
      
      return NextResponse.json({
        error: "OCR 백엔드가 실행되지 않았습니다.\n\n다음 명령어로 백엔드를 시작하세요:\ncd ocr_service && ./start_ocr.sh\n\n또는 npm run dev:ocr"
      }, { status: 503 })
    }
    
    console.log("[OCR Proxy] ✅ 백엔드 연결 확인됨")
    
    // 2. FormData 파싱
    const formData = await request.formData()
    const files = formData.getAll("files")
    
    if (!files.length) {
      console.error("[OCR Proxy] ❌ 업로드된 파일이 없습니다")
      return NextResponse.json({
        error: "업로드된 파일이 없습니다."
      }, { status: 400 })
    }
    
    console.log(`[OCR Proxy] 📁 파일 개수: ${files.length}`)
    
    // 파일 정보 로깅
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      if (file instanceof File) {
        console.log(`[OCR Proxy] 📄 파일 ${i + 1}: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`)
      }
    }
    
    // 3. PaddleOCR 백엔드로 전달 (재시도 로직 포함)
    console.log("[OCR Proxy] 🔄 백엔드로 요청 전송 중...")
    
    const result = await callPaddleOCR(formData, 3)
    
    if (result.error && result.texts.length === 0) {
      console.error("[OCR Proxy] ❌ OCR 실패:", result.error)
      return NextResponse.json({
        error: result.error
      }, { status: 500 })
    }
    
    // 4. 성공
    const successCount = result.texts.filter(t => t && t.length > 0).length
    const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(2)
    
    console.log(`[OCR Proxy] ✅ OCR 완료: ${successCount}/${result.texts.length} 성공`)
    console.log(`[OCR Proxy] ⏱️  처리 시간: ${elapsedTime}초`)
    console.log("[OCR Proxy] " + "=".repeat(70))
    
    return NextResponse.json({
      texts: result.texts,
      error: result.error
    })
    
  } catch (error) {
    const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(2)
    console.error("[OCR Proxy] ❌ 예상치 못한 오류")
    console.error("[OCR Proxy] 오류 내용:", error)
    console.error(`[OCR Proxy] ⏱️  경과 시간: ${elapsedTime}초`)
    console.error("[OCR Proxy] " + "=".repeat(70))
    
    const message = error instanceof Error ? error.message : "OCR 처리 중 오류가 발생했습니다."
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
