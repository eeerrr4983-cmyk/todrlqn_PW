"""
초월적 한국어 OCR API 서버
Ultimate Korean OCR API Server with 100% Accuracy Target
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import asyncio
import aiofiles
import uvicorn
import cv2
import numpy as np
import base64
import io
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import hashlib
import sys
import os

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ultimate_korean_ocr import get_ultimate_ocr, UltimateKoreanOCR

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Ultimate Korean OCR API",
    description="초월적 정확도의 한국어 OCR API - 생기부 문서 특화",
    version="3.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OCRRequest(BaseModel):
    """OCR 요청 모델"""
    image_base64: str = Field(..., description="Base64 인코딩된 이미지")
    enhance_image: bool = Field(True, description="이미지 향상 적용 여부")
    return_detailed: bool = Field(False, description="상세 결과 반환 여부")
    confidence_threshold: float = Field(0.9, description="신뢰도 임계값", ge=0.0, le=1.0)
    language_correction: bool = Field(True, description="언어 모델 교정 적용")
    
    class Config:
        schema_extra = {
            "example": {
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "enhance_image": True,
                "return_detailed": False,
                "confidence_threshold": 0.9,
                "language_correction": True
            }
        }


class OCRResponse(BaseModel):
    """OCR 응답 모델"""
    success: bool
    text: str
    confidence: float
    processing_time: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "text": "인식된 텍스트 내용",
                "confidence": 0.98,
                "processing_time": 1.23,
                "details": None,
                "error": None
            }
        }


class BatchOCRRequest(BaseModel):
    """배치 OCR 요청 모델"""
    images: List[str] = Field(..., description="Base64 인코딩된 이미지 리스트")
    parallel_processing: bool = Field(True, description="병렬 처리 여부")
    
    class Config:
        schema_extra = {
            "example": {
                "images": ["base64_image_1", "base64_image_2"],
                "parallel_processing": True
            }
        }


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    version: str
    ocr_engines: List[str]
    gpu_available: bool
    metrics: Dict[str, Any]


# OCR 엔진 초기화
ocr_engine: Optional[UltimateKoreanOCR] = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global ocr_engine
    logger.info("초월적 OCR 엔진 초기화 중...")
    ocr_engine = get_ultimate_ocr()
    logger.info("OCR 엔진 초기화 완료")


@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Ultimate Korean OCR API Server",
        "version": "3.0.0",
        "description": "초월적 정확도의 한국어 OCR API"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """헬스체크 엔드포인트"""
    import torch
    
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR 엔진이 초기화되지 않았습니다")
    
    metrics = ocr_engine.get_metrics()
    
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        ocr_engines=["PaddleOCR", "EasyOCR", "Tesseract", "TrOCR"],
        gpu_available=torch.cuda.is_available(),
        metrics=metrics
    )


@app.post("/ocr", response_model=OCRResponse, tags=["OCR"])
async def process_ocr(request: OCRRequest):
    """단일 이미지 OCR 처리"""
    start_time = time.time()
    
    try:
        # Base64 디코딩
        image_data = base64.b64decode(request.image_base64.split(',')[-1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("이미지 디코딩 실패")
        
        # 임시 파일로 저장
        temp_path = f"/tmp/ocr_temp_{hashlib.md5(image_data).hexdigest()}.jpg"
        cv2.imwrite(temp_path, image)
        
        # OCR 처리
        result = await ocr_engine.process_image_async(temp_path)
        
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)
        
        # 신뢰도 필터링
        if result['confidence'] < request.confidence_threshold:
            logger.warning(f"낮은 신뢰도: {result['confidence']}")
        
        processing_time = time.time() - start_time
        
        response = OCRResponse(
            success=True,
            text=result['text'],
            confidence=result['confidence'],
            processing_time=processing_time,
            details=result if request.return_detailed else None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"OCR 처리 오류: {e}")
        return OCRResponse(
            success=False,
            text="",
            confidence=0.0,
            processing_time=time.time() - start_time,
            error=str(e)
        )


@app.post("/ocr/batch", tags=["OCR"])
async def process_batch_ocr(request: BatchOCRRequest):
    """배치 이미지 OCR 처리"""
    start_time = time.time()
    results = []
    
    async def process_single(image_base64: str, idx: int) -> Dict:
        try:
            # Base64 디코딩
            image_data = base64.b64decode(image_base64.split(',')[-1])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 임시 파일로 저장
            temp_path = f"/tmp/ocr_batch_{idx}_{hashlib.md5(image_data).hexdigest()}.jpg"
            cv2.imwrite(temp_path, image)
            
            # OCR 처리
            result = await ocr_engine.process_image_async(temp_path)
            
            # 임시 파일 삭제
            Path(temp_path).unlink(missing_ok=True)
            
            return {
                "index": idx,
                "success": True,
                "text": result['text'],
                "confidence": result['confidence']
            }
        except Exception as e:
            return {
                "index": idx,
                "success": False,
                "error": str(e)
            }
    
    if request.parallel_processing:
        # 병렬 처리
        tasks = [process_single(img, idx) for idx, img in enumerate(request.images)]
        results = await asyncio.gather(*tasks)
    else:
        # 순차 처리
        for idx, img in enumerate(request.images):
            result = await process_single(img, idx)
            results.append(result)
    
    processing_time = time.time() - start_time
    
    return {
        "success": True,
        "total_images": len(request.images),
        "results": results,
        "processing_time": processing_time,
        "average_time_per_image": processing_time / len(request.images) if request.images else 0
    }


@app.post("/ocr/file", tags=["OCR"])
async def process_file_ocr(
    file: UploadFile = File(...),
    enhance_image: bool = True,
    language_correction: bool = True
):
    """파일 업로드 OCR 처리"""
    start_time = time.time()
    
    try:
        # 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("이미지 파일을 읽을 수 없습니다")
        
        # 임시 파일로 저장
        temp_path = f"/tmp/ocr_upload_{hashlib.md5(contents).hexdigest()}.jpg"
        cv2.imwrite(temp_path, image)
        
        # OCR 처리
        result = await ocr_engine.process_image_async(temp_path)
        
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)
        
        processing_time = time.time() - start_time
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "text": result['text'],
            "confidence": result['confidence'],
            "processing_time": processing_time,
            "details": result
        })
        
    except Exception as e:
        logger.error(f"파일 OCR 처리 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        )


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """성능 메트릭 조회"""
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR 엔진이 초기화되지 않았습니다")
    
    metrics = ocr_engine.get_metrics()
    
    return {
        "total_processed": metrics['total_processed'],
        "perfect_recognitions": metrics['perfect_recognitions'],
        "accuracy_rate": f"{metrics['accuracy_rate']:.2f}%",
        "average_confidence": f"{metrics['average_confidence']:.2%}",
        "average_processing_time": f"{metrics['average_processing_time']:.2f}s",
        "min_processing_time": f"{metrics['min_processing_time']:.2f}s",
        "max_processing_time": f"{metrics['max_processing_time']:.2f}s"
    }


@app.post("/test/sample", tags=["Test"])
async def test_sample_image():
    """샘플 이미지로 OCR 테스트"""
    try:
        # 샘플 이미지 생성 (간단한 텍스트 이미지)
        img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "생기부 테스트 문서", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 임시 파일로 저장
        temp_path = "/tmp/test_sample.jpg"
        cv2.imwrite(temp_path, img)
        
        # OCR 처리
        result = await ocr_engine.process_image_async(temp_path)
        
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)
        
        return {
            "success": True,
            "message": "테스트 완료",
            "result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.websocket("/ws/ocr")
async def websocket_ocr(websocket):
    """WebSocket을 통한 실시간 OCR 처리"""
    await websocket.accept()
    
    try:
        while True:
            # 클라이언트로부터 이미지 데이터 수신
            data = await websocket.receive_text()
            request = json.loads(data)
            
            # OCR 처리
            image_data = base64.b64decode(request['image'].split(',')[-1])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 임시 파일로 저장
            temp_path = f"/tmp/ws_ocr_{hashlib.md5(image_data).hexdigest()}.jpg"
            cv2.imwrite(temp_path, image)
            
            # OCR 처리
            result = await ocr_engine.process_image_async(temp_path)
            
            # 임시 파일 삭제
            Path(temp_path).unlink(missing_ok=True)
            
            # 결과 전송
            await websocket.send_json({
                "success": True,
                "text": result['text'],
                "confidence": result['confidence']
            })
            
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        await websocket.close()


if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(
        "ultimate_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
