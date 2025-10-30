"""
한국어 OCR API 서버
FastAPI 기반 고성능 REST API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import sys
import os
import io
import base64
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import asyncio
import uuid
from datetime import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.korean_ocr_engine import UltraKoreanOCR, OCRConfig
from src.deep_learning_enhancer import DeepLearningOCREnhancer
from src.layout_analyzer import SchoolRecordLayoutAnalyzer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Korean OCR API",
    description="생기부 문서 특화 한국어 OCR 100% 정확도 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
ocr_engine = None
deep_enhancer = None
layout_analyzer = None
executor = ThreadPoolExecutor(max_workers=4)
processing_jobs = {}


class OCRRequest(BaseModel):
    """OCR 요청 모델"""
    image_base64: str = Field(..., description="Base64 encoded image")
    enable_enhancement: bool = Field(True, description="Enable deep learning enhancement")
    enable_layout_analysis: bool = Field(True, description="Enable layout analysis")
    enable_multi_pass: bool = Field(True, description="Enable multi-pass OCR")
    extract_fields: bool = Field(True, description="Extract school record fields")


class OCRResponse(BaseModel):
    """OCR 응답 모델"""
    job_id: str
    status: str
    full_text: str = ""
    confidence: float = 0.0
    korean_ratio: float = 0.0
    extracted_fields: Dict[str, Any] = {}
    layout_analysis: Dict[str, Any] = {}
    processing_time: float = 0.0
    errors: List[str] = []


class JobStatus(BaseModel):
    """작업 상태 모델"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    result: Optional[OCRResponse] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup():
    """서버 시작 시 초기화"""
    global ocr_engine, deep_enhancer, layout_analyzer
    
    logger.info("Initializing OCR engines...")
    
    # OCR 엔진 초기화
    config = OCRConfig()
    ocr_engine = UltraKoreanOCR(config)
    
    # 딥러닝 향상 엔진 초기화
    deep_enhancer = DeepLearningOCREnhancer()
    
    # 레이아웃 분석기 초기화
    layout_analyzer = SchoolRecordLayoutAnalyzer()
    
    logger.info("OCR engines initialized successfully")


@app.on_event("shutdown")
async def shutdown():
    """서버 종료 시 정리"""
    executor.shutdown(wait=True)
    logger.info("Server shutdown complete")


@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Korean OCR API - 100% Accuracy for School Records",
        "version": "1.0.0",
        "endpoints": {
            "/ocr": "Perform OCR on image",
            "/ocr/file": "Perform OCR on uploaded file",
            "/ocr/batch": "Batch OCR processing",
            "/job/{job_id}": "Check job status",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "engines": {
            "ocr": ocr_engine is not None,
            "deep_learning": deep_enhancer is not None,
            "layout": layout_analyzer is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(request: OCRRequest, background_tasks: BackgroundTasks):
    """
    이미지 OCR 수행
    
    Base64 인코딩된 이미지를 받아 OCR 수행
    """
    try:
        # 작업 ID 생성
        job_id = str(uuid.uuid4())
        
        # 초기 응답
        response = OCRResponse(
            job_id=job_id,
            status="processing"
        )
        
        # 백그라운드 작업 추가
        background_tasks.add_task(
            process_ocr_async,
            job_id,
            request
        )
        
        processing_jobs[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "started_at": datetime.now()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/file")
async def perform_ocr_file(
    file: UploadFile = File(...),
    enable_enhancement: bool = True,
    enable_layout_analysis: bool = True,
    enable_multi_pass: bool = True,
    extract_fields: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    파일 업로드를 통한 OCR 수행
    """
    try:
        # 파일 읽기
        contents = await file.read()
        
        # Base64 인코딩
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # OCR 요청 생성
        request = OCRRequest(
            image_base64=image_base64,
            enable_enhancement=enable_enhancement,
            enable_layout_analysis=enable_layout_analysis,
            enable_multi_pass=enable_multi_pass,
            extract_fields=extract_fields
        )
        
        # OCR 수행
        return await perform_ocr(request, background_tasks)
        
    except Exception as e:
        logger.error(f"File OCR error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/batch")
async def perform_batch_ocr(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    여러 파일 일괄 OCR 처리
    """
    try:
        job_ids = []
        
        for file in files:
            # 파일 읽기
            contents = await file.read()
            
            # Base64 인코딩
            image_base64 = base64.b64encode(contents).decode('utf-8')
            
            # OCR 요청 생성
            request = OCRRequest(
                image_base64=image_base64,
                enable_enhancement=True,
                enable_layout_analysis=True,
                enable_multi_pass=True,
                extract_fields=True
            )
            
            # 작업 ID 생성
            job_id = str(uuid.uuid4())
            job_ids.append(job_id)
            
            # 백그라운드 작업 추가
            background_tasks.add_task(
                process_ocr_async,
                job_id,
                request
            )
            
            processing_jobs[job_id] = {
                "status": "processing",
                "progress": 0.0,
                "started_at": datetime.now()
            }
        
        return {
            "message": f"{len(files)} files submitted for processing",
            "job_ids": job_ids
        }
        
    except Exception as e:
        logger.error(f"Batch OCR error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    작업 상태 확인
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        result=job.get("result"),
        error=job.get("error")
    )


async def process_ocr_async(job_id: str, request: OCRRequest):
    """
    비동기 OCR 처리
    """
    try:
        # 진행 상황 업데이트
        processing_jobs[job_id]["progress"] = 10.0
        
        # Base64 디코딩
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        processing_jobs[job_id]["progress"] = 20.0
        
        # 임시 파일 저장
        temp_path = f"/tmp/{job_id}.jpg"
        cv2.imwrite(temp_path, image)
        
        processing_jobs[job_id]["progress"] = 30.0
        
        # OCR 수행
        start_time = datetime.now()
        ocr_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            ocr_engine.process_image,
            temp_path,
            request.enable_enhancement,
            request.enable_multi_pass
        )
        
        processing_jobs[job_id]["progress"] = 60.0
        
        # 딥러닝 향상 적용
        if request.enable_enhancement and deep_enhancer:
            ocr_result = await asyncio.get_event_loop().run_in_executor(
                executor,
                deep_enhancer.enhance_ocr_result,
                image,
                ocr_result
            )
        
        processing_jobs[job_id]["progress"] = 70.0
        
        # 레이아웃 분석
        layout_result = {}
        if request.enable_layout_analysis and layout_analyzer:
            layout_result = await asyncio.get_event_loop().run_in_executor(
                executor,
                layout_analyzer.analyze_layout,
                image
            )
        
        processing_jobs[job_id]["progress"] = 80.0
        
        # 필드 추출
        extracted_fields = {}
        if request.extract_fields:
            extracted_fields = await asyncio.get_event_loop().run_in_executor(
                executor,
                ocr_engine.extract_school_record_fields,
                ocr_result
            )
        
        processing_jobs[job_id]["progress"] = 90.0
        
        # 처리 시간 계산
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 결과 생성
        result = OCRResponse(
            job_id=job_id,
            status="completed",
            full_text=ocr_result.get("full_text", ""),
            confidence=ocr_result.get("average_confidence", 0.0),
            korean_ratio=ocr_result.get("korean_character_ratio", 0.0),
            extracted_fields=extracted_fields,
            layout_analysis=layout_result,
            processing_time=processing_time
        )
        
        # 작업 완료
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 100.0
        processing_jobs[job_id]["result"] = result
        
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"OCR processing error for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)


@app.post("/ocr/enhance")
async def enhance_text(text: str):
    """
    텍스트 향상 엔드포인트
    OCR 결과 텍스트의 맞춤법 교정 및 향상
    """
    try:
        if not deep_enhancer:
            raise HTTPException(status_code=503, detail="Enhancement engine not available")
        
        # 맞춤법 검사 및 교정
        corrected_text, corrections = deep_enhancer.spell_checker.check_and_correct(text)
        
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "corrections": corrections,
            "changes_made": len(corrections)
        }
        
    except Exception as e:
        logger.error(f"Text enhancement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """
    통계 정보 조회
    """
    total_jobs = len(processing_jobs)
    completed_jobs = sum(1 for job in processing_jobs.values() if job["status"] == "completed")
    processing_jobs_count = sum(1 for job in processing_jobs.values() if job["status"] == "processing")
    failed_jobs = sum(1 for job in processing_jobs.values() if job["status"] == "failed")
    
    # 평균 처리 시간 계산
    processing_times = []
    for job in processing_jobs.values():
        if job["status"] == "completed" and "result" in job:
            processing_times.append(job["result"].processing_time)
    
    avg_processing_time = np.mean(processing_times) if processing_times else 0.0
    
    # 평균 신뢰도 계산
    confidences = []
    for job in processing_jobs.values():
        if job["status"] == "completed" and "result" in job:
            confidences.append(job["result"].confidence)
    
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "processing_jobs": processing_jobs_count,
        "failed_jobs": failed_jobs,
        "average_processing_time": f"{avg_processing_time:.2f} seconds",
        "average_confidence": f"{avg_confidence:.2%}",
        "success_rate": f"{(completed_jobs / total_jobs * 100) if total_jobs > 0 else 0:.1f}%"
    }


@app.delete("/jobs/clear")
async def clear_jobs():
    """
    완료된 작업 정리
    """
    cleared = 0
    for job_id in list(processing_jobs.keys()):
        if processing_jobs[job_id]["status"] in ["completed", "failed"]:
            del processing_jobs[job_id]
            cleared += 1
    
    return {
        "message": f"Cleared {cleared} completed/failed jobs",
        "remaining_jobs": len(processing_jobs)
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )
