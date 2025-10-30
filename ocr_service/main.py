"""
PaddleOCR-VL FastAPI 서버 - 공식 API 스펙 준수
=====================================================
한국어 100% 정확도를 위한 최첨단 PaddleOCR-VL 백엔드 서비스

주요 기능:
- PaddleOCR-VL: 109개 언어 지원 (한국어 포함)
- REST API: POST /layout-parsing (공식 스펙)
- Base64 이미지 입력 지원
- 문서 전처리 (방향 보정, 왜곡 보정)
- 레이아웃 검출 및 구조화된 텍스트 추출
- JSON/Markdown 결과 반환
"""

import os
import sys
import io
import traceback
import logging
import tempfile
import base64
import uuid
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

try:
    from .preprocessing import KoreanOCRPreprocessor, PreprocessingReport
except ImportError:  # pragma: no cover
    from preprocessing import KoreanOCRPreprocessor, PreprocessingReport

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# PaddleOCR-VL 파이프라인
paddleocr_vl = None
ocr_available = False
preprocessor = KoreanOCRPreprocessor()

# ============================================================================
# Pydantic 모델 (공식 API 스펙)
# ============================================================================

class LayoutParsingRequest(BaseModel):
    """레이아웃 파싱 요청 모델 (공식 API 스펙)"""
    file: str = Field(..., description="Base64 인코딩된 이미지 또는 URL")
    fileType: Optional[int] = Field(None, description="0=PDF, 1=이미지")
    useDocUnwarping: Optional[bool] = Field(None, description="이미지 왜곡 보정")
    useLayoutDetection: Optional[bool] = Field(None, description="레이아웃 검출")
    useChartRecognition: Optional[bool] = Field(None, description="차트 인식")
    layoutThreshold: Optional[Union[float, Dict]] = Field(None, description="레이아웃 임계값")
    layoutNms: Optional[bool] = Field(None, description="NMS 사용")
    layoutUnclipRatio: Optional[Union[float, list, Dict]] = Field(None, description="박스 확장 비율")
    layoutMergeBboxesMode: Optional[Union[str, Dict]] = Field(None, description="박스 병합 모드")
    promptLabel: Optional[Union[str, Dict]] = Field(None, description="프롬프트 레이블")
    formatBlockContent: Optional[bool] = Field(None, description="블록 내용 포맷")
    repetitionPenalty: Optional[float] = Field(None, description="반복 패널티")
    temperature: Optional[float] = Field(None, description="Temperature")
    topP: Optional[float] = Field(None, description="Top-P")
    minPixels: Optional[int] = Field(None, description="최소 픽셀 수")
    maxPixels: Optional[int] = Field(None, description="최대 픽셀 수")
    prettifyMarkdown: Optional[bool] = Field(True, description="Markdown 미화")
    showFormulaNumber: Optional[bool] = Field(False, description="수식 번호 표시")
    visualize: Optional[bool] = Field(None, description="시각화 결과 반환")

class MarkdownResult(BaseModel):
    """Markdown 결과"""
    text: str
    images: Dict[str, str]
    isStart: bool
    isEnd: bool

class LayoutParsingResultItem(BaseModel):
    """레이아웃 파싱 결과 항목"""
    prunedResult: Dict[str, Any]
    markdown: MarkdownResult
    outputImages: Optional[Dict[str, str]] = None
    inputImage: Optional[str] = None

class LayoutParsingResponse(BaseModel):
    """레이아웃 파싱 응답 모델 (공식 API 스펙)"""
    logId: str
    errorCode: int
    errorMsg: str
    result: Optional[Dict[str, Any]] = None

# ============================================================================
# PaddleOCR-VL 초기화
# ============================================================================

def init_paddleocr_vl():
    """PaddleOCR-VL 초기화 (한국어 최적화)"""
    global paddleocr_vl, ocr_available
    try:
        logger.info("=" * 80)
        logger.info("🚀 PaddleOCR-VL 초기화 중...")
        logger.info("🎯 한국어 100% 정확도 목표 - 109개 언어 지원")
        logger.info("=" * 80)
        
        # PaddleOCR-VL import
        try:
            from paddleocr import PaddleOCRVL
            logger.info("✅ PaddleOCRVL 모듈 import 성공")
        except ImportError as e:
            logger.error("❌ PaddleOCRVL을 import할 수 없습니다.")
            logger.error(f"오류: {e}")
            logger.error("\n다음 명령어로 설치하세요:")
            logger.error("  python -m pip install paddlepaddle-gpu==3.2.0")
            logger.error("  python -m pip install -U \"paddleocr[doc-parser]\"")
            return False
        
        # PaddleOCR-VL 초기화
        logger.info("\n⚙️  PaddleOCR-VL 파라미터 설정 중...")
        logger.info("   🔹 문서 방향 자동 분류: ON")
        logger.info("   🔹 이미지 왜곡 보정: ON")
        logger.info("   🔹 레이아웃 검출: ON")
        logger.info("   🔹 한국어 최적 해상도: 256×256 ~ 4096×4096")

        device_setting = os.getenv("PADDLE_DEVICE")
        precision_pref = os.getenv("PADDLE_PRECISION")

        if device_setting:
            logger.info(f"   🔹 환경 변수 지정 디바이스: {device_setting}")

        try:
            if not device_setting:
                import paddle  # type: ignore
                if (
                    hasattr(paddle.device, "is_compiled_with_cuda")
                    and paddle.device.is_compiled_with_cuda()
                    and paddle.device.cuda.device_count() > 0
                ):
                    device_setting = "gpu:0"
                else:
                    device_setting = "cpu"
            else:
                device_setting = device_setting.lower()
        except Exception as detection_error:
            logger.warning(f"   ⚠️  디바이스 자동 감지 중 경고: {detection_error}")
            device_setting = "cpu"

        if not device_setting:
            device_setting = "cpu"

        if precision_pref:
            precision_pref = precision_pref.lower()

        precision_mode = precision_pref or ("fp16" if device_setting.startswith("gpu") else "fp32")
        if precision_mode not in {"fp16", "fp32"}:
            logger.warning(f"   ⚠️  지원되지 않는 precision '{precision_mode}' -> 기본값으로 조정")
            precision_mode = "fp16" if device_setting.startswith("gpu") else "fp32"

        enable_mkldnn = device_setting.startswith("cpu")
        enable_hpi = device_setting.startswith("gpu")

        logger.info(f"   🔹 선택된 디바이스: {device_setting}")
        logger.info(f"   🔹 Precision 모드: {precision_mode}")
        logger.info(f"   🔹 MKL-DNN 사용: {enable_mkldnn}")
        logger.info(f"   🔹 HPI 가속: {enable_hpi}")

        def _initialise_pipeline(target_device: str, precision: str, mkldnn: bool, hpi: bool):
            return PaddleOCRVL(
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_layout_detection=True,
                layout_threshold=0.5,
                layout_nms=True,
                layout_unclip_ratio=1.5,
                use_chart_recognition=False,
                format_block_content=True,
                device=target_device,
                enable_hpi=hpi,
                precision=precision,
                enable_mkldnn=mkldnn,
            )

        try:
            paddleocr_vl = _initialise_pipeline(device_setting, precision_mode, enable_mkldnn, enable_hpi)
        except Exception as init_error:
            if device_setting.startswith("gpu"):
                logger.warning("   ⚠️  GPU 모드 초기화 실패, CPU 모드로 재시도합니다.")
                try:
                    paddleocr_vl = _initialise_pipeline("cpu", "fp32", True, False)
                    device_setting = "cpu"
                    precision_mode = "fp32"
                    enable_mkldnn = True
                    enable_hpi = False
                except Exception as cpu_error:
                    logger.error(f"❌ CPU 재시도 실패: {cpu_error}")
                    return False
            else:
                logger.error(f"❌ PaddleOCR-VL 초기화 실패: {init_error}")
                return False

        logger.info(f"   🔹 최종 디바이스 구성: {device_setting} / {precision_mode}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ PaddleOCR-VL 초기화 완료!")
        logger.info("🎯 한국어 고정확도 모드 활성화")
        logger.info("🚀 109개 언어 지원 - 100% 정확도 목표")
        logger.info("=" * 80 + "\n")
        
        ocr_available = True
        return True
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error(f"❌ PaddleOCR-VL 초기화 실패: {e}")
        logger.error(f"상세 오류:\n{traceback.format_exc()}")
        logger.error("=" * 80 + "\n")
        ocr_available = False
        return False

# ============================================================================
# FastAPI 앱 생성
# ============================================================================

app = FastAPI(
    title="PaddleOCR-VL Korean API",
    description="한국어 100% 정확도 OCR 서비스 - PaddleOCR-VL REST API",
    version="6.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 헬퍼 함수
# ============================================================================

def decode_base64_image(base64_str: str) -> bytes:
    """Base64 문자열을 이미지 바이트로 디코딩"""
    try:
        # data:image/... 접두사 제거
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Base64 디코딩 실패: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid Base64 image: {str(e)}")

def process_ocr_result(result_obj, prettify: bool = True, show_formula: bool = False) -> Dict[str, Any]:
    """PaddleOCR-VL 결과를 공식 API 형식으로 변환"""
    try:
        # 결과 객체에서 JSON 추출
        if hasattr(result_obj, 'json'):
            pruned_result = result_obj.json
        else:
            pruned_result = {}
        
        # Markdown 추출
        markdown_text = ""
        markdown_images = {}
        is_start = True
        is_end = True
        
        if hasattr(result_obj, 'markdown') and result_obj.markdown:
            markdown_data = result_obj.markdown
            if isinstance(markdown_data, dict):
                markdown_text = markdown_data.get('markdown_texts', '')
                markdown_images = markdown_data.get('markdown_images', {})
            elif isinstance(markdown_data, str):
                markdown_text = markdown_data
        
        # 시각화 이미지 추출
        output_images = None
        if hasattr(result_obj, 'img') and result_obj.img:
            output_images = {}
            for key, img_array in result_obj.img.items():
                # NumPy 배열을 JPEG로 변환 후 Base64 인코딩
                try:
                    from PIL import Image
                    import io
                    
                    if isinstance(img_array, np.ndarray):
                        img = Image.fromarray(img_array)
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        output_images[key] = img_base64
                except Exception as e:
                    logger.warning(f"이미지 변환 실패 ({key}): {e}")
        
        return {
            "prunedResult": pruned_result,
            "markdown": {
                "text": markdown_text,
                "images": markdown_images,
                "isStart": is_start,
                "isEnd": is_end
            },
            "outputImages": output_images,
            "inputImage": None
        }
        
    except Exception as e:
        logger.error(f"결과 처리 실패: {e}")
        logger.error(f"상세: {traceback.format_exc()}")
        raise

# ============================================================================
# API 엔드포인트
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 PaddleOCR-VL 초기화"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 PaddleOCR-VL FastAPI 서버 시작")
    logger.info("🎯 한국어 100% 정확도 OCR 서비스")
    logger.info("🔧 REST API: POST /layout-parsing")
    logger.info("=" * 80)
    logger.info("")
    
    success = init_paddleocr_vl()
    
    if not success:
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning("⚠️  PaddleOCR-VL 초기화 실패")
        logger.warning("⚠️  OCR 기능이 제한될 수 있습니다")
        logger.warning("=" * 80)
        logger.warning("")
    else:
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ 서버가 정상적으로 시작되었습니다!")
        logger.info(f"📡 엔드포인트: http://0.0.0.0:8000")
        logger.info(f"📚 API 문서: http://0.0.0.0:8000/docs")
        logger.info(f"🔗 레이아웃 파싱: POST /layout-parsing")
        logger.info("=" * 80)
        logger.info("")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "status": "ok",
        "service": "PaddleOCR-VL Korean API",
        "version": "6.0.0",
        "korean_optimized": True,
        "target_accuracy": "100%",
        "ocr_available": ocr_available,
        "supported_languages": 109,
        "endpoints": {
            "health": "GET /health",
            "layout_parsing": "POST /layout-parsing",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy" if ocr_available else "degraded",
        "paddleocr_vl": "initialized" if ocr_available else "not_initialized",
        "service": "PaddleOCR-VL Korean API",
        "ready": ocr_available,
        "version": "6.0.0"
    }

@app.post("/layout-parsing")
async def layout_parsing(request: LayoutParsingRequest):
    """
    레이아웃 파싱 API (공식 스펙)
    
    Args:
        request: LayoutParsingRequest 모델
        
    Returns:
        LayoutParsingResponse: 파싱 결과
    """
    log_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"📥 레이아웃 파싱 요청 (logId: {log_id})")
        logger.info("=" * 80)
        
        # OCR 사용 가능 여부 확인
        if not ocr_available or paddleocr_vl is None:
            logger.error("❌ PaddleOCR-VL이 초기화되지 않았습니다.")
            return JSONResponse(
                status_code=503,
                content={
                    "logId": log_id,
                    "errorCode": 503,
                    "errorMsg": "OCR 서비스가 초기화되지 않았습니다."
                }
            )
        
        # Base64 이미지 디코딩
        logger.info("🔍 이미지 디코딩 중...")
        raw_image_bytes = decode_base64_image(request.file)
        logger.info(f"   원본 이미지 크기: {len(raw_image_bytes) / 1024:.2f} KB")

        processed_bytes = raw_image_bytes
        preprocess_report: Optional[PreprocessingReport] = None
        try:
            processed_bytes, preprocess_report = preprocessor.preprocess(raw_image_bytes)
            logger.info("   🛠️  전처리 단계: %s", " -> ".join(preprocess_report.steps))
            logger.info(
                "   📐 전처리 해상도: %s -> %s (scale x%.2f)",
                preprocess_report.original_shape,
                preprocess_report.processed_shape,
                preprocess_report.scale_factor,
            )
            logger.info(
                "   📊 전처리 지표 - 밝기: %.3f | 대비: %.3f | 선명도: %.2f | 노이즈지수: %.3f",
                preprocess_report.brightness,
                preprocess_report.contrast,
                preprocess_report.sharpness,
                preprocess_report.noise_index,
            )
        except Exception as preprocess_error:
            logger.warning(f"   ⚠️  전처리 실패, 원본 이미지를 사용합니다: {preprocess_error}")
            preprocess_report = None
            processed_bytes = raw_image_bytes

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', mode='wb') as tmp_file:
            tmp_file.write(processed_bytes)
            tmp_path = tmp_file.name

        logger.info(f"   💾 임시 파일: {tmp_path}")
        
        try:
            # PaddleOCR-VL 실행
            logger.info("🔍 PaddleOCR-VL 텍스트 추출 중...")
            
            # 파라미터 준비 (None이 아닌 값만 전달)
            predict_params = {
                "input": tmp_path,
            }
            
            # 선택적 파라미터 추가
            if request.useDocUnwarping is not None:
                predict_params["use_doc_unwarping"] = request.useDocUnwarping
            if request.useLayoutDetection is not None:
                predict_params["use_layout_detection"] = request.useLayoutDetection
            if request.useChartRecognition is not None:
                predict_params["use_chart_recognition"] = request.useChartRecognition
            if request.layoutThreshold is not None:
                predict_params["layout_threshold"] = request.layoutThreshold
            if request.layoutNms is not None:
                predict_params["layout_nms"] = request.layoutNms
            if request.layoutUnclipRatio is not None:
                predict_params["layout_unclip_ratio"] = request.layoutUnclipRatio
            if request.layoutMergeBboxesMode is not None:
                predict_params["layout_merge_bboxes_mode"] = request.layoutMergeBboxesMode
            if request.promptLabel is not None:
                predict_params["prompt_label"] = request.promptLabel
            if request.formatBlockContent is not None:
                predict_params["format_block_content"] = request.formatBlockContent
            if request.repetitionPenalty is not None:
                predict_params["repetition_penalty"] = request.repetitionPenalty
            if request.temperature is not None:
                predict_params["temperature"] = request.temperature
            if request.topP is not None:
                predict_params["top_p"] = request.topP
            if request.minPixels is not None:
                predict_params["min_pixels"] = request.minPixels
            if request.maxPixels is not None:
                predict_params["max_pixels"] = request.maxPixels
            
            # 한국어 최적화 기본값 (요청에 없으면 설정)
            if "min_pixels" not in predict_params:
                predict_params["min_pixels"] = 256 * 256
            if "max_pixels" not in predict_params:
                predict_params["max_pixels"] = 4096 * 4096
            if "repetition_penalty" not in predict_params:
                predict_params["repetition_penalty"] = 1.1
            if "temperature" not in predict_params:
                predict_params["temperature"] = 0.3
            if "top_p" not in predict_params:
                predict_params["top_p"] = 0.9
            
            logger.info(f"   파라미터: {list(predict_params.keys())}")
            
            # predict 호출
            def convert_ocr_outputs(ocr_output, run_tag: str) -> Tuple[List[Dict[str, Any]], int]:
                items: List[Dict[str, Any]] = []
                total_chars = 0

                if not ocr_output:
                    return items, total_chars

                output_list = ocr_output if isinstance(ocr_output, list) else [ocr_output]

                for idx, result_obj in enumerate(output_list):
                    logger.info('   [%s] 처리 중: 결과 %d/%d', run_tag, idx + 1, len(output_list))

                    result_item = process_ocr_result(
                        result_obj,
                        prettify=request.prettifyMarkdown,
                        show_formula=request.showFormulaNumber
                    )

                    items.append(result_item)

                    markdown_data = result_item.get('markdown', {}) if isinstance(result_item, dict) else {}
                    markdown_text = ''
                    if isinstance(markdown_data, dict):
                        markdown_text = markdown_data.get('text') or ''
                    char_count = len(markdown_text)
                    total_chars += char_count

                    if char_count:
                        preview = markdown_text.replace('\n', ' ')
                        if len(preview) > 160:
                            preview = preview[:160] + '…'
                        logger.info('      ✅ 텍스트 추출 성공: %d 글자', char_count)
                        logger.info('      📝 미리보기: %s', preview)
                    else:
                        logger.warning('      ⚠️  텍스트 추출 실패')

                return items, total_chars

            ocr_results = paddleocr_vl.predict(**predict_params)

            result_count_logged = len(ocr_results) if isinstance(ocr_results, list) else 1
            logger.info('   📊 1차 OCR 결과 개수: %d', result_count_logged)

            layout_parsing_results, total_characters = convert_ocr_outputs(ocr_results, 'primary')

            if not layout_parsing_results:
                logger.error('❌ OCR 결과가 비어있습니다.')
                return JSONResponse(
                    status_code=500,
                    content={
                        'logId': log_id,
                        'errorCode': 500,
                        'errorMsg': 'OCR 처리 결과가 없습니다.'
                    }
                )

            fallback_executed = False
            fallback_improved = False
            primary_characters = total_characters

            fallback_threshold_env = os.getenv('KOREAN_OCR_FALLBACK_THRESHOLD')
            try:
                fallback_threshold = int(fallback_threshold_env) if fallback_threshold_env else 18
            except ValueError:
                logger.warning('   ⚠️  KOREAN_OCR_FALLBACK_THRESHOLD 환경 변수는 정수가 아니므로 기본값 18을 사용합니다.')
                fallback_threshold = 18

            if total_characters < fallback_threshold:
                fallback_executed = True
                logger.info('   🔁 한국어 초정밀 재추론 수행 (fallback)')
                fallback_params = dict(predict_params)
                fallback_params['layout_threshold'] = 0.35
                fallback_params['layout_unclip_ratio'] = 2.0
                fallback_params['temperature'] = 0.2
                fallback_params['top_p'] = 0.85
                fallback_params['repetition_penalty'] = 1.05

                if preprocess_report:
                    target_pixels = preprocess_report.processed_shape[0] * preprocess_report.processed_shape[1]
                    computed_min = int(target_pixels * 0.5)
                    computed_max = int(target_pixels * 1.1)

                    existing_min = fallback_params.get('min_pixels')
                    existing_max = fallback_params.get('max_pixels')

                    if isinstance(existing_min, (int, float)):
                        computed_min = max(computed_min, int(existing_min))
                    if isinstance(existing_max, (int, float)):
                        computed_max = max(computed_max, int(existing_max))

                    fallback_params['min_pixels'] = computed_min
                    fallback_params['max_pixels'] = computed_max

                try:
                    fallback_results = paddleocr_vl.predict(**fallback_params)
                    fallback_items, fallback_characters = convert_ocr_outputs(fallback_results, 'korean_fallback')

                    if fallback_items and fallback_characters > total_characters:
                        layout_parsing_results = fallback_items
                        ocr_results = fallback_results
                        total_characters = fallback_characters
                        fallback_improved = True
                        logger.info(
                            '   ✅ 재추론 결과 채택: 글자수 %d → %d',
                            primary_characters,
                            fallback_characters,
                        )
                    else:
                        logger.info('   ℹ️  재추론 결과가 개선되지 않아 원본 결과를 유지합니다.')
                except Exception as retry_error:
                    logger.warning('   ⚠️  재추론 중 오류가 발생했습니다: %s', retry_error)
            
            # 성공 응답
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info('')
            logger.info('=' * 80)
            logger.info('✅ 레이아웃 파싱 완료 (logId: %s)', log_id)
            logger.info('⏱️  처리 시간: %.2f초', elapsed_time)
            logger.info('📊 결과 개수: %d', len(layout_parsing_results))
            logger.info('📝 총 추출 글자 수: %d', total_characters)
            logger.info('=' * 80)
            logger.info('')

            data_info: Dict[str, Any] = {
                'processingTime': elapsed_time,
                'imageSize': len(raw_image_bytes),
                'processedImageSize': len(processed_bytes),
                'resultCount': len(layout_parsing_results),
                'totalCharacters': total_characters,
            }
            if preprocess_report:
                data_info['preprocessing'] = preprocess_report.as_dict()
            data_info['fallback'] = {
                'executed': fallback_executed,
                'improved': fallback_improved,
                'threshold': fallback_threshold,
                'primaryCharacters': primary_characters,
                'finalCharacters': total_characters,
            }

            return {
                'logId': log_id,
                'errorCode': 0,
                'errorMsg': 'Success',
                'result': {
                    'layoutParsingResults': layout_parsing_results,
                    'dataInfo': data_info,
                }
            }
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    logger.debug(f"   🗑️  임시 파일 삭제: {tmp_path}")
                except Exception as e:
                    logger.warning(f"   ⚠️  임시 파일 삭제 실패: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"❌ 예상치 못한 오류 (logId: {log_id})")
        logger.error(f"오류: {e}")
        logger.error(f"상세:\n{traceback.format_exc()}")
        logger.error("=" * 80)
        logger.error("")
        
        return JSONResponse(
            status_code=500,
            content={
                "logId": log_id,
                "errorCode": 500,
                "errorMsg": f"Internal server error: {str(e)}"
            }
        )

# ============================================================================
# OCR API (기존 호환성)
# ============================================================================

@app.post("/ocr")
async def extract_text_legacy(files: List[UploadFile]):
    """
    기존 OCR API (하위 호환성)
    
    Args:
        files: 업로드된 파일 리스트
        
    Returns:
        dict: OCR 결과
    """
    log_id = str(uuid.uuid4())
    
    try:
        logger.info(f"📥 OCR 요청 (레거시 API, logId: {log_id}): {len(files)}개 파일")
        
        if not ocr_available or paddleocr_vl is None:
            raise HTTPException(
                status_code=503,
                detail="OCR 서비스가 초기화되지 않았습니다."
            )
        
        results = []
        
        for idx, file in enumerate(files):
            logger.info(f"   파일 {idx + 1}/{len(files)}: {file.filename}")
            
            # 파일 읽기
            content = await file.read()
            
            # Base64 인코딩
            base64_image = base64.b64encode(content).decode('utf-8')
            
            # LayoutParsingRequest 생성
            request = LayoutParsingRequest(file=base64_image, fileType=1)
            
            # layout_parsing 호출
            response = await layout_parsing(request)
            
            # 텍스트 추출
            if response.get("errorCode") == 0:
                result_data = response.get("result", {})
                parsing_results = result_data.get("layoutParsingResults", [])
                
                if parsing_results:
                    markdown_text = parsing_results[0]["markdown"]["text"]
                    results.append(markdown_text)
                else:
                    results.append("")
            else:
                results.append("")
        
        return {
            "texts": results,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"❌ OCR 오류 (logId: {log_id}): {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 서버 실행
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("OCR_PORT", "8000"))
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🚀 PaddleOCR-VL 서버 시작 중... (포트: {port})")
    logger.info(f"🎯 한국어 100% 정확도 OCR 모드")
    logger.info(f"🔧 REST API: POST /layout-parsing")
    logger.info("=" * 80)
    logger.info("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
