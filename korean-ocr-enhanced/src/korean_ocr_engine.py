"""
한국어 OCR 인식률 100% 달성을 위한 고성능 PaddleOCR 엔진
생기부 문서 특화 최적화 버전
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import re
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """OCR 설정 클래스"""
    use_angle_cls: bool = True
    lang: str = 'korean'
    use_gpu: bool = torch.cuda.is_available()
    det_model_dir: str = './models/det'
    rec_model_dir: str = './models/rec'
    cls_model_dir: str = './models/cls'
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    det_db_unclip_ratio: float = 1.8
    max_batch_size: int = 10
    use_mp: bool = True
    total_process_num: int = mp.cpu_count()
    enable_mkldnn: bool = True
    cpu_threads: int = 10
    rec_batch_num: int = 6
    drop_score: float = 0.3
    use_space_char: bool = True
    precision: str = 'fp32'
    
    # 한국어 특화 설정
    korean_char_whitelist: str = None  # 한글, 숫자, 특수문자 화이트리스트
    min_text_size: int = 10
    max_text_size: int = 200
    
    # 생기부 특화 설정
    document_type: str = 'school_record'
    enable_layout_analysis: bool = True
    enable_table_detection: bool = True
    

class KoreanTextPostProcessor:
    """한국어 텍스트 후처리 프로세서"""
    
    def __init__(self):
        self.korean_pattern = re.compile(r'[가-힣]+')
        self.number_pattern = re.compile(r'[0-9]+')
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        
        # 한국어 자주 틀리는 패턴 교정 사전
        self.correction_dict = {
            '습니다': ['숩니다', '슴니다', '습니디', '승니다'],
            '있습니다': ['있숩니다', '있슴니다', '있습니디'],
            '합니다': ['함니다', '합니디', '한니다'],
            '학생': ['학샘', '학쌩', '학셍'],
            '선생님': ['선샘님', '선생닝', '선셍님'],
            '성실': ['성싣', '성실', '섬실'],
            '우수': ['우쑤', '우수', '우슈'],
            '참여': ['참어', '참며', '찬여'],
            '활동': ['활똥', '활동', '활둥'],
            '교과': ['교가', '굥과', '교콰'],
        }
        
        # 한글 자모 분리 및 재조합을 위한 설정
        self.CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    def correct_common_errors(self, text: str) -> str:
        """일반적인 OCR 오류 교정"""
        corrected = text
        
        # 교정 사전 기반 수정
        for correct, errors in self.correction_dict.items():
            for error in errors:
                corrected = corrected.replace(error, correct)
        
        # 특수 패턴 교정
        corrected = re.sub(r'(\d+)\s*학\s*년', r'\1학년', corrected)
        corrected = re.sub(r'(\d+)\s*반', r'\1반', corrected)
        corrected = re.sub(r'(\d+)\s*번', r'\1번', corrected)
        
        # 띄어쓰기 교정
        corrected = self._correct_spacing(corrected)
        
        return corrected
    
    def _correct_spacing(self, text: str) -> str:
        """띄어쓰기 교정"""
        # 기본 띄어쓰기 규칙 적용
        text = re.sub(r'([가-힣])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([가-힣])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def decompose_hangul(self, char: str) -> Tuple[str, str, str]:
        """한글 자모 분리"""
        if not self.korean_pattern.match(char):
            return None, None, None
            
        code = ord(char) - 0xAC00
        jong = code % 28
        jung = ((code - jong) // 28) % 21
        cho = ((code - jong) // 28) // 21
        
        return (self.CHOSUNG_LIST[cho], 
                self.JUNGSUNG_LIST[jung], 
                self.JONGSUNG_LIST[jong])
    
    def compose_hangul(self, cho: str, jung: str, jong: str = '') -> str:
        """한글 자모 조합"""
        try:
            cho_idx = self.CHOSUNG_LIST.index(cho)
            jung_idx = self.JUNGSUNG_LIST.index(jung)
            jong_idx = self.JONGSUNG_LIST.index(jong) if jong else 0
            
            code = 0xAC00 + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx
            return chr(code)
        except:
            return ''


class AdvancedImagePreprocessor:
    """고급 이미지 전처리 클래스"""
    
    def __init__(self):
        self.target_dpi = 300
        self.min_text_height = 20
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 품질 향상"""
        # 1. 초해상도 업스케일링 (ESRGAN 방식)
        enhanced = self._super_resolution(image)
        
        # 2. 노이즈 제거
        denoised = self._denoise(enhanced)
        
        # 3. 선명도 향상
        sharpened = self._sharpen(denoised)
        
        # 4. 대비 향상
        contrasted = self._enhance_contrast(sharpened)
        
        # 5. 이진화 최적화
        binarized = self._adaptive_binarization(contrasted)
        
        # 6. 기울기 보정
        deskewed = self._deskew(binarized)
        
        # 7. 모폴로지 연산
        morphed = self._morphological_operations(deskewed)
        
        return morphed
    
    def _super_resolution(self, image: np.ndarray, scale: int = 2) -> np.ndarray:
        """초해상도 업스케일링"""
        height, width = image.shape[:2]
        new_height, new_width = height * scale, width * scale
        
        # Lanczos 보간법 사용
        upscaled = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Edge-preserving filter 적용
        upscaled = cv2.edgePreservingFilter(upscaled, flags=2, sigma_s=50, sigma_r=0.4)
        
        return upscaled
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        # Non-local means denoising
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        # Bilateral filter for edge preservation
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        return denoised
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """선명도 향상"""
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Custom sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(sharpened, -1, kernel)
        
        return sharpened
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """대비 향상"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def _adaptive_binarization(self, image: np.ndarray) -> np.ndarray:
        """적응적 이진화"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Otsu's thresholding with Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive thresholding for fine details
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Combine both methods
        combined = cv2.bitwise_and(binary, adaptive)
        
        return combined
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """기울기 보정"""
        # Hough 변환을 이용한 기울기 검출
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    rows, cols = image.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                    deskewed = cv2.warpAffine(image, M, (cols, rows), 
                                            flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REPLICATE)
                    return deskewed
        
        return image
    
    def _morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """모폴로지 연산"""
        # 텍스트 영역 강화를 위한 모폴로지 연산
        kernel = np.ones((2, 2), np.uint8)
        
        # Closing to connect broken characters
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Opening to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened


class UltraKoreanOCR:
    """궁극의 한국어 OCR 엔진"""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.preprocessor = AdvancedImagePreprocessor()
        self.postprocessor = KoreanTextPostProcessor()
        
        # PaddleOCR 초기화 (한국어 특화 설정)
        self.ocr_engine = PaddleOCR(
            use_angle_cls=self.config.use_angle_cls,
            lang=self.config.lang,
            use_gpu=self.config.use_gpu,
            det_db_thresh=self.config.det_db_thresh,
            det_db_box_thresh=self.config.det_db_box_thresh,
            det_db_unclip_ratio=self.config.det_db_unclip_ratio,
            max_batch_size=self.config.max_batch_size,
            use_mp=self.config.use_mp,
            total_process_num=self.config.total_process_num,
            enable_mkldnn=self.config.enable_mkldnn,
            cpu_threads=self.config.cpu_threads,
            rec_batch_num=self.config.rec_batch_num,
            drop_score=self.config.drop_score,
            use_space_char=self.config.use_space_char,
            precision=self.config.precision,
            show_log=False
        )
        
        # 결과 캐싱을 위한 딕셔너리
        self.cache = {}
        
        # 멀티프로세싱 풀 초기화
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count() * 2)
        
    def process_image(self, image_path: str, 
                     enable_enhancement: bool = True,
                     enable_multi_pass: bool = True) -> Dict[str, Any]:
        """
        이미지에서 텍스트 추출 (100% 정확도 목표)
        
        Args:
            image_path: 이미지 파일 경로
            enable_enhancement: 이미지 향상 활성화 여부
            enable_multi_pass: 다중 패스 OCR 활성화 여부
            
        Returns:
            OCR 결과 딕셔너리
        """
        # 캐시 확인
        image_hash = self._get_image_hash(image_path)
        if image_hash in self.cache:
            logger.info(f"Using cached result for {image_path}")
            return self.cache[image_hash]
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # 이미지 전처리
        if enable_enhancement:
            logger.info("Applying advanced image enhancement...")
            enhanced_image = self.preprocessor.enhance_image(image)
        else:
            enhanced_image = image
        
        # OCR 실행
        results = []
        
        if enable_multi_pass:
            # 다중 패스 OCR (여러 설정으로 여러 번 실행)
            results = self._multi_pass_ocr(enhanced_image)
        else:
            # 단일 패스 OCR
            results = self._single_pass_ocr(enhanced_image)
        
        # 결과 통합 및 후처리
        final_result = self._merge_and_postprocess_results(results)
        
        # 캐싱
        self.cache[image_hash] = final_result
        
        return final_result
    
    def _get_image_hash(self, image_path: str) -> str:
        """이미지 해시 생성"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _single_pass_ocr(self, image: np.ndarray) -> List[Dict]:
        """단일 패스 OCR"""
        result = self.ocr_engine.ocr(image, cls=True)
        return [self._parse_ocr_result(result)]
    
    def _multi_pass_ocr(self, image: np.ndarray) -> List[Dict]:
        """다중 패스 OCR - 여러 설정으로 실행하여 최적 결과 도출"""
        results = []
        
        # Pass 1: 기본 설정
        result1 = self.ocr_engine.ocr(image, cls=True)
        results.append(self._parse_ocr_result(result1))
        
        # Pass 2: 회전된 이미지 (90도)
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        result2 = self.ocr_engine.ocr(rotated_90, cls=True)
        results.append(self._parse_ocr_result(result2))
        
        # Pass 3: 대비 강화된 이미지
        enhanced = self.preprocessor._enhance_contrast(image)
        result3 = self.ocr_engine.ocr(enhanced, cls=True)
        results.append(self._parse_ocr_result(result3))
        
        # Pass 4: 이진화된 이미지
        binary = self.preprocessor._adaptive_binarization(image)
        result4 = self.ocr_engine.ocr(binary, cls=True)
        results.append(self._parse_ocr_result(result4))
        
        return results
    
    def _parse_ocr_result(self, result: List) -> Dict:
        """OCR 결과 파싱"""
        if not result or not result[0]:
            return {"texts": [], "boxes": [], "scores": []}
        
        texts = []
        boxes = []
        scores = []
        
        for line in result[0]:
            if line:
                box = line[0]
                text = line[1][0] if len(line) > 1 and line[1] else ""
                score = line[1][1] if len(line) > 1 and len(line[1]) > 1 else 0.0
                
                texts.append(text)
                boxes.append(box)
                scores.append(score)
        
        return {
            "texts": texts,
            "boxes": boxes,
            "scores": scores
        }
    
    def _merge_and_postprocess_results(self, results: List[Dict]) -> Dict[str, Any]:
        """결과 통합 및 후처리"""
        # 모든 결과에서 최고 신뢰도 텍스트 선택
        merged_texts = []
        merged_boxes = []
        merged_scores = []
        
        # 각 위치별로 최적 텍스트 선택
        all_texts = []
        for result in results:
            all_texts.extend(result.get("texts", []))
        
        # 중복 제거 및 신뢰도 기반 선택
        unique_texts = list(set(all_texts))
        
        for text in unique_texts:
            # 각 텍스트에 대한 최고 신뢰도 찾기
            best_score = 0
            best_box = None
            
            for result in results:
                texts = result.get("texts", [])
                scores = result.get("scores", [])
                boxes = result.get("boxes", [])
                
                for i, t in enumerate(texts):
                    if t == text and scores[i] > best_score:
                        best_score = scores[i]
                        best_box = boxes[i]
            
            if best_score > self.config.drop_score:
                # 후처리 적용
                corrected_text = self.postprocessor.correct_common_errors(text)
                merged_texts.append(corrected_text)
                merged_boxes.append(best_box)
                merged_scores.append(best_score)
        
        # 박스 위치 기준으로 정렬 (위에서 아래, 왼쪽에서 오른쪽)
        sorted_indices = self._sort_by_position(merged_boxes)
        
        final_result = {
            "texts": [merged_texts[i] for i in sorted_indices],
            "boxes": [merged_boxes[i] for i in sorted_indices],
            "scores": [merged_scores[i] for i in sorted_indices],
            "full_text": " ".join([merged_texts[i] for i in sorted_indices]),
            "average_confidence": np.mean(merged_scores) if merged_scores else 0,
            "total_characters": sum(len(text) for text in merged_texts),
            "korean_character_ratio": self._calculate_korean_ratio(merged_texts)
        }
        
        return final_result
    
    def _sort_by_position(self, boxes: List) -> List[int]:
        """박스 위치 기준 정렬"""
        if not boxes:
            return []
        
        # 각 박스의 중심점 계산
        centers = []
        for i, box in enumerate(boxes):
            if box and len(box) >= 4:
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                centers.append((i, center_x, center_y))
        
        # Y 좌표로 먼저 정렬, 그 다음 X 좌표로 정렬
        centers.sort(key=lambda x: (x[2], x[1]))
        
        return [c[0] for c in centers]
    
    def _calculate_korean_ratio(self, texts: List[str]) -> float:
        """한국어 문자 비율 계산"""
        total_chars = 0
        korean_chars = 0
        
        for text in texts:
            total_chars += len(text)
            korean_chars += len(re.findall(r'[가-힣]', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """배치 처리"""
        results = []
        
        # 멀티프로세싱으로 병렬 처리
        with self.process_pool as pool:
            futures = [pool.submit(self.process_image, path) for path in image_paths]
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    results.append({"error": str(e)})
        
        return results
    
    def extract_school_record_fields(self, ocr_result: Dict[str, Any]) -> Dict[str, str]:
        """생기부 특정 필드 추출"""
        text = ocr_result.get("full_text", "")
        
        fields = {
            "학년": self._extract_grade(text),
            "반": self._extract_class(text),
            "번호": self._extract_number(text),
            "이름": self._extract_name(text),
            "교과활동": self._extract_subject_activities(text),
            "행동특성": self._extract_behavior_characteristics(text),
            "진로희망": self._extract_career_aspiration(text),
            "창의적체험활동": self._extract_creative_activities(text),
            "독서활동": self._extract_reading_activities(text),
            "봉사활동": self._extract_volunteer_activities(text),
            "수상경력": self._extract_awards(text),
            "자격증": self._extract_certifications(text),
        }
        
        return fields
    
    def _extract_grade(self, text: str) -> str:
        """학년 추출"""
        pattern = r'(\d+)\s*학년'
        match = re.search(pattern, text)
        return match.group(1) if match else ""
    
    def _extract_class(self, text: str) -> str:
        """반 추출"""
        pattern = r'(\d+)\s*반'
        match = re.search(pattern, text)
        return match.group(1) if match else ""
    
    def _extract_number(self, text: str) -> str:
        """번호 추출"""
        pattern = r'(\d+)\s*번'
        match = re.search(pattern, text)
        return match.group(1) if match else ""
    
    def _extract_name(self, text: str) -> str:
        """이름 추출"""
        pattern = r'성명\s*[:：]?\s*([가-힣]+)|이름\s*[:：]?\s*([가-힣]+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1) or match.group(2)
        return ""
    
    def _extract_subject_activities(self, text: str) -> str:
        """교과활동 추출"""
        pattern = r'교과\s*활동.*?(?=행동|진로|창의|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""
    
    def _extract_behavior_characteristics(self, text: str) -> str:
        """행동특성 추출"""
        pattern = r'행동\s*특성.*?(?=진로|창의|독서|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""
    
    def _extract_career_aspiration(self, text: str) -> str:
        """진로희망 추출"""
        pattern = r'진로\s*희망.*?(?=창의|독서|봉사|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""
    
    def _extract_creative_activities(self, text: str) -> str:
        """창의적체험활동 추출"""
        pattern = r'창의적\s*체험\s*활동.*?(?=독서|봉사|수상|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""
    
    def _extract_reading_activities(self, text: str) -> str:
        """독서활동 추출"""
        pattern = r'독서\s*활동.*?(?=봉사|수상|자격|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""
    
    def _extract_volunteer_activities(self, text: str) -> str:
        """봉사활동 추출"""
        pattern = r'봉사\s*활동.*?(?=수상|자격|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""
    
    def _extract_awards(self, text: str) -> str:
        """수상경력 추출"""
        pattern = r'수상\s*경력.*?(?=자격|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""
    
    def _extract_certifications(self, text: str) -> str:
        """자격증 추출"""
        pattern = r'자격증.*?$'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False)
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    # 테스트 코드
    config = OCRConfig()
    ocr = UltraKoreanOCR(config)
    
    # 테스트 이미지 처리
    test_image = "./data/test_school_record.jpg"
    result = ocr.process_image(test_image)
    
    print("OCR Result:")
    print(f"Full Text: {result['full_text']}")
    print(f"Average Confidence: {result['average_confidence']:.2%}")
    print(f"Korean Character Ratio: {result['korean_character_ratio']:.2%}")
    
    # 생기부 필드 추출
    fields = ocr.extract_school_record_fields(result)
    print("\nExtracted Fields:")
    for field, value in fields.items():
        print(f"{field}: {value}")