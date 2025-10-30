"""
생기부 문서 레이아웃 분석 시스템
테이블, 섹션, 필드 자동 인식 및 구조화
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import ndimage
from sklearn.cluster import DBSCAN
import json

logger = logging.getLogger(__name__)


class LayoutElementType(Enum):
    """레이아웃 요소 타입"""
    TITLE = "title"
    SUBTITLE = "subtitle"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FIELD_LABEL = "field_label"
    FIELD_VALUE = "field_value"
    SIGNATURE = "signature"
    STAMP = "stamp"
    IMAGE = "image"
    CHECKBOX = "checkbox"
    FORM_FIELD = "form_field"


@dataclass
class LayoutElement:
    """레이아웃 요소"""
    type: LayoutElementType
    bbox: List[int]  # [x1, y1, x2, y2]
    text: str = ""
    confidence: float = 0.0
    children: List['LayoutElement'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            "type": self.type.value,
            "bbox": self.bbox,
            "text": self.text,
            "confidence": self.confidence,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }


class SchoolRecordLayoutAnalyzer:
    """학교생활기록부 레이아웃 분석기"""
    
    def __init__(self):
        self.min_line_length = 100
        self.min_table_cells = 4
        self.field_patterns = self._init_field_patterns()
        
    def _init_field_patterns(self) -> Dict[str, List[str]]:
        """생기부 필드 패턴 초기화"""
        return {
            "personal_info": [
                "성명", "이름", "학번", "생년월일", "주민등록번호",
                "성별", "주소", "연락처", "전화번호", "보호자"
            ],
            "academic_info": [
                "학년", "반", "번호", "담임", "학기", "학년도"
            ],
            "grades": [
                "교과", "과목", "단위수", "원점수", "평균", "표준편차",
                "석차", "등급", "성취도", "이수단위", "학점"
            ],
            "activities": [
                "창의적체험활동", "자율활동", "동아리활동", "봉사활동",
                "진로활동", "특별활동", "행사활동"
            ],
            "behavior": [
                "행동특성", "종합의견", "행동발달", "인성", "생활태도"
            ],
            "attendance": [
                "출결", "출석", "결석", "지각", "조퇴", "결과"
            ],
            "awards": [
                "수상", "표창", "상장", "수상명", "수상일자", "수여기관"
            ],
            "certificates": [
                "자격증", "인증", "취득일", "발급기관", "자격번호"
            ],
            "career": [
                "진로", "희망", "적성", "흥미", "특기", "장래희망"
            ],
            "reading": [
                "독서", "도서명", "저자", "독서활동", "독후감"
            ]
        }
    
    def analyze_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """
        문서 레이아웃 분석
        
        Args:
            image: 입력 이미지
            
        Returns:
            레이아웃 분석 결과
        """
        # 전처리
        preprocessed = self._preprocess_image(image)
        
        # 텍스트 영역 검출
        text_regions = self._detect_text_regions(preprocessed)
        
        # 라인 검출
        horizontal_lines, vertical_lines = self._detect_lines(preprocessed)
        
        # 테이블 검출
        tables = self._detect_tables(horizontal_lines, vertical_lines)
        
        # 레이아웃 요소 분류
        layout_elements = self._classify_layout_elements(
            text_regions, tables, image
        )
        
        # 계층 구조 생성
        hierarchy = self._build_hierarchy(layout_elements)
        
        # 필드 매핑
        mapped_fields = self._map_fields(hierarchy)
        
        return {
            "layout_elements": [elem.to_dict() for elem in layout_elements],
            "hierarchy": hierarchy,
            "mapped_fields": mapped_fields,
            "tables": [table.to_dict() for table in tables],
            "statistics": self._calculate_statistics(layout_elements)
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, 
                                   np.ones((2, 2), np.uint8))
        
        return denoised
    
    def _detect_text_regions(self, image: np.ndarray) -> List[LayoutElement]:
        """텍스트 영역 검출"""
        regions = []
        
        # MSER(Maximally Stable Extremal Regions) 검출기
        mser = cv2.MSER_create()
        regions_coords, _ = mser.detectRegions(image)
        
        # 바운딩 박스 생성
        for coords in regions_coords:
            x, y, w, h = cv2.boundingRect(coords)
            
            # 너무 작거나 큰 영역 제외
            if w < 10 or h < 10 or w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
                continue
            
            element = LayoutElement(
                type=LayoutElementType.PARAGRAPH,
                bbox=[x, y, x + w, y + h]
            )
            regions.append(element)
        
        # 중복 제거 및 병합
        regions = self._merge_overlapping_regions(regions)
        
        return regions
    
    def _detect_lines(self, image: np.ndarray) -> Tuple[List, List]:
        """수평선과 수직선 검출"""
        # 수평선 검출
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 수직선 검출
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Hough 변환으로 라인 검출
        h_lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, 100, 
                                 minLineLength=self.min_line_length, maxLineGap=20)
        v_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 100, 
                                 minLineLength=self.min_line_length, maxLineGap=20)
        
        h_lines = [] if h_lines is None else h_lines.tolist()
        v_lines = [] if v_lines is None else v_lines.tolist()
        
        return h_lines, v_lines
    
    def _detect_tables(self, h_lines: List, v_lines: List) -> List[LayoutElement]:
        """테이블 검출"""
        tables = []
        
        if not h_lines or not v_lines:
            return tables
        
        # 교차점 찾기
        intersections = []
        for h_line in h_lines:
            x1, y1, x2, y2 = h_line[0] if isinstance(h_line[0], list) else h_line
            for v_line in v_lines:
                x3, y3, x4, y4 = v_line[0] if isinstance(v_line[0], list) else v_line
                
                # 교차 여부 확인
                if self._lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                    # 교차점 계산
                    intersection = self._get_intersection_point(
                        x1, y1, x2, y2, x3, y3, x4, y4
                    )
                    if intersection:
                        intersections.append(intersection)
        
        # 교차점 클러스터링으로 테이블 영역 찾기
        if intersections:
            intersections_array = np.array(intersections)
            clustering = DBSCAN(eps=50, min_samples=4).fit(intersections_array)
            
            # 각 클러스터에 대해 테이블 생성
            for label in set(clustering.labels_):
                if label == -1:  # 노이즈 제외
                    continue
                
                cluster_points = intersections_array[clustering.labels_ == label]
                
                # 바운딩 박스 계산
                x_min, y_min = cluster_points.min(axis=0).astype(int)
                x_max, y_max = cluster_points.max(axis=0).astype(int)
                
                table = LayoutElement(
                    type=LayoutElementType.TABLE,
                    bbox=[x_min, y_min, x_max, y_max],
                    metadata={"cell_count": len(cluster_points)}
                )
                
                # 테이블 셀 검출
                cells = self._detect_table_cells(
                    x_min, y_min, x_max, y_max, h_lines, v_lines
                )
                table.children = cells
                
                tables.append(table)
        
        return tables
    
    def _detect_table_cells(self, x_min: int, y_min: int, x_max: int, y_max: int,
                           h_lines: List, v_lines: List) -> List[LayoutElement]:
        """테이블 셀 검출"""
        cells = []
        
        # 테이블 영역 내의 수평선과 수직선 필터링
        table_h_lines = []
        table_v_lines = []
        
        for h_line in h_lines:
            x1, y1, x2, y2 = h_line[0] if isinstance(h_line[0], list) else h_line
            if y_min <= y1 <= y_max and y_min <= y2 <= y_max:
                table_h_lines.append((min(x1, x2), y1, max(x1, x2), y2))
        
        for v_line in v_lines:
            x1, y1, x2, y2 = v_line[0] if isinstance(v_line[0], list) else v_line
            if x_min <= x1 <= x_max and x_min <= x2 <= x_max:
                table_v_lines.append((x1, min(y1, y2), x2, max(y1, y2)))
        
        # 정렬
        table_h_lines.sort(key=lambda x: x[1])
        table_v_lines.sort(key=lambda x: x[0])
        
        # 셀 생성
        for i in range(len(table_h_lines) - 1):
            for j in range(len(table_v_lines) - 1):
                cell_x1 = table_v_lines[j][0]
                cell_y1 = table_h_lines[i][1]
                cell_x2 = table_v_lines[j + 1][0]
                cell_y2 = table_h_lines[i + 1][1]
                
                cell = LayoutElement(
                    type=LayoutElementType.TABLE_CELL,
                    bbox=[cell_x1, cell_y1, cell_x2, cell_y2],
                    metadata={"row": i, "col": j}
                )
                cells.append(cell)
        
        return cells
    
    def _lines_intersect(self, x1: float, y1: float, x2: float, y2: float,
                        x3: float, y3: float, x4: float, y4: float) -> bool:
        """두 선분이 교차하는지 확인"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A = (x1, y1)
        B = (x2, y2)
        C = (x3, y3)
        D = (x4, y4)
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def _get_intersection_point(self, x1: float, y1: float, x2: float, y2: float,
                               x3: float, y3: float, x4: float, y4: float) -> Optional[Tuple[float, float]]:
        """두 선분의 교차점 계산"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 0.001:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        if 0 <= t <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def _merge_overlapping_regions(self, regions: List[LayoutElement]) -> List[LayoutElement]:
        """중복 영역 병합"""
        if not regions:
            return regions
        
        merged = []
        used = [False] * len(regions)
        
        for i, region1 in enumerate(regions):
            if used[i]:
                continue
            
            # 현재 영역과 겹치는 모든 영역 찾기
            overlapping = [region1]
            used[i] = True
            
            for j, region2 in enumerate(regions[i+1:], i+1):
                if used[j]:
                    continue
                
                if self._regions_overlap(region1.bbox, region2.bbox):
                    overlapping.append(region2)
                    used[j] = True
            
            # 병합
            if len(overlapping) > 1:
                merged_bbox = self._merge_bboxes([r.bbox for r in overlapping])
                merged_element = LayoutElement(
                    type=region1.type,
                    bbox=merged_bbox
                )
                merged.append(merged_element)
            else:
                merged.append(region1)
        
        return merged
    
    def _regions_overlap(self, bbox1: List[int], bbox2: List[int]) -> bool:
        """두 영역이 겹치는지 확인"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return not (x1_max < x2_min or x2_max < x1_min or 
                   y1_max < y2_min or y2_max < y1_min)
    
    def _merge_bboxes(self, bboxes: List[List[int]]) -> List[int]:
        """여러 바운딩 박스 병합"""
        x_mins = [bbox[0] for bbox in bboxes]
        y_mins = [bbox[1] for bbox in bboxes]
        x_maxs = [bbox[2] for bbox in bboxes]
        y_maxs = [bbox[3] for bbox in bboxes]
        
        return [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]
    
    def _classify_layout_elements(self, text_regions: List[LayoutElement],
                                 tables: List[LayoutElement],
                                 image: np.ndarray) -> List[LayoutElement]:
        """레이아웃 요소 분류"""
        all_elements = text_regions + tables
        
        # 위치와 크기 기반 분류
        for element in all_elements:
            if element.type == LayoutElementType.TABLE:
                continue
            
            x1, y1, x2, y2 = element.bbox
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 제목 판단 (상단, 중앙 정렬, 큰 폰트)
            if y1 < image.shape[0] * 0.15 and abs(center_x - image.shape[1]/2) < 100:
                element.type = LayoutElementType.TITLE
            
            # 페이지 번호 판단 (하단, 작은 크기)
            elif y2 > image.shape[0] * 0.95 and width < 100:
                element.type = LayoutElementType.PAGE_NUMBER
            
            # 서명 영역 판단 (하단 우측, 특정 크기)
            elif y2 > image.shape[0] * 0.85 and x2 > image.shape[1] * 0.7:
                element.type = LayoutElementType.SIGNATURE
            
            # 필드 라벨/값 판단 (작은 높이, 수평 배열)
            elif height < 50 and width < image.shape[1] * 0.3:
                # 콜론이나 특정 패턴으로 라벨/값 구분
                element.type = LayoutElementType.FIELD_LABEL
        
        return all_elements
    
    def _build_hierarchy(self, elements: List[LayoutElement]) -> Dict:
        """계층 구조 생성"""
        hierarchy = {
            "root": {
                "type": "document",
                "children": []
            }
        }
        
        # Y 좌표로 정렬
        sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))
        
        current_section = None
        
        for element in sorted_elements:
            if element.type == LayoutElementType.TITLE:
                # 새 섹션 시작
                current_section = {
                    "type": "section",
                    "title": element.text,
                    "bbox": element.bbox,
                    "children": []
                }
                hierarchy["root"]["children"].append(current_section)
            
            elif current_section:
                current_section["children"].append(element.to_dict())
            else:
                hierarchy["root"]["children"].append(element.to_dict())
        
        return hierarchy
    
    def _map_fields(self, hierarchy: Dict) -> Dict[str, Any]:
        """필드 매핑"""
        mapped = {}
        
        def extract_fields(node: Dict, parent_key: str = ""):
            if "children" in node:
                for child in node["children"]:
                    if isinstance(child, dict):
                        # 필드 타입별로 분류
                        field_type = self._identify_field_type(child.get("text", ""))
                        if field_type:
                            if field_type not in mapped:
                                mapped[field_type] = []
                            mapped[field_type].append({
                                "text": child.get("text", ""),
                                "bbox": child.get("bbox", []),
                                "confidence": child.get("confidence", 0.0)
                            })
                        
                        # 재귀적으로 하위 노드 탐색
                        extract_fields(child, field_type or parent_key)
        
        extract_fields(hierarchy["root"])
        
        return mapped
    
    def _identify_field_type(self, text: str) -> Optional[str]:
        """텍스트로부터 필드 타입 식별"""
        if not text:
            return None
        
        for field_type, patterns in self.field_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return field_type
        
        return None
    
    def _calculate_statistics(self, elements: List[LayoutElement]) -> Dict:
        """통계 계산"""
        stats = {
            "total_elements": len(elements),
            "element_types": {},
            "average_confidence": 0.0,
            "table_count": 0,
            "total_cells": 0
        }
        
        confidences = []
        
        for element in elements:
            # 타입별 카운트
            element_type = element.type.value
            if element_type not in stats["element_types"]:
                stats["element_types"][element_type] = 0
            stats["element_types"][element_type] += 1
            
            # 신뢰도 수집
            if element.confidence > 0:
                confidences.append(element.confidence)
            
            # 테이블 통계
            if element.type == LayoutElementType.TABLE:
                stats["table_count"] += 1
                stats["total_cells"] += len(element.children)
        
        if confidences:
            stats["average_confidence"] = np.mean(confidences)
        
        return stats


class FormFieldExtractor:
    """양식 필드 추출기"""
    
    def __init__(self):
        self.checkbox_detector = CheckboxDetector()
        self.signature_detector = SignatureDetector()
        
    def extract_form_fields(self, image: np.ndarray, 
                           layout: Dict) -> Dict[str, Any]:
        """양식 필드 추출"""
        fields = {}
        
        # 체크박스 검출
        checkboxes = self.checkbox_detector.detect(image)
        fields["checkboxes"] = checkboxes
        
        # 서명 영역 검출
        signatures = self.signature_detector.detect(image)
        fields["signatures"] = signatures
        
        # 텍스트 필드 추출
        text_fields = self._extract_text_fields(layout)
        fields["text_fields"] = text_fields
        
        return fields
    
    def _extract_text_fields(self, layout: Dict) -> List[Dict]:
        """텍스트 필드 추출"""
        text_fields = []
        
        for element in layout.get("layout_elements", []):
            if element.get("type") in ["field_label", "field_value"]:
                text_fields.append({
                    "type": element["type"],
                    "text": element.get("text", ""),
                    "bbox": element.get("bbox", []),
                    "confidence": element.get("confidence", 0.0)
                })
        
        return text_fields


class CheckboxDetector:
    """체크박스 검출기"""
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """체크박스 검출"""
        checkboxes = []
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 사각형 검출
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 근사 다각형
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 4개의 꼭짓점을 가진 사각형인지 확인
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 체크박스 크기 필터링 (10x10 ~ 30x30)
                if 10 <= w <= 30 and 10 <= h <= 30:
                    # 정사각형 비율 확인
                    aspect_ratio = w / h
                    if 0.9 <= aspect_ratio <= 1.1:
                        # 체크 여부 확인
                        roi = gray[y:y+h, x:x+w]
                        is_checked = self._is_checked(roi)
                        
                        checkboxes.append({
                            "bbox": [x, y, x+w, y+h],
                            "checked": is_checked
                        })
        
        return checkboxes
    
    def _is_checked(self, roi: np.ndarray) -> bool:
        """체크박스가 체크되었는지 확인"""
        # 평균 픽셀 값으로 판단
        mean_val = np.mean(roi)
        
        # 체크된 경우 내부가 어두움
        return mean_val < 128


class SignatureDetector:
    """서명 검출기"""
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """서명 영역 검출"""
        signatures = []
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 연결된 컴포넌트 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # 서명 영역 크기 필터링
            if 50 <= w <= 200 and 20 <= h <= 100:
                # 가로세로 비율 확인
                aspect_ratio = w / h
                if 1.5 <= aspect_ratio <= 5.0:
                    # 밀도 확인 (서명은 보통 연속적인 선)
                    roi = edges[y:y+h, x:x+w]
                    density = np.sum(roi > 0) / (w * h)
                    
                    if 0.05 <= density <= 0.3:
                        signatures.append({
                            "bbox": [x, y, x+w, y+h],
                            "confidence": density
                        })
        
        return signatures


if __name__ == "__main__":
    # 테스트 코드
    analyzer = SchoolRecordLayoutAnalyzer()
    
    # 테스트 이미지 생성
    test_image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # 샘플 테이블 그리기
    cv2.rectangle(test_image, (50, 100), (750, 400), (0, 0, 0), 2)
    cv2.line(test_image, (50, 150), (750, 150), (0, 0, 0), 2)
    cv2.line(test_image, (200, 100), (200, 400), (0, 0, 0), 2)
    cv2.line(test_image, (400, 100), (400, 400), (0, 0, 0), 2)
    
    # 레이아웃 분석
    result = analyzer.analyze_layout(test_image)
    
    print("Layout Analysis Result:")
    print(f"Total elements: {len(result['layout_elements'])}")
    print(f"Tables found: {result['statistics']['table_count']}")
    print(f"Element types: {result['statistics']['element_types']}")
    
    # 양식 필드 추출
    form_extractor = FormFieldExtractor()
    form_fields = form_extractor.extract_form_fields(test_image, result)
    
    print("\nForm Fields:")
    print(f"Checkboxes: {len(form_fields['checkboxes'])}")
    print(f"Signatures: {len(form_fields['signatures'])}")
    print(f"Text fields: {len(form_fields['text_fields'])}")
