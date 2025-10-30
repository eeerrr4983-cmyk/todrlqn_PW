"""
딥러닝 기반 한국어 텍스트 인식 향상 모듈
Transformer 기반 모델과 CNN 결합으로 정확도 극대화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    VisionEncoderDecoderModel,
    TrOCRProcessor
)
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class KoreanCharacterCNN(nn.Module):
    """한글 문자 인식 특화 CNN 모델"""
    
    def __init__(self, num_classes: int = 11172):  # 한글 유니코드 범위
        super(KoreanCharacterCNN, self).__init__()
        
        # 특징 추출 레이어
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Attention 메커니즘
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # 분류 레이어
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        # 특징 추출
        features = self.features(x)
        
        # Global Average Pooling
        gap = F.adaptive_avg_pool2d(features, (4, 4))
        gap = gap.view(gap.size(0), -1)
        
        # Attention 적용
        attention_weights = self.attention(gap.view(-1, 512))
        attended_features = gap * attention_weights.view(-1, 1)
        
        # 분류
        output = self.classifier(attended_features)
        return output


class KoreanTextCRNN(nn.Module):
    """한국어 텍스트 시퀀스 인식 CRNN 모델"""
    
    def __init__(self, img_height: int = 32, nc: int = 1, 
                 nclass: int = 11172, nh: int = 256):
        super(KoreanTextCRNN, self).__init__()
        
        # CNN 특징 추출기
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        
        # RNN
        self.rnn = nn.LSTM(512, nh, num_layers=2, bidirectional=True, batch_first=True)
        
        # 출력 레이어
        self.fc = nn.Linear(nh * 2, nclass)
        
    def forward(self, x):
        # CNN 특징 추출
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "Height of conv must be 1"
        
        # CNN to RNN
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        
        # RNN
        output, _ = self.rnn(conv)
        
        # 출력
        output = self.fc(output)
        return output


class TransformerOCR(nn.Module):
    """Transformer 기반 OCR 모델"""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, num_classes: int = 11172):
        super(TransformerOCR, self).__init__()
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 입력 임베딩
        self.embedding = nn.Linear(512, d_model)
        
        # 출력 레이어
        self.fc = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model
        
    def forward(self, src):
        # 임베딩
        src = self.embedding(src) * np.sqrt(self.d_model)
        
        # Position encoding 적용
        src = self.pos_encoder(src)
        
        # Transformer encoding
        output = self.transformer_encoder(src)
        
        # 분류
        output = self.fc(output)
        
        return output


class PositionalEncoding(nn.Module):
    """위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class EnsembleOCR:
    """앙상블 OCR 모델 - 여러 모델의 예측을 결합"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        # GPU 사용 가능 여부 확인
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델을 디바이스로 이동
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        앙상블 예측
        
        Args:
            image: 입력 이미지
            
        Returns:
            예측 텍스트와 신뢰도
        """
        # 이미지 전처리
        tensor = self._preprocess_image(image)
        tensor = tensor.to(self.device)
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                output = model(tensor)
                prob = F.softmax(output, dim=-1)
                
                # Top-k 예측
                values, indices = torch.topk(prob, k=5, dim=-1)
                
                predictions.append(indices)
                confidences.append(values * weight)
        
        # 예측 결합 (가중 투표)
        combined_prediction = self._weighted_voting(predictions, confidences)
        
        # 문자로 변환
        text = self._indices_to_text(combined_prediction)
        confidence = self._calculate_confidence(confidences)
        
        return text, confidence
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 크기 조정
        resized = cv2.resize(gray, (128, 32))
        
        # 정규화
        normalized = resized.astype(np.float32) / 255.0
        
        # 텐서 변환
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def _weighted_voting(self, predictions: List[torch.Tensor], 
                        confidences: List[torch.Tensor]) -> torch.Tensor:
        """가중 투표"""
        # 각 위치별로 최대 신뢰도를 가진 예측 선택
        combined = torch.zeros_like(predictions[0][0])
        
        for i in range(len(predictions[0][0])):
            votes = {}
            
            for pred, conf in zip(predictions, confidences):
                for j in range(pred.shape[-1]):
                    char_idx = pred[0, i, j].item()
                    confidence = conf[0, i, j].item()
                    
                    if char_idx not in votes:
                        votes[char_idx] = 0
                    votes[char_idx] += confidence
            
            # 최대 투표수를 받은 문자 선택
            if votes:
                best_char = max(votes, key=votes.get)
                combined[i] = best_char
        
        return combined
    
    def _indices_to_text(self, indices: torch.Tensor) -> str:
        """인덱스를 텍스트로 변환"""
        text = ""
        for idx in indices:
            if idx.item() > 0:  # 0은 padding
                # 한글 유니코드 변환 (가 = 0xAC00)
                char = chr(0xAC00 + idx.item() - 1)
                text += char
        return text
    
    def _calculate_confidence(self, confidences: List[torch.Tensor]) -> float:
        """평균 신뢰도 계산"""
        total_conf = 0
        count = 0
        
        for conf_tensor in confidences:
            total_conf += conf_tensor.max(dim=-1)[0].mean().item()
            count += 1
        
        return total_conf / count if count > 0 else 0.0


class KoreanSpellChecker:
    """한국어 맞춤법 검사 및 교정"""
    
    def __init__(self):
        # 한국어 맞춤법 사전 로드 (실제로는 더 큰 사전 필요)
        self.dictionary = self._load_dictionary()
        self.common_errors = self._load_common_errors()
        
    def _load_dictionary(self) -> set:
        """사전 로드"""
        # 기본 한국어 단어 사전 (예시)
        return {
            "학생", "선생님", "학교", "교실", "수업", "과목", "국어", "영어", "수학",
            "과학", "사회", "역사", "체육", "음악", "미술", "기술", "가정", "도덕",
            "창의적", "체험", "활동", "봉사", "동아리", "자율", "진로", "학년", "반",
            "번호", "이름", "성명", "출석", "결석", "지각", "조퇴", "수행평가",
            "중간고사", "기말고사", "성적", "등급", "평균", "표준편차", "석차"
        }
    
    def _load_common_errors(self) -> dict:
        """일반적인 오류 패턴"""
        return {
            "숩니다": "습니다",
            "슴니다": "습니다",
            "함니다": "합니다",
            "었읍니다": "었습니다",
            "됬": "됐",
            "됫": "됐",
            "햇": "했",
            "엇": "었",
            "는데": "는데",
            "는대": "는데",
            "어떻해": "어떻게",
            "어떻개": "어떻게",
        }
    
    def check_and_correct(self, text: str) -> Tuple[str, List[Dict]]:
        """
        맞춤법 검사 및 교정
        
        Args:
            text: 입력 텍스트
            
        Returns:
            교정된 텍스트와 수정 내역
        """
        corrections = []
        corrected = text
        
        # 일반적인 오류 교정
        for error, correct in self.common_errors.items():
            if error in corrected:
                corrected = corrected.replace(error, correct)
                corrections.append({
                    "original": error,
                    "corrected": correct,
                    "type": "common_error"
                })
        
        # 띄어쓰기 교정
        corrected, spacing_corrections = self._correct_spacing(corrected)
        corrections.extend(spacing_corrections)
        
        # 조사 교정
        corrected, particle_corrections = self._correct_particles(corrected)
        corrections.extend(particle_corrections)
        
        return corrected, corrections
    
    def _correct_spacing(self, text: str) -> Tuple[str, List[Dict]]:
        """띄어쓰기 교정"""
        corrections = []
        
        # 숫자와 한글 사이 띄어쓰기
        import re
        pattern = r'(\d)([가-힣])'
        if re.search(pattern, text):
            text = re.sub(pattern, r'\1 \2', text)
            corrections.append({
                "type": "spacing",
                "description": "숫자와 한글 사이 띄어쓰기"
            })
        
        pattern = r'([가-힣])(\d)'
        if re.search(pattern, text):
            text = re.sub(pattern, r'\1 \2', text)
            corrections.append({
                "type": "spacing",
                "description": "한글과 숫자 사이 띄어쓰기"
            })
        
        # 중복 공백 제거
        if '  ' in text:
            text = re.sub(r'\s+', ' ', text)
            corrections.append({
                "type": "spacing",
                "description": "중복 공백 제거"
            })
        
        return text, corrections
    
    def _correct_particles(self, text: str) -> Tuple[str, List[Dict]]:
        """조사 교정"""
        corrections = []
        
        # 은/는 교정
        import re
        
        # 받침이 있는 글자 뒤의 '는' -> '은'
        pattern = r'([가-힣][ㄱ-ㅎ])는'
        matches = re.finditer(pattern, text)
        for match in matches:
            char = match.group(1)
            if self._has_final_consonant(char):
                text = text.replace(f"{char}는", f"{char}은")
                corrections.append({
                    "original": f"{char}는",
                    "corrected": f"{char}은",
                    "type": "particle"
                })
        
        # 이/가 교정
        pattern = r'([가-힣])가'
        matches = re.finditer(pattern, text)
        for match in matches:
            char = match.group(1)
            if self._has_final_consonant(char):
                text = text.replace(f"{char}가", f"{char}이")
                corrections.append({
                    "original": f"{char}가",
                    "corrected": f"{char}이",
                    "type": "particle"
                })
        
        return text, corrections
    
    def _has_final_consonant(self, char: str) -> bool:
        """받침 유무 확인"""
        if not char or not ('가' <= char <= '힣'):
            return False
        
        code = ord(char) - 0xAC00
        return (code % 28) != 0


class DeepLearningOCREnhancer:
    """딥러닝 기반 OCR 향상 통합 클래스"""
    
    def __init__(self):
        # 모델 초기화
        self.cnn_model = KoreanCharacterCNN()
        self.crnn_model = KoreanTextCRNN()
        self.transformer_model = TransformerOCR()
        
        # 앙상블 모델
        self.ensemble = EnsembleOCR(
            models=[self.cnn_model, self.crnn_model, self.transformer_model],
            weights=[0.3, 0.4, 0.3]
        )
        
        # 맞춤법 검사기
        self.spell_checker = KoreanSpellChecker()
        
        logger.info("Deep Learning OCR Enhancer initialized")
    
    def enhance_ocr_result(self, image: np.ndarray, 
                          paddle_result: Dict) -> Dict:
        """
        딥러닝으로 OCR 결과 향상
        
        Args:
            image: 원본 이미지
            paddle_result: PaddleOCR 결과
            
        Returns:
            향상된 OCR 결과
        """
        enhanced_texts = []
        enhanced_scores = []
        
        # 각 텍스트 영역에 대해 딥러닝 재인식
        boxes = paddle_result.get("boxes", [])
        texts = paddle_result.get("texts", [])
        scores = paddle_result.get("scores", [])
        
        for i, box in enumerate(boxes):
            # 텍스트 영역 추출
            roi = self._extract_roi(image, box)
            
            if roi is not None:
                # 딥러닝 예측
                dl_text, dl_confidence = self.ensemble.predict(roi)
                
                # PaddleOCR 결과와 비교
                paddle_text = texts[i] if i < len(texts) else ""
                paddle_score = scores[i] if i < len(scores) else 0.0
                
                # 더 높은 신뢰도를 가진 결과 선택
                if dl_confidence > paddle_score:
                    final_text = dl_text
                    final_score = dl_confidence
                else:
                    final_text = paddle_text
                    final_score = paddle_score
                
                # 맞춤법 검사 및 교정
                corrected_text, corrections = self.spell_checker.check_and_correct(final_text)
                
                enhanced_texts.append(corrected_text)
                enhanced_scores.append(final_score)
            else:
                enhanced_texts.append(texts[i] if i < len(texts) else "")
                enhanced_scores.append(scores[i] if i < len(scores) else 0.0)
        
        # 결과 반환
        return {
            "texts": enhanced_texts,
            "boxes": boxes,
            "scores": enhanced_scores,
            "full_text": " ".join(enhanced_texts),
            "average_confidence": np.mean(enhanced_scores) if enhanced_scores else 0.0,
            "enhancements_applied": True
        }
    
    def _extract_roi(self, image: np.ndarray, box: List) -> Optional[np.ndarray]:
        """박스 영역 추출"""
        try:
            # 박스 좌표를 정수로 변환
            pts = np.array(box, dtype=np.int32)
            
            # 최소/최대 좌표 찾기
            x_min = max(0, pts[:, 0].min())
            x_max = min(image.shape[1], pts[:, 0].max())
            y_min = max(0, pts[:, 1].min())
            y_max = min(image.shape[0], pts[:, 1].max())
            
            # ROI 추출
            roi = image[y_min:y_max, x_min:x_max]
            
            if roi.size == 0:
                return None
            
            return roi
            
        except Exception as e:
            logger.error(f"Error extracting ROI: {e}")
            return None


if __name__ == "__main__":
    # 테스트 코드
    enhancer = DeepLearningOCREnhancer()
    
    # 테스트 이미지와 PaddleOCR 결과 (예시)
    test_image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
    paddle_result = {
        "texts": ["학생", "선셍님", "학교"],
        "boxes": [[[10, 10], [50, 10], [50, 30], [10, 30]],
                 [[60, 10], [120, 10], [120, 30], [60, 30]],
                 [[130, 10], [180, 10], [180, 30], [130, 30]]],
        "scores": [0.95, 0.85, 0.90]
    }
    
    # 향상 적용
    enhanced = enhancer.enhance_ocr_result(test_image, paddle_result)
    
    print("Enhanced OCR Result:")
    print(f"Texts: {enhanced['texts']}")
    print(f"Average Confidence: {enhanced['average_confidence']:.2%}")
