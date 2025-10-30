# PaddleOCR-VL Docker 이미지
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 종속성 파일 복사
COPY requirements.txt .

# 기본 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# PaddlePaddle CPU 버전 설치 (Docker 환경)
RUN python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# PaddleOCR 설치
RUN python -m pip install -U "paddleocr[doc-parser]"

# 애플리케이션 코드 복사
COPY main.py .

# 포트 노출
EXPOSE 8000

# 환경 변수 설정
ENV OCR_PORT=8000

# 서버 실행
CMD ["python", "main.py"]
