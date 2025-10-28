#!/bin/bash

# Docker entrypoint script for Ultra Korean OCR

echo "🚀 Starting Ultra Korean OCR System..."

# Start API server in background
echo "Starting API server..."
python api/ultra_server.py &
API_PID=$!

# Wait for API to be ready
sleep 5

# Start web interface
echo "Starting web interface..."
streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0 &
WEB_PID=$!

echo "✅ All services started!"
echo "📌 API: http://localhost:8000"
echo "🌐 Web: http://localhost:8501"

# Keep container running
wait $API_PID $WEB_PID