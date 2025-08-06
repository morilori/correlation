#!/bin/bash
# Start both backend and frontend for the BERT Attention Visualization website
set -e

# --- Backend setup ---
cd backend
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
echo "Installing backend dependencies..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt
cd ..

# --- Frontend setup ---
cd frontend
if [ ! -d "node_modules" ]; then
  echo "Installing frontend npm dependencies..."
  npm install
fi
cd ..

# --- Kill any process using port 8000 (backend default) ---
if lsof -i :8000 >/dev/null 2>&1; then
  echo "Port 8000 in use. Killing process..."
  lsof -ti :8000 | xargs kill -9 2>/dev/null || true
fi

# --- Start backend ---
cd backend
source venv/bin/activate
echo "Starting FastAPI backend on http://localhost:8000..."
uvicorn app.main:app --reload &
BACKEND_PID=$!
cd ..

# --- Start frontend ---
cd frontend
echo "Starting React frontend on http://localhost:5173..."
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Both services are starting up..."
echo "📊 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
