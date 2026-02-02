#!/bin/bash
# Start both backend and frontend servers for the Data Viewer

echo "Starting Robotics Dataset Viewer..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start backend with multiple workers (prevents one stuck request from blocking all others)
echo -e "${BLUE}Starting Backend (FastAPI with 4 workers)...${NC}"
cd "$SCRIPT_DIR/backend"
source "$SCRIPT_DIR/.venv/bin/activate"
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 2

# Start frontend
echo -e "${BLUE}Starting Frontend (Next.js)...${NC}"
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Robotics Dataset Viewer is running!  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo ""
echo "  Open http://localhost:3000 in your browser"
echo ""
echo "  Press Ctrl+C to stop both servers"
echo ""

# Wait for Ctrl+C
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
