#!/usr/bin/env bash
# Run backend (FastAPI) and frontend (Next.js) dev servers together.
# Aborts if the required ports are already in use.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND_PORT="${PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
HOST="${HOST:-0.0.0.0}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

check_port() {
    local port=$1
    local label=$2
    if lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
        echo -e "${RED}error:${NC} port $port (${label}) is already in use." >&2
        echo "  Override with $3=<port> or stop the process using that port." >&2
        exit 1
    fi
}

check_port "$BACKEND_PORT" "backend" "PORT"
check_port "$FRONTEND_PORT" "frontend" "FRONTEND_PORT"

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
    echo -e "${RED}error:${NC} no virtualenv at $ROOT_DIR/.venv. Run 'make install' first." >&2
    exit 1
fi
if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
    echo -e "${RED}error:${NC} frontend/node_modules missing. Run 'make install' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

# Run children in their own process groups so we can signal the whole tree on exit
# (npm spawns next-server as a grandchild that SIGTERM on npm alone does not reap).
set -m

echo -e "${BLUE}Starting backend (FastAPI) on ${HOST}:${BACKEND_PORT}...${NC}"
(
    cd "$ROOT_DIR/backend"
    exec gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker \
        --bind "${HOST}:${BACKEND_PORT}" --timeout 300
) &
BACKEND_PID=$!

echo -e "${BLUE}Starting frontend (Next.js) on :${FRONTEND_PORT}...${NC}"
(
    cd "$ROOT_DIR/frontend"
    # BACKEND_PORT is read by next.config.ts to wire up the /api/* rewrite proxy.
    PORT="$FRONTEND_PORT" BACKEND_PORT="$BACKEND_PORT" exec npm run dev
) &
FRONTEND_PID=$!

cleanup() {
    trap - INT TERM EXIT
    echo
    echo "Stopping servers..."
    # Kill entire process group of each child (negative PID in kill).
    kill -TERM -- "-$BACKEND_PID" "-$FRONTEND_PID" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$BACKEND_PID" "-$FRONTEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Robotics Dataset Viewer is running!  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "  Backend:  http://localhost:${BACKEND_PORT}"
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
echo
echo "  Press Ctrl+C to stop."
echo

wait
