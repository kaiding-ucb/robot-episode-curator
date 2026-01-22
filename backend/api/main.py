"""
FastAPI application for the Robotics Dataset Viewer.

Provides API endpoints for:
- Dataset listing and metadata
- Episode browsing and frame streaming
- Quality metrics (Phase 2)
- Downloads management
"""
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import datasets, episodes, downloads, quality, compare

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Data root directory (configurable via environment)
# Default to ../data since backend runs from backend/ directory
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "../data")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Robotics Dataset Viewer",
    description="API for viewing and analyzing robotics datasets",
    version="0.1.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(episodes.router, prefix="/api/episodes", tags=["episodes"])
app.include_router(downloads.router, prefix="/api/downloads", tags=["downloads"])
app.include_router(quality.router, prefix="/api/quality", tags=["quality"])
app.include_router(compare.router, prefix="/api/compare", tags=["compare"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Robotics Dataset Viewer API",
        "version": "0.1.0",
    }


@app.get("/api/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "data_root": str(DATA_ROOT),
        "data_root_exists": DATA_ROOT.exists(),
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Store data root in app state for access by routes
app.state.data_root = DATA_ROOT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
