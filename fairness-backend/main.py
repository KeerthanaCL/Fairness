"""
FastAPI Backend for Fairness Evaluation Platform
Production-ready backend that integrates with the React frontend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path

from app.api import upload, analysis, config, explanations
# from app.core.database import init_db  # Temporarily disabled for in-memory storage
from app.core.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Fairness Evaluation Platform API",
    description="Production-ready API for ML fairness analysis and bias detection",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for file serving
uploads_path = Path("uploads")
uploads_path.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include API routers
app.include_router(upload.router, prefix="/api/upload", tags=["File Upload"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(config.router, prefix="/api/config", tags=["Configuration"])
app.include_router(explanations.router, prefix="/api/explanations", tags=["AI Explanations"])

@app.on_event("startup")
async def startup_event():
    """Initialize database and create necessary directories on startup"""
    # await init_db()  # Temporarily disabled for in-memory storage
    
    # Create upload directories
    for directory in ["uploads/datasets", "uploads/models", "uploads/temp"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "fairness-evaluation-platform",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check_simple():
    """Simple health check endpoint for frontend compatibility"""
    return {
        "status": "healthy",
        "service": "fairness-evaluation-platform",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fairness Evaluation Platform API",
        "docs": "/api/docs",
        "health": "/api/health",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
