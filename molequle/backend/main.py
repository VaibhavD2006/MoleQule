from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API routers and database
from app.api import jobs
from app.api import enhanced_jobs
from app.models.database import create_tables

# Initialize database tables
create_tables()

app = FastAPI(
    title="MoleQule API",
    description="Quantum-Enhanced Drug Discovery Platform API",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://molequle.vercel.app"  # Production frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
app.include_router(enhanced_jobs.router, prefix="/api/v1", tags=["enhanced"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MoleQule API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "database": "connected",  # TODO: Add actual DB check
            "ml_service": "connected"  # TODO: Add actual ML service check
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 