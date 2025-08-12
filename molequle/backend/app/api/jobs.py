from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import uuid
import os
from typing import Optional

from ..services.job_service import JobService

router = APIRouter()
job_service = JobService()

@router.post("/upload-molecule")
async def upload_molecule(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload molecular file and start processing"""
    try:
        # Validate file type
        allowed_types = [
            "text/plain",  # SMILES files
            "chemical/x-mol",  # MOL files
            "chemical/x-xyz",  # XYZ files
            "application/octet-stream"  # Generic binary files
        ]
        
        if file.content_type not in allowed_types:
            # Check file extension as fallback
            filename = file.filename.lower()
            if not any(filename.endswith(ext) for ext in ['.smi', '.smiles', '.mol', '.xyz']):
                raise HTTPException(400, "Invalid file type. Supported formats: SMILES (.smi, .smiles), MOL (.mol), XYZ (.xyz)")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        print(f"Processing upload for job: {job_id}")
        print(f"File: {file.filename}, Type: {file.content_type}")
        
        # Save file
        file_path = await job_service.save_file(file, job_id)
        print(f"File saved to: {file_path}")
        
        # Create job record
        job = await job_service.create_job(job_id, file_path, file.filename)
        print(f"Job created with status: {job.status}")
        
        # Start background processing
        if background_tasks:
            background_tasks.add_task(job_service.process_molecule_with_ml, job_id, file_path)
            print(f"Background task added for job: {job_id}")
        else:
            print("Warning: No background tasks available")
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "message": "File uploaded successfully. Processing started.",
            "filename": file.filename
        })
        
    except Exception as e:
        print(f"Upload failed for file {file.filename}: {str(e)}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@router.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get processing results for a job"""
    try:
        job = await job_service.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        
        if job.status == "completed":
            analogs = await job_service.get_analogs(job_id)
            return JSONResponse({
                "job_id": job_id,
                "status": job.status,
                "analogs": analogs,
                "total_analogs": len(analogs),
                "processing_time": job.processing_time,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            })
        elif job.status == "failed":
            return JSONResponse({
                "job_id": job_id,
                "status": job.status,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            })
        else:
            return JSONResponse({
                "job_id": job_id,
                "status": job.status,
                "message": "Processing in progress...",
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None
            })
            
    except Exception as e:
        raise HTTPException(500, f"Failed to get results: {str(e)}")

@router.get("/download/{job_id}/{format}")
async def download_results(job_id: str, format: str):
    """Download results in specified format (CSV, JSON)"""
    try:
        # Validate format
        if format.lower() not in ["csv", "json"]:
            raise HTTPException(400, "Unsupported format. Use 'csv' or 'json'")
        
        # Generate download file
        file_path = await job_service.generate_download(job_id, format)
        
        # Return file response
        filename = f"analogs_{job_id}.{format.lower()}"
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="text/csv" if format.lower() == "csv" else "application/json"
        )
        
    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")

@router.get("/jobs")
async def list_jobs(limit: int = 10, offset: int = 0):
    """List recent jobs (for future dashboard)"""
    try:
        # This would be implemented for a dashboard
        # For now, return a simple response
        return JSONResponse({
            "message": "Job listing endpoint - to be implemented",
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        raise HTTPException(500, f"Failed to list jobs: {str(e)}")

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    try:
        # This would be implemented for cleanup
        # For now, return a simple response
        return JSONResponse({
            "message": f"Job {job_id} deletion - to be implemented"
        })
    except Exception as e:
        raise HTTPException(500, f"Failed to delete job: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for the jobs API"""
    return JSONResponse({
        "status": "healthy",
        "service": "jobs-api",
        "version": "1.0.0"
    }) 