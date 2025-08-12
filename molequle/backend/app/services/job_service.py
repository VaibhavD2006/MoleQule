import os
import uuid
import shutil
from datetime import datetime
from typing import List, Optional
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
import requests
import json

from ..models.database import Job, Analog, SessionLocal

class JobService:
    """Service for managing job operations"""
    
    def __init__(self):
        # Use absolute path for upload directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.upload_dir = os.path.join(base_dir, "uploads")
        self.ml_service_url = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        print(f"Upload directory: {self.upload_dir}")
    
    async def save_file(self, file: UploadFile, job_id: str) -> str:
        """Save uploaded file to local storage"""
        try:
            # Create job-specific directory
            job_dir = os.path.join(self.upload_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)
            
            # Save file with absolute path
            file_path = os.path.join(job_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Return absolute path
            abs_file_path = os.path.abspath(file_path)
            print(f"Saved file: {abs_file_path}")
            return abs_file_path
        except Exception as e:
            raise HTTPException(500, f"Failed to save file: {str(e)}")
    
    async def create_job(self, job_id: str, file_path: str, filename: str, user_id: Optional[str] = None) -> Job:
        """Create a new job record in the database"""
        try:
            db = SessionLocal()
            
            # Determine file format
            file_format = self._determine_file_format(filename)
            
            job = Job(
                job_id=job_id,
                user_id=user_id,
                input_file=file_path,
                input_format=file_format,
                original_filename=filename,
                status="pending"
            )
            
            db.add(job)
            db.commit()
            db.refresh(job)
            db.close()
            
            return job
        except Exception as e:
            raise HTTPException(500, f"Failed to create job: {str(e)}")
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        try:
            db = SessionLocal()
            job = db.query(Job).filter(Job.job_id == job_id).first()
            db.close()
            return job
        except Exception as e:
            raise HTTPException(500, f"Failed to get job: {str(e)}")
    
    async def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None):
        """Update job status"""
        try:
            db = SessionLocal()
            job = db.query(Job).filter(Job.job_id == job_id).first()
            
            if not job:
                raise HTTPException(404, "Job not found")
            
            job.status = status
            
            if status == "processing":
                job.started_at = datetime.utcnow()
            elif status == "completed":
                job.completed_at = datetime.utcnow()
                if job.started_at:
                    job.processing_time = (job.completed_at - job.started_at).total_seconds()
            elif status == "failed":
                job.error_message = error_message
                job.completed_at = datetime.utcnow()
            
            db.commit()
            db.close()
        except Exception as e:
            raise HTTPException(500, f"Failed to update job status: {str(e)}")
    
    async def save_analogs(self, job_id: str, analogs_data: List[dict]):
        """Save generated analogs to database"""
        try:
            db = SessionLocal()
            
            for analog_data in analogs_data:
                analog = Analog(
                    job_id=job_id,
                    analog_id=analog_data.get("analog_id"),
                    smiles=analog_data.get("smiles"),
                    binding_affinity=analog_data.get("binding_affinity"),
                    final_score=analog_data.get("final_score"),
                    rank=analog_data.get("rank"),
                    energy=analog_data.get("energy"),
                    homo_lumo_gap=analog_data.get("homo_lumo_gap"),
                    dipole_moment=analog_data.get("dipole_moment")
                )
                db.add(analog)
            
            db.commit()
            db.close()
        except Exception as e:
            raise HTTPException(500, f"Failed to save analogs: {str(e)}")
    
    async def get_analogs(self, job_id: str) -> List[dict]:
        """Get analogs for a job"""
        try:
            db = SessionLocal()
            analogs = db.query(Analog).filter(Analog.job_id == job_id).order_by(Analog.rank).all()
            
            analogs_data = []
            for analog in analogs:
                analogs_data.append({
                    "analog_id": analog.analog_id,
                    "smiles": analog.smiles,
                    "binding_affinity": analog.binding_affinity,
                    "final_score": analog.final_score,
                    "rank": analog.rank,
                    "energy": analog.energy,
                    "homo_lumo_gap": analog.homo_lumo_gap,
                    "dipole_moment": analog.dipole_moment
                })
            
            db.close()
            return analogs_data
        except Exception as e:
            raise HTTPException(500, f"Failed to get analogs: {str(e)}")
    
    async def process_molecule_with_ml(self, job_id: str, file_path: str):
        """Send molecule to ML service for processing"""
        try:
            print(f"Starting ML processing for job: {job_id}")
            print(f"File path: {file_path}")
            
            # Update job status to processing
            await self.update_job_status(job_id, "processing")
            print(f"Job status updated to processing")
            
            # Verify file still exists
            if not os.path.exists(file_path):
                error_msg = f"File not found during processing: {file_path}"
                print(f"Error: {error_msg}")
                await self.update_job_status(job_id, "failed", error_msg)
                return
            
            # Prepare request to ML service
            ml_request = {
                "job_id": job_id,
                "input_file_path": file_path
            }
            
            print(f"Sending request to ML service: {self.ml_service_url}")
            print(f"Request data: {ml_request}")
            
            # Send request to ML service
            response = requests.post(
                f"{self.ml_service_url}/process-molecule",
                json=ml_request,
                timeout=300  # 5 minutes timeout
            )
            
            print(f"ML service response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"ML service result: {result}")
                
                if result.get("status") == "completed":
                    # Save analogs to database
                    analogs = result.get("analogs", [])
                    print(f"Saving {len(analogs)} analogs to database")
                    await self.save_analogs(job_id, analogs)
                    
                    # Update job status
                    await self.update_job_status(job_id, "completed")
                    print(f"Job {job_id} completed successfully")
                else:
                    error_msg = result.get("error", "Unknown error from ML service")
                    print(f"ML service returned error: {error_msg}")
                    await self.update_job_status(job_id, "failed", error_msg)
            else:
                error_msg = f"ML service error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                print(f"Error: {error_msg}")
                await self.update_job_status(job_id, "failed", error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "ML service request timed out"
            print(f"Error: {error_msg}")
            await self.update_job_status(job_id, "failed", error_msg)
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to ML service"
            print(f"Error: {error_msg}")
            await self.update_job_status(job_id, "failed", error_msg)
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(f"Error: {error_msg}")
            await self.update_job_status(job_id, "failed", error_msg)
    
    async def generate_download(self, job_id: str, format: str) -> str:
        """Generate download URL for results"""
        try:
            analogs = await self.get_analogs(job_id)
            
            if not analogs:
                raise HTTPException(404, "No analogs found for this job")
            
            # Create download file
            download_dir = os.path.join(self.upload_dir, job_id, "downloads")
            os.makedirs(download_dir, exist_ok=True)
            
            if format.lower() == "csv":
                file_path = os.path.join(download_dir, f"analogs_{job_id}.csv")
                await self._create_csv_download(analogs, file_path)
            elif format.lower() == "json":
                file_path = os.path.join(download_dir, f"analogs_{job_id}.json")
                await self._create_json_download(analogs, file_path)
            else:
                raise HTTPException(400, f"Unsupported format: {format}")
            
            return file_path
        except Exception as e:
            raise HTTPException(500, f"Failed to generate download: {str(e)}")
    
    async def _create_csv_download(self, analogs: List[dict], file_path: str):
        """Create CSV download file"""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['rank', 'analog_id', 'smiles', 'binding_affinity', 'final_score', 'energy', 'homo_lumo_gap', 'dipole_moment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for analog in analogs:
                writer.writerow(analog)
    
    async def _create_json_download(self, analogs: List[dict], file_path: str):
        """Create JSON download file"""
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(analogs, jsonfile, indent=2)
    
    def _determine_file_format(self, filename: str) -> str:
        """Determine file format from filename"""
        ext = filename.lower().split('.')[-1]
        
        if ext in ['smi', 'smiles']:
            return "SMILES"
        elif ext == 'mol':
            return "MOL"
        elif ext == 'xyz':
            return "XYZ"
        else:
            return "UNKNOWN" 