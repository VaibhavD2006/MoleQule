# 🚀 MoleQule SaaS MVP Build Plan
## Quantum-Enhanced Drug Discovery Platform

### 🎯 **Phase 1: MVP Goal**
Build a B2B SaaS platform where users can upload molecular files (SMILES, XYZ, MOL), run them through the quantum-enhanced cisplatin model, and receive optimized analogs for download.

---

## 📋 **Phase 1: Project Setup & Architecture**

### **1.1 Project Structure**
```
molequle-saas/
├── frontend/                 # React + Next.js + Tailwind
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/          # Next.js pages
│   │   ├── hooks/          # Custom React hooks
│   │   ├── utils/          # Helper functions
│   │   └── styles/         # Tailwind CSS
│   ├── public/             # Static assets
│   └── package.json
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   ├── ml_engine/      # ML model integration
│   │   └── utils/          # Helper functions
│   ├── requirements.txt
│   └── main.py
├── ml_service/             # Quantum model microservice
│   ├── models/             # Model files
│   ├── quantum_dock/       # Your existing quantum_dock code
│   ├── api/                # Model API endpoints
│   └── requirements.txt
├── docker-compose.yml      # Local development
└── README.md
```

### **1.2 Technology Stack**
- **Frontend**: React 18 + Next.js 14 + Tailwind CSS + TypeScript
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **ML Service**: Python + Your existing quantum_dock code
- **Authentication**: Clerk.dev (simple integration)
- **File Storage**: AWS S3 or Supabase Storage
- **Deployment**: Vercel (frontend) + Railway/Render (backend)

---

## 🏗️ **Phase 2: Backend Development**

### **2.1 FastAPI Backend Setup**
**File: `backend/main.py`**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="MoleQule API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "MoleQule API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### **2.2 Database Models**
**File: `backend/app/models/database.py`**
```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://user:password@localhost/molequle"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    job_id = Column(String, unique=True, index=True)
    input_file = Column(String)  # S3 path
    input_format = Column(String)  # SMILES, MOL, XYZ
    status = Column(String)  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result_file = Column(String, nullable=True)  # S3 path
    error_message = Column(Text, nullable=True)

class Analog(Base):
    __tablename__ = "analogs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, index=True)
    analog_id = Column(String)
    smiles = Column(String)
    binding_affinity = Column(Float)
    final_score = Column(Float)
    rank = Column(Integer)
```

### **2.3 API Endpoints**
**File: `backend/app/api/jobs.py`**
```python
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os
from ..services.job_service import JobService
from ..services.ml_service import MLService

router = APIRouter()
job_service = JobService()
ml_service = MLService()

@router.post("/upload-molecule")
async def upload_molecule(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload molecular file and start processing"""
    try:
        # Validate file type
        allowed_types = ["text/plain", "chemical/x-mol", "chemical/x-xyz"]
        if file.content_type not in allowed_types:
            raise HTTPException(400, "Invalid file type")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save file to S3
        file_path = await job_service.save_file(file, job_id)
        
        # Create job record
        job = await job_service.create_job(job_id, file_path, file.filename)
        
        # Start background processing
        background_tasks.add_task(ml_service.process_molecule, job_id, file_path)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "message": "File uploaded successfully. Processing started."
        })
        
    except Exception as e:
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
                "download_url": job.result_file
            })
        else:
            return JSONResponse({
                "job_id": job_id,
                "status": job.status,
                "message": "Processing in progress..."
            })
            
    except Exception as e:
        raise HTTPException(500, f"Failed to get results: {str(e)}")

@router.get("/download/{job_id}/{format}")
async def download_results(job_id: str, format: str):
    """Download results in specified format (CSV, SDF, MOL)"""
    try:
        download_url = await job_service.generate_download(job_id, format)
        return JSONResponse({"download_url": download_url})
    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")
```

---

## 🤖 **Phase 3: ML Service Integration**

### **3.1 ML Service Wrapper**
**File: `ml_service/api/model_api.py`**
```python
from fastapi import FastAPI, HTTPException
import sys
import os
import json
from pathlib import Path

# Add quantum_dock to path
sys.path.append(str(Path(__file__).parent.parent / "quantum_dock"))

from quantum_dock.agent_core.enhanced_analog_generator import generate_enhanced_analogs
from quantum_dock.vqe_engine.vqe_runner import run_vqe_descriptors
from quantum_dock.qnn_model.qnn_predictor import QNNPredictor
from quantum_dock.agent_core.scoring_engine import calculate_final_score, rank_analogs

app = FastAPI(title="MoleQule ML Service")

class CisplatinModel:
    def __init__(self):
        self.qnn_model = QNNPredictor(n_features=3, n_layers=2)
        # Load pre-trained model
        self.qnn_model.load_model("models/trained_qnn_model.pkl")
    
    def process_molecule(self, input_file_path: str) -> dict:
        """Process molecule through the full pipeline"""
        try:
            # Step 1: Generate analogs
            analogs = generate_enhanced_analogs(input_file_path)
            
            # Step 2: Run VQE for each analog
            results = []
            for analog in analogs:
                # Run VQE simulation
                descriptors = run_vqe_descriptors(analog['xyz_path'])
                
                # Predict binding affinity
                binding_affinity = self.qnn_model.predict([
                    descriptors['energy'],
                    descriptors['homo_lumo_gap'],
                    descriptors['dipole_moment']
                ])
                
                # Calculate final score
                final_score = calculate_final_score(
                    binding_affinity=binding_affinity,
                    resistance_score=0.1,  # Default values for MVP
                    toxicity_score=0.05
                )
                
                results.append({
                    'analog_id': analog['analog_id'],
                    'smiles': analog['smiles'],
                    'binding_affinity': binding_affinity,
                    'final_score': final_score,
                    'descriptors': descriptors
                })
            
            # Step 3: Rank results
            ranked_results = rank_analogs(results)
            
            return {
                'status': 'completed',
                'analogs': ranked_results,
                'total_analogs': len(ranked_results)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

# Initialize model
cisplatin_model = CisplatinModel()

@app.post("/process-molecule")
async def process_molecule(job_id: str, input_file_path: str):
    """Process molecule through cisplatin model"""
    try:
        results = cisplatin_model.process_molecule(input_file_path)
        return results
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

---

## 🎨 **Phase 4: Frontend Development**

### **4.1 Next.js Setup**
**File: `frontend/package.json`**
```json
{
  "name": "molequle-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "@clerk/nextjs": "^4.29.0",
    "axios": "^1.6.0",
    "react-dropzone": "^14.2.3",
    "3dmol": "^2.0.2",
    "lucide-react": "^0.294.0",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32"
  },
  "devDependencies": {
    "@types/node": "^20.8.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.2.0",
    "eslint": "^8.52.0",
    "eslint-config-next": "14.0.0"
  }
}
```

### **4.2 Main Upload Page**
**File: `frontend/src/pages/upload.tsx`**
```tsx
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { useRouter } from 'next/router';

interface UploadState {
  isUploading: boolean;
  jobId: string | null;
  error: string | null;
}

export default function UploadPage() {
  const [uploadState, setUploadState] = useState<UploadState>({
    isUploading: false,
    jobId: null,
    error: null
  });
  const router = useRouter();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploadState({ isUploading: true, jobId: null, error: null });

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/upload-molecule', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadState({
          isUploading: false,
          jobId: data.job_id,
          error: null
        });
        // Redirect to results page
        router.push(`/results/${data.job_id}`);
      } else {
        throw new Error(data.message || 'Upload failed');
      }
    } catch (error) {
      setUploadState({
        isUploading: false,
        jobId: null,
        error: error.message
      });
    }
  }, [router]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.smi', '.smiles'],
      'chemical/x-mol': ['.mol'],
      'chemical/x-xyz': ['.xyz']
    },
    multiple: false
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              MoleQule
            </h1>
            <p className="text-xl text-gray-600">
              Upload your molecular structure to generate optimized cisplatin analogs
            </p>
          </div>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-blue-400 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            
            {uploadState.isUploading ? (
              <div className="space-y-4">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                <p className="text-gray-600">Uploading and processing...</p>
              </div>
            ) : (
              <div className="space-y-4">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <div>
                  <p className="text-lg font-medium text-gray-900">
                    {isDragActive ? 'Drop your file here' : 'Upload molecular structure'}
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    Supports SMILES, MOL, and XYZ formats
                  </p>
                </div>
              </div>
            )}
          </div>

          {uploadState.error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
                <p className="text-red-800">{uploadState.error}</p>
              </div>
            </div>
          )}

          {uploadState.jobId && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 text-green-400 mr-2" />
                <p className="text-green-800">
                  Upload successful! Job ID: {uploadState.jobId}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

### **4.3 Results Page**
**File: `frontend/src/pages/results/[jobId].tsx`**
```tsx
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Download, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react';

interface Analog {
  analog_id: string;
  smiles: string;
  binding_affinity: number;
  final_score: number;
  rank: number;
}

interface ResultsData {
  job_id: string;
  status: string;
  analogs: Analog[];
  download_url?: string;
}

export default function ResultsPage() {
  const router = useRouter();
  const { jobId } = router.query;
  const [results, setResults] = useState<ResultsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = async () => {
    if (!jobId) return;

    try {
      const response = await fetch(`http://localhost:8000/results/${jobId}`);
      const data = await response.json();

      if (response.ok) {
        setResults(data);
        setError(null);
      } else {
        setError(data.message || 'Failed to fetch results');
      }
    } catch (err) {
      setError('Network error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
    // Poll for updates if status is pending
    const interval = setInterval(() => {
      if (results?.status === 'pending') {
        fetchResults();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [jobId, results?.status]);

  const handleDownload = async (format: string) => {
    try {
      const response = await fetch(`http://localhost:8000/download/${jobId}/${format}`);
      const data = await response.json();
      
      if (response.ok && data.download_url) {
        window.open(data.download_url, '_blank');
      }
    } catch (err) {
      setError('Download failed');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={fetchResults}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <RefreshCw className="h-4 w-4 inline mr-2" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!results) return null;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Results</h1>
                <p className="text-gray-600">Job ID: {results.job_id}</p>
              </div>
              <div className="flex items-center space-x-4">
                {results.status === 'completed' && (
                  <>
                    <button
                      onClick={() => handleDownload('csv')}
                      className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download CSV
                    </button>
                    <button
                      onClick={() => handleDownload('sdf')}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download SDF
                    </button>
                  </>
                )}
                <div className={`flex items-center px-3 py-1 rounded-full text-sm ${
                  results.status === 'completed' 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-yellow-100 text-yellow-800'
                }`}>
                  {results.status === 'completed' ? (
                    <CheckCircle className="h-4 w-4 mr-1" />
                  ) : (
                    <RefreshCw className="h-4 w-4 mr-1 animate-spin" />
                  )}
                  {results.status}
                </div>
              </div>
            </div>
          </div>

          {/* Results Table */}
          {results.status === 'completed' && results.analogs && (
            <div className="bg-white rounded-lg shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">
                  Generated Analogs ({results.analogs.length})
                </h2>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Rank
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Analog ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        SMILES
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Binding Affinity (kcal/mol)
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Final Score
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.analogs.map((analog, index) => (
                      <tr key={analog.analog_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {analog.rank}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {analog.analog_id}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900 font-mono">
                          {analog.smiles}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {analog.binding_affinity.toFixed(2)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {analog.final_score.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {results.status === 'pending' && (
            <div className="bg-white rounded-lg shadow-sm p-12 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <h2 className="text-lg font-semibold text-gray-900 mb-2">
                Processing your molecule...
              </h2>
              <p className="text-gray-600">
                This may take a few minutes. We'll automatically refresh the results.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

---

## 🔧 **Phase 5: Integration & Testing**

### **5.1 Docker Setup**
**File: `docker-compose.yml`**
```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/molequle
      - ML_SERVICE_URL=http://ml_service:8001
    depends_on:
      - db

  ml_service:
    build: ./ml_service
    ports:
      - "8001:8001"
    volumes:
      - ./ml_service/models:/app/models
      - ./quantum_dock:/app/quantum_dock

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=molequle
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### **5.2 Environment Configuration**
**File: `.env.local`**
```env
# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_key

# Backend
DATABASE_URL=postgresql://user:password@localhost:5432/molequle
ML_SERVICE_URL=http://localhost:8001
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_S3_BUCKET=molequle-uploads

# ML Service
MODEL_PATH=/app/models
QUANTUM_DOCK_PATH=/app/quantum_dock
```

---

## 🚀 **Phase 6: Deployment**

### **6.1 Vercel Deployment (Frontend)**
1. Connect GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### **6.2 Railway Deployment (Backend)**
1. Connect GitHub repository to Railway
2. Set environment variables
3. Deploy automatically

### **6.3 Production Checklist**
- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] SSL certificates installed
- [ ] Monitoring and logging setup
- [ ] Error tracking (Sentry) configured

---

## 📈 **Future Extensibility**

### **Model Registry System**
**File: `backend/app/models/model_registry.py`**
```python
MODEL_REGISTRY = {
    "cisplatin_v1": {
        "name": "Cisplatin Analog Generator",
        "type": "generation",
        "input_formats": ["SMILES", "MOL", "XYZ"],
        "output_formats": ["CSV", "SDF"],
        "service_url": "http://ml_service:8001",
        "description": "Generate optimized cisplatin analogs using quantum computing"
    },
    "binding_pred_v2": {
        "name": "Binding Affinity Predictor",
        "type": "prediction",
        "input_formats": ["SMILES", "PDB"],
        "output_formats": ["CSV", "JSON"],
        "service_url": "http://binding_service:8002",
        "description": "Predict binding affinities for drug-receptor interactions"
    }
}
```

### **Plugin Architecture**
- Each new model becomes a separate microservice
- Standardized API interface for all models
- Frontend dynamically loads model options
- Easy addition of new features without code changes

---

## ⏱️ **Build Timeline (2-3 Weeks)**

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Backend + ML Integration | API endpoints, database, ML service wrapper |
| 2 | Frontend + UI | Upload page, results page, basic styling |
| 3 | Integration + Deployment | Docker setup, testing, production deployment |

This structure provides a solid foundation for the MVP while ensuring easy extensibility for future features and models.

