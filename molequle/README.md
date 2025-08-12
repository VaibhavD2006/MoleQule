# 🧬 MoleQule - Quantum-Enhanced Drug Discovery Platform

A B2B SaaS platform that enables users to upload molecular structures and generate optimized cisplatin analogs using quantum computing techniques.

## 🚀 Features

- **Molecular File Upload**: Support for SMILES (.smi, .smiles), MOL (.mol), and XYZ (.xyz) formats
- **Quantum-Enhanced Processing**: Uses VQE simulations and QNN models for accurate predictions
- **Real-time Results**: Live status updates and progress tracking
- **Download Results**: Export analogs in CSV or JSON formats
- **Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS

## 🏗️ Architecture

```
molequle/
├── frontend/          # React + Next.js UI
├── backend/           # FastAPI backend API
├── ml_service/        # Quantum model microservice
├── quantum_dock/      # Your existing quantum models
└── docker-compose.yml # Local development setup
```

## 🛠️ Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ (for local backend development)

### Option 1: Docker (Recommended)

1. **Clone and navigate to the project**:
   ```bash
   cd molequle
   ```

2. **Start all services**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - ML Service: http://localhost:8001

### Option 2: Local Development

#### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend**:
   ```bash
   python main.py
   ```

#### ML Service Setup

1. **Navigate to ml_service directory**:
   ```bash
   cd ml_service
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the ML service**:
   ```bash
   python main.py
   ```

#### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the frontend**:
   ```bash
   npm run dev
   ```

## 📁 Project Structure

### Backend (`backend/`)
- `main.py` - FastAPI application entry point
- `app/api/` - API endpoints
- `app/models/` - Database models
- `app/services/` - Business logic
- `requirements.txt` - Python dependencies

### ML Service (`ml_service/`)
- `main.py` - ML service with quantum_dock integration
- `requirements.txt` - ML-specific dependencies

### Frontend (`frontend/`)
- `src/pages/` - Next.js pages
- `src/components/` - React components
- `src/styles/` - Tailwind CSS styles
- `package.json` - Node.js dependencies

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Backend
DATABASE_URL=sqlite:///./molequle.db
ML_SERVICE_URL=http://localhost:8001
UPLOAD_DIR=uploads

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000

# ML Service
ML_SERVICE_PORT=8001
```

### Database

The application uses SQLite by default for simplicity. For production, consider using PostgreSQL:

```env
DATABASE_URL=postgresql://user:password@localhost/molequle
```

## 🧪 Usage

1. **Upload a Molecular File**:
   - Navigate to http://localhost:3000
   - Drag and drop or click to upload a molecular file
   - Supported formats: SMILES (.smi, .smiles), MOL (.mol), XYZ (.xyz)

2. **Monitor Processing**:
   - The system will automatically process your molecule
   - View real-time status updates
   - Processing typically takes 2-5 minutes

3. **View Results**:
   - Once complete, view ranked analogs
   - See binding affinities, scores, and molecular descriptors
   - Download results in CSV or JSON format

## 🔬 Technical Details

### Quantum Pipeline

1. **Analog Generation**: Uses your existing `quantum_dock/agent_core/` for systematic analog generation
2. **VQE Simulation**: Runs quantum chemistry simulations via `quantum_dock/vqe_engine/`
3. **QNN Prediction**: Predicts binding affinities using trained quantum neural networks
4. **Scoring & Ranking**: Applies biological context and ranks analogs by effectiveness

### API Endpoints

- `POST /api/v1/upload-molecule` - Upload molecular file
- `GET /api/v1/results/{job_id}` - Get processing results
- `GET /api/v1/download/{job_id}/{format}` - Download results

## 🚀 Deployment

### Production Setup

1. **Environment Configuration**:
   - Set production environment variables
   - Configure database (PostgreSQL recommended)
   - Set up file storage (AWS S3 or similar)

2. **Deploy Services**:
   - Frontend: Deploy to Vercel or similar
   - Backend: Deploy to Railway, Render, or similar
   - ML Service: Deploy to cloud with GPU support

3. **SSL & Security**:
   - Configure HTTPS
   - Set up authentication (Clerk, Auth0, etc.)
   - Implement rate limiting

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure quantum_dock dependencies are installed
2. **Port Conflicts**: Check if ports 3000, 8000, 8001 are available
3. **File Permissions**: Ensure upload directories are writable
4. **Memory Issues**: ML processing requires significant RAM

### Logs

- Backend logs: `docker-compose logs backend`
- ML Service logs: `docker-compose logs ml_service`
- Frontend logs: `docker-compose logs frontend`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on your existing quantum_dock platform
- Uses quantum computing libraries (PennyLane, PySCF)
- Modern web technologies (React, FastAPI, Tailwind CSS)

---

**Note**: This is a research prototype. Not intended for clinical use. 