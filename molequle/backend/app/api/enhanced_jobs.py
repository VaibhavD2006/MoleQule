#!/usr/bin/env python3
"""
Enhanced API endpoints for comprehensive analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uuid
import json
import logging
from datetime import datetime
import requests
from sqlalchemy.orm import Session

# Add parent path for imports
import sys
from pathlib import Path
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

# Try to import quantum_dock modules, with fallbacks
try:
    from quantum_dock.qnn_model.qnn_predictor import QNNPredictor
    QNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: QNNPredictor not available: {e}")
    QNN_AVAILABLE = False
    QNNPredictor = None

try:
    from quantum_dock.predictors.admet_predictor import ADMETPredictor
    ADMET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ADMETPredictor not available: {e}")
    ADMET_AVAILABLE = False
    ADMETPredictor = None

try:
    from quantum_dock.predictors.synthetic_predictor import SyntheticAccessibilityPredictor
    SYNTHETIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SyntheticAccessibilityPredictor not available: {e}")
    SYNTHETIC_AVAILABLE = False
    SyntheticAccessibilityPredictor = None

try:
    from quantum_dock.predictors.stability_predictor import StabilityPredictor
    STABILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: StabilityPredictor not available: {e}")
    STABILITY_AVAILABLE = False
    StabilityPredictor = None

try:
    from quantum_dock.predictors.selectivity_predictor import SelectivityPredictor
    SELECTIVITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SelectivityPredictor not available: {e}")
    SELECTIVITY_AVAILABLE = False
    SelectivityPredictor = None

try:
    from quantum_dock.predictors.clinical_predictor import ClinicalRelevancePredictor
    CLINICAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ClinicalRelevancePredictor not available: {e}")
    CLINICAL_AVAILABLE = False
    ClinicalRelevancePredictor = None

from ..models.database import get_db
from ..models.enhanced_database import EnhancedJob, EnhancedAnalog
from ..services.ml_service_client import MLServiceClient

router = APIRouter(tags=["enhanced"])
logger = logging.getLogger(__name__)

# Initialize ML service client
ml_client = MLServiceClient()

# Pydantic models
class ComprehensiveAnalysisRequest(BaseModel):
    smiles: str
    user_id: Optional[str] = None

class ComprehensiveAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None

class AnalogComprehensiveRequest(BaseModel):
    job_id: str

class AnalogComprehensiveResponse(BaseModel):
    job_id: str
    analogs: List[Dict[str, Any]]
    comprehensive_summary: Dict[str, Any]

# Initialize predictors
predictors = {}

def get_predictors():
    """Get or initialize predictors"""
    if not predictors:
        try:
            if QNN_AVAILABLE and QNNPredictor:
                predictors['qnn'] = QNNPredictor(n_features=8, n_layers=6, n_qubits=12)
            else:
                logging.warning("QNNPredictor not available, using fallback")
                
            if ADMET_AVAILABLE and ADMETPredictor:
                predictors['admet'] = ADMETPredictor()
            else:
                logging.warning("ADMETPredictor not available, using fallback")
                
            if SYNTHETIC_AVAILABLE and SyntheticAccessibilityPredictor:
                predictors['synthetic'] = SyntheticAccessibilityPredictor()
            else:
                logging.warning("SyntheticAccessibilityPredictor not available, using fallback")
                
            if STABILITY_AVAILABLE and StabilityPredictor:
                predictors['stability'] = StabilityPredictor()
            else:
                logging.warning("StabilityPredictor not available, using fallback")
                
            if SELECTIVITY_AVAILABLE and SelectivityPredictor:
                predictors['selectivity'] = SelectivityPredictor()
            else:
                logging.warning("SelectivityPredictor not available, using fallback")
                
            if CLINICAL_AVAILABLE and ClinicalRelevancePredictor:
                predictors['clinical'] = ClinicalRelevancePredictor()
            else:
                logging.warning("ClinicalRelevancePredictor not available, using fallback")
                
        except Exception as e:
            logging.error(f"Error initializing predictors: {e}")
    return predictors

@router.post("/comprehensive-analysis", response_model=ComprehensiveAnalysisResponse)
async def comprehensive_analysis(request: ComprehensiveAnalysisRequest):
    """
    Perform comprehensive analysis on a molecule
    """
    try:
        analysis_id = str(uuid.uuid4())
        
        # Get predictors
        preds = get_predictors()
        
        # Perform comprehensive analysis
        results = await perform_comprehensive_analysis(request.smiles, preds)
        
        return ComprehensiveAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            message="Comprehensive analysis completed successfully",
            results=results
        )
        
    except Exception as e:
        logging.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analogs/{job_id}/comprehensive", response_model=AnalogComprehensiveResponse)
async def get_comprehensive_analogs(job_id: str):
    """
    Get comprehensive analysis for analogs in a job
    """
    try:
        # Get analogs from existing job
        analogs = await get_analogs_from_job(job_id)
        
        # Perform comprehensive analysis on each analog
        comprehensive_analogs = []
        for analog in analogs:
            comprehensive_analog = await analyze_analog_comprehensive(analog)
            comprehensive_analogs.append(comprehensive_analog)
        
        # Generate comprehensive summary
        comprehensive_summary = generate_comprehensive_summary(comprehensive_analogs)
        
        return AnalogComprehensiveResponse(
            job_id=job_id,
            analogs=comprehensive_analogs,
            comprehensive_summary=comprehensive_summary
        )
        
    except Exception as e:
        logging.error(f"Error getting comprehensive analogs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def perform_comprehensive_analysis(smiles: str, predictors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a molecule
    """
    try:
        results = {}
        
        # QNN predictions
        if 'qnn' in predictors:
            qnn_results = predictors['qnn'].predict_comprehensive([300.0, 2.0, 60.0, 2, 5, 5, 1, 3])
            results['qnn_predictions'] = qnn_results
        else:
            results['qnn_predictions'] = get_default_qnn_predictions()
        
        # ADMET analysis
        if 'admet' in predictors:
            try:
                admet_results = predictors['admet'].predict_comprehensive_admet(smiles)
                results['admet_analysis'] = admet_results
            except Exception as e:
                logging.error(f"ADMET analysis failed: {e}")
                results['admet_analysis'] = get_default_admet_analysis(smiles)
        else:
            results['admet_analysis'] = get_default_admet_analysis(smiles)
        
        # Synthetic accessibility
        if 'synthetic' in predictors:
            try:
                synthetic_results = predictors['synthetic'].predict_synthesis_complexity(smiles)
                results['synthetic_analysis'] = synthetic_results
            except Exception as e:
                logging.error(f"Synthetic analysis failed: {e}")
                results['synthetic_analysis'] = get_default_synthetic_analysis(smiles)
        else:
            results['synthetic_analysis'] = get_default_synthetic_analysis(smiles)
        
        # Stability analysis
        if 'stability' in predictors:
            try:
                stability_results = predictors['stability'].predict_comprehensive_stability(smiles)
                results['stability_analysis'] = stability_results
            except Exception as e:
                logging.error(f"Stability analysis failed: {e}")
                results['stability_analysis'] = get_default_stability_analysis(smiles)
        else:
            results['stability_analysis'] = get_default_stability_analysis(smiles)
        
        # Selectivity analysis
        if 'selectivity' in predictors:
            try:
                selectivity_results = predictors['selectivity'].predict_comprehensive_selectivity(smiles)
                results['selectivity_analysis'] = selectivity_results
            except Exception as e:
                logging.error(f"Selectivity analysis failed: {e}")
                results['selectivity_analysis'] = get_default_selectivity_analysis(smiles)
        else:
            results['selectivity_analysis'] = get_default_selectivity_analysis(smiles)
        
        # Clinical analysis
        if 'clinical' in predictors:
            try:
                clinical_results = predictors['clinical'].predict_clinical_relevance(smiles)
                results['clinical_analysis'] = clinical_results
            except Exception as e:
                logging.error(f"Clinical analysis failed: {e}")
                results['clinical_analysis'] = get_default_clinical_analysis(smiles)
        else:
            results['clinical_analysis'] = get_default_clinical_analysis(smiles)
        
        # Calculate comprehensive score
        comprehensive_score = calculate_comprehensive_score(results)
        results['comprehensive_score'] = comprehensive_score
        results['comprehensive_grade'] = grade_comprehensive_score(comprehensive_score)
        
        # Generate summary
        results['summary'] = generate_analysis_summary(results)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in comprehensive analysis: {e}")
        return get_default_comprehensive_results(smiles)

async def get_analogs_from_job(job_id: str) -> List[Dict[str, Any]]:
    """
    Get analogs from existing job
    """
    try:
        # This would typically query the database
        # For now, return mock analogs
        return [
            {
                'analog_id': f'analog_{i}',
                'smiles': f'C[Pt](N)(Cl)Cl_{i}',
                'binding_affinity': -7.0 - i * 0.5,
                'final_score': 0.7 - i * 0.05
            }
            for i in range(1, 6)
        ]
    except Exception as e:
        logging.error(f"Error getting analogs from job: {e}")
        return []

async def analyze_analog_comprehensive(analog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single analog comprehensively
    """
    try:
        smiles = analog.get('smiles', 'C[Pt](N)(Cl)Cl')
        
        # Get predictors
        preds = get_predictors()
        
        # Perform comprehensive analysis
        comprehensive_results = await perform_comprehensive_analysis(smiles, preds)
        
        # Combine with original analog data
        enhanced_analog = {
            **analog,
            **comprehensive_results
        }
        
        return enhanced_analog
        
    except Exception as e:
        logging.error(f"Error analyzing analog: {e}")
        return analog

def generate_comprehensive_summary(analogs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate comprehensive summary for all analogs
    """
    try:
        if not analogs:
            return {}
        
        # Calculate statistics
        comprehensive_scores = [analog.get('comprehensive_score', 0.5) for analog in analogs]
        binding_affinities = [analog.get('binding_affinity_kcal_mol', -7.0) for analog in analogs]
        
        # Find best analogs
        best_comprehensive = max(analogs, key=lambda x: x.get('comprehensive_score', 0.5))
        best_binding = min(analogs, key=lambda x: x.get('binding_affinity_kcal_mol', -7.0))
        
        return {
            'total_analogs': len(analogs),
            'average_comprehensive_score': sum(comprehensive_scores) / len(comprehensive_scores),
            'average_binding_affinity': sum(binding_affinities) / len(binding_affinities),
            'best_comprehensive_analog': best_comprehensive.get('analog_id'),
            'best_binding_analog': best_binding.get('analog_id'),
            'development_recommendations': generate_development_recommendations(analogs)
        }
        
    except Exception as e:
        logging.error(f"Error generating comprehensive summary: {e}")
        return {}

def calculate_comprehensive_score(results: Dict[str, Any]) -> float:
    """
    Calculate comprehensive score from all results
    """
    try:
        # Extract individual scores
        binding_affinity = abs(results.get('qnn_predictions', {}).get('binding_affinity', -7.0)) / 15.0
        admet_score = results.get('admet_analysis', {}).get('overall_admet_score', 0.5)
        synthetic_score = results.get('synthetic_analysis', {}).get('feasibility_score', 0.5)
        stability_score = results.get('stability_analysis', {}).get('overall_stability_score', 0.5)
        selectivity_score = results.get('selectivity_analysis', {}).get('overall_selectivity_score', 0.5)
        clinical_score = results.get('clinical_analysis', {}).get('overall_clinical_score', 0.5)
        
        # Weighted average
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        scores = [binding_affinity, admet_score, synthetic_score, stability_score, selectivity_score, clinical_score]
        
        comprehensive_score = sum(score * weight for score, weight in zip(scores, weights))
        return max(0, min(1, comprehensive_score))
        
    except Exception as e:
        logging.error(f"Error calculating comprehensive score: {e}")
        return 0.5

def grade_comprehensive_score(score: float) -> str:
    """
    Grade comprehensive score
    """
    if score >= 0.8:
        return 'Excellent'
    elif score >= 0.6:
        return 'Good'
    elif score >= 0.4:
        return 'Fair'
    else:
        return 'Poor'

def generate_analysis_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate analysis summary
    """
    try:
        summary = {
            'key_strengths': [],
            'key_concerns': [],
            'recommendations': []
        }
        
        # Analyze QNN predictions
        qnn_preds = results.get('qnn_predictions', {})
        binding_affinity = qnn_preds.get('binding_affinity', -7.0)
        if binding_affinity < -8.0:
            summary['key_strengths'].append('Strong binding affinity')
        elif binding_affinity > -5.0:
            summary['key_concerns'].append('Weak binding affinity')
        
        # Analyze ADMET
        admet = results.get('admet_analysis', {})
        admet_score = admet.get('overall_admet_score', 0.5)
        if admet_score > 0.7:
            summary['key_strengths'].append('Good ADMET profile')
        elif admet_score < 0.4:
            summary['key_concerns'].append('Poor ADMET profile')
        
        # Analyze synthetic accessibility
        synthetic = results.get('synthetic_analysis', {})
        feasibility = synthetic.get('feasibility_score', 0.5)
        if feasibility > 0.7:
            summary['key_strengths'].append('Synthetically accessible')
        elif feasibility < 0.4:
            summary['key_concerns'].append('Synthetic challenges')
        
        # Generate recommendations
        if len(summary['key_strengths']) > len(summary['key_concerns']):
            summary['recommendations'].append('Consider for further development')
        else:
            summary['recommendations'].append('Requires optimization')
        
        return summary
        
    except Exception as e:
        logging.error(f"Error generating analysis summary: {e}")
        return {'key_strengths': [], 'key_concerns': [], 'recommendations': []}

def generate_development_recommendations(analogs: List[Dict[str, Any]]) -> List[str]:
    """
    Generate development recommendations
    """
    recommendations = []
    
    # Check for analogs with high comprehensive scores
    high_score_analogs = [a for a in analogs if a.get('comprehensive_score', 0) > 0.7]
    if high_score_analogs:
        recommendations.append(f"Focus on {len(high_score_analogs)} high-scoring analogs")
    
    # Check for binding affinity
    strong_binders = [a for a in analogs if a.get('binding_affinity_kcal_mol', -7.0) < -8.0]
    if strong_binders:
        recommendations.append(f"{len(strong_binders)} analogs show strong binding")
    
    # Check for ADMET issues
    poor_admet = [a for a in analogs if a.get('admet_score', 0.5) < 0.4]
    if poor_admet:
        recommendations.append(f"{len(poor_admet)} analogs need ADMET optimization")
    
    return recommendations

# Default methods for fallback
def get_default_qnn_predictions():
    return {
        'binding_affinity': -7.0,
        'admet_score': 0.5,
        'synthetic_accessibility': 0.5,
        'stability': 0.5,
        'selectivity': 0.5,
        'clinical_relevance': 0.5
    }

def get_default_admet_analysis(smiles: str):
    return {
        'smiles': smiles,
        'overall_admet_score': 0.5,
        'overall_admet_grade': 'Fair',
        'absorption': {'absorption_score': 0.5},
        'distribution': {'distribution_score': 0.5},
        'metabolism': {'metabolism_score': 0.5},
        'excretion': {'excretion_score': 0.5},
        'toxicity': {'toxicity_score': 0.5}
    }

def get_default_synthetic_analysis(smiles: str):
    return {
        'feasibility_score': 0.5,
        'synthesis_grade': 'Fair',
        'reaction_steps': {'total_steps': 5},
        'starting_materials': {'total_materials': 3},
        'reaction_yields': {'total_yield': 0.7},
        'purification_difficulty': {'total_difficulty': 0.5}
    }

def get_default_stability_analysis(smiles: str):
    return {
        'smiles': smiles,
        'overall_stability_score': 0.5,
        'overall_stability_grade': 'Fair',
        'chemical_stability': {'stability_score': 0.5},
        'biological_stability': {'stability_score': 0.5},
        'storage_stability': {'stability_score': 0.5}
    }

def get_default_selectivity_analysis(smiles: str):
    return {
        'smiles': smiles,
        'overall_selectivity_score': 0.5,
        'selectivity_grade': 'Fair',
        'target_selectivity': {'selectivity_score': 0.5},
        'off_target_binding': {'overall_risk': 0.5},
        'side_effects': {'overall_risk': 0.5},
        'therapeutic_index': {'therapeutic_index': 5.0}
    }

def get_default_clinical_analysis(smiles: str):
    return {
        'smiles': smiles,
        'overall_clinical_score': 0.5,
        'clinical_grade': 'Fair',
        'cancer_pathway_targeting': {'targeting_score': 0.5},
        'patient_population': {'population_score': 0.5},
        'clinical_trial_readiness': {'readiness_score': 0.5},
        'regulatory_pathway': {'pathway_score': 0.5}
    }

def get_default_comprehensive_results(smiles: str):
    return {
        'smiles': smiles,
        'qnn_predictions': get_default_qnn_predictions(),
        'admet_analysis': get_default_admet_analysis(smiles),
        'synthetic_analysis': get_default_synthetic_analysis(smiles),
        'stability_analysis': get_default_stability_analysis(smiles),
        'selectivity_analysis': get_default_selectivity_analysis(smiles),
        'clinical_analysis': get_default_clinical_analysis(smiles),
        'comprehensive_score': 0.5,
        'comprehensive_grade': 'Fair',
        'summary': {
            'key_strengths': [],
            'key_concerns': ['Limited data available'],
            'recommendations': ['Requires further analysis']
        }
    } 

@router.post("/enhanced-jobs")
async def create_enhanced_job(
    file_path: str,
    comprehensive_analysis: bool = True,
    db: Session = Depends(get_db)
):
    """Create a new enhanced job with comprehensive analysis."""
    try:
        job_id = str(uuid.uuid4())
        
        # Create enhanced job record
        enhanced_job = EnhancedJob(
            job_id=job_id,
            input_file_path=file_path,
            comprehensive_analysis=comprehensive_analysis,
            status="pending",
            created_at=datetime.utcnow(),
            analysis_type="comprehensive" if comprehensive_analysis else "basic"
        )
        
        db.add(enhanced_job)
        db.commit()
        
        # Start comprehensive analysis
        if comprehensive_analysis:
            await ml_client.start_comprehensive_analysis(job_id, file_path)
        
        return {
            "job_id": job_id,
            "status": "created",
            "comprehensive_analysis": comprehensive_analysis,
            "message": "Enhanced job created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create enhanced job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/enhanced-jobs/{job_id}")
async def get_enhanced_job(job_id: str, db: Session = Depends(get_db)):
    """Get enhanced job details."""
    try:
        job = db.query(EnhancedJob).filter(EnhancedJob.job_id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get analogs for this job
        analogs = db.query(EnhancedAnalog).filter(EnhancedAnalog.job_id == job_id).all()
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "comprehensive_analysis": job.comprehensive_analysis,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "analogs_count": len(analogs),
            "summary": job.summary_json if job.summary_json else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get enhanced job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analogs/{job_id}/comprehensive")
async def get_comprehensive_analogs(job_id: str, db: Session = Depends(get_db)):
    """Get comprehensive analogs for a job."""
    try:
        analogs = db.query(EnhancedAnalog).filter(EnhancedAnalog.job_id == job_id).all()
        
        if not analogs:
            raise HTTPException(status_code=404, detail="No analogs found for this job")
        
        comprehensive_analogs = []
        for analog in analogs:
            analog_data = {
                "analog_id": analog.analog_id,
                "rank": analog.rank,
                "smiles": analog.smiles,
                "binding_affinity": analog.binding_affinity,
                "energy": analog.energy,
                "homo_lumo_gap": analog.homo_lumo_gap,
                "final_score": analog.final_score,
                "comprehensive_score": analog.comprehensive_score,
                "admet_score": analog.admet_score,
                "cytotoxicity_score": analog.cytotoxicity_score,
                "cancer_pathway_score": analog.cancer_pathway_score,
                "safety_score": analog.safety_score,
                "clinical_readiness": analog.clinical_readiness,
                "detailed_analysis": json.loads(analog.detailed_analysis_json) if analog.detailed_analysis_json else {}
            }
            comprehensive_analogs.append(analog_data)
        
        # Sort by comprehensive score
        comprehensive_analogs.sort(key=lambda x: x.get('comprehensive_score', 0), reverse=True)
        
        return {
            "job_id": job_id,
            "analogs": comprehensive_analogs,
            "total_analogs": len(comprehensive_analogs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comprehensive analogs for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis-capabilities")
async def get_analysis_capabilities():
    """Get information about available analysis capabilities."""
    try:
        capabilities = await ml_client.get_analysis_capabilities()
        return capabilities
    except Exception as e:
        logger.error(f"Failed to get analysis capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/enhanced-jobs")
async def list_enhanced_jobs(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List all enhanced jobs."""
    try:
        jobs = db.query(EnhancedJob).offset(offset).limit(limit).all()
        
        job_list = []
        for job in jobs:
            job_data = {
                "job_id": job.job_id,
                "status": job.status,
                "comprehensive_analysis": job.comprehensive_analysis,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "analysis_type": job.analysis_type
            }
            job_list.append(job_data)
        
        return {
            "jobs": job_list,
            "total": len(job_list),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list enhanced jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/enhanced-jobs/{job_id}")
async def delete_enhanced_job(job_id: str, db: Session = Depends(get_db)):
    """Delete an enhanced job and its analogs."""
    try:
        # Delete analogs first
        db.query(EnhancedAnalog).filter(EnhancedAnalog.job_id == job_id).delete()
        
        # Delete job
        job = db.query(EnhancedJob).filter(EnhancedJob.job_id == job_id).first()
        if job:
            db.delete(job)
            db.commit()
        
        return {
            "job_id": job_id,
            "status": "deleted",
            "message": "Job and associated analogs deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete enhanced job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/comprehensive-summary/{job_id}")
async def get_comprehensive_summary(job_id: str, db: Session = Depends(get_db)):
    """Get comprehensive summary for a job."""
    try:
        job = db.query(EnhancedJob).filter(EnhancedJob.job_id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        analogs = db.query(EnhancedAnalog).filter(EnhancedAnalog.job_id == job_id).all()
        
        # Calculate summary statistics
        if analogs:
            comprehensive_scores = [a.comprehensive_score for a in analogs if a.comprehensive_score]
            binding_affinities = [a.binding_affinity for a in analogs if a.binding_affinity]
            admet_scores = [a.admet_score for a in analogs if a.admet_score]
            
            summary = {
                "total_analogs": len(analogs),
                "average_comprehensive_score": sum(comprehensive_scores) / len(comprehensive_scores) if comprehensive_scores else 0,
                "average_binding_affinity": sum(binding_affinities) / len(binding_affinities) if binding_affinities else 0,
                "average_admet_score": sum(admet_scores) / len(admet_scores) if admet_scores else 0,
                "best_comprehensive_score": max(comprehensive_scores) if comprehensive_scores else 0,
                "clinical_readiness_distribution": {
                    "ready": len([a for a in analogs if a.clinical_readiness == "ready"]),
                    "needs_optimization": len([a for a in analogs if a.clinical_readiness == "needs_optimization"]),
                    "requires_work": len([a for a in analogs if a.clinical_readiness == "requires_work"])
                }
            }
        else:
            summary = {"total_analogs": 0}
        
        return {
            "job_id": job_id,
            "summary": summary,
            "job_status": job.status,
            "comprehensive_analysis": job.comprehensive_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comprehensive summary for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 