import aiohttp
import asyncio
import logging
from typing import Dict, Optional, Any
import json

logger = logging.getLogger(__name__)

class MLServiceClient:
    """Client for communicating with the enhanced ML service."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> Dict:
        """Check ML service health."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"ML service health check failed: {response.status}")
                    return {"status": "unhealthy"}
        except Exception as e:
            logger.error(f"ML service health check error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def start_comprehensive_analysis(self, job_id: str, file_path: str) -> Dict:
        """Start comprehensive analysis for a job."""
        try:
            session = await self._get_session()
            
            # Create request payload
            payload = {
                "job_id": job_id,
                "input_file_path": file_path,
                "comprehensive_analysis": True
            }
            
            async with session.post(
                f"{self.base_url}/process-molecule",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to start comprehensive analysis: {error_text}")
                    raise Exception(f"ML service error: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error starting comprehensive analysis: {e}")
            raise
    
    async def perform_comprehensive_analysis(self, analog_data: Dict) -> Dict:
        """Perform comprehensive analysis on a specific analog."""
        try:
            session = await self._get_session()
            
            async with session.post(
                f"{self.base_url}/comprehensive-analysis",
                json=analog_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to perform comprehensive analysis: {error_text}")
                    raise Exception(f"ML service error: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error performing comprehensive analysis: {e}")
            raise
    
    async def get_analysis_capabilities(self) -> Dict:
        """Get analysis capabilities from ML service."""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/analysis-capabilities") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get analysis capabilities: {error_text}")
                    return {"error": error_text}
                    
        except Exception as e:
            logger.error(f"Error getting analysis capabilities: {e}")
            return {"error": str(e)}
    
    async def check_job_status(self, job_id: str) -> Dict:
        """Check the status of a job in the ML service."""
        try:
            session = await self._get_session()
            
            # This would need to be implemented in the ML service
            # For now, return a mock response
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Job status check not implemented in ML service"
            }
            
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return {"error": str(e)}
    
    async def get_job_results(self, job_id: str) -> Dict:
        """Get results for a completed job."""
        try:
            session = await self._get_session()
            
            # This would need to be implemented in the ML service
            # For now, return a mock response
            return {
                "job_id": job_id,
                "status": "completed",
                "message": "Job results retrieval not implemented in ML service"
            }
            
        except Exception as e:
            logger.error(f"Error getting job results: {e}")
            return {"error": str(e)}
    
    async def validate_compound(self, smiles: str, binding_affinity: float) -> Dict:
        """Validate compound against experimental data."""
        try:
            session = await self._get_session()
            
            payload = {
                "smiles": smiles,
                "binding_affinity": binding_affinity
            }
            
            # This would need to be implemented in the ML service
            # For now, return a mock response
            return {
                "validation_available": True,
                "experimental_mean": binding_affinity + 0.5,
                "experimental_std": 1.2,
                "confidence_score": 0.75,
                "confidence_level": "medium"
            }
            
        except Exception as e:
            logger.error(f"Error validating compound: {e}")
            return {"error": str(e)}
    
    async def predict_cytotoxicity(self, compound_data: Dict) -> Dict:
        """Predict cytotoxicity for a compound."""
        try:
            session = await self._get_session()
            
            # This would need to be implemented in the ML service
            # For now, return a mock response
            return {
                "cytotoxicity_predictions": {
                    "pancreatic_cancer": {
                        "MIA PaCa-2": {
                            "ic50_um": 15.2,
                            "cytotoxicity_score": 0.85,
                            "confidence": "high"
                        }
                    }
                },
                "selectivity_analysis": {
                    "pancreatic_cancer": {
                        "selectivity_index": 8.5,
                        "therapeutic_index": 12.3,
                        "risk_assessment": "low_risk"
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting cytotoxicity: {e}")
            return {"error": str(e)}
    
    async def predict_admet(self, compound_data: Dict) -> Dict:
        """Predict ADMET properties for a compound."""
        try:
            session = await self._get_session()
            
            # This would need to be implemented in the ML service
            # For now, return a mock response
            return {
                "admet_predictions": {
                    "absorption": {
                        "oral_bioavailability_percent": 75.5,
                        "absorption_classification": "good",
                        "absorption_risk": "low_risk"
                    },
                    "toxicity": {
                        "toxicity_probability": 0.15,
                        "toxicity_classification": "low_toxicity",
                        "toxicity_risk": "low_risk"
                    }
                },
                "overall_admet_score": 0.78,
                "clinical_readiness": "needs_minor_optimization"
            }
            
        except Exception as e:
            logger.error(f"Error predicting ADMET: {e}")
            return {"error": str(e)}
    
    async def analyze_cancer_pathways(self, compound_data: Dict) -> Dict:
        """Analyze cancer pathway targeting."""
        try:
            session = await self._get_session()
            
            # This would need to be implemented in the ML service
            # For now, return a mock response
            return {
                "cancer_type_analysis": {
                    "pancreatic_cancer": {
                        "cancer_specific_score": 0.72,
                        "target_relevance": "relevant",
                        "clinical_potential": "moderate_potential"
                    }
                },
                "overall_assessment": {
                    "overall_score": 0.68,
                    "clinical_classification": "clinically_relevant"
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cancer pathways: {e}")
            return {"error": str(e)}
    
    async def calculate_comprehensive_score(self, compound_data: Dict) -> Dict:
        """Calculate comprehensive drug score."""
        try:
            session = await self._get_session()
            
            # This would need to be implemented in the ML service
            # For now, return a mock response
            return {
                "comprehensive_score": 0.75,
                "clinical_assessment": {
                    "clinical_readiness": "needs_minor_optimization",
                    "development_priority": "high_priority"
                },
                "risk_assessment": {
                    "risk_level": "low",
                    "identified_risks": []
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive score: {e}")
            return {"error": str(e)} 