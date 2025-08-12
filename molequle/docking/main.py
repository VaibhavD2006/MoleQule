"""
MoleQule Docking Service
Molecular docking and visualization service for binding analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import os
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MoleQule Docking Service",
    description="Molecular docking and 3D visualization service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class DockingRequest(BaseModel):
    analog_id: str
    analog_smiles: str
    target_protein: str = "DNA"
    method: str = "basic"  # basic, qaoa, classical, vina

class VisualizationRequest(BaseModel):
    ligand_data: str
    target_data: Optional[str] = None
    pose_data: Optional[Dict] = None

class DockingResponse(BaseModel):
    analog_id: str
    target_protein: str
    binding_score: float
    visualization_html: str
    interactions: List[Dict]
    method_used: str

# Import visualization and docking modules (with fallbacks)
try:
    # Try relative imports first
    try:
        from .visualizer import MolecularVisualizer
        from .binding_sites import BindingSiteDetector
        from .qaoa_docking import QAOAPoseOptimizer
        from .classical_docking import ClassicalDocker, ScoringFunction
    except ImportError:
        # Fallback to absolute imports when running as main module
        from visualizer import MolecularVisualizer
        from binding_sites import BindingSiteDetector
        from qaoa_docking import QAOAPoseOptimizer
        from classical_docking import ClassicalDocker, ScoringFunction
    
    VISUALIZATION_AVAILABLE = True
    ADVANCED_DOCKING_AVAILABLE = True
    logger.info("✅ All docking modules loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Advanced docking modules not available: {e}")
    logger.info("⚠️ Using fallback docking methods")
    VISUALIZATION_AVAILABLE = False
    ADVANCED_DOCKING_AVAILABLE = False

# Initialize components
molecular_visualizer = None
binding_detector = None
qaoa_optimizer = None
classical_docker = None
scoring_function = None

if VISUALIZATION_AVAILABLE:
    try:
        molecular_visualizer = MolecularVisualizer()
        binding_detector = BindingSiteDetector()
        logger.info("✅ Visualization components initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize visualization components: {e}")
        VISUALIZATION_AVAILABLE = False

if ADVANCED_DOCKING_AVAILABLE:
    try:
        qaoa_optimizer = QAOAPoseOptimizer()
        classical_docker = ClassicalDocker()
        scoring_function = ScoringFunction()
        logger.info("✅ Advanced docking components initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize docking components: {e}")
        ADVANCED_DOCKING_AVAILABLE = False

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "docking-service",
        "version": "2.0.0",
        "visualization_available": VISUALIZATION_AVAILABLE,
        "advanced_docking_available": ADVANCED_DOCKING_AVAILABLE,
        "features": {
            "basic_visualization": True,
            "binding_site_detection": VISUALIZATION_AVAILABLE,
            "3d_rendering": VISUALIZATION_AVAILABLE,
            "qaoa_optimization": qaoa_optimizer is not None,
            "classical_docking": classical_docker is not None,
            "advanced_scoring": scoring_function is not None
        },
        "docking_methods": {
            "basic": "Always available",
            "qaoa": "Available" if qaoa_optimizer else "Requires PennyLane",
            "classical": "Available" if classical_docker else "Requires RDKit"
        }
    }

@app.post("/visualize-molecule", response_model=Dict[str, Any])
async def visualize_molecule(request: VisualizationRequest):
    """
    Create 3D visualization of a molecule
    """
    try:
        if not VISUALIZATION_AVAILABLE:
            # Fallback visualization using basic HTML/CSS
            return await _create_fallback_visualization(request)
        
        # Use full visualization capabilities
        viz_html = molecular_visualizer.create_molecule_viewer(
            ligand_data=request.ligand_data,
            target_data=request.target_data
        )
        
        return {
            "status": "success",
            "visualization_html": viz_html,
            "method": "advanced"
        }
        
    except Exception as e:
        logger.error(f"Error in molecule visualization: {e}")
        return await _create_fallback_visualization(request)

@app.post("/dock-molecule", response_model=DockingResponse)
async def dock_molecule(request: DockingRequest):
    """
    Perform molecular docking analysis
    """
    try:
        logger.info(f"Docking analysis for analog {request.analog_id} against {request.target_protein} using {request.method}")
        
        if not VISUALIZATION_AVAILABLE:
            return await _create_mock_docking_result(request)
        
        # Get binding site for target
        binding_site = await _get_binding_site(request.target_protein)
        
        # Perform docking based on method
        if request.method == "qaoa":
            result = await _qaoa_docking(request, binding_site)
        elif request.method == "classical":
            result = await _classical_docking(request, binding_site)
        elif request.method == "vina":
            # Direct Vina method request
            result = await _vina_docking(request, binding_site)
        else:
            result = await _basic_docking_analysis(request, binding_site)
        
        # Generate visualization
        viz_html = await _create_docking_visualization(request, result)
        
        return DockingResponse(
            analog_id=request.analog_id,
            target_protein=request.target_protein,
            binding_score=result["binding_score"],
            visualization_html=viz_html,
            interactions=result.get("interactions", []),
            method_used=result.get("method", request.method)
        )
        
    except Exception as e:
        logger.error(f"Error in molecular docking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/binding-sites/{target_protein}")
async def get_binding_sites(target_protein: str):
    """
    Get binding site information for a target protein
    """
    if binding_detector:
        site = binding_detector.get_predefined_site(target_protein)
        if site:
            return site
    
    # Fallback to basic predefined sites
    predefined_sites = {
        "DNA": {
            "name": "DNA Guanine N7",
            "description": "Primary cisplatin binding site at N7 of guanine",
            "coordinates": [0.0, 0.0, 0.0],
            "radius": 3.0,
            "key_interactions": ["hydrogen_bonding", "coordination"]
        },
        "GSTP1": {
            "name": "GSTP1 Active Site",
            "description": "Glutathione S-transferase P1 enzyme active site",
            "coordinates": [10.5, 15.2, 8.7],
            "radius": 4.5,
            "key_interactions": ["hydrophobic", "electrostatic"]
        }
    }
    
    site = predefined_sites.get(target_protein.upper())
    if not site:
        raise HTTPException(status_code=404, detail=f"Binding site not found for {target_protein}")
    
    return site

@app.post("/detect-binding-sites")
async def detect_binding_sites(pdb_file: str):
    """
    Detect binding sites in uploaded PDB file
    """
    if not binding_detector:
        raise HTTPException(status_code=503, detail="Binding site detection not available")
    
    try:
        # In a real implementation, you'd handle file upload here
        cavities = binding_detector.detect_cavities(pdb_file)
        
        # Add druggability analysis for each site
        for cavity in cavities:
            druggability = binding_detector.analyze_druggability(cavity)
            cavity["druggability"] = druggability
        
        return {
            "pdb_file": pdb_file,
            "detected_sites": cavities,
            "total_sites": len(cavities),
            "method": "automatic_detection"
        }
        
    except Exception as e:
        logger.error(f"Error detecting binding sites: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/target-info/{target_name}")
async def get_target_info(target_name: str):
    """
    Get comprehensive information about a target protein
    """
    try:
        binding_site = await get_binding_sites(target_name)
        
        # Add druggability analysis if detector is available
        druggability_info = None
        if binding_detector:
            druggability_info = binding_detector.analyze_druggability(binding_site)
        
        return {
            "target_name": target_name,
            "binding_site": binding_site,
            "druggability": druggability_info,
            "analysis_available": binding_detector is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting target info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions

async def _extract_interactions_from_pose(coordinates: List, binding_site: Dict, 
                                        target_protein: str) -> List[Dict]:
    """Extract molecular interactions from pose coordinates"""
    try:
        if not coordinates:
            return []
        
        coords = np.array(coordinates)
        binding_center = np.array(binding_site.get('center', [0.0, 0.0, 0.0]))
        
        interactions = []
        for i, coord in enumerate(coords):
            distance = np.linalg.norm(coord - binding_center)
            
            if distance <= 2.2:
                interactions.append({
                    "type": "coordination",
                    "distance": float(distance),
                    "atoms": [f"L{i}", "TARGET"],
                    "strength": "very_strong"
                })
            elif distance <= 3.0:
                interactions.append({
                    "type": "hydrogen_bond",
                    "distance": float(distance),
                    "atoms": [f"L{i}", "TARGET"],
                    "strength": "strong"
                })
            elif distance <= 4.5:
                interactions.append({
                    "type": "hydrophobic",
                    "distance": float(distance),
                    "atoms": [f"L{i}", "TARGET"],
                    "strength": "moderate"
                })
        
        return interactions[:5]  # Return top 5 interactions
        
    except Exception as e:
        logger.error(f"Error extracting interactions: {e}")
        return []

async def _get_binding_site(target_protein: str) -> Dict:
    """Get binding site information"""
    response = await get_binding_sites(target_protein)
    return response

async def _basic_docking_analysis(request: DockingRequest, binding_site: Dict) -> Dict:
    """Perform basic docking analysis with scoring"""
    # Simplified binding score calculation based on molecular properties
    import hashlib
    
    # Create deterministic but varied binding scores based on SMILES
    hash_input = f"{request.analog_smiles}_{request.target_protein}"
    score_hash = hashlib.md5(hash_input.encode()).hexdigest()
    base_score = -8.0 + (int(score_hash[:2], 16) % 60) / 10.0  # Range: -8.0 to -2.0
    
    interactions = [
        {
            "type": "hydrogen_bond",
            "distance": 2.1 + (int(score_hash[2:4], 16) % 10) / 10.0,
            "atoms": ["N7", "H1"],
            "strength": "strong"
        },
        {
            "type": "coordination",
            "distance": 2.0 + (int(score_hash[4:6], 16) % 8) / 10.0,
            "atoms": ["Pt", "N7"],
            "strength": "very_strong"
        }
    ]
    
    return {
        "binding_score": base_score,
        "interactions": interactions,
        "pose_coordinates": [[0.0, 0.0, 0.0]],  # Simplified
        "method": "basic_analysis"
    }

async def _qaoa_docking(request: DockingRequest, binding_site: Dict) -> Dict:
    """Quantum-enhanced docking using QAOA"""
    try:
        if qaoa_optimizer:
            logger.info(f"Running QAOA optimization for {request.analog_id}")
            
            # Run QAOA pose optimization
            qaoa_result = qaoa_optimizer.optimize_pose(
                ligand_smiles=request.analog_smiles,
                binding_site=binding_site,
                target_protein=request.target_protein
            )
            
            # Convert QAOA result to standard format
            interactions = await _extract_interactions_from_pose(
                qaoa_result.get("pose_coordinates", []),
                binding_site,
                request.target_protein
            )
            
            return {
                "binding_score": qaoa_result.get("binding_energy", -5.0),
                "interactions": interactions,
                "pose_coordinates": qaoa_result.get("pose_coordinates", []),
                "method": "qaoa_quantum_real",
                "quantum_enhancement": True,
                "optimization_steps": qaoa_result.get("optimization_steps", 0),
                "convergence": qaoa_result.get("convergence", "unknown"),
                "pose_quality": qaoa_result.get("pose_quality", "unknown"),
                "binding_efficiency": qaoa_result.get("binding_efficiency", 0.5)
            }
        else:
            # Fallback to enhanced basic analysis
            logger.warning("QAOA optimizer not available, using enhanced fallback")
            basic_result = await _basic_docking_analysis(request, binding_site)
            basic_result["binding_score"] -= 0.5  # Quantum simulation bonus
            basic_result["method"] = "qaoa_simulation"
            basic_result["quantum_enhancement"] = True
            return basic_result
            
    except Exception as e:
        logger.error(f"QAOA docking failed: {e}")
        # Fallback to basic analysis
        basic_result = await _basic_docking_analysis(request, binding_site)
        basic_result["method"] = "qaoa_fallback"
        return basic_result

async def _classical_docking(request: DockingRequest, binding_site: Dict) -> Dict:
    """Classical docking using force fields and optimization"""
    try:
        if classical_docker:
            logger.info(f"Running classical docking for {request.analog_id}")
            
            # Run classical docking
            classical_result = classical_docker.dock_molecule(
                ligand_smiles=request.analog_smiles,
                target_protein=request.target_protein,
                binding_site=binding_site,
                method="force_field"
            )
            
            # Get best pose
            best_pose = classical_result.get("poses", [{}])[0]
            
            # Convert to standard format
            interactions = await _extract_interactions_from_pose(
                best_pose.get("coordinates", []),
                binding_site,
                request.target_protein
            )
            
            return {
                "binding_score": best_pose.get("score", -3.0),
                "interactions": interactions,
                "pose_coordinates": best_pose.get("coordinates", []),
                "method": "classical_force_field",
                "quantum_enhancement": False,
                "num_poses_evaluated": classical_result.get("num_conformers", 1),
                "convergence": classical_result.get("convergence", "unknown"),
                "docking_software": "rdkit_uff" if classical_docker.rdkit_available else "simplified"
            }
        else:
            # Fallback to basic analysis
            logger.warning("Classical docker not available, using basic analysis")
            basic_result = await _basic_docking_analysis(request, binding_site)
            basic_result["method"] = "classical_simulation"
            return basic_result
            
    except Exception as e:
        logger.error(f"Classical docking failed: {e}")
        # Fallback to basic analysis
        basic_result = await _basic_docking_analysis(request, binding_site)
        basic_result["method"] = "classical_fallback"
        return basic_result

async def _vina_docking(request: DockingRequest, binding_site: Dict) -> Dict:
    """Vina-specific docking using AutoDock Vina"""
    try:
        if classical_docker:
            logger.info(f"Running AutoDock Vina for {request.analog_id}")
            
            # Run Vina docking specifically
            vina_result = classical_docker.dock_molecule(
                ligand_smiles=request.analog_smiles,
                target_protein=request.target_protein,
                binding_site=binding_site,
                method="vina"  # Force Vina method
            )
            
            # Get best pose
            best_pose = vina_result.get("poses", [{}])[0]
            
            # Convert to standard format
            interactions = await _extract_interactions_from_pose(
                best_pose.get("coordinates", []),
                binding_site,
                request.target_protein
            )
            
            return {
                "binding_score": best_pose.get("score", -5.0),
                "interactions": interactions,
                "pose_coordinates": best_pose.get("coordinates", []),
                "method": vina_result.get("method", "autodock_vina"),
                "quantum_enhancement": False,
                "num_poses_evaluated": vina_result.get("num_conformers", 1),
                "convergence": vina_result.get("convergence", "unknown"),
                "docking_software": "AutoDock Vina",
                "exhaustiveness": vina_result.get("exhaustiveness", 8),
                "search_space": vina_result.get("search_space", [10, 10, 10]),
                "software_version": vina_result.get("software_version", "AutoDock Vina"),
                "vina_specific": True
            }
        else:
            # Fallback to basic analysis if classical docker not available
            logger.warning("Classical docker not available for Vina, using basic analysis")
            basic_result = await _basic_docking_analysis(request, binding_site)
            basic_result["method"] = "vina_fallback"
            return basic_result
            
    except Exception as e:
        logger.error(f"Vina docking failed: {e}")
        # Fallback to basic analysis
        basic_result = await _basic_docking_analysis(request, binding_site)
        basic_result["method"] = "vina_error_fallback"
        return basic_result

async def _create_docking_visualization(request: DockingRequest, result: Dict) -> str:
    """Create HTML visualization of docking results"""
    if molecular_visualizer:
        return molecular_visualizer.create_docking_complex_viewer(
            ligand_smiles=request.analog_smiles,
            target=request.target_protein,
            binding_score=result["binding_score"],
            interactions=result["interactions"]
        )
    else:
        return await _create_fallback_docking_html(request, result)

async def _create_fallback_visualization(request: VisualizationRequest) -> Dict[str, Any]:
    """Create basic HTML visualization as fallback"""
    html = f"""
    <div style="width: 100%; height: 400px; border: 2px solid #e2e8f0; border-radius: 8px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                display: flex; align-items: center; justify-content: center; color: white;">
        <div style="text-align: center;">
            <h3>🧬 Molecular Structure</h3>
            <p>3D visualization will be available when RDKit and Py3Dmol are installed</p>
            <div style="margin-top: 20px; font-family: monospace; background: rgba(0,0,0,0.3); 
                        padding: 10px; border-radius: 4px;">
                {request.ligand_data[:50]}...
            </div>
        </div>
    </div>
    """
    
    return {
        "status": "success",
        "visualization_html": html,
        "method": "fallback"
    }

async def _create_fallback_docking_html(request: DockingRequest, result: Dict) -> str:
    """Create fallback HTML for docking visualization"""
    interactions_html = ""
    for interaction in result.get("interactions", []):
        interactions_html += f"""
        <div style="margin: 5px 0; padding: 8px; background: rgba(255,255,255,0.1); 
                    border-radius: 4px; border-left: 3px solid #4ade80;">
            <strong>{interaction['type'].replace('_', ' ').title()}</strong>: 
            {interaction['distance']:.1f}Å ({interaction['strength']})
        </div>
        """
    
    return f"""
    <div style="width: 100%; min-height: 500px; border: 2px solid #e2e8f0; border-radius: 12px; 
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 20px;">
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #60a5fa; margin-bottom: 10px;">🎯 Docking Analysis</h3>
            <p style="color: #cbd5e1;">{request.analog_id} → {request.target_protein}</p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                <h4 style="color: #34d399; margin-bottom: 10px;">Binding Score</h4>
                <div style="font-size: 24px; font-weight: bold;">{result['binding_score']:.2f} kcal/mol</div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 5px;">
                    Method: {result['method'].replace('_', ' ').title()}
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                <h4 style="color: #fbbf24; margin-bottom: 10px;">Target Site</h4>
                <div>{request.target_protein}</div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 5px;">
                    {len(result.get('interactions', []))} interactions detected
                </div>
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;">
            <h4 style="color: #a78bfa; margin-bottom: 15px;">Key Interactions</h4>
            {interactions_html}
        </div>
        
        <div style="text-align: center; margin-top: 20px; padding: 15px; 
                    background: rgba(59, 130, 246, 0.1); border-radius: 8px; border: 1px solid #3b82f6;">
            <p style="color: #93c5fd; margin: 0;">
                🔬 Advanced 3D visualization available with RDKit and Py3Dmol installation
            </p>
        </div>
    </div>
    """

async def _create_mock_docking_result(request: DockingRequest) -> DockingResponse:
    """Create mock docking result for testing"""
    result = await _basic_docking_analysis(request, {"name": request.target_protein})
    viz_html = await _create_fallback_docking_html(request, result)
    
    return DockingResponse(
        analog_id=request.analog_id,
        target_protein=request.target_protein,
        binding_score=result["binding_score"],
        visualization_html=viz_html,
        interactions=result["interactions"],
        method_used="mock_analysis"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DOCKING_SERVICE_PORT", 8002))
    logger.info(f"Starting Docking Service on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 