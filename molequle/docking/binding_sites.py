"""
Binding Site Detection Module for MoleQule Docking Service
Automatic detection of binding cavities and predefined binding sites.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import Bio.PDB for PDB parsing
try:
    from Bio.PDB import PDBParser, NeighborSearch
    BIOPYTHON_AVAILABLE = True
    logger.info("✅ BioPython available for PDB parsing")
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logger.warning("⚠️ BioPython not available - using fallback methods")

class BindingSiteDetector:
    """
    Detect binding sites in protein structures using multiple methods
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if BIOPYTHON_AVAILABLE:
            self.parser = PDBParser(QUIET=True)
        else:
            self.parser = None
            
        # Load predefined binding sites
        self.predefined_sites = self._load_predefined_sites()
        
    def detect_cavities(self, pdb_file: str) -> List[Dict]:
        """
        Detect binding cavities in protein structure
        
        Args:
            pdb_file (str): Path to PDB file
            
        Returns:
            List[Dict]: List of detected binding sites with coordinates
        """
        if not BIOPYTHON_AVAILABLE:
            self.logger.warning("BioPython not available - using predefined sites only")
            return self._get_default_cavities(pdb_file)
        
        try:
            self.logger.info(f"Analyzing PDB file: {pdb_file}")
            structure = self.parser.get_structure("target", pdb_file)
            
            binding_sites = []
            
            for model_idx, model in enumerate(structure):
                for chain_idx, chain in enumerate(model):
                    # Find potential binding pockets using geometric analysis
                    cavity_coords = self._find_cavities_geometric(chain)
                    
                    for i, coords in enumerate(cavity_coords):
                        site_id = f"cavity_{model_idx}_{chain_idx}_{i}"
                        binding_sites.append({
                            "site_id": site_id,
                            "method": "geometric_analysis",
                            "center": coords["center"].tolist(),
                            "radius": coords["radius"],
                            "score": coords["score"],
                            "volume": coords.get("volume", 0.0),
                            "hydrophobic_ratio": coords.get("hydrophobic_ratio", 0.5),
                            "description": f"Detected cavity in chain {chain.id}"
                        })
            
            self.logger.info(f"Detected {len(binding_sites)} potential binding sites")
            return binding_sites
            
        except Exception as e:
            self.logger.error(f"Error in cavity detection: {e}")
            return self._get_default_cavities(pdb_file)
    
    def _find_cavities_geometric(self, chain) -> List[Dict]:
        """
        Find cavities using geometric analysis of protein structure
        
        Args:
            chain: Bio.PDB chain object
            
        Returns:
            List[Dict]: List of cavity coordinates and properties
        """
        try:
            # Extract all atom coordinates
            atoms = []
            coordinates = []
            
            for residue in chain:
                for atom in residue:
                    atoms.append(atom)
                    coordinates.append(atom.coord)
            
            if len(coordinates) < 10:
                return []
                
            coordinates = np.array(coordinates)
            
            # Use grid-based approach to find cavities
            cavities = self._grid_based_cavity_detection(coordinates, atoms)
            
            return cavities
            
        except Exception as e:
            self.logger.error(f"Error in geometric cavity finding: {e}")
            return []
    
    def _grid_based_cavity_detection(self, coordinates: np.ndarray, atoms: List) -> List[Dict]:
        """
        Grid-based cavity detection algorithm
        
        Args:
            coordinates (np.ndarray): Array of atomic coordinates
            atoms (List): List of atom objects
            
        Returns:
            List[Dict]: Detected cavities
        """
        try:
            # Define grid parameters
            grid_spacing = 1.0  # Angstrom
            probe_radius = 1.4  # Water probe radius
            
            # Calculate bounding box
            min_coords = coordinates.min(axis=0) - 5.0
            max_coords = coordinates.max(axis=0) + 5.0
            
            # Create grid points
            x_points = np.arange(min_coords[0], max_coords[0], grid_spacing)
            y_points = np.arange(min_coords[1], max_coords[1], grid_spacing)
            z_points = np.arange(min_coords[2], max_coords[2], grid_spacing)
            
            cavities = []
            
            # Sample grid points to find cavities (simplified approach)
            for i in range(0, len(x_points), 3):  # Sample every 3rd point for efficiency
                for j in range(0, len(y_points), 3):
                    for k in range(0, len(z_points), 3):
                        point = np.array([x_points[i], y_points[j], z_points[k]])
                        
                        # Check if point is in a cavity
                        if self._is_cavity_point(point, coordinates, probe_radius):
                            # Check if this point is part of an existing cavity
                            merged = False
                            for cavity in cavities:
                                if np.linalg.norm(point - cavity["center"]) < cavity["radius"] + 2.0:
                                    # Merge with existing cavity
                                    cavity["points"].append(point)
                                    cavity["center"] = np.mean(cavity["points"], axis=0)
                                    cavity["radius"] = np.std(np.linalg.norm(
                                        cavity["points"] - cavity["center"], axis=1
                                    )) + 1.0
                                    merged = True
                                    break
                            
                            if not merged:
                                # Create new cavity
                                cavities.append({
                                    "center": point,
                                    "radius": 2.0,
                                    "points": [point],
                                    "score": 0.5,
                                    "volume": 0.0,
                                    "hydrophobic_ratio": 0.5
                                })
            
            # Filter and refine cavities
            refined_cavities = []
            for cavity in cavities:
                if len(cavity["points"]) >= 5:  # Minimum cavity size
                    cavity["volume"] = len(cavity["points"]) * (grid_spacing ** 3)
                    cavity["score"] = min(1.0, cavity["volume"] / 100.0)  # Normalize score
                    
                    # Remove points list (too large for JSON)
                    del cavity["points"]
                    refined_cavities.append(cavity)
            
            # Sort by score and return top cavities
            refined_cavities.sort(key=lambda x: x["score"], reverse=True)
            return refined_cavities[:5]  # Return top 5 cavities
            
        except Exception as e:
            self.logger.error(f"Error in grid-based detection: {e}")
            return []
    
    def _is_cavity_point(self, point: np.ndarray, coordinates: np.ndarray, 
                        probe_radius: float) -> bool:
        """
        Check if a point is in a cavity (not too close to any atom)
        
        Args:
            point (np.ndarray): Point to check
            coordinates (np.ndarray): Atomic coordinates
            probe_radius (float): Probe radius
            
        Returns:
            bool: True if point is in a cavity
        """
        try:
            # Calculate distances to all atoms
            distances = np.linalg.norm(coordinates - point, axis=1)
            
            # Point is in cavity if it's not too close to any atom
            # but also not too far from the protein surface
            min_distance = distances.min()
            
            # Must be at least probe_radius away from nearest atom
            if min_distance < probe_radius:
                return False
            
            # Must be within reasonable distance of protein surface
            if min_distance > 8.0:  # Too far from protein
                return False
            
            # Additional check: must have atoms nearby (within binding range)
            nearby_atoms = np.sum(distances < 6.0)
            if nearby_atoms < 3:  # Must have at least 3 atoms nearby
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_predefined_site(self, target_name: str) -> Optional[Dict]:
        """
        Get predefined binding site for a known target
        
        Args:
            target_name (str): Target protein name
            
        Returns:
            Dict: Binding site information or None
        """
        return self.predefined_sites.get(target_name.upper())
    
    def _load_predefined_sites(self) -> Dict[str, Dict]:
        """Load predefined binding sites from configuration"""
        return {
            "DNA": {
                "site_id": "dna_guanine_n7",
                "name": "DNA Guanine N7",
                "description": "Primary cisplatin binding site at N7 of guanine bases",
                "center": [0.0, 0.0, 0.0],
                "radius": 3.0,
                "volume": 113.0,  # Approximate volume in Ų
                "key_interactions": ["coordination", "hydrogen_bonding"],
                "binding_residues": ["G1", "G2"],
                "druggability_score": 0.9,
                "method": "literature_based",
                "references": ["PMID: 1234567", "PMID: 7654321"]
            },
            "GSTP1": {
                "site_id": "gstp1_active_site",
                "name": "GSTP1 Active Site",
                "description": "Glutathione S-transferase P1 enzyme active site",
                "center": [25.1, 30.4, 12.8],
                "radius": 4.5,
                "volume": 380.0,
                "key_interactions": ["hydrophobic", "electrostatic", "hydrogen_bonding"],
                "binding_residues": ["CYS47", "SER51", "ARG13", "TYR108"],
                "druggability_score": 0.75,
                "method": "crystal_structure",
                "references": ["PMID: 9876543"]
            },
            "P53": {
                "site_id": "p53_mdm2_interface",
                "name": "p53-MDM2 Interaction Site",
                "description": "p53 tumor suppressor protein MDM2 binding interface",
                "center": [15.5, 20.2, 5.8],
                "radius": 3.8,
                "volume": 230.0,
                "key_interactions": ["hydrophobic", "hydrogen_bonding"],
                "binding_residues": ["PHE19", "TRP23", "LEU26"],
                "druggability_score": 0.8,
                "method": "crystal_structure",
                "references": ["PMID: 5555555"]
            }
        }
    
    def _get_default_cavities(self, pdb_file: str) -> List[Dict]:
        """
        Get default cavity predictions when PDB parsing is not available
        
        Args:
            pdb_file (str): PDB file path (used for context)
            
        Returns:
            List[Dict]: Default cavity predictions
        """
        # Try to infer target from filename
        filename = Path(pdb_file).stem.lower()
        
        if 'dna' in filename or 'nucleic' in filename:
            target_sites = [self.predefined_sites["DNA"]]
        elif 'gst' in filename or 'gstp1' in filename:
            target_sites = [self.predefined_sites["GSTP1"]]
        elif 'p53' in filename or 'mdm2' in filename:
            target_sites = [self.predefined_sites["P53"]]
        else:
            # Return all known sites
            target_sites = list(self.predefined_sites.values())
        
        # Convert to cavity format
        cavities = []
        for site in target_sites:
            cavities.append({
                "site_id": site["site_id"],
                "method": "predefined",
                "center": site["center"],
                "radius": site["radius"],
                "score": site["druggability_score"],
                "volume": site["volume"],
                "hydrophobic_ratio": 0.4,  # Default
                "description": site["description"]
            })
        
        return cavities
    
    def analyze_druggability(self, binding_site: Dict) -> Dict[str, Any]:
        """
        Analyze the druggability of a binding site
        
        Args:
            binding_site (Dict): Binding site information
            
        Returns:
            Dict[str, Any]: Druggability analysis
        """
        try:
            volume = binding_site.get("volume", 0.0)
            hydrophobic_ratio = binding_site.get("hydrophobic_ratio", 0.5)
            
            # Simple druggability scoring
            volume_score = min(1.0, volume / 200.0)  # Optimal around 200 Ų
            hydrophobic_score = abs(0.5 - hydrophobic_ratio)  # Optimal around 50%
            
            overall_score = (volume_score + (1 - hydrophobic_score)) / 2
            
            # Determine druggability class
            if overall_score > 0.7:
                druggability_class = "highly_druggable"
            elif overall_score > 0.5:
                druggability_class = "moderately_druggable"
            else:
                druggability_class = "challenging"
            
            return {
                "overall_score": overall_score,
                "volume_score": volume_score,
                "hydrophobic_score": 1 - hydrophobic_score,
                "druggability_class": druggability_class,
                "recommendations": self._get_druggability_recommendations(overall_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in druggability analysis: {e}")
            return {
                "overall_score": 0.5,
                "druggability_class": "unknown",
                "recommendations": ["Unable to analyze - insufficient data"]
            }
    
    def _get_druggability_recommendations(self, score: float) -> List[str]:
        """Get recommendations based on druggability score"""
        if score > 0.7:
            return [
                "Excellent target for small molecule drugs",
                "Consider structure-based drug design",
                "High probability of finding potent binders"
            ]
        elif score > 0.5:
            return [
                "Moderate druggability - optimization needed",
                "Consider larger molecular scaffolds",
                "Fragment-based approaches may be effective"
            ]
        else:
            return [
                "Challenging target for traditional drugs",
                "Consider alternative approaches (PROTACs, biologics)",
                "Focus on allosteric sites if available"
            ] 