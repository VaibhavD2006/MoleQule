"""
Analog Generator Module for QuantumDock
Generates cisplatin analogs with systematic substitutions.
"""

import logging
from typing import Dict, List, Any
from pathlib import Path
import uuid

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    import warnings
    # Suppress RDKit warnings about UFF atom types
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    logging.warning("RDKit not installed. Molecular generation will be limited.")
    Chem = None


def _safe_optimize_molecule(mol):
    """
    Safely optimize molecule geometry, handling problematic atom types like Pt.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        bool: True if optimization was successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if not mol:
        return False
    
    try:
        # Check for problematic atoms that UFF doesn't handle well
        problematic_atoms = {'Pt', 'Au', 'Pd', 'Ir', 'Rh', 'Ru', 'Os'}
        atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        has_problematic_atoms = bool(atom_symbols & problematic_atoms)
        
        if has_problematic_atoms:
            logger.debug(f"Molecule contains problematic atoms for UFF: {atom_symbols & problematic_atoms}")
            # For metal complexes, just embed without optimization
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == 0:
                logger.debug("Successfully embedded molecule without UFF optimization")
                return True
            else:
                logger.warning("Failed to embed molecule even without optimization")
                return False
        else:
            # For organic molecules, use UFF optimization
            embed_result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if embed_result == 0:
                # Suppress UFF warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    opt_result = AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                    if opt_result == 0:
                        logger.debug("Successfully optimized molecule with UFF")
                        return True
                    else:
                        logger.debug("UFF optimization did not converge, but embedded coordinates are usable")
                        return True
            else:
                logger.warning("Failed to embed molecule")
                return False
                
    except Exception as e:
        logger.error(f"Error in molecule optimization: {e}")
        return False


def generate_analogs(base_smiles: str, substitutions: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Generate cisplatin analogs based on context file substitutions.
    
    Args:
        base_smiles (str): Base SMILES string for cisplatin
        substitutions (Dict[str, List[str]]): Dictionary of substitution patterns
        
    Returns:
        List[Dict[str, Any]]: List of generated analog structures
    """
    logger = logging.getLogger(__name__)
    
    if not Chem:
        logger.error("RDKit not available. Cannot generate analogs.")
        return []
    
    analogs = []
    
    try:
        # Parse base molecule
        base_mol = Chem.MolFromSmiles(base_smiles)
        if not base_mol:
            logger.error(f"Invalid base SMILES: {base_smiles}")
            return []
        
        logger.info(f"Generating analogs from base: {base_smiles}")
        
        # Generate analogs by substitution
        for substitution_type, replacement_groups in substitutions.items():
            for replacement in replacement_groups:
                try:
                    # Generate analog (simplified approach)
                    analog_smiles = _apply_substitution(base_smiles, substitution_type, replacement)
                    
                    if analog_smiles:
                        analog_mol = Chem.MolFromSmiles(analog_smiles)
                        
                        if analog_mol and validate_molecule(analog_mol):
                            # Generate 3D coordinates
                            xyz_path = _generate_3d_structure(analog_mol, replacement)
                            
                            analog_data = {
                                "id": str(uuid.uuid4()),
                                "smiles": analog_smiles,
                                "substitution_type": substitution_type,
                                "replacement": replacement,
                                "xyz_path": xyz_path,
                                "molecular_formula": CalcMolFormula(analog_mol),
                                "molecular_weight": Descriptors.MolWt(analog_mol)
                            }
                            
                            analogs.append(analog_data)
                            logger.debug(f"Generated analog: {analog_smiles}")
                
                except Exception as e:
                    logger.warning(f"Error generating analog with {replacement}: {e}")
                    continue
        
        logger.info(f"Generated {len(analogs)} valid analogs")
        return analogs
        
    except Exception as e:
        logger.error(f"Error in analog generation: {e}")
        return []


def _apply_substitution(base_smiles: str, substitution_type: str, replacement: str) -> str:
    """
    Apply substitution to base SMILES string.
    
    Args:
        base_smiles (str): Base SMILES string
        substitution_type (str): Type of substitution (e.g., 'ammine', 'chloride')
        replacement (str): Replacement group
        
    Returns:
        str: Modified SMILES string
    """
    # Simplified substitution logic
    # In a real implementation, this would use more sophisticated chemistry
    
    substitution_map = {
        "ammine": {
            "methyl": base_smiles.replace("N", "NC"),
            "ethyl": base_smiles.replace("N", "NCC"),
            "propyl": base_smiles.replace("N", "NCCC"),
            "isopropyl": base_smiles.replace("N", "NC(C)C"),
            "tert-butyl": base_smiles.replace("N", "NC(C)(C)C")
        },
        "chloride": {
            "bromide": base_smiles.replace("Cl", "Br"),
            "iodide": base_smiles.replace("Cl", "I"),
            "acetate": base_smiles.replace("Cl", "OC(=O)C"),
            "nitrate": base_smiles.replace("Cl", "ONO2"),
            "hydroxide": base_smiles.replace("Cl", "O")
        }
    }
    
    if substitution_type in substitution_map and replacement in substitution_map[substitution_type]:
        return substitution_map[substitution_type][replacement]
    
    return base_smiles


def smiles_to_xyz(smiles: str) -> str:
    """
    Convert SMILES string to XYZ coordinates.
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        str: XYZ coordinate string
    """
    if not Chem:
        return ""
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return ""
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates with safe optimization
        if not _safe_optimize_molecule(mol):
            logging.getLogger(__name__).warning(f"Failed to optimize molecule: {smiles}")
            return ""
        
        # Convert to XYZ format
        xyz_block = Chem.MolToXYZBlock(mol)
        return xyz_block
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error converting SMILES to XYZ: {e}")
        return ""


def _generate_3d_structure(mol, replacement: str) -> str:
    """
    Generate 3D structure file for molecule.
    
    Args:
        mol: RDKit molecule object
        replacement (str): Replacement group identifier
        
    Returns:
        str: Path to generated XYZ file
    """
    try:
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates with safe optimization
        if not _safe_optimize_molecule(mol):
            logging.getLogger(__name__).warning(f"Failed to optimize molecule for replacement: {replacement}")
            return ""
        
        # Create filename
        filename = f"analog_{replacement}_{uuid.uuid4().hex[:8]}.xyz"
        filepath = Path("molecules/generated_analogs") / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write XYZ file
        xyz_block = Chem.MolToXYZBlock(mol)
        with open(filepath, 'w') as f:
            f.write(xyz_block)
        
        return str(filepath)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error generating 3D structure: {e}")
        return ""


def validate_molecule(mol_structure) -> bool:
    """
    Validate molecule structure.
    
    Args:
        mol_structure: RDKit molecule object
        
    Returns:
        bool: True if molecule is valid, False otherwise
    """
    if not mol_structure:
        return False
    
    try:
        # Basic validation checks
        if mol_structure.GetNumAtoms() == 0:
            return False
        
        # Check for reasonable molecular weight
        mw = Descriptors.MolWt(mol_structure)
        if mw < 50 or mw > 2000:  # Reasonable range for drug-like molecules
            return False
        
        # Check for presence of platinum (essential for cisplatin analogs)
        has_platinum = any(atom.GetSymbol() == 'Pt' for atom in mol_structure.GetAtoms())
        if not has_platinum:
            logging.getLogger(__name__).warning("Molecule does not contain platinum")
            return False
        
        # Additional validation can be added here
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error validating molecule: {e}")
        return False


def calculate_diversity_score(analogs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate diversity scores for generated analogs.
    
    Args:
        analogs (List[Dict[str, Any]]): List of analog structures
        
    Returns:
        Dict[str, float]: Diversity scores for each analog
    """
    if not Chem:
        return {}
    
    diversity_scores = {}
    
    try:
        # Calculate Tanimoto similarity between analogs
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors
        
        fps = []
        for analog in analogs:
            mol = Chem.MolFromSmiles(analog["smiles"])
            if mol:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
                fps.append((analog["id"], fp))
        
        # Calculate diversity as 1 - average similarity to all others
        for i, (analog_id, fp1) in enumerate(fps):
            similarities = []
            for j, (_, fp2) in enumerate(fps):
                if i != j:
                    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                    similarities.append(sim)
            
            if similarities:
                diversity_scores[analog_id] = 1.0 - (sum(similarities) / len(similarities))
            else:
                diversity_scores[analog_id] = 1.0
        
        return diversity_scores
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error calculating diversity scores: {e}")
        return {} 