"""
VQE Engine Utilities
Geometry parsing, xyz generation, and PySCF setup utilities.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from pyscf import gto
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


def parse_xyz_file(xyz_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Parse XYZ file to extract atomic symbols and coordinates.
    
    Args:
        xyz_path (str): Path to XYZ file
        
    Returns:
        Tuple[List[str], np.ndarray]: Atomic symbols and coordinates
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
        
        # Parse number of atoms
        n_atoms = int(lines[0].strip())
        
        atoms = []
        coordinates = []
        
        # Parse atomic data (skip first two lines)
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return atoms, np.array(coordinates)
        
    except Exception as e:
        logger.error(f"Error parsing XYZ file {xyz_path}: {e}")
        raise


def generate_xyz_from_smiles(smiles: str, output_path: str) -> str:
    """
    Generate XYZ file from SMILES string.
    
    Args:
        smiles (str): SMILES string
        output_path (str): Output XYZ file path
        
    Returns:
        str: Path to generated XYZ file
    """
    logger = logging.getLogger(__name__)
    
    if not RDKIT_AVAILABLE:
        logger.error("RDKit not available for SMILES to XYZ conversion")
        raise ImportError("RDKit is required for SMILES to XYZ conversion")
    
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write XYZ file
        xyz_block = Chem.MolToXYZBlock(mol)
        with open(output_path, 'w') as f:
            f.write(xyz_block)
        
        logger.info(f"Generated XYZ file: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating XYZ from SMILES: {e}")
        raise


def setup_pyscf_molecule(atoms: List[str], coordinates: np.ndarray, 
                        basis: str = "6-31g") -> Any:
    """
    Setup PySCF molecule object.
    
    Args:
        atoms (List[str]): List of atomic symbols
        coordinates (np.ndarray): Atomic coordinates
        basis (str): Basis set name
        
    Returns:
        Molecule object for PySCF calculations
    """
    logger = logging.getLogger(__name__)
    
    if not PYSCF_AVAILABLE:
        logger.error("PySCF not available for molecule setup")
        raise ImportError("PySCF is required for quantum chemistry calculations")
    
    try:
        # Build molecule string
        mol_str = ""
        for i, atom in enumerate(atoms):
            mol_str += f"{atom} {coordinates[i][0]:.6f} {coordinates[i][1]:.6f} {coordinates[i][2]:.6f}; "
        
        # Create molecule object
        mol = gto.Mole()
        mol.atom = mol_str
        mol.basis = basis
        mol.spin = 0  # Assuming closed shell
        mol.charge = 0  # Assuming neutral
        mol.build()
        
        logger.debug(f"PySCF molecule created with {len(atoms)} atoms")
        return mol
        
    except Exception as e:
        logger.error(f"Error setting up PySCF molecule: {e}")
        raise


def validate_xyz_file(xyz_path: str) -> bool:
    """
    Validate XYZ file format and content.
    
    Args:
        xyz_path (str): Path to XYZ file
        
    Returns:
        bool: True if valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not Path(xyz_path).exists():
            logger.error(f"XYZ file not found: {xyz_path}")
            return False
        
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            logger.error("XYZ file too short")
            return False
        
        # Check first line (number of atoms)
        try:
            n_atoms = int(lines[0].strip())
        except ValueError:
            logger.error("Invalid number of atoms in XYZ file")
            return False
        
        # Check if we have enough lines
        if len(lines) < n_atoms + 2:
            logger.error("XYZ file has insufficient lines")
            return False
        
        # Validate atomic data
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            if len(parts) < 4:
                logger.error(f"Invalid atomic data at line {i + 1}")
                return False
            
            # Check if coordinates are valid numbers
            try:
                float(parts[1])
                float(parts[2])
                float(parts[3])
            except ValueError:
                logger.error(f"Invalid coordinates at line {i + 1}")
                return False
        
        logger.debug(f"XYZ file validation passed: {xyz_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating XYZ file: {e}")
        return False


def get_molecular_properties(atoms: List[str], coordinates: np.ndarray) -> Dict[str, Any]:
    """
    Calculate basic molecular properties.
    
    Args:
        atoms (List[str]): List of atomic symbols
        coordinates (np.ndarray): Atomic coordinates
        
    Returns:
        Dict[str, Any]: Dictionary of molecular properties
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Basic properties
        properties = {
            'n_atoms': len(atoms),
            'atomic_symbols': atoms,
            'center_of_mass': _calculate_center_of_mass(atoms, coordinates),
            'molecular_formula': _get_molecular_formula(atoms),
            'total_charge': _estimate_total_charge(atoms)
        }
        
        # Geometric properties
        properties['bond_lengths'] = _calculate_bond_lengths(coordinates)
        properties['molecular_size'] = _calculate_molecular_size(coordinates)
        
        return properties
        
    except Exception as e:
        logger.error(f"Error calculating molecular properties: {e}")
        return {}


def _calculate_center_of_mass(atoms: List[str], coordinates: np.ndarray) -> np.ndarray:
    """Calculate center of mass."""
    atomic_masses = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'F': 18.998, 'Cl': 35.45, 'Br': 79.904, 'I': 126.90,
        'Pt': 195.08
    }
    
    masses = np.array([atomic_masses.get(atom, 12.0) for atom in atoms])
    total_mass = np.sum(masses)
    
    com = np.sum(coordinates * masses.reshape(-1, 1), axis=0) / total_mass
    return com


def _get_molecular_formula(atoms: List[str]) -> str:
    """Get molecular formula."""
    from collections import Counter
    
    atom_counts = Counter(atoms)
    
    # Sort by conventional order
    formula_order = ['C', 'H', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'Pt']
    
    formula = ""
    for atom in formula_order:
        if atom in atom_counts:
            count = atom_counts[atom]
            if count == 1:
                formula += atom
            else:
                formula += f"{atom}{count}"
    
    # Add any remaining atoms
    for atom, count in atom_counts.items():
        if atom not in formula_order:
            if count == 1:
                formula += atom
            else:
                formula += f"{atom}{count}"
    
    return formula


def _estimate_total_charge(atoms: List[str]) -> int:
    """Estimate total molecular charge."""
    # Simplified charge estimation
    # In practice, would need more sophisticated approach
    return 0


def _calculate_bond_lengths(coordinates: np.ndarray) -> List[float]:
    """Calculate all pairwise distances."""
    n_atoms = len(coordinates)
    bond_lengths = []
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            bond_lengths.append(distance)
    
    return bond_lengths


def _calculate_molecular_size(coordinates: np.ndarray) -> float:
    """Calculate molecular size as maximum distance."""
    if len(coordinates) < 2:
        return 0.0
    
    max_distance = 0.0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            max_distance = max(max_distance, distance)
    
    return max_distance 