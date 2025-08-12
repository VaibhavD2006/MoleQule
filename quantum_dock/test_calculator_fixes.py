#!/usr/bin/env python3
"""
Test script to verify VQE runner fixes for cisplatin analogs.
Tests calculator selection and fallback mechanisms.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from vqe_engine.vqe_runner import _get_best_ase_calculator, ASE_AVAILABLE, EMT_SUPPORTED_ELEMENTS
from vqe_engine.vqe_runner import _estimate_energy_from_atoms, _estimate_dipole_moment, _estimate_homo_lumo_gap
import numpy as np

def test_calculator_selection():
    """Test ASE calculator selection for different atomic compositions."""
    print("Testing ASE calculator selection...")
    
    if not ASE_AVAILABLE:
        print("ASE not available - skipping calculator tests")
        return
    
    # Test cases
    test_molecules = [
        (['H', 'H'], "Simple H2 molecule"),
        (['C', 'H', 'H', 'H', 'H'], "Methane"),
        (['Pt', 'Cl', 'Cl', 'N', 'H', 'H', 'H'], "Cisplatin-like"),
        (['Pt', 'N', 'N', 'Cl', 'Cl'], "Cisplatin"),
        (['Au', 'Cl', 'Cl', 'Cl'], "Gold complex"),
        (['Unknown', 'Element'], "Unknown elements")
    ]
    
    for atoms, description in test_molecules:
        print(f"\n{description}: {atoms}")
        
        # Check EMT support
        element_set = set(atoms)
        emt_supported = element_set.issubset(EMT_SUPPORTED_ELEMENTS)
        print(f"  EMT supported: {emt_supported}")
        
        # Test calculator selection
        try:
            calc = _get_best_ase_calculator(atoms)
            if calc is not None:
                calc_type = type(calc).__name__
                print(f"  Selected calculator: {calc_type}")
            else:
                print(f"  No calculator available")
        except Exception as e:
            print(f"  Calculator selection error: {e}")


def test_estimation_functions():
    """Test enhanced estimation functions."""
    print("\n" + "="*50)
    print("Testing estimation functions...")
    
    # Test molecules
    test_molecules = [
        (['Pt', 'N', 'N', 'Cl', 'Cl'], np.array([
            [0.0, 0.0, 0.0],    # Pt
            [2.0, 0.0, 0.0],    # N
            [-2.0, 0.0, 0.0],   # N  
            [0.0, 2.0, 0.0],    # Cl
            [0.0, -2.0, 0.0]    # Cl
        ]), "Cisplatin"),
        
        (['C', 'H', 'H', 'H', 'H'], np.array([
            [0.0, 0.0, 0.0],    # C
            [1.0, 1.0, 1.0],    # H
            [-1.0, 1.0, -1.0],  # H
            [1.0, -1.0, -1.0],  # H
            [-1.0, -1.0, 1.0]   # H
        ]), "Methane"),
        
        (['Au', 'Cl', 'Cl', 'Cl'], np.array([
            [0.0, 0.0, 0.0],    # Au
            [2.0, 0.0, 0.0],    # Cl
            [0.0, 2.0, 0.0],    # Cl
            [0.0, 0.0, 2.0]     # Cl
        ]), "Gold complex")
    ]
    
    for atoms, coordinates, description in test_molecules:
        print(f"\n{description}:")
        
        # Test energy estimation
        try:
            energy = _estimate_energy_from_atoms(atoms)
            print(f"  Estimated energy: {energy:.2f} Hartree")
        except Exception as e:
            print(f"  Energy estimation error: {e}")
        
        # Test HOMO-LUMO gap estimation
        try:
            gap = _estimate_homo_lumo_gap(atoms)
            print(f"  Estimated HOMO-LUMO gap: {gap:.2f} eV")
        except Exception as e:
            print(f"  HOMO-LUMO estimation error: {e}")
        
        # Test dipole moment estimation
        try:
            dipole = _estimate_dipole_moment(atoms, coordinates)
            print(f"  Estimated dipole moment: {dipole:.2f} Debye")
        except Exception as e:
            print(f"  Dipole estimation error: {e}")


def test_element_support():
    """Test element support information."""
    print("\n" + "="*50)
    print("Element support information:")
    
    cisplatin_elements = {'Pt', 'N', 'H', 'Cl'}
    print(f"Cisplatin elements: {cisplatin_elements}")
    print(f"EMT supports all cisplatin elements: {cisplatin_elements.issubset(EMT_SUPPORTED_ELEMENTS)}")
    
    common_drug_elements = {'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'}
    print(f"Common drug elements: {common_drug_elements}")
    print(f"EMT supports all common drug elements: {common_drug_elements.issubset(EMT_SUPPORTED_ELEMENTS)}")
    
    unsupported = common_drug_elements - EMT_SUPPORTED_ELEMENTS
    if unsupported:
        print(f"EMT unsupported elements: {unsupported}")


def main():
    """Run all tests."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("VQE Runner Calculator Fixes Test")
    print("=" * 50)
    
    # Test element support
    test_element_support()
    
    # Test calculator selection
    test_calculator_selection()
    
    # Test estimation functions
    test_estimation_functions()
    
    print("\n" + "="*50)
    print("Test completed!")


if __name__ == "__main__":
    main() 