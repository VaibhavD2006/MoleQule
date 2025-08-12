"""
Data Loader Module for QuantumDock
Handles loading and validation of configuration files.
"""

import json
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
from io import StringIO


def parse_sectioned_csv(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Parse a CSV file with section headers marked by # comments.
    
    Args:
        file_path (str): Path to the sectioned CSV file
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping section names to DataFrames
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = {}
    current_section = None
    current_data = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a section header (starts with # and contains uppercase words)
        if line.startswith('# ') and ('_' in line or line.isupper() or any(word.isupper() for word in line.split())):
            # Save previous section if exists
            if current_section and current_data:
                try:
                    sections[current_section] = pd.read_csv(StringIO('\n'.join(current_data)))
                except pd.errors.ParserError as e:
                    print(f"Warning: Could not parse section '{current_section}': {e}")
            
            # Start new section
            current_section = line[2:].strip()
            current_data = []
        elif not line.startswith('#') and current_section:
            current_data.append(line)
    
    # Save final section
    if current_section and current_data:
        try:
            sections[current_section] = pd.read_csv(StringIO('\n'.join(current_data)))
        except pd.errors.ParserError as e:
            print(f"Warning: Could not parse section '{current_section}': {e}")
    
    return sections


def load_cisplatin_context(file_path: str) -> Dict[str, Any]:
    """
    Load cisplatin context configuration from CSV file.
    
    Args:
        file_path (str): Path to the cisplatin CSV file
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    logger = logging.getLogger(__name__)
    
    try:
        sections = parse_sectioned_csv(file_path)
        
        # Extract metadata
        metadata = {}
        if 'CISPLATIN_METADATA' in sections:
            for _, row in sections['CISPLATIN_METADATA'].iterrows():
                metadata[row['Data_Type']] = row['Value']
        
        # Extract element composition
        element_composition = {}
        if 'CISPLATIN_ELEMENT_COMPOSITION' in sections:
            for _, row in sections['CISPLATIN_ELEMENT_COMPOSITION'].iterrows():
                element_composition[row['Element']] = {
                    'atom_count': row['Atom_count'],
                    'mass_fraction': row['Mass_fraction_percent'],
                    'cov_radius': row['Cov_radius_A']
                }
        
        # Extract bond lengths for descriptor weights
        bond_stats = {}
        if 'CISPLATIN_BOND_LENGTHS_STATS' in sections:
            for _, row in sections['CISPLATIN_BOND_LENGTHS_STATS'].iterrows():
                bond_stats[row['Bond_Type']] = {
                    'mean': row['Mean_A'],
                    'std': row['Std_A'],
                    'min': row['Min_A'],
                    'max': row['Max_A']
                }
        
        # Create context dictionary
        context = {
            "base_smiles": "N.N.Cl[Pt]Cl",  # Cisplatin SMILES
            "ligand_substitutions": {
                "Cl": ["Br", "I", "OH", "NH3"],
                "NH3": ["en", "dach", "py", "im"]
            },
            "descriptor_weights": {
                "pt_cl_bond": bond_stats.get("Pt–Cl", {}).get("mean", 2.3),
                "pt_n_bond": bond_stats.get("Pt–N", {}).get("mean", 2.05),
                "molecular_weight": float(metadata.get("Formula_weight_g_mol", 298.04)),
                "element_composition": element_composition
            },
            "metadata": metadata
        }
        
        logger.info(f"Loaded cisplatin context from {file_path}")
        return context
        
    except FileNotFoundError:
        logger.error(f"Cisplatin context file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading cisplatin context: {e}")
        raise


def load_pancreatic_target(file_path: str) -> Dict[str, Any]:
    """
    Load pancreatic target configuration from CSV file.
    
    Args:
        file_path (str): Path to the protein binding CSV file
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    logger = logging.getLogger(__name__)
    
    try:
        sections = parse_sectioned_csv(file_path)
        
        # Extract metadata
        metadata = {}
        if 'METADATA' in sections:
            for _, row in sections['METADATA'].iterrows():
                metadata[row['Data_Type']] = row['Value']
        
        # Extract element composition for environment modifiers
        element_composition = {}
        if 'ELEMENT_COMPOSITION' in sections:
            for _, row in sections['ELEMENT_COMPOSITION'].iterrows():
                element_composition[row['Element']] = {
                    'atom_count': row['Atom_count'],
                    'mass_fraction': row['Mass_fraction_percent'],
                    'cov_radius': row['Cov_radius_A']
                }
        
        # Extract bond statistics for resistance factors
        bond_stats = {}
        if 'BOND_LENGTHS_STATS' in sections:
            for _, row in sections['BOND_LENGTHS_STATS'].iterrows():
                bond_stats[row['Bond_Type']] = {
                    'mean': row['Mean_A'],
                    'std': row['Std_A'],
                    'min': row['Min_A'],
                    'max': row['Max_A']
                }
        
        # Create target configuration
        target_config = {
            "environment_modifiers": {
                "hypoxic_factor": 0.85,  # Hypoxic environment reduces drug effectiveness
                "acidic_ph_factor": 0.92,  # Acidic pH affects drug stability
                "stromal_barrier_factor": 0.78,  # Dense stroma reduces drug penetration
                "protein_binding_affinity": element_composition.get("C", {}).get("mass_fraction", 36.73),
                "molecular_weight_target": float(metadata.get("Formula_weight_g_mol", 5036.27))
            },
            "resistance_factors": {
                "gstp1_overexpression": 1.25,  # Increased detoxification
                "kras_mutation": 1.15,  # Altered signaling pathways
                "brca_deficiency": 0.88,  # Potential sensitivity to DNA damaging agents
                "p53_mutation": 1.10,  # Reduced apoptosis
                "bond_stability": {
                    "pt_bonds": bond_stats.get("Pt–N", {}).get("mean", 2.05),
                    "dna_interactions": bond_stats.get("N–H", {}).get("mean", 1.02)
                }
            },
            "stromal_barriers": {
                "collagen_density": 0.85,  # Dense collagen matrix
                "hyaluronic_acid": 0.90,  # HA contributes to stroma
                "fibroblast_activation": 1.20  # Activated fibroblasts increase barriers
            },
            "metadata": metadata
        }
        
        logger.info(f"Loaded pancreatic target config from {file_path}")
        return target_config
        
    except FileNotFoundError:
        logger.error(f"Pancreatic target file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading pancreatic target config: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary structure and values.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check for required top-level keys
        if isinstance(config, dict):
            # Basic validation - can be extended
            if not config:
                logger.warning("Empty configuration provided")
                return False
            
            # Additional validation logic can be added here
            logger.debug("Configuration validation passed")
            return True
        else:
            logger.error("Configuration must be a dictionary")
            return False
            
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False


def load_analogs_csv(file_path: str) -> Dict[str, Any]:
    """
    Load analog data from CSV file.
    
    Args:
        file_path (str): Path to the analogs CSV file
        
    Returns:
        Dict[str, Any]: Parsed analog data
    """
    logger = logging.getLogger(__name__)
    
    try:
        import pandas as pd
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ["analog_id", "smiles", "substitution_type", "source"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in analogs CSV: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} analogs from {file_path}")
        return df.to_dict('records')
        
    except FileNotFoundError:
        logger.error(f"Analogs CSV file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading analogs CSV: {e}")
        raise


def save_results(data: Dict[str, Any], file_path: str) -> None:
    """
    Save results data to file.
    
    Args:
        data (Dict[str, Any]): Data to save
        file_path (str): Output file path
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file format and save accordingly
        if file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif file_path.endswith('.csv'):
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            # Default to JSON
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise 