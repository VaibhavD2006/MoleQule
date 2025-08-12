#!/usr/bin/env python3
"""
Synthetic Accessibility Predictor for MoleQule
Predicts synthesis complexity, feasibility, and cost
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available. Some molecular descriptors will be limited.")
    RDKIT_AVAILABLE = False

class SyntheticAccessibilityPredictor:
    """
    Predict synthetic accessibility and feasibility
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Synthesis complexity thresholds
        self.complexity_thresholds = {
            'reaction_steps': {'low': 3, 'medium': 6, 'high': 10},
            'starting_materials': {'low': 2, 'medium': 4, 'high': 6},
            'reaction_yields': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
            'purification_difficulty': {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        }
        
        # Cost factors
        self.cost_factors = {
            'starting_materials': {'low': 100, 'medium': 500, 'high': 2000},
            'reaction_steps': {'low': 50, 'medium': 200, 'high': 500},
            'purification': {'low': 100, 'medium': 300, 'high': 800}
        }
    
    def predict_synthesis_complexity(self, smiles: str) -> Dict[str, Any]:
        """
        Predict comprehensive synthesis complexity
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            Dict[str, Any]: Synthesis complexity analysis
        """
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_synthesis_analysis()
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._get_default_synthesis_analysis()
            
            # Predict various synthesis aspects
            reaction_steps = self._predict_reaction_steps(mol)
            starting_materials = self._predict_starting_materials(mol)
            reaction_yields = self._predict_reaction_yields(mol)
            purification_difficulty = self._predict_purification_difficulty(mol)
            
            # Calculate feasibility score
            feasibility_score = self._calculate_feasibility_score(
                reaction_steps, starting_materials, reaction_yields, purification_difficulty
            )
            
            # Calculate total cost
            total_cost = self._calculate_total_cost(
                reaction_steps, starting_materials, purification_difficulty
            )
            
            # Determine synthesis grade
            synthesis_grade = self._grade_synthesis(feasibility_score)
            
            return {
                'reaction_steps': reaction_steps,
                'starting_materials': starting_materials,
                'reaction_yields': reaction_yields,
                'purification_difficulty': purification_difficulty,
                'feasibility_score': feasibility_score,
                'total_cost': total_cost,
                'synthesis_grade': synthesis_grade,
                'synthesis_pathway': self._predict_synthesis_pathway(mol),
                'scale_up_potential': self._predict_scale_up_potential(mol)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting synthesis complexity: {e}")
            return self._get_default_synthesis_analysis()
    
    def _predict_reaction_steps(self, mol) -> Dict[str, Any]:
        """Predict number of reaction steps"""
        mw = Descriptors.MolWt(mol)
        complexity = Descriptors.BertzCT(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        
        # Base steps calculation
        base_steps = 2  # Minimum steps
        
        # Add complexity-based steps
        complexity_steps = int(complexity / 100)
        
        # Add functional group steps
        functional_groups = self._count_functional_groups(mol)
        functional_steps = functional_groups * 0.5
        
        # Add stereochemistry steps
        stereochemistry_steps = self._count_stereocenters(mol) * 0.3
        
        total_steps = base_steps + complexity_steps + functional_steps + stereochemistry_steps
        
        # Determine complexity level
        if total_steps <= self.complexity_thresholds['reaction_steps']['low']:
            complexity_level = 'low'
        elif total_steps <= self.complexity_thresholds['reaction_steps']['medium']:
            complexity_level = 'medium'
        else:
            complexity_level = 'high'
        
        return {
            'total_steps': int(total_steps),
            'complexity_level': complexity_level,
            'base_steps': base_steps,
            'complexity_steps': complexity_steps,
            'functional_steps': functional_steps,
            'stereochemistry_steps': stereochemistry_steps
        }
    
    def _predict_starting_materials(self, mol) -> Dict[str, Any]:
        """Predict starting material requirements"""
        mw = Descriptors.MolWt(mol)
        complexity = Descriptors.BertzCT(mol)
        
        # Estimate number of starting materials
        base_materials = 2
        
        # Add complexity-based materials
        complexity_materials = int(complexity / 200)
        
        # Add functional group materials
        functional_materials = self._count_functional_groups(mol) * 0.3
        
        total_materials = base_materials + complexity_materials + functional_materials
        
        # Determine availability
        availability_score = self._predict_material_availability(mol)
        
        return {
            'total_materials': int(total_materials),
            'availability_score': availability_score,
            'base_materials': base_materials,
            'complexity_materials': complexity_materials,
            'functional_materials': functional_materials
        }
    
    def _predict_reaction_yields(self, mol) -> Dict[str, Any]:
        """Predict reaction yields"""
        complexity = Descriptors.BertzCT(mol)
        stereocenters = self._count_stereocenters(mol)
        
        # Base yield
        base_yield = 0.8
        
        # Complexity penalty
        complexity_penalty = min(0.3, complexity / 1000)
        
        # Stereochemistry penalty
        stereochemistry_penalty = stereocenters * 0.05
        
        # Functional group penalty
        functional_penalty = self._count_functional_groups(mol) * 0.02
        
        total_yield = base_yield - complexity_penalty - stereochemistry_penalty - functional_penalty
        total_yield = max(0.1, min(0.95, total_yield))
        
        # Determine yield level
        if total_yield >= self.complexity_thresholds['reaction_yields']['high']:
            yield_level = 'high'
        elif total_yield >= self.complexity_thresholds['reaction_yields']['medium']:
            yield_level = 'medium'
        else:
            yield_level = 'low'
        
        return {
            'total_yield': total_yield,
            'yield_level': yield_level,
            'base_yield': base_yield,
            'complexity_penalty': complexity_penalty,
            'stereochemistry_penalty': stereochemistry_penalty,
            'functional_penalty': functional_penalty
        }
    
    def _predict_purification_difficulty(self, mol) -> Dict[str, Any]:
        """Predict purification difficulty"""
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        complexity = Descriptors.BertzCT(mol)
        
        # Base difficulty
        base_difficulty = 0.3
        
        # Molecular weight penalty
        mw_penalty = max(0, (mw - 300) / 200) * 0.2
        
        # Lipophilicity penalty
        logp_penalty = max(0, abs(logp - 2.0) / 2.0) * 0.2
        
        # Complexity penalty
        complexity_penalty = min(0.3, complexity / 1000)
        
        total_difficulty = base_difficulty + mw_penalty + logp_penalty + complexity_penalty
        total_difficulty = max(0.1, min(0.9, total_difficulty))
        
        # Determine difficulty level
        if total_difficulty <= self.complexity_thresholds['purification_difficulty']['low']:
            difficulty_level = 'low'
        elif total_difficulty <= self.complexity_thresholds['purification_difficulty']['medium']:
            difficulty_level = 'medium'
        else:
            difficulty_level = 'high'
        
        return {
            'total_difficulty': total_difficulty,
            'difficulty_level': difficulty_level,
            'base_difficulty': base_difficulty,
            'mw_penalty': mw_penalty,
            'logp_penalty': logp_penalty,
            'complexity_penalty': complexity_penalty
        }
    
    def _calculate_feasibility_score(self, reaction_steps, starting_materials, reaction_yields, purification_difficulty):
        """Calculate overall synthesis feasibility score"""
        feasibility = 1.0
        
        # Reaction steps penalty
        steps_penalty = max(0, (reaction_steps['total_steps'] - 5) / 10)
        feasibility -= steps_penalty * 0.3
        
        # Starting materials penalty
        materials_penalty = max(0, (starting_materials['total_materials'] - 3) / 5)
        feasibility -= materials_penalty * 0.2
        
        # Yield penalty
        yield_penalty = (0.8 - reaction_yields['total_yield']) * 0.3
        feasibility -= yield_penalty
        
        # Purification penalty
        purification_penalty = purification_difficulty['total_difficulty'] * 0.2
        feasibility -= purification_penalty
        
        return max(0, min(1, feasibility))
    
    def _calculate_total_cost(self, reaction_steps, starting_materials, purification_difficulty):
        """Calculate total synthesis cost"""
        # Starting materials cost
        materials_cost = starting_materials['total_materials'] * self.cost_factors['starting_materials']['medium']
        
        # Reaction steps cost
        steps_cost = reaction_steps['total_steps'] * self.cost_factors['reaction_steps']['medium']
        
        # Purification cost
        purification_cost = purification_difficulty['total_difficulty'] * self.cost_factors['purification']['medium']
        
        total_cost = materials_cost + steps_cost + purification_cost
        
        return total_cost
    
    def _predict_synthesis_pathway(self, mol) -> List[str]:
        """Predict synthesis pathway steps"""
        pathway = []
        
        # Add basic synthesis steps
        pathway.append("Starting material preparation")
        pathway.append("Core structure formation")
        
        # Add functional group modifications
        functional_groups = self._count_functional_groups(mol)
        if functional_groups > 0:
            pathway.append("Functional group introduction")
        
        # Add stereochemistry steps
        stereocenters = self._count_stereocenters(mol)
        if stereocenters > 0:
            pathway.append("Stereochemistry control")
        
        pathway.append("Purification and isolation")
        pathway.append("Characterization")
        
        return pathway
    
    def _predict_scale_up_potential(self, mol) -> Dict[str, Any]:
        """Predict scale-up potential"""
        complexity = Descriptors.BertzCT(mol)
        stereocenters = self._count_stereocenters(mol)
        
        # Scale-up feasibility
        scale_up_feasibility = 1.0
        scale_up_feasibility -= min(0.4, complexity / 1000)
        scale_up_feasibility -= stereocenters * 0.05
        scale_up_feasibility = max(0.2, min(1, scale_up_feasibility))
        
        # Cost scaling
        cost_scaling = 1.0 + (complexity / 500)
        
        return {
            'scale_up_feasibility': scale_up_feasibility,
            'cost_scaling': cost_scaling,
            'recommended_scale': 'laboratory' if scale_up_feasibility < 0.6 else 'pilot' if scale_up_feasibility < 0.8 else 'industrial'
        }
    
    def _count_functional_groups(self, mol) -> int:
        """Count functional groups"""
        # Simplified functional group counting
        return Descriptors.NumHDonors(mol) + Descriptors.NumHAcceptors(mol)
    
    def _count_stereocenters(self, mol) -> int:
        """Count stereocenters"""
        return Descriptors.NumRotatableBonds(mol) // 2
    
    def _predict_material_availability(self, mol) -> float:
        """Predict starting material availability"""
        # Simplified availability prediction
        return np.random.uniform(0.6, 0.9)
    
    def _grade_synthesis(self, feasibility_score: float) -> str:
        """Grade synthesis feasibility"""
        if feasibility_score >= 0.8:
            return 'Excellent'
        elif feasibility_score >= 0.6:
            return 'Good'
        elif feasibility_score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'
    
    def _get_default_synthesis_analysis(self):
        """Get default synthesis analysis"""
        return {
            'reaction_steps': {'total_steps': 5, 'complexity_level': 'medium'},
            'starting_materials': {'total_materials': 3, 'availability_score': 0.7},
            'reaction_yields': {'total_yield': 0.7, 'yield_level': 'medium'},
            'purification_difficulty': {'total_difficulty': 0.5, 'difficulty_level': 'medium'},
            'feasibility_score': 0.6,
            'total_cost': 1500,
            'synthesis_grade': 'Good',
            'synthesis_pathway': ['Starting material preparation', 'Core structure formation', 'Purification and isolation'],
            'scale_up_potential': {'scale_up_feasibility': 0.7, 'cost_scaling': 1.2, 'recommended_scale': 'pilot'}
        } 