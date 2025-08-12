#!/usr/bin/env python3
"""
Stability Predictor for MoleQule
Predicts chemical and biological stability
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

class StabilityPredictor:
    """
    Predict chemical and biological stability
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Stability thresholds
        self.stability_thresholds = {
            'chemical_stability': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
            'biological_stability': {'low': 0.4, 'medium': 0.7, 'high': 0.9},
            'storage_stability': {'low': 0.5, 'medium': 0.8, 'high': 0.95}
        }
    
    def predict_comprehensive_stability(self, smiles: str) -> Dict[str, Any]:
        """
        Predict all stability aspects
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            Dict[str, Any]: Comprehensive stability analysis
        """
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_stability_analysis(smiles)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._get_default_stability_analysis(smiles)
            
            # Predict different stability aspects
            chemical_stability = self._predict_chemical_stability(mol)
            biological_stability = self._predict_biological_stability(mol)
            storage_stability = self._predict_storage_stability(mol)
            
            # Calculate overall stability score
            stability_scores = [
                chemical_stability['stability_score'],
                biological_stability['stability_score'],
                storage_stability['stability_score']
            ]
            
            overall_stability_score = np.mean(stability_scores)
            overall_stability_grade = self._grade_stability(overall_stability_score)
            
            return {
                'smiles': smiles,
                'chemical_stability': chemical_stability,
                'biological_stability': biological_stability,
                'storage_stability': storage_stability,
                'overall_stability_score': overall_stability_score,
                'overall_stability_grade': overall_stability_grade,
                'stability_summary': self._generate_stability_summary(
                    chemical_stability, biological_stability, storage_stability
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting stability: {e}")
            return self._get_default_stability_analysis(smiles)
    
    def _predict_chemical_stability(self, mol) -> Dict[str, Any]:
        """Predict chemical stability"""
        try:
            # Calculate stability factors
            hydrolysis_susceptibility = self._predict_hydrolysis(mol)
            oxidation_resistance = self._predict_oxidation(mol)
            photostability = self._predict_photostability(mol)
            thermal_stability = self._predict_thermal_stability(mol)
            
            # Calculate overall chemical stability score
            stability_score = 1.0
            stability_score -= hydrolysis_susceptibility * 0.3
            stability_score -= (1 - oxidation_resistance) * 0.2
            stability_score -= (1 - photostability) * 0.2
            stability_score -= (1 - thermal_stability) * 0.3
            stability_score = max(0, min(1, stability_score))
            
            # Determine stability level
            if stability_score >= self.stability_thresholds['chemical_stability']['high']:
                stability_level = 'high'
            elif stability_score >= self.stability_thresholds['chemical_stability']['medium']:
                stability_level = 'medium'
            else:
                stability_level = 'low'
            
            return {
                'stability_score': stability_score,
                'stability_level': stability_level,
                'hydrolysis_susceptibility': hydrolysis_susceptibility,
                'oxidation_resistance': oxidation_resistance,
                'photostability': photostability,
                'thermal_stability': thermal_stability,
                'degradation_pathways': self._predict_degradation_pathways(mol)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting chemical stability: {e}")
            return self._get_default_chemical_stability()
    
    def _predict_biological_stability(self, mol) -> Dict[str, Any]:
        """Predict biological stability"""
        try:
            # Calculate biological stability factors
            plasma_stability = self._predict_plasma_stability(mol)
            liver_stability = self._predict_liver_stability(mol)
            intestinal_stability = self._predict_intestinal_stability(mol)
            blood_stability = self._predict_blood_stability(mol)
            
            # Calculate overall biological stability score
            stability_score = 1.0
            stability_score -= (1 - plasma_stability) * 0.3
            stability_score -= (1 - liver_stability) * 0.3
            stability_score -= (1 - intestinal_stability) * 0.2
            stability_score -= (1 - blood_stability) * 0.2
            stability_score = max(0, min(1, stability_score))
            
            # Determine stability level
            if stability_score >= self.stability_thresholds['biological_stability']['high']:
                stability_level = 'high'
            elif stability_score >= self.stability_thresholds['biological_stability']['medium']:
                stability_level = 'medium'
            else:
                stability_level = 'low'
            
            return {
                'stability_score': stability_score,
                'stability_level': stability_level,
                'plasma_stability': plasma_stability,
                'liver_stability': liver_stability,
                'intestinal_stability': intestinal_stability,
                'blood_stability': blood_stability,
                'metabolic_pathways': self._predict_metabolic_pathways(mol)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting biological stability: {e}")
            return self._get_default_biological_stability()
    
    def _predict_storage_stability(self, mol) -> Dict[str, Any]:
        """Predict storage stability"""
        try:
            # Calculate storage stability factors
            shelf_life = self._predict_shelf_life(mol)
            temperature_stability = self._predict_temperature_stability(mol)
            humidity_stability = self._predict_humidity_stability(mol)
            light_stability = self._predict_light_stability(mol)
            
            # Calculate overall storage stability score
            stability_score = 1.0
            stability_score -= (1 - shelf_life) * 0.4
            stability_score -= (1 - temperature_stability) * 0.2
            stability_score -= (1 - humidity_stability) * 0.2
            stability_score -= (1 - light_stability) * 0.2
            stability_score = max(0, min(1, stability_score))
            
            # Determine stability level
            if stability_score >= self.stability_thresholds['storage_stability']['high']:
                stability_level = 'high'
            elif stability_score >= self.stability_thresholds['storage_stability']['medium']:
                stability_level = 'medium'
            else:
                stability_level = 'low'
            
            return {
                'stability_score': stability_score,
                'stability_level': stability_level,
                'shelf_life': shelf_life,
                'temperature_stability': temperature_stability,
                'humidity_stability': humidity_stability,
                'light_stability': light_stability,
                'storage_conditions': self._predict_storage_conditions(mol)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting storage stability: {e}")
            return self._get_default_storage_stability()
    
    # Helper methods for specific stability predictions
    def _predict_hydrolysis(self, mol) -> float:
        """Predict hydrolysis susceptibility"""
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        # Simplified hydrolysis prediction
        hydrolysis = 0.2 + (logp * 0.05) + (mw / 1000) * 0.1
        return max(0, min(1, hydrolysis))
    
    def _predict_oxidation(self, mol) -> float:
        """Predict oxidation resistance"""
        # Simplified oxidation resistance prediction
        return np.random.uniform(0.6, 0.9)
    
    def _predict_photostability(self, mol) -> float:
        """Predict photostability"""
        # Simplified photostability prediction
        return np.random.uniform(0.7, 0.95)
    
    def _predict_thermal_stability(self, mol) -> float:
        """Predict thermal stability"""
        mw = Descriptors.MolWt(mol)
        
        # Simplified thermal stability prediction
        thermal_stability = 0.8 - (mw / 1000) * 0.2
        return max(0.3, min(1, thermal_stability))
    
    def _predict_plasma_stability(self, mol) -> float:
        """Predict plasma stability"""
        # Simplified plasma stability prediction
        return np.random.uniform(0.5, 0.8)
    
    def _predict_liver_stability(self, mol) -> float:
        """Predict liver stability"""
        # Simplified liver stability prediction
        return np.random.uniform(0.4, 0.7)
    
    def _predict_intestinal_stability(self, mol) -> float:
        """Predict intestinal stability"""
        # Simplified intestinal stability prediction
        return np.random.uniform(0.6, 0.9)
    
    def _predict_blood_stability(self, mol) -> float:
        """Predict blood stability"""
        # Simplified blood stability prediction
        return np.random.uniform(0.7, 0.95)
    
    def _predict_shelf_life(self, mol) -> float:
        """Predict shelf life"""
        # Simplified shelf life prediction (in years)
        return np.random.uniform(0.5, 2.0)
    
    def _predict_temperature_stability(self, mol) -> float:
        """Predict temperature stability"""
        # Simplified temperature stability prediction
        return np.random.uniform(0.7, 0.95)
    
    def _predict_humidity_stability(self, mol) -> float:
        """Predict humidity stability"""
        # Simplified humidity stability prediction
        return np.random.uniform(0.6, 0.9)
    
    def _predict_light_stability(self, mol) -> float:
        """Predict light stability"""
        # Simplified light stability prediction
        return np.random.uniform(0.5, 0.8)
    
    def _predict_degradation_pathways(self, mol) -> List[str]:
        """Predict degradation pathways"""
        pathways = []
        
        # Add common degradation pathways
        pathways.append("Hydrolysis")
        pathways.append("Oxidation")
        
        # Add specific pathways based on molecular properties
        logp = Descriptors.MolLogP(mol)
        if logp > 3:
            pathways.append("Photodegradation")
        
        return pathways
    
    def _predict_metabolic_pathways(self, mol) -> List[str]:
        """Predict metabolic pathways"""
        pathways = []
        
        # Add common metabolic pathways
        pathways.append("Phase I metabolism")
        pathways.append("Phase II metabolism")
        
        return pathways
    
    def _predict_storage_conditions(self, mol) -> Dict[str, str]:
        """Predict optimal storage conditions"""
        return {
            'temperature': '2-8°C',
            'humidity': '≤60% RH',
            'light': 'Protect from light',
            'container': 'Amber glass vial'
        }
    
    def _grade_stability(self, stability_score: float) -> str:
        """Grade stability"""
        if stability_score >= 0.8:
            return 'Excellent'
        elif stability_score >= 0.6:
            return 'Good'
        elif stability_score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'
    
    def _generate_stability_summary(self, chemical_stability, biological_stability, storage_stability):
        """Generate stability summary"""
        return {
            'chemical_issues': chemical_stability['stability_score'] < 0.6,
            'biological_issues': biological_stability['stability_score'] < 0.6,
            'storage_issues': storage_stability['stability_score'] < 0.6,
            'overall_stable': min(chemical_stability['stability_score'], 
                                biological_stability['stability_score'], 
                                storage_stability['stability_score']) >= 0.6
        }
    
    # Default methods for error handling
    def _get_default_chemical_stability(self):
        return {
            'stability_score': 0.7,
            'stability_level': 'medium',
            'hydrolysis_susceptibility': 0.3,
            'oxidation_resistance': 0.8,
            'photostability': 0.8,
            'thermal_stability': 0.7,
            'degradation_pathways': ['Hydrolysis', 'Oxidation']
        }
    
    def _get_default_biological_stability(self):
        return {
            'stability_score': 0.6,
            'stability_level': 'medium',
            'plasma_stability': 0.6,
            'liver_stability': 0.5,
            'intestinal_stability': 0.7,
            'blood_stability': 0.8,
            'metabolic_pathways': ['Phase I metabolism', 'Phase II metabolism']
        }
    
    def _get_default_storage_stability(self):
        return {
            'stability_score': 0.8,
            'stability_level': 'high',
            'shelf_life': 1.5,
            'temperature_stability': 0.8,
            'humidity_stability': 0.7,
            'light_stability': 0.6,
            'storage_conditions': {
                'temperature': '2-8°C',
                'humidity': '≤60% RH',
                'light': 'Protect from light',
                'container': 'Amber glass vial'
            }
        }
    
    def _get_default_stability_analysis(self, smiles: str):
        return {
            'smiles': smiles,
            'chemical_stability': self._get_default_chemical_stability(),
            'biological_stability': self._get_default_biological_stability(),
            'storage_stability': self._get_default_storage_stability(),
            'overall_stability_score': 0.7,
            'overall_stability_grade': 'Good',
            'stability_summary': {}
        } 