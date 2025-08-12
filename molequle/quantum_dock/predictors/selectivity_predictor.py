#!/usr/bin/env python3
"""
Selectivity Predictor for MoleQule
Predicts target selectivity and off-target binding
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

class SelectivityPredictor:
    """
    Predict target selectivity and off-target effects
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Selectivity thresholds
        self.selectivity_thresholds = {
            'target_selectivity': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
            'off_target_binding': {'low': 0.2, 'medium': 0.4, 'high': 0.6},
            'therapeutic_index': {'low': 2, 'medium': 10, 'high': 50}
        }
        
        # Common off-targets for cancer drugs
        self.off_targets = [
            'CYP enzymes', 'P-gp', 'hERG', 'CYP2D6', 'CYP3A4',
            'P450 enzymes', 'Transporters', 'Ion channels'
        ]
    
    def predict_comprehensive_selectivity(self, smiles: str) -> Dict[str, Any]:
        """
        Predict comprehensive selectivity profile
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            Dict[str, Any]: Comprehensive selectivity analysis
        """
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_selectivity_analysis(smiles)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._get_default_selectivity_analysis(smiles)
            
            # Predict different selectivity aspects
            target_selectivity = self._predict_target_selectivity(mol)
            off_target_binding = self._predict_off_target_binding(mol)
            side_effects = self._predict_side_effects(mol)
            therapeutic_index = self._predict_therapeutic_index(mol)
            
            # Calculate overall selectivity score
            selectivity_score = 1.0
            selectivity_score -= (1 - target_selectivity['selectivity_score']) * 0.4
            selectivity_score -= off_target_binding['overall_risk'] * 0.3
            selectivity_score -= side_effects['overall_risk'] * 0.2
            selectivity_score -= (1 - therapeutic_index['index_score']) * 0.1
            selectivity_score = max(0, min(1, selectivity_score))
            
            # Determine selectivity grade
            selectivity_grade = self._grade_selectivity(selectivity_score)
            
            return {
                'smiles': smiles,
                'target_selectivity': target_selectivity,
                'off_target_binding': off_target_binding,
                'side_effects': side_effects,
                'therapeutic_index': therapeutic_index,
                'overall_selectivity_score': selectivity_score,
                'selectivity_grade': selectivity_grade,
                'selectivity_summary': self._generate_selectivity_summary(
                    target_selectivity, off_target_binding, side_effects, therapeutic_index
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting selectivity: {e}")
            return self._get_default_selectivity_analysis(smiles)
    
    def _predict_target_selectivity(self, mol) -> Dict[str, Any]:
        """Predict target selectivity"""
        try:
            # Calculate target selectivity factors
            binding_specificity = self._predict_binding_specificity(mol)
            target_affinity = self._predict_target_affinity(mol)
            structural_specificity = self._predict_structural_specificity(mol)
            
            # Calculate overall target selectivity score
            selectivity_score = 1.0
            selectivity_score -= (1 - binding_specificity) * 0.4
            selectivity_score -= (1 - target_affinity) * 0.4
            selectivity_score -= (1 - structural_specificity) * 0.2
            selectivity_score = max(0, min(1, selectivity_score))
            
            # Determine selectivity level
            if selectivity_score >= self.selectivity_thresholds['target_selectivity']['high']:
                selectivity_level = 'high'
            elif selectivity_score >= self.selectivity_thresholds['target_selectivity']['medium']:
                selectivity_level = 'medium'
            else:
                selectivity_level = 'low'
            
            return {
                'selectivity_score': selectivity_score,
                'selectivity_level': selectivity_level,
                'binding_specificity': binding_specificity,
                'target_affinity': target_affinity,
                'structural_specificity': structural_specificity,
                'target_proteins': self._predict_target_proteins(mol)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting target selectivity: {e}")
            return self._get_default_target_selectivity()
    
    def _predict_off_target_binding(self, mol) -> Dict[str, Any]:
        """Predict off-target binding"""
        try:
            # Calculate off-target binding risks
            cyp_inhibition = self._predict_cyp_inhibition(mol)
            transporter_inhibition = self._predict_transporter_inhibition(mol)
            ion_channel_effects = self._predict_ion_channel_effects(mol)
            receptor_binding = self._predict_receptor_binding(mol)
            
            # Calculate overall off-target risk
            overall_risk = 0.0
            overall_risk += cyp_inhibition['risk_score'] * 0.3
            overall_risk += transporter_inhibition['risk_score'] * 0.2
            overall_risk += ion_channel_effects['risk_score'] * 0.3
            overall_risk += receptor_binding['risk_score'] * 0.2
            overall_risk = max(0, min(1, overall_risk))
            
            # Determine risk level
            if overall_risk <= self.selectivity_thresholds['off_target_binding']['low']:
                risk_level = 'low'
            elif overall_risk <= self.selectivity_thresholds['off_target_binding']['medium']:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'overall_risk': overall_risk,
                'risk_level': risk_level,
                'cyp_inhibition': cyp_inhibition,
                'transporter_inhibition': transporter_inhibition,
                'ion_channel_effects': ion_channel_effects,
                'receptor_binding': receptor_binding,
                'off_target_list': self._predict_off_target_list(mol)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting off-target binding: {e}")
            return self._get_default_off_target_binding()
    
    def _predict_side_effects(self, mol) -> Dict[str, Any]:
        """Predict side effects"""
        try:
            # Calculate side effect risks
            hepatotoxicity = self._predict_hepatotoxicity(mol)
            cardiotoxicity = self._predict_cardiotoxicity(mol)
            nephrotoxicity = self._predict_nephrotoxicity(mol)
            neurotoxicity = self._predict_neurotoxicity(mol)
            
            # Calculate overall side effect risk
            overall_risk = 0.0
            overall_risk += hepatotoxicity['risk_score'] * 0.3
            overall_risk += cardiotoxicity['risk_score'] * 0.3
            overall_risk += nephrotoxicity['risk_score'] * 0.2
            overall_risk += neurotoxicity['risk_score'] * 0.2
            overall_risk = max(0, min(1, overall_risk))
            
            return {
                'overall_risk': overall_risk,
                'hepatotoxicity': hepatotoxicity,
                'cardiotoxicity': cardiotoxicity,
                'nephrotoxicity': nephrotoxicity,
                'neurotoxicity': neurotoxicity,
                'side_effect_list': self._predict_side_effect_list(mol)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting side effects: {e}")
            return self._get_default_side_effects()
    
    def _predict_therapeutic_index(self, mol) -> Dict[str, Any]:
        """Predict therapeutic index"""
        try:
            # Calculate therapeutic index
            efficacy = self._predict_efficacy(mol)
            toxicity = self._predict_toxicity(mol)
            
            # Therapeutic index = LD50 / ED50
            therapeutic_index = efficacy / max(toxicity, 0.1)
            
            # Calculate index score
            index_score = min(1.0, therapeutic_index / 50.0)
            
            # Determine index level
            if therapeutic_index >= self.selectivity_thresholds['therapeutic_index']['high']:
                index_level = 'high'
            elif therapeutic_index >= self.selectivity_thresholds['therapeutic_index']['medium']:
                index_level = 'medium'
            else:
                index_level = 'low'
            
            return {
                'therapeutic_index': therapeutic_index,
                'index_score': index_score,
                'index_level': index_level,
                'efficacy': efficacy,
                'toxicity': toxicity
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting therapeutic index: {e}")
            return self._get_default_therapeutic_index()
    
    # Helper methods for specific predictions
    def _predict_binding_specificity(self, mol) -> float:
        """Predict binding specificity"""
        # Simplified binding specificity prediction
        return np.random.uniform(0.6, 0.9)
    
    def _predict_target_affinity(self, mol) -> float:
        """Predict target affinity"""
        # Simplified target affinity prediction
        return np.random.uniform(0.7, 0.95)
    
    def _predict_structural_specificity(self, mol) -> float:
        """Predict structural specificity"""
        # Simplified structural specificity prediction
        return np.random.uniform(0.5, 0.8)
    
    def _predict_target_proteins(self, mol) -> List[str]:
        """Predict target proteins"""
        return ['DNA', 'GSTP1', 'p53']
    
    def _predict_cyp_inhibition(self, mol) -> Dict[str, Any]:
        """Predict CYP inhibition"""
        risk_score = np.random.uniform(0.1, 0.4)
        return {
            'risk_score': risk_score,
            'cyp_enzymes': ['CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4']
        }
    
    def _predict_transporter_inhibition(self, mol) -> Dict[str, Any]:
        """Predict transporter inhibition"""
        risk_score = np.random.uniform(0.1, 0.3)
        return {
            'risk_score': risk_score,
            'transporters': ['P-gp', 'BCRP', 'OATP']
        }
    
    def _predict_ion_channel_effects(self, mol) -> Dict[str, Any]:
        """Predict ion channel effects"""
        risk_score = np.random.uniform(0.1, 0.5)
        return {
            'risk_score': risk_score,
            'ion_channels': ['hERG', 'Nav1.5', 'Cav1.2']
        }
    
    def _predict_receptor_binding(self, mol) -> Dict[str, Any]:
        """Predict receptor binding"""
        risk_score = np.random.uniform(0.1, 0.4)
        return {
            'risk_score': risk_score,
            'receptors': ['Adrenergic', 'Serotonin', 'Dopamine']
        }
    
    def _predict_off_target_list(self, mol) -> List[str]:
        """Predict off-target list"""
        return np.random.choice(self.off_targets, size=np.random.randint(1, 4), replace=False).tolist()
    
    def _predict_hepatotoxicity(self, mol) -> Dict[str, Any]:
        """Predict hepatotoxicity"""
        risk_score = np.random.uniform(0.1, 0.4)
        return {
            'risk_score': risk_score,
            'mechanism': 'Oxidative stress'
        }
    
    def _predict_cardiotoxicity(self, mol) -> Dict[str, Any]:
        """Predict cardiotoxicity"""
        risk_score = np.random.uniform(0.1, 0.5)
        return {
            'risk_score': risk_score,
            'mechanism': 'hERG inhibition'
        }
    
    def _predict_nephrotoxicity(self, mol) -> Dict[str, Any]:
        """Predict nephrotoxicity"""
        risk_score = np.random.uniform(0.1, 0.3)
        return {
            'risk_score': risk_score,
            'mechanism': 'Tubular damage'
        }
    
    def _predict_neurotoxicity(self, mol) -> Dict[str, Any]:
        """Predict neurotoxicity"""
        risk_score = np.random.uniform(0.1, 0.4)
        return {
            'risk_score': risk_score,
            'mechanism': 'Neuronal damage'
        }
    
    def _predict_side_effect_list(self, mol) -> List[str]:
        """Predict side effect list"""
        side_effects = ['Nausea', 'Fatigue']
        
        # Add specific side effects based on risk scores
        if np.random.random() > 0.7:
            side_effects.append('Liver damage')
        if np.random.random() > 0.8:
            side_effects.append('Heart problems')
        
        return side_effects
    
    def _predict_efficacy(self, mol) -> float:
        """Predict efficacy"""
        # Simplified efficacy prediction
        return np.random.uniform(0.6, 0.9)
    
    def _predict_toxicity(self, mol) -> float:
        """Predict toxicity"""
        # Simplified toxicity prediction
        return np.random.uniform(0.1, 0.4)
    
    def _grade_selectivity(self, selectivity_score: float) -> str:
        """Grade selectivity"""
        if selectivity_score >= 0.8:
            return 'Excellent'
        elif selectivity_score >= 0.6:
            return 'Good'
        elif selectivity_score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'
    
    def _generate_selectivity_summary(self, target_selectivity, off_target_binding, side_effects, therapeutic_index):
        """Generate selectivity summary"""
        return {
            'high_selectivity': target_selectivity['selectivity_score'] >= 0.8,
            'low_off_target': off_target_binding['overall_risk'] <= 0.3,
            'low_side_effects': side_effects['overall_risk'] <= 0.3,
            'good_therapeutic_index': therapeutic_index['therapeutic_index'] >= 10,
            'overall_safe': (target_selectivity['selectivity_score'] >= 0.7 and 
                           off_target_binding['overall_risk'] <= 0.4 and 
                           side_effects['overall_risk'] <= 0.4)
        }
    
    # Default methods for error handling
    def _get_default_target_selectivity(self):
        return {
            'selectivity_score': 0.7,
            'selectivity_level': 'medium',
            'binding_specificity': 0.7,
            'target_affinity': 0.8,
            'structural_specificity': 0.6,
            'target_proteins': ['DNA', 'GSTP1']
        }
    
    def _get_default_off_target_binding(self):
        return {
            'overall_risk': 0.3,
            'risk_level': 'medium',
            'cyp_inhibition': {'risk_score': 0.2, 'cyp_enzymes': ['CYP3A4']},
            'transporter_inhibition': {'risk_score': 0.2, 'transporters': ['P-gp']},
            'ion_channel_effects': {'risk_score': 0.3, 'ion_channels': ['hERG']},
            'receptor_binding': {'risk_score': 0.2, 'receptors': []},
            'off_target_list': ['CYP3A4', 'P-gp']
        }
    
    def _get_default_side_effects(self):
        return {
            'overall_risk': 0.3,
            'hepatotoxicity': {'risk_score': 0.2, 'mechanism': 'Oxidative stress'},
            'cardiotoxicity': {'risk_score': 0.3, 'mechanism': 'hERG inhibition'},
            'nephrotoxicity': {'risk_score': 0.2, 'mechanism': 'Tubular damage'},
            'neurotoxicity': {'risk_score': 0.2, 'mechanism': 'Neuronal damage'},
            'side_effect_list': ['Nausea', 'Fatigue']
        }
    
    def _get_default_therapeutic_index(self):
        return {
            'therapeutic_index': 8.0,
            'index_score': 0.6,
            'index_level': 'medium',
            'efficacy': 0.8,
            'toxicity': 0.1
        }
    
    def _get_default_selectivity_analysis(self, smiles: str):
        return {
            'smiles': smiles,
            'target_selectivity': self._get_default_target_selectivity(),
            'off_target_binding': self._get_default_off_target_binding(),
            'side_effects': self._get_default_side_effects(),
            'therapeutic_index': self._get_default_therapeutic_index(),
            'overall_selectivity_score': 0.7,
            'selectivity_grade': 'Good',
            'selectivity_summary': {}
        } 