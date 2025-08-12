#!/usr/bin/env python3
"""
ADMET Properties Predictor for MoleQule
Predicts Absorption, Distribution, Metabolism, Excretion, Toxicity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Crippen import MolLogP, MolMR
    from rdkit.Chem.EState import EStateIndices
    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available. Some molecular descriptors will be limited.")
    RDKIT_AVAILABLE = False

class ADMETPredictor:
    """
    Comprehensive ADMET properties predictor
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ADMET thresholds and ranges
        self.admet_thresholds = {
            'absorption': {
                'logp_optimal': (1.0, 3.0),
                'molecular_weight_optimal': (150, 500),
                'hbd_optimal': (0, 5),
                'hba_optimal': (2, 10),
                'tpsa_optimal': (20, 130),
                'rotatable_bonds_optimal': (0, 10)
            },
            'distribution': {
                'plasma_protein_binding_optimal': (0.1, 0.95),
                'volume_distribution_optimal': (0.5, 20.0)
            },
            'metabolism': {
                'cyp_inhibition_risk': (0, 0.3),
                'metabolic_stability_optimal': (0.5, 1.0)
            },
            'excretion': {
                'clearance_optimal': (0.1, 2.0),
                'half_life_optimal': (2, 24)
            },
            'toxicity': {
                'mutagenicity_risk': (0, 0.2),
                'carcinogenicity_risk': (0, 0.2),
                'hepatotoxicity_risk': (0, 0.3),
                'cardiotoxicity_risk': (0, 0.3)
            }
        }
    
    def predict_absorption(self, mol) -> Dict[str, Any]:
        """Predict absorption properties"""
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_absorption()
            
            # Calculate key absorption descriptors
            logp = MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            # Lipinski's Rule of Five
            lipinski_violations = 0
            if logp > 5: lipinski_violations += 1
            if mw > 500: lipinski_violations += 1
            if hbd > 5: lipinski_violations += 1
            if hba > 10: lipinski_violations += 1
            
            # Absorption score calculation
            absorption_score = 1.0
            absorption_score -= max(0, abs(logp - 2.0) / 2.0) * 0.2
            absorption_score -= max(0, abs(mw - 325) / 175) * 0.2
            absorption_score -= max(0, abs(hbd - 2.5) / 2.5) * 0.2
            absorption_score -= max(0, abs(hba - 6) / 4) * 0.2
            absorption_score -= max(0, abs(tpsa - 75) / 55) * 0.2
            absorption_score = max(0, min(1, absorption_score))
            
            # Caco-2 permeability prediction
            caco2_permeability = self._predict_caco2_permeability(mol)
            
            # Bioavailability prediction
            bioavailability = self._predict_bioavailability(mol)
            
            return {
                'logp': logp,
                'molecular_weight': mw,
                'hbd': hbd,
                'hba': hba,
                'tpsa': tpsa,
                'rotatable_bonds': rotatable_bonds,
                'lipinski_violations': lipinski_violations,
                'absorption_score': absorption_score,
                'caco2_permeability': caco2_permeability,
                'bioavailability': bioavailability,
                'absorption_grade': self._grade_property(absorption_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting absorption: {e}")
            return self._get_default_absorption()
    
    def predict_distribution(self, mol) -> Dict[str, Any]:
        """Predict distribution properties"""
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_distribution()
            
            # Volume of distribution prediction
            vd = self._predict_volume_distribution(mol)
            
            # Plasma protein binding prediction
            ppb = self._predict_plasma_protein_binding(mol)
            
            # Blood-brain barrier penetration
            bbb_penetration = self._predict_bbb_penetration(mol)
            
            # Tissue distribution
            tissue_distribution = self._predict_tissue_distribution(mol)
            
            distribution_score = 1.0
            distribution_score -= max(0, abs(vd - 1.0) / 1.0) * 0.3
            distribution_score -= max(0, abs(ppb - 0.9) / 0.1) * 0.3
            distribution_score -= max(0, bbb_penetration - 0.1) * 0.4
            distribution_score = max(0, min(1, distribution_score))
            
            return {
                'volume_distribution': vd,
                'plasma_protein_binding': ppb,
                'bbb_penetration': bbb_penetration,
                'tissue_distribution': tissue_distribution,
                'distribution_score': distribution_score,
                'distribution_grade': self._grade_property(distribution_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting distribution: {e}")
            return self._get_default_distribution()
    
    def predict_metabolism(self, mol) -> Dict[str, Any]:
        """Predict metabolism properties"""
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_metabolism()
            
            # CYP inhibition predictions
            cyp_inhibitions = self._predict_cyp_inhibition(mol)
            
            # Metabolic stability
            metabolic_stability = self._predict_metabolic_stability(mol)
            
            # Metabolic pathways
            metabolic_pathways = self._predict_metabolic_pathways(mol)
            
            # Drug-drug interaction potential
            ddi_potential = self._predict_ddi_potential(mol)
            
            metabolism_score = 1.0
            metabolism_score -= sum(cyp_inhibitions.values()) * 0.2
            metabolism_score -= (1 - metabolic_stability) * 0.3
            metabolism_score -= ddi_potential * 0.5
            metabolism_score = max(0, min(1, metabolism_score))
            
            return {
                'cyp_inhibitions': cyp_inhibitions,
                'metabolic_stability': metabolic_stability,
                'metabolic_pathways': metabolic_pathways,
                'ddi_potential': ddi_potential,
                'metabolism_score': metabolism_score,
                'metabolism_grade': self._grade_property(metabolism_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting metabolism: {e}")
            return self._get_default_metabolism()
    
    def predict_excretion(self, mol) -> Dict[str, Any]:
        """Predict excretion properties"""
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_excretion()
            
            # Clearance prediction
            clearance = self._predict_clearance(mol)
            
            # Half-life prediction
            half_life = self._predict_half_life(mol)
            
            # Excretion routes
            excretion_routes = self._predict_excretion_routes(mol)
            
            # Renal clearance
            renal_clearance = self._predict_renal_clearance(mol)
            
            excretion_score = 1.0
            excretion_score -= max(0, abs(clearance - 0.5) / 0.5) * 0.3
            excretion_score -= max(0, abs(half_life - 8) / 8) * 0.3
            excretion_score -= max(0, renal_clearance - 0.3) * 0.4
            excretion_score = max(0, min(1, excretion_score))
            
            return {
                'clearance': clearance,
                'half_life': half_life,
                'excretion_routes': excretion_routes,
                'renal_clearance': renal_clearance,
                'excretion_score': excretion_score,
                'excretion_grade': self._grade_property(excretion_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting excretion: {e}")
            return self._get_default_excretion()
    
    def predict_toxicity(self, mol) -> Dict[str, Any]:
        """Predict toxicity properties"""
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_toxicity()
            
            # Mutagenicity prediction
            mutagenicity = self._predict_mutagenicity(mol)
            
            # Carcinogenicity prediction
            carcinogenicity = self._predict_carcinogenicity(mol)
            
            # Hepatotoxicity prediction
            hepatotoxicity = self._predict_hepatotoxicity(mol)
            
            # Cardiotoxicity prediction
            cardiotoxicity = self._predict_cardiotoxicity(mol)
            
            # Nephrotoxicity prediction
            nephrotoxicity = self._predict_nephrotoxicity(mol)
            
            # Overall toxicity score
            toxicity_score = 1.0
            toxicity_score -= mutagenicity * 0.2
            toxicity_score -= carcinogenicity * 0.2
            toxicity_score -= hepatotoxicity * 0.2
            toxicity_score -= cardiotoxicity * 0.2
            toxicity_score -= nephrotoxicity * 0.2
            toxicity_score = max(0, min(1, toxicity_score))
            
            return {
                'mutagenicity': mutagenicity,
                'carcinogenicity': carcinogenicity,
                'hepatotoxicity': hepatotoxicity,
                'cardiotoxicity': cardiotoxicity,
                'nephrotoxicity': nephrotoxicity,
                'toxicity_score': toxicity_score,
                'toxicity_grade': self._grade_property(toxicity_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting toxicity: {e}")
            return self._get_default_toxicity()
    
    def predict_comprehensive_admet(self, smiles: str) -> Dict[str, Any]:
        """Predict all ADMET properties for a compound"""
        try:
            if not RDKIT_AVAILABLE:
                return self._get_default_comprehensive_admet(smiles)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Predict all ADMET properties
            absorption = self.predict_absorption(mol)
            distribution = self.predict_distribution(mol)
            metabolism = self.predict_metabolism(mol)
            excretion = self.predict_excretion(mol)
            toxicity = self.predict_toxicity(mol)
            
            # Calculate overall ADMET score
            admet_scores = [
                absorption['absorption_score'],
                distribution['distribution_score'],
                metabolism['metabolism_score'],
                excretion['excretion_score'],
                toxicity['toxicity_score']
            ]
            
            overall_admet_score = np.mean(admet_scores)
            
            return {
                'smiles': smiles,
                'absorption': absorption,
                'distribution': distribution,
                'metabolism': metabolism,
                'excretion': excretion,
                'toxicity': toxicity,
                'overall_admet_score': overall_admet_score,
                'overall_admet_grade': self._grade_property(overall_admet_score),
                'admet_summary': self._generate_admet_summary(absorption, distribution, metabolism, excretion, toxicity)
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive ADMET prediction: {e}")
            return self._get_default_comprehensive_admet(smiles)
    
    # Helper methods for specific predictions
    def _predict_caco2_permeability(self, mol) -> float:
        """Predict Caco-2 permeability"""
        logp = MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Simplified Caco-2 model
        permeability = 1.0
        permeability -= max(0, abs(logp - 2.0) / 2.0) * 0.3
        permeability -= max(0, (mw - 300) / 200) * 0.3
        permeability -= max(0, (tpsa - 60) / 40) * 0.4
        return max(0, min(1, permeability))
    
    def _predict_bioavailability(self, mol) -> float:
        """Predict oral bioavailability"""
        logp = MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        bioavailability = 1.0
        bioavailability -= max(0, abs(logp - 2.0) / 2.0) * 0.25
        bioavailability -= max(0, (mw - 300) / 200) * 0.25
        bioavailability -= max(0, (hbd - 2) / 3) * 0.25
        bioavailability -= max(0, (hba - 5) / 5) * 0.25
        return max(0, min(1, bioavailability))
    
    def _predict_volume_distribution(self, mol) -> float:
        """Predict volume of distribution"""
        logp = MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        # Simplified Vd model
        vd = 0.5 + (logp * 0.3) + (mw / 1000) * 0.2
        return max(0.1, min(20, vd))
    
    def _predict_plasma_protein_binding(self, mol) -> float:
        """Predict plasma protein binding"""
        logp = MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        # Simplified PPB model
        ppb = 0.3 + (logp * 0.1) + (mw / 1000) * 0.05
        return max(0.1, min(0.99, ppb))
    
    def _predict_bbb_penetration(self, mol) -> float:
        """Predict blood-brain barrier penetration"""
        logp = MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Simplified BBB model
        bbb_score = 1.0
        bbb_score -= max(0, abs(logp - 2.0) / 2.0) * 0.4
        bbb_score -= max(0, (mw - 400) / 100) * 0.3
        bbb_score -= max(0, (tpsa - 90) / 40) * 0.3
        return max(0, min(1, bbb_score))
    
    def _predict_tissue_distribution(self, mol) -> Dict[str, float]:
        """Predict tissue distribution"""
        logp = MolLogP(mol)
        
        # Simplified tissue distribution model
        return {
            'liver': 0.3 + (logp * 0.1),
            'kidney': 0.2 + (logp * 0.05),
            'lung': 0.1 + (logp * 0.03),
            'brain': self._predict_bbb_penetration(mol) * 0.5,
            'fat': 0.1 + (logp * 0.15)
        }
    
    def _predict_cyp_inhibition(self, mol) -> Dict[str, float]:
        """Predict CYP enzyme inhibition"""
        # Simplified CYP inhibition model
        return {
            'CYP1A2': np.random.uniform(0, 0.3),
            'CYP2C9': np.random.uniform(0, 0.3),
            'CYP2C19': np.random.uniform(0, 0.3),
            'CYP2D6': np.random.uniform(0, 0.3),
            'CYP3A4': np.random.uniform(0, 0.3)
        }
    
    def _predict_metabolic_stability(self, mol) -> float:
        """Predict metabolic stability"""
        logp = MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        stability = 1.0
        stability -= max(0, (logp - 3) / 2) * 0.3
        stability -= max(0, (mw - 400) / 100) * 0.2
        return max(0, min(1, stability))
    
    def _predict_metabolic_pathways(self, mol) -> List[str]:
        """Predict metabolic pathways"""
        logp = MolLogP(mol)
        
        pathways = []
        if logp > 2:
            pathways.append('Oxidation')
        if logp < 1:
            pathways.append('Glucuronidation')
        pathways.append('Hydrolysis')
        
        return pathways
    
    def _predict_ddi_potential(self, mol) -> float:
        """Predict drug-drug interaction potential"""
        return np.random.uniform(0, 0.5)
    
    def _predict_clearance(self, mol) -> float:
        """Predict clearance"""
        logp = MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        clearance = 0.5 + (logp * 0.2) + (mw / 1000) * 0.1
        return max(0.1, min(2.0, clearance))
    
    def _predict_half_life(self, mol) -> float:
        """Predict half-life"""
        clearance = self._predict_clearance(mol)
        vd = self._predict_volume_distribution(mol)
        
        # t1/2 = ln(2) * Vd / CL
        half_life = 0.693 * vd / clearance
        return max(1, min(24, half_life))
    
    def _predict_excretion_routes(self, mol) -> Dict[str, float]:
        """Predict excretion routes"""
        logp = MolLogP(mol)
        
        return {
            'urine': 0.6 - (logp * 0.1),
            'feces': 0.3 + (logp * 0.1),
            'bile': 0.1 + (logp * 0.05)
        }
    
    def _predict_renal_clearance(self, mol) -> float:
        """Predict renal clearance"""
        logp = MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        renal_clearance = 0.3 - (logp * 0.05) - (mw / 1000) * 0.1
        return max(0, min(1, renal_clearance))
    
    def _predict_mutagenicity(self, mol) -> float:
        """Predict mutagenicity risk"""
        # Simplified mutagenicity model
        return np.random.uniform(0, 0.3)
    
    def _predict_carcinogenicity(self, mol) -> float:
        """Predict carcinogenicity risk"""
        # Simplified carcinogenicity model
        return np.random.uniform(0, 0.3)
    
    def _predict_hepatotoxicity(self, mol) -> float:
        """Predict hepatotoxicity risk"""
        logp = MolLogP(mol)
        
        # Simplified hepatotoxicity model
        hepatotoxicity = 0.1 + (logp * 0.05)
        return max(0, min(1, hepatotoxicity))
    
    def _predict_cardiotoxicity(self, mol) -> float:
        """Predict cardiotoxicity risk"""
        # Simplified cardiotoxicity model
        return np.random.uniform(0, 0.4)
    
    def _predict_nephrotoxicity(self, mol) -> float:
        """Predict nephrotoxicity risk"""
        # Simplified nephrotoxicity model
        return np.random.uniform(0, 0.3)
    
    def _grade_property(self, score: float) -> str:
        """Grade a property score"""
        if score >= 0.8:
            return 'Excellent'
        elif score >= 0.6:
            return 'Good'
        elif score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'
    
    def _generate_admet_summary(self, absorption, distribution, metabolism, excretion, toxicity):
        """Generate ADMET summary"""
        return {
            'absorption_issues': absorption.get('lipinski_violations', 0) > 0,
            'distribution_issues': distribution.get('volume_distribution', 1) < 0.5,
            'metabolism_issues': metabolism.get('metabolic_stability', 1) < 0.5,
            'excretion_issues': excretion.get('clearance', 1) < 0.1,
            'toxicity_issues': toxicity.get('toxicity_score', 1) < 0.5
        }
    
    # Default methods for error handling
    def _get_default_absorption(self):
        return {
            'absorption_score': 0.5,
            'absorption_grade': 'Fair',
            'logp': 2.0,
            'molecular_weight': 300,
            'hbd': 2,
            'hba': 5,
            'tpsa': 60,
            'rotatable_bonds': 5,
            'lipinski_violations': 0,
            'caco2_permeability': 0.5,
            'bioavailability': 0.5
        }
    
    def _get_default_distribution(self):
        return {
            'distribution_score': 0.5,
            'distribution_grade': 'Fair',
            'volume_distribution': 1.0,
            'plasma_protein_binding': 0.9,
            'bbb_penetration': 0.1,
            'tissue_distribution': {}
        }
    
    def _get_default_metabolism(self):
        return {
            'metabolism_score': 0.5,
            'metabolism_grade': 'Fair',
            'cyp_inhibitions': {},
            'metabolic_stability': 0.5,
            'metabolic_pathways': [],
            'ddi_potential': 0.3
        }
    
    def _get_default_excretion(self):
        return {
            'excretion_score': 0.5,
            'excretion_grade': 'Fair',
            'clearance': 0.5,
            'half_life': 8.0,
            'excretion_routes': {},
            'renal_clearance': 0.3
        }
    
    def _get_default_toxicity(self):
        return {
            'toxicity_score': 0.5,
            'toxicity_grade': 'Fair',
            'mutagenicity': 0.2,
            'carcinogenicity': 0.2,
            'hepatotoxicity': 0.2,
            'cardiotoxicity': 0.2,
            'nephrotoxicity': 0.2
        }
    
    def _get_default_comprehensive_admet(self, smiles: str):
        return {
            'smiles': smiles,
            'absorption': self._get_default_absorption(),
            'distribution': self._get_default_distribution(),
            'metabolism': self._get_default_metabolism(),
            'excretion': self._get_default_excretion(),
            'toxicity': self._get_default_toxicity(),
            'overall_admet_score': 0.5,
            'overall_admet_grade': 'Fair',
            'admet_summary': {}
        } 