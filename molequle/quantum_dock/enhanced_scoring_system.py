import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDrugScore:
    """
    Enhanced comprehensive scoring system that integrates all drug discovery metrics
    into a unified clinical score for real-world applicability.
    """
    
    def __init__(self):
        self.scoring_weights = {
            'binding_affinity': 0.15,
            'cell_cytotoxicity': 0.20,
            'admet_score': 0.25,
            'cancer_pathway': 0.20,
            'safety_profile': 0.20
        }
        
        self.clinical_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'moderate': 0.4,
            'poor': 0.2
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_comprehensive_score(self, compound_data: Dict) -> Dict:
        """
        Calculate comprehensive drug score integrating all metrics.
        
        Args:
            compound_data: Dictionary containing all compound analysis results
            
        Returns:
            Dictionary with comprehensive scoring results
        """
        try:
            # Extract individual scores
            scores = self._extract_individual_scores(compound_data)
            
            # Calculate weighted comprehensive score
            comprehensive_score = self._calculate_weighted_score(scores)
            
            # Assess clinical readiness
            clinical_assessment = self._assess_clinical_readiness(scores, comprehensive_score)
            
            # Generate development recommendations
            development_recommendations = self._generate_development_recommendations(scores)
            
            # Calculate risk assessment
            risk_assessment = self._calculate_risk_assessment(scores)
            
            return {
                'comprehensive_score': comprehensive_score,
                'individual_scores': scores,
                'clinical_assessment': clinical_assessment,
                'development_recommendations': development_recommendations,
                'risk_assessment': risk_assessment,
                'clinical_implications': self._get_clinical_implications(comprehensive_score, scores)
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive scoring failed: {e}")
            return {'error': str(e)}
    
    def _extract_individual_scores(self, compound_data: Dict) -> Dict:
        """Extract individual scores from compound analysis data."""
        scores = {}
        
        # Binding affinity score (normalize to 0-1 scale)
        binding_affinity = compound_data.get('binding_affinity', -7.0)
        scores['binding_affinity'] = self._normalize_binding_affinity(binding_affinity)
        
        # Cell cytotoxicity score
        cytotoxicity_data = compound_data.get('cytotoxicity_predictions', {})
        scores['cell_cytotoxicity'] = self._calculate_cytotoxicity_score(cytotoxicity_data)
        
        # ADMET score
        admet_data = compound_data.get('admet_predictions', {})
        scores['admet_score'] = self._calculate_admet_score(admet_data)
        
        # Cancer pathway score
        cancer_data = compound_data.get('cancer_pathway_analysis', {})
        scores['cancer_pathway'] = self._calculate_cancer_pathway_score(cancer_data)
        
        # Safety profile score
        safety_data = compound_data.get('toxicity_predictions', {})
        scores['safety_profile'] = self._calculate_safety_score(safety_data)
        
        return scores
    
    def _normalize_binding_affinity(self, binding_affinity: float) -> float:
        """Normalize binding affinity to 0-1 scale."""
        # Convert from kcal/mol to normalized score
        # -10 kcal/mol = 1.0, -5 kcal/mol = 0.5, 0 kcal/mol = 0.0
        normalized = max(0, min(1, (binding_affinity + 10) / 10))
        return normalized
    
    def _calculate_cytotoxicity_score(self, cytotoxicity_data: Dict) -> float:
        """Calculate cytotoxicity score from cell-based data."""
        if not cytotoxicity_data:
            return 0.5  # Default moderate score
        
        try:
            # Extract cancer cell IC50 values
            cancer_ic50s = []
            for cancer_type, cell_data in cytotoxicity_data.get('cytotoxicity_predictions', {}).items():
                if cancer_type != 'normal_cells':
                    for cell_line, data in cell_data.items():
                        cancer_ic50s.append(data.get('ic50_um', 50))
            
            if not cancer_ic50s:
                return 0.5
            
            # Calculate average IC50 and convert to score
            avg_ic50 = np.mean(cancer_ic50s)
            cytotoxicity_score = max(0, min(1, 1 - (avg_ic50 / 100)))
            
            return cytotoxicity_score
            
        except Exception as e:
            self.logger.warning(f"Cytotoxicity score calculation failed: {e}")
            return 0.5
    
    def _calculate_admet_score(self, admet_data: Dict) -> float:
        """Calculate ADMET score from ADMET predictions."""
        if not admet_data:
            return 0.5  # Default moderate score
        
        try:
            # Extract ADMET scores
            absorption_score = admet_data.get('absorption', {}).get('oral_bioavailability_percent', 50) / 100
            toxicity_score = 1 - admet_data.get('toxicity', {}).get('toxicity_probability', 0.5)
            metabolism_score = admet_data.get('metabolism', {}).get('metabolic_stability_score', 0.5)
            
            # Calculate weighted ADMET score
            admet_score = (0.4 * absorption_score + 0.4 * toxicity_score + 0.2 * metabolism_score)
            
            return max(0, min(1, admet_score))
            
        except Exception as e:
            self.logger.warning(f"ADMET score calculation failed: {e}")
            return 0.5
    
    def _calculate_cancer_pathway_score(self, cancer_data: Dict) -> float:
        """Calculate cancer pathway score from pathway analysis."""
        if not cancer_data:
            return 0.5  # Default moderate score
        
        try:
            # Extract pathway scores
            pathway_scores = []
            for cancer_type, analysis in cancer_data.get('cancer_type_analysis', {}).items():
                pathway_scores.append(analysis.get('cancer_specific_score', 0.5))
            
            if not pathway_scores:
                return 0.5
            
            # Calculate average pathway score
            cancer_pathway_score = np.mean(pathway_scores)
            
            return max(0, min(1, cancer_pathway_score))
            
        except Exception as e:
            self.logger.warning(f"Cancer pathway score calculation failed: {e}")
            return 0.5
    
    def _calculate_safety_score(self, safety_data: Dict) -> float:
        """Calculate safety profile score."""
        if not safety_data:
            return 0.5  # Default moderate score
        
        try:
            # Extract safety metrics
            toxicity_prob = safety_data.get('toxicity_probability', 0.5)
            safety_score = 1 - toxicity_prob  # Invert toxicity to get safety
            
            return max(0, min(1, safety_score))
            
        except Exception as e:
            self.logger.warning(f"Safety score calculation failed: {e}")
            return 0.5
    
    def _calculate_weighted_score(self, scores: Dict) -> float:
        """Calculate weighted comprehensive score."""
        weighted_score = 0
        
        for metric, score in scores.items():
            weight = self.scoring_weights.get(metric, 0.2)
            weighted_score += score * weight
        
        return max(0, min(1, weighted_score))
    
    def _assess_clinical_readiness(self, scores: Dict, comprehensive_score: float) -> Dict:
        """Assess clinical readiness based on comprehensive score and individual metrics."""
        # Check if all critical metrics meet minimum thresholds
        critical_metrics = {
            'binding_affinity': 0.4,  # At least moderate binding
            'cell_cytotoxicity': 0.5,  # At least moderate cytotoxicity
            'admet_score': 0.6,        # Good ADMET profile required
            'safety_profile': 0.7      # High safety required
        }
        
        failed_metrics = []
        for metric, threshold in critical_metrics.items():
            if scores.get(metric, 0) < threshold:
                failed_metrics.append(metric)
        
        # Determine clinical readiness
        if comprehensive_score >= 0.8 and len(failed_metrics) == 0:
            readiness = 'ready_for_clinical_trials'
        elif comprehensive_score >= 0.6 and len(failed_metrics) <= 1:
            readiness = 'needs_minor_optimization'
        elif comprehensive_score >= 0.4 and len(failed_metrics) <= 2:
            readiness = 'needs_significant_optimization'
        else:
            readiness = 'requires_major_redesign'
        
        return {
            'clinical_readiness': readiness,
            'comprehensive_score': comprehensive_score,
            'failed_critical_metrics': failed_metrics,
            'readiness_level': self._get_readiness_level(comprehensive_score)
        }
    
    def _get_readiness_level(self, score: float) -> str:
        """Get readiness level based on comprehensive score."""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'moderate'
        else:
            return 'poor'
    
    def _generate_development_recommendations(self, scores: Dict) -> Dict:
        """Generate development recommendations based on individual scores."""
        recommendations = {
            'priority_actions': [],
            'optimization_targets': [],
            'risk_mitigation': [],
            'development_timeline': self._estimate_development_timeline(scores)
        }
        
        # Check binding affinity
        if scores.get('binding_affinity', 0) < 0.6:
            recommendations['optimization_targets'].append('improve_binding_affinity')
            recommendations['priority_actions'].append('conduct_structure_activity_relationship_studies')
        
        # Check cytotoxicity
        if scores.get('cell_cytotoxicity', 0) < 0.5:
            recommendations['optimization_targets'].append('enhance_cytotoxicity')
            recommendations['priority_actions'].append('screen_additional_cancer_cell_lines')
        
        # Check ADMET
        if scores.get('admet_score', 0) < 0.6:
            recommendations['optimization_targets'].append('optimize_admet_properties')
            recommendations['priority_actions'].append('conduct_admet_studies')
        
        # Check safety
        if scores.get('safety_profile', 0) < 0.7:
            recommendations['risk_mitigation'].append('conduct_toxicity_studies')
            recommendations['priority_actions'].append('implement_safety_monitoring')
        
        # Check cancer pathway
        if scores.get('cancer_pathway', 0) < 0.5:
            recommendations['optimization_targets'].append('improve_cancer_pathway_targeting')
            recommendations['priority_actions'].append('conduct_pathway_analysis')
        
        return recommendations
    
    def _estimate_development_timeline(self, scores: Dict) -> str:
        """Estimate development timeline based on scores."""
        avg_score = np.mean(list(scores.values()))
        
        if avg_score >= 0.7:
            return '2-3_years_to_clinical_trials'
        elif avg_score >= 0.5:
            return '3-5_years_to_clinical_trials'
        elif avg_score >= 0.3:
            return '5-7_years_to_clinical_trials'
        else:
            return '7-10_years_to_clinical_trials'
    
    def _calculate_risk_assessment(self, scores: Dict) -> Dict:
        """Calculate comprehensive risk assessment."""
        risks = []
        risk_level = 'low'
        
        # Assess individual risks
        if scores.get('binding_affinity', 0) < 0.4:
            risks.append('poor_binding_affinity')
        
        if scores.get('cell_cytotoxicity', 0) < 0.3:
            risks.append('low_cytotoxicity')
        
        if scores.get('admet_score', 0) < 0.5:
            risks.append('poor_admet_profile')
        
        if scores.get('safety_profile', 0) < 0.6:
            risks.append('safety_concerns')
        
        if scores.get('cancer_pathway', 0) < 0.4:
            risks.append('poor_cancer_targeting')
        
        # Determine overall risk level
        if len(risks) >= 4:
            risk_level = 'very_high'
        elif len(risks) >= 3:
            risk_level = 'high'
        elif len(risks) >= 2:
            risk_level = 'moderate'
        elif len(risks) >= 1:
            risk_level = 'low'
        else:
            risk_level = 'very_low'
        
        return {
            'risk_level': risk_level,
            'identified_risks': risks,
            'risk_mitigation_strategies': self._get_risk_mitigation_strategies(risks)
        }
    
    def _get_risk_mitigation_strategies(self, risks: List[str]) -> List[str]:
        """Get risk mitigation strategies for identified risks."""
        strategies = []
        
        risk_strategies = {
            'poor_binding_affinity': 'conduct_structure_optimization',
            'low_cytotoxicity': 'screen_additional_cell_lines',
            'poor_admet_profile': 'conduct_admet_optimization',
            'safety_concerns': 'implement_safety_monitoring',
            'poor_cancer_targeting': 'conduct_pathway_analysis'
        }
        
        for risk in risks:
            if risk in risk_strategies:
                strategies.append(risk_strategies[risk])
        
        return strategies
    
    def _get_clinical_implications(self, comprehensive_score: float, scores: Dict) -> Dict:
        """Get clinical implications of the comprehensive score."""
        return {
            'clinical_potential': self._assess_clinical_potential(comprehensive_score),
            'patient_population': self._determine_patient_population(scores),
            'dosing_strategy': self._suggest_dosing_strategy(scores),
            'monitoring_requirements': self._get_monitoring_requirements(scores),
            'combination_therapy': self._suggest_combination_therapy(scores)
        }
    
    def _assess_clinical_potential(self, score: float) -> str:
        """Assess clinical potential based on comprehensive score."""
        if score >= 0.8:
            return 'high_clinical_potential'
        elif score >= 0.6:
            return 'moderate_clinical_potential'
        elif score >= 0.4:
            return 'low_clinical_potential'
        else:
            return 'unlikely_clinical_success'
    
    def _determine_patient_population(self, scores: Dict) -> str:
        """Determine suitable patient population."""
        safety_score = scores.get('safety_profile', 0.5)
        admet_score = scores.get('admet_score', 0.5)
        
        if safety_score >= 0.8 and admet_score >= 0.7:
            return 'broad_patient_population'
        elif safety_score >= 0.6 and admet_score >= 0.5:
            return 'targeted_patient_population'
        else:
            return 'restricted_patient_population'
    
    def _suggest_dosing_strategy(self, scores: Dict) -> str:
        """Suggest dosing strategy based on scores."""
        admet_score = scores.get('admet_score', 0.5)
        safety_score = scores.get('safety_profile', 0.5)
        
        if admet_score >= 0.7 and safety_score >= 0.8:
            return 'standard_dosing'
        elif admet_score >= 0.5 and safety_score >= 0.6:
            return 'conservative_dosing'
        else:
            return 'individualized_dosing'
    
    def _get_monitoring_requirements(self, scores: Dict) -> List[str]:
        """Get monitoring requirements based on scores."""
        requirements = []
        
        if scores.get('safety_profile', 0.5) < 0.7:
            requirements.append('liver_function_monitoring')
            requirements.append('kidney_function_monitoring')
        
        if scores.get('admet_score', 0.5) < 0.6:
            requirements.append('drug_level_monitoring')
        
        if scores.get('cell_cytotoxicity', 0.5) < 0.5:
            requirements.append('efficacy_monitoring')
        
        return requirements
    
    def _suggest_combination_therapy(self, scores: Dict) -> List[str]:
        """Suggest combination therapy partners."""
        suggestions = []
        
        if scores.get('binding_affinity', 0.5) < 0.6:
            suggestions.append('dna_damage_agents')
        
        if scores.get('cell_cytotoxicity', 0.5) < 0.5:
            suggestions.append('cytotoxic_agents')
        
        if scores.get('cancer_pathway', 0.5) < 0.5:
            suggestions.append('targeted_therapies')
        
        return suggestions
    
    def get_validation_summary(self) -> Dict:
        """Get summary of comprehensive scoring system."""
        return {
            'scoring_metrics': list(self.scoring_weights.keys()),
            'clinical_relevance': 'high',
            'validation_status': 'active',
            'integration_level': 'comprehensive'
        } 