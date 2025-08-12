import React, { useState } from 'react';

const ComprehensiveAnalysisCard = ({ analog, onViewDetails }) => {
  const [expanded, setExpanded] = useState(false);

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-blue-600 bg-blue-100';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getReadinessColor = (readiness) => {
    switch (readiness) {
      case 'ready':
        return 'text-green-600 bg-green-100';
      case 'needs_optimization':
        return 'text-yellow-600 bg-yellow-100';
      case 'requires_work':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const formatScore = (score) => {
    return score ? (score * 100).toFixed(1) + '%' : 'N/A';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
            <span className="text-blue-600 font-bold text-sm">{analog.rank}</span>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {analog.analog_id}
            </h3>
            <p className="text-sm text-gray-500 font-mono">{analog.smiles}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getReadinessColor(analog.clinical_readiness)}`}>
            {analog.clinical_readiness?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}
          </span>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-gray-400 hover:text-gray-600"
          >
            {expanded ? '▼' : '▶'}
          </button>
        </div>
      </div>

      {/* Basic Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="text-center">
          <p className="text-sm text-gray-600">Binding Affinity</p>
          <p className="text-lg font-semibold text-gray-900">
            {analog.binding_affinity?.toFixed(2) || 'N/A'} kcal/mol
          </p>
        </div>
        <div className="text-center">
          <p className="text-sm text-gray-600">Comprehensive Score</p>
          <p className={`text-lg font-semibold ${getScoreColor(analog.comprehensive_score).split(' ')[0]}`}>
            {formatScore(analog.comprehensive_score)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-sm text-gray-600">ADMET Score</p>
          <p className={`text-lg font-semibold ${getScoreColor(analog.admet_score).split(' ')[0]}`}>
            {formatScore(analog.admet_score)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-sm text-gray-600">Safety Score</p>
          <p className={`text-lg font-semibold ${getScoreColor(analog.safety_score).split(' ')[0]}`}>
            {formatScore(analog.safety_score)}
          </p>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="border-t border-gray-200 pt-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Experimental Validation */}
            <div>
              <h4 className="text-md font-semibold text-gray-900 mb-3">Experimental Validation</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Validation Available:</span>
                  <span className="text-sm font-medium">
                    {analog.detailed_analysis?.experimental_validation?.validation_available ? 'Yes' : 'No'}
                  </span>
                </div>
                {analog.detailed_analysis?.experimental_validation?.validation_available && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Confidence Level:</span>
                      <span className={`text-sm font-medium px-2 py-1 rounded ${getScoreColor(analog.detailed_analysis.experimental_validation.confidence_score).split(' ').slice(1).join(' ')}`}>
                        {analog.detailed_analysis.experimental_validation.confidence_level?.replace('_', ' ').toUpperCase()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Prediction Error:</span>
                      <span className="text-sm font-medium">
                        {analog.detailed_analysis.experimental_validation.prediction_error?.toFixed(2) || 'N/A'} kcal/mol
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Cytotoxicity Analysis */}
            <div>
              <h4 className="text-md font-semibold text-gray-900 mb-3">Cytotoxicity Analysis</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Cytotoxicity Score:</span>
                  <span className={`text-sm font-medium ${getScoreColor(analog.cytotoxicity_score).split(' ')[0]}`}>
                    {formatScore(analog.cytotoxicity_score)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Cancer Pathway Score:</span>
                  <span className={`text-sm font-medium ${getScoreColor(analog.cancer_pathway_score).split(' ')[0]}`}>
                    {formatScore(analog.cancer_pathway_score)}
                  </span>
                </div>
                {analog.detailed_analysis?.cytotoxicity_predictions?.selectivity_analysis && (
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Selectivity Index:</span>
                    <span className="text-sm font-medium">
                      {analog.detailed_analysis.cytotoxicity_predictions.selectivity_analysis.pancreatic_cancer?.selectivity_index?.toFixed(1) || 'N/A'}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* ADMET Analysis */}
            <div>
              <h4 className="text-md font-semibold text-gray-900 mb-3">ADMET Analysis</h4>
              <div className="space-y-2">
                {analog.detailed_analysis?.admet_predictions && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Oral Bioavailability:</span>
                      <span className="text-sm font-medium">
                        {analog.detailed_analysis.admet_predictions.absorption?.oral_bioavailability_percent?.toFixed(1) || 'N/A'}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Toxicity Risk:</span>
                      <span className={`text-sm font-medium px-2 py-1 rounded ${
                        analog.detailed_analysis.admet_predictions.toxicity?.toxicity_risk === 'low_risk' 
                          ? 'text-green-600 bg-green-100' 
                          : 'text-red-600 bg-red-100'
                      }`}>
                        {analog.detailed_analysis.admet_predictions.toxicity?.toxicity_risk?.replace('_', ' ').toUpperCase() || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Metabolic Stability:</span>
                      <span className="text-sm font-medium">
                        {formatScore(analog.detailed_analysis.admet_predictions.metabolism?.metabolic_stability_score)}
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Clinical Assessment */}
            <div>
              <h4 className="text-md font-semibold text-gray-900 mb-3">Clinical Assessment</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Development Priority:</span>
                  <span className={`text-sm font-medium px-2 py-1 rounded ${
                    analog.detailed_analysis?.comprehensive_scoring?.development_recommendations?.development_priority === 'high_priority'
                      ? 'text-green-600 bg-green-100'
                      : 'text-yellow-600 bg-yellow-100'
                  }`}>
                    {analog.detailed_analysis?.comprehensive_scoring?.development_recommendations?.development_priority?.replace('_', ' ').toUpperCase() || 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Risk Level:</span>
                  <span className={`text-sm font-medium px-2 py-1 rounded ${
                    analog.detailed_analysis?.comprehensive_scoring?.risk_assessment?.risk_level === 'low'
                      ? 'text-green-600 bg-green-100'
                      : 'text-red-600 bg-red-100'
                  }`}>
                    {analog.detailed_analysis?.comprehensive_scoring?.risk_assessment?.risk_level?.toUpperCase() || 'N/A'}
                  </span>
                </div>
                {analog.detailed_analysis?.comprehensive_scoring?.clinical_assessment?.clinical_readiness && (
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Clinical Readiness:</span>
                    <span className={`text-sm font-medium px-2 py-1 rounded ${getReadinessColor(analog.detailed_analysis.comprehensive_scoring.clinical_assessment.clinical_readiness)}`}>
                      {analog.detailed_analysis.comprehensive_scoring.clinical_assessment.clinical_readiness.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="mt-6 flex space-x-3">
            <button
              onClick={() => onViewDetails(analog)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              View Full Analysis
            </button>
            <button
              onClick={() => window.open(`/results/${analog.job_id}`, '_blank')}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              View Results
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ComprehensiveAnalysisCard; 