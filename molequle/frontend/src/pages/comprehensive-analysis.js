import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ComprehensiveAnalysis = () => {
  const [smiles, setSmiles] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!smiles.trim()) {
      setError('Please enter a SMILES string');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_URL}/api/v1/comprehensive-analysis`, {
        smiles: smiles.trim()
      });

      setAnalysis(response.data.results);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const getGradeColor = (grade) => {
    switch (grade) {
      case 'Excellent': return 'text-green-600 bg-green-100';
      case 'Good': return 'text-blue-600 bg-blue-100';
      case 'Fair': return 'text-yellow-600 bg-yellow-100';
      case 'Poor': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-blue-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Comprehensive Drug Analysis
        </h1>
        <p className="text-gray-600">
          Analyze molecules with comprehensive ADMET, synthetic accessibility, stability, selectivity, and clinical relevance predictions.
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="smiles" className="block text-sm font-medium text-gray-700 mb-2">
              SMILES String
            </label>
            <input
              type="text"
              id="smiles"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="Enter SMILES string (e.g., N[Pt](N)(Cl)Cl)"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Analyzing...' : 'Analyze Molecule'}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
      </div>

      {/* Analysis Results */}
      {analysis && (
        <div className="space-y-8">
          {/* Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Comprehensive Score</h3>
              <div className={`text-3xl font-bold ${getScoreColor(analysis.comprehensive_score)}`}>
                {(analysis.comprehensive_score * 100).toFixed(1)}%
              </div>
              <div className={`inline-block px-2 py-1 rounded-full text-sm font-medium ${getGradeColor(analysis.comprehensive_grade)}`}>
                {analysis.comprehensive_grade}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Binding Affinity</h3>
              <div className="text-3xl font-bold text-blue-600">
                {analysis.qnn_predictions?.binding_affinity?.toFixed(2) || 'N/A'} kcal/mol
              </div>
              <div className="text-sm text-gray-600">Lower is better</div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">ADMET Score</h3>
              <div className={`text-3xl font-bold ${getScoreColor(analysis.admet_analysis?.overall_admet_score)}`}>
                {((analysis.admet_analysis?.overall_admet_score || 0.5) * 100).toFixed(1)}%
              </div>
              <div className={`inline-block px-2 py-1 rounded-full text-sm font-medium ${getGradeColor(analysis.admet_analysis?.overall_admet_grade)}`}>
                {analysis.admet_analysis?.overall_admet_grade || 'Fair'}
              </div>
            </div>
          </div>

          {/* Detailed Analysis */}
          <div className="bg-white rounded-lg shadow-md">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Analysis</h3>
              
              {/* ADMET Properties */}
              <div className="mb-8">
                <h4 className="text-md font-semibold text-gray-900 mb-3">ADMET Properties</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                  {['absorption', 'distribution', 'metabolism', 'excretion', 'toxicity'].map((property) => (
                    <div key={property} className="bg-gray-50 rounded-lg p-4">
                      <h5 className="font-medium text-gray-900 capitalize mb-2">{property}</h5>
                      <div className={`text-2xl font-bold ${getScoreColor(analysis.admet_analysis?.[property]?.[`${property}_score`] || 0.5)}`}>
                        {((analysis.admet_analysis?.[property]?.[`${property}_score`] || 0.5) * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Synthetic Accessibility */}
              <div className="mb-8">
                <h4 className="text-md font-semibold text-gray-900 mb-3">Synthetic Accessibility</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Feasibility Score</h5>
                    <div className={`text-2xl font-bold ${getScoreColor(analysis.synthetic_analysis?.feasibility_score || 0.5)}`}>
                      {((analysis.synthetic_analysis?.feasibility_score || 0.5) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Reaction Steps</h5>
                    <div className="text-2xl font-bold text-blue-600">
                      {analysis.synthetic_analysis?.reaction_steps?.total_steps || 'N/A'}
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Starting Materials</h5>
                    <div className="text-2xl font-bold text-blue-600">
                      {analysis.synthetic_analysis?.starting_materials?.total_materials || 'N/A'}
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Reaction Yield</h5>
                    <div className="text-2xl font-bold text-blue-600">
                      {((analysis.synthetic_analysis?.reaction_yields?.total_yield || 0.7) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Stability Analysis */}
              <div className="mb-8">
                <h4 className="text-md font-semibold text-gray-900 mb-3">Stability Analysis</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {['chemical', 'biological', 'storage'].map((type) => (
                    <div key={type} className="bg-gray-50 rounded-lg p-4">
                      <h5 className="font-medium text-gray-900 capitalize mb-2">{type} Stability</h5>
                      <div className={`text-2xl font-bold ${getScoreColor(analysis.stability_analysis?.[`${type}_stability`]?.stability_score || 0.5)}`}>
                        {((analysis.stability_analysis?.[`${type}_stability`]?.stability_score || 0.5) * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Selectivity Analysis */}
              <div className="mb-8">
                <h4 className="text-md font-semibold text-gray-900 mb-3">Selectivity Analysis</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Target Selectivity</h5>
                    <div className={`text-2xl font-bold ${getScoreColor(analysis.selectivity_analysis?.target_selectivity?.selectivity_score || 0.5)}`}>
                      {((analysis.selectivity_analysis?.target_selectivity?.selectivity_score || 0.5) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Off-target Risk</h5>
                    <div className={`text-2xl font-bold ${getScoreColor(1 - (analysis.selectivity_analysis?.off_target_binding?.overall_risk || 0.5))}`}>
                      {((1 - (analysis.selectivity_analysis?.off_target_binding?.overall_risk || 0.5)) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Side Effect Risk</h5>
                    <div className={`text-2xl font-bold ${getScoreColor(1 - (analysis.selectivity_analysis?.side_effects?.overall_risk || 0.5))}`}>
                      {((1 - (analysis.selectivity_analysis?.side_effects?.overall_risk || 0.5)) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Therapeutic Index</h5>
                    <div className="text-2xl font-bold text-blue-600">
                      {analysis.selectivity_analysis?.therapeutic_index?.therapeutic_index?.toFixed(1) || 'N/A'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Clinical Analysis */}
              <div>
                <h4 className="text-md font-semibold text-gray-900 mb-3">Clinical Relevance</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Pathway Targeting</h5>
                    <div className={`text-2xl font-bold ${getScoreColor(analysis.clinical_analysis?.cancer_pathway_targeting?.targeting_score || 0.5)}`}>
                      {((analysis.clinical_analysis?.cancer_pathway_targeting?.targeting_score || 0.5) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Patient Population</h5>
                    <div className={`text-2xl font-bold ${getScoreColor(analysis.clinical_analysis?.patient_population?.population_score || 0.5)}`}>
                      {((analysis.clinical_analysis?.patient_population?.population_score || 0.5) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Trial Readiness</h5>
                    <div className={`text-2xl font-bold ${getScoreColor(analysis.clinical_analysis?.clinical_trial_readiness?.readiness_score || 0.5)}`}>
                      {((analysis.clinical_analysis?.clinical_trial_readiness?.readiness_score || 0.5) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Regulatory Pathway</h5>
                    <div className={`text-2xl font-bold ${getScoreColor(analysis.clinical_analysis?.regulatory_pathway?.pathway_score || 0.5)}`}>
                      {((analysis.clinical_analysis?.regulatory_pathway?.pathway_score || 0.5) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Summary */}
          {analysis.summary && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Summary</h3>
              
              {analysis.summary.key_strengths && analysis.summary.key_strengths.length > 0 && (
                <div className="mb-4">
                  <h4 className="font-medium text-green-700 mb-2">Key Strengths</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {analysis.summary.key_strengths.map((strength, index) => (
                      <li key={index} className="text-green-600">{strength}</li>
                    ))}
                  </ul>
                </div>
              )}

              {analysis.summary.key_concerns && analysis.summary.key_concerns.length > 0 && (
                <div className="mb-4">
                  <h4 className="font-medium text-red-700 mb-2">Key Concerns</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {analysis.summary.key_concerns.map((concern, index) => (
                      <li key={index} className="text-red-600">{concern}</li>
                    ))}
                  </ul>
                </div>
              )}

              {analysis.summary.recommendations && analysis.summary.recommendations.length > 0 && (
                <div>
                  <h4 className="font-medium text-blue-700 mb-2">Recommendations</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {analysis.summary.recommendations.map((recommendation, index) => (
                      <li key={index} className="text-blue-600">{recommendation}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ComprehensiveAnalysis; 