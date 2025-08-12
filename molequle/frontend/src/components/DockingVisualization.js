import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DockingVisualization = ({ analog, onClose }) => {
  const [dockingResults, setDockingResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState('DNA');
  const [selectedMethod, setSelectedMethod] = useState('basic');
  const [error, setError] = useState(null);
  const [targetInfo, setTargetInfo] = useState(null);
  const [showTargetInfo, setShowTargetInfo] = useState(false);

  const targetOptions = [
    { value: 'DNA', label: 'DNA (Guanine N7)', description: 'Primary cisplatin binding site' },
    { value: 'GSTP1', label: 'GSTP1 Enzyme', description: 'Glutathione S-transferase P1 active site' }
  ];

  const methodOptions = [
    { value: 'basic', label: 'Basic Analysis', description: 'Rapid binding assessment that quickly evaluates how well your analog fits into the target binding site. Provides fundamental binding scores and identifies key molecular interactions like hydrogen bonds and coordination sites. Best for initial screening and fast results.' },
    { value: 'qaoa', label: 'QAOA Quantum', description: 'Advanced quantum computing approach that explores multiple binding configurations simultaneously to find optimal drug-target interactions. Uses quantum algorithms to discover binding poses that classical methods might miss, potentially revealing stronger binding affinities and novel interaction patterns.' },
    { value: 'classical', label: 'Classical Force Field', description: 'Comprehensive molecular mechanics simulation that models realistic atomic forces and molecular flexibility. Calculates detailed binding energies by considering how atoms attract and repel each other, providing highly accurate predictions of drug stability and binding strength in biological conditions.' },
    { value: 'grid_search', label: 'Grid Search', description: 'Systematic exploration method that tests thousands of different binding orientations and positions around the target site. Ensures no potential binding pose is missed by methodically sampling the entire binding pocket, ideal for discovering unexpected binding modes.' },
    { value: 'vina', label: 'AutoDock Vina', description: 'Industry-standard pharmaceutical docking engine used by major drug companies worldwide. Provides publication-quality results with proven accuracy for drug development. Generates multiple binding poses ranked by pharmaceutical relevance, including detailed interaction analysis and binding affinity predictions.' }
  ];

  const fetchTargetInfo = async (targetName) => {
    try {
      const response = await axios.get(`http://localhost:8002/target-info/${targetName}`);
      setTargetInfo(response.data);
    } catch (error) {
      console.error('Failed to fetch target info:', error);
    }
  };

  // Fetch target info when target changes
  useEffect(() => {
    if (selectedTarget) {
      fetchTargetInfo(selectedTarget);
    }
  }, [selectedTarget]);

  const runDocking = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log(`Running docking for ${analog.analog_id} against ${selectedTarget}`);
      
      const response = await axios.post('http://localhost:8002/dock-molecule', {
        analog_id: analog.analog_id,
        analog_smiles: analog.smiles,
        target_protein: selectedTarget,
        method: selectedMethod
      });
      
      setDockingResults(response.data);
      console.log('Docking completed:', response.data);
      
    } catch (error) {
      console.error('Docking failed:', error);
      setError(error.response?.data?.detail || 'Docking analysis failed');
    }
    
    setLoading(false);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-6xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b border-gray-200">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              🎯 Molecular Docking Analysis
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              Analyze binding interactions for {analog.analog_id}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl"
          >
            ×
          </button>
        </div>

        {/* Configuration Section */}
        <div className="p-6 border-b border-gray-200 bg-gray-50">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Target Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Protein
              </label>
              <select
                value={selectedTarget}
                onChange={(e) => setSelectedTarget(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                disabled={loading}
              >
                {targetOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {targetOptions.find(opt => opt.value === selectedTarget)?.description}
              </p>
            </div>

            {/* Method Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Docking Method
              </label>
              <select
                value={selectedMethod}
                onChange={(e) => setSelectedMethod(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                disabled={loading}
              >
                {methodOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {methodOptions.find(opt => opt.value === selectedMethod)?.description}
              </p>
            </div>
          </div>

          {/* Target Information */}
          {targetInfo && (
            <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex justify-between items-start">
                <div>
                  <h4 className="font-medium text-blue-900 mb-2">
                    {targetInfo.binding_site?.name || `${selectedTarget} Target`}
                  </h4>
                  <p className="text-sm text-blue-700 mb-2">
                    {targetInfo.binding_site?.description}
                  </p>
                  {targetInfo.druggability && (
                    <div className="text-xs text-blue-600">
                      <span className="font-medium">Druggability:</span>{' '}
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        targetInfo.druggability.druggability_class === 'highly_druggable' ? 'bg-green-200 text-green-800' :
                        targetInfo.druggability.druggability_class === 'moderately_druggable' ? 'bg-yellow-200 text-yellow-800' :
                        'bg-red-200 text-red-800'
                      }`}>
                        {targetInfo.druggability.druggability_class.replace('_', ' ')}
                      </span>{' '}
                      (Score: {targetInfo.druggability.overall_score.toFixed(2)})
                    </div>
                  )}
                </div>
                <button
                  onClick={() => setShowTargetInfo(!showTargetInfo)}
                  className="text-blue-600 hover:text-blue-800 text-sm"
                >
                  {showTargetInfo ? 'Hide Details' : 'Show Details'}
                </button>
              </div>
              
              {showTargetInfo && targetInfo.druggability && (
                <div className="mt-3 pt-3 border-t border-blue-200">
                  <h5 className="font-medium text-blue-900 mb-2">Druggability Analysis</h5>
                  <div className="grid md:grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="font-medium">Volume Score:</span> {targetInfo.druggability.volume_score.toFixed(2)}
                    </div>
                    <div>
                      <span className="font-medium">Hydrophobic Score:</span> {targetInfo.druggability.hydrophobic_score.toFixed(2)}
                    </div>
                  </div>
                  {targetInfo.druggability.recommendations && (
                    <div className="mt-2">
                      <span className="font-medium text-blue-900">Recommendations:</span>
                      <ul className="mt-1 text-xs text-blue-700 list-disc list-inside">
                        {targetInfo.druggability.recommendations.slice(0, 2).map((rec, index) => (
                          <li key={index}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Run Button */}
          <div className="mt-6 flex justify-center">
            <button
              onClick={runDocking}
              disabled={loading}
              className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 ${
                loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transform hover:scale-105'
              } text-white`}
            >
              {loading ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Running Docking Analysis...
                </div>
              ) : (
                'Run Docking Analysis'
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-6 bg-red-50 border-l-4 border-red-400">
            <div className="flex">
              <div className="text-red-400 mr-3">⚠️</div>
              <div>
                <h3 className="text-red-800 font-medium">Docking Error</h3>
                <p className="text-red-700 text-sm mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {dockingResults && (
          <div className="p-6">
            {/* Results Summary */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Binding Analysis Results
              </h3>
              
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="text-green-800 font-semibold">Binding Score</div>
                  <div className="text-2xl font-bold text-green-900">
                    {dockingResults.binding_score.toFixed(2)} kcal/mol
                  </div>
                  <div className="text-xs text-green-600 mt-1">
                    Lower values indicate stronger binding
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="text-blue-800 font-semibold">Target</div>
                  <div className="text-lg font-bold text-blue-900">
                    {dockingResults.target_protein}
                  </div>
                  <div className="text-xs text-blue-600 mt-1">
                    {targetOptions.find(opt => opt.value === dockingResults.target_protein)?.description}
                  </div>
                </div>

                <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                  <div className="text-purple-800 font-semibold">Method</div>
                  <div className="text-lg font-bold text-purple-900">
                    {dockingResults.method_used.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </div>
                  <div className="text-xs text-purple-600 mt-1">
                    {dockingResults.method_used.includes('qaoa') ? 'Quantum Enhanced' : 
                     dockingResults.method_used.includes('classical') ? 'Force Field' :
                     dockingResults.method_used.includes('vina') ? 'AutoDock Vina' : 'Classical Method'}
                  </div>
                  {dockingResults.quantum_enhancement && (
                    <div className="text-xs text-purple-700 mt-1 font-medium">
                      ⚛️ Quantum Optimized
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Visualization */}
            <div className="mb-6">
              <h4 className="text-md font-semibold text-gray-900 mb-3">
                3D Binding Visualization
              </h4>
              <div 
                className="border border-gray-200 rounded-lg overflow-hidden"
                dangerouslySetInnerHTML={{ __html: dockingResults.visualization_html }}
              />
            </div>

            {/* Interactions Details */}
            {dockingResults.interactions && dockingResults.interactions.length > 0 && (
              <div>
                <h4 className="text-md font-semibold text-gray-900 mb-3">
                  Key Molecular Interactions
                </h4>
                <div className="space-y-3">
                  {dockingResults.interactions.map((interaction, index) => (
                    <div key={index} className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <div className="font-medium text-gray-900">
                            {interaction.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </div>
                          <div className="text-sm text-gray-600 mt-1">
                            {interaction.atoms && interaction.atoms.length > 0 && (
                              <span>Atoms: {interaction.atoms.join(' ↔ ')}</span>
                            )}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-semibold text-gray-900">
                            {interaction.distance.toFixed(1)}Å
                          </div>
                          <div className={`text-xs px-2 py-1 rounded-full ${
                            interaction.strength === 'very_strong' ? 'bg-green-200 text-green-800' :
                            interaction.strength === 'strong' ? 'bg-yellow-200 text-yellow-800' :
                            'bg-gray-200 text-gray-800'
                          }`}>
                            {interaction.strength.replace('_', ' ')}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Additional Info */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h5 className="font-medium text-blue-900 mb-2">
                  💡 Understanding Your Results
                </h5>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• <strong>Binding Score:</strong> More negative values indicate stronger predicted binding affinity</li>
                  <li>• <strong>Distance:</strong> Shorter distances generally indicate stronger interactions</li>
                  <li>• <strong>Interaction Types:</strong> Hydrogen bonds and coordination bonds are key for drug activity</li>
                  {dockingResults.method_used.includes('qaoa') && (
                    <li>• <strong>Quantum Enhancement:</strong> QAOA optimization may find better binding poses than classical methods</li>
                  )}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default DockingVisualization; 