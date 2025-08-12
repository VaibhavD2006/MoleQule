import React, { useState, useEffect } from 'react';

export default function UserOnboarding({ isOpen, onClose, userType = 'new' }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState([]);

  const onboardingSteps = [
    {
      id: 'welcome',
      title: 'Welcome to MoleQule! 🧬',
      description: 'Your quantum-enhanced drug discovery platform',
      content: (
        <div className="space-y-4">
          <div className="text-center">
            <span className="text-6xl">🧬</span>
            <h2 className="text-2xl font-bold text-gray-900 mt-4">Welcome to MoleQule</h2>
            <p className="text-gray-600 mt-2">
              The world's first quantum-enhanced drug discovery platform
            </p>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-900 mb-2">What you can do:</h3>
            <ul className="text-blue-800 space-y-1">
              <li>• Upload molecular structures (SMILES, MOL, XYZ)</li>
              <li>• Generate optimized drug analogs</li>
              <li>• Perform quantum-enhanced molecular docking</li>
              <li>• Analyze binding affinities and drug properties</li>
              <li>• Validate results against experimental data</li>
            </ul>
          </div>
        </div>
      )
    },
    {
      id: 'upload',
      title: 'Upload Your Molecules 📁',
      description: 'Learn how to upload and process molecular structures',
      content: (
        <div className="space-y-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-gray-900 mb-3">Supported File Formats:</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-3 bg-white rounded border">
                <span className="text-2xl">📝</span>
                <div className="font-medium text-gray-900">SMILES</div>
                <div className="text-sm text-gray-600">Text format</div>
              </div>
              <div className="text-center p-3 bg-white rounded border">
                <span className="text-2xl">🧪</span>
                <div className="font-medium text-gray-900">MOL</div>
                <div className="text-sm text-gray-600">Molecular structure</div>
              </div>
              <div className="text-center p-3 bg-white rounded border">
                <span className="text-2xl">📍</span>
                <div className="font-medium text-gray-900">XYZ</div>
                <div className="text-sm text-gray-600">3D coordinates</div>
              </div>
            </div>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold text-green-900 mb-2">Upload Process:</h3>
            <ol className="text-green-800 space-y-2">
              <li>1. Navigate to the Upload page</li>
              <li>2. Drag and drop your file or click to browse</li>
              <li>3. Select your target protein (DNA, GSTP1, etc.)</li>
              <li>4. Choose docking method (Basic, QAOA, AutoDock)</li>
              <li>5. Click "Process Molecule" to start analysis</li>
            </ol>
          </div>

          <div className="bg-yellow-50 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-900 mb-2">💡 Tips:</h3>
            <ul className="text-yellow-800 space-y-1">
              <li>• Ensure your molecule contains valid chemical structures</li>
              <li>• For best results, use molecules with known experimental data</li>
              <li>• Platinum-based compounds work particularly well</li>
              <li>• Processing typically takes 2-5 minutes</li>
            </ul>
          </div>
        </div>
      )
    },
    {
      id: 'docking',
      title: 'Molecular Docking ⚛️',
      description: 'Understand our quantum-enhanced docking methods',
      content: (
        <div className="space-y-4">
          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-900 mb-3">Available Docking Methods:</h3>
            <div className="space-y-3">
              <div className="bg-white p-3 rounded border">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900">Basic Analysis</div>
                    <div className="text-sm text-gray-600">Fast preliminary assessment</div>
                  </div>
                  <span className="text-purple-600 font-medium">~30 sec</span>
                </div>
              </div>
              <div className="bg-white p-3 rounded border">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900">QAOA Quantum</div>
                    <div className="text-sm text-gray-600">Quantum-enhanced optimization</div>
                  </div>
                  <span className="text-purple-600 font-medium">~3 min</span>
                </div>
              </div>
              <div className="bg-white p-3 rounded border">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900">AutoDock Vina</div>
                    <div className="text-sm text-gray-600">Industry standard docking</div>
                  </div>
                  <span className="text-purple-600 font-medium">~5 min</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-900 mb-2">Target Proteins:</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="text-center p-2 bg-white rounded border">
                <div className="font-medium text-gray-900">DNA</div>
                <div className="text-xs text-gray-600">Primary target</div>
              </div>
              <div className="text-center p-2 bg-white rounded border">
                <div className="font-medium text-gray-900">GSTP1</div>
                <div className="text-xs text-gray-600">Resistance protein</div>
              </div>
              <div className="text-center p-2 bg-white rounded border">
                <div className="font-medium text-gray-900">TP53</div>
                <div className="text-xs text-gray-600">Tumor suppressor</div>
              </div>
              <div className="text-center p-2 bg-white rounded border">
                <div className="font-medium text-gray-900">KRAS</div>
                <div className="text-xs text-gray-600">Oncogene</div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold text-green-900 mb-2">Performance Metrics:</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-green-800">Binding Affinity:</span>
                <span className="font-medium">-8.2 to -6.2 kcal/mol</span>
              </div>
              <div className="flex justify-between">
                <span className="text-green-800">Pose Accuracy:</span>
                <span className="font-medium">75-95%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-green-800">R² Score:</span>
                <span className="font-medium">0.75-0.81</span>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'results',
      title: 'Understanding Results 📊',
      description: 'Learn how to interpret your analysis results',
      content: (
        <div className="space-y-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-900 mb-3">Key Metrics Explained:</h3>
            <div className="space-y-3">
              <div>
                <div className="font-medium text-gray-900">Binding Affinity (kcal/mol)</div>
                <div className="text-sm text-gray-600">
                  Measures how strongly the molecule binds to the target. 
                  More negative values indicate stronger binding.
                </div>
              </div>
              <div>
                <div className="font-medium text-gray-900">Final Score</div>
                <div className="text-sm text-gray-600">
                  Composite score combining binding affinity, toxicity, and drug-likeness.
                  Higher values (0-1) indicate better drug candidates.
                </div>
              </div>
              <div>
                <div className="font-medium text-gray-900">Energy (Hartree)</div>
                <div className="text-sm text-gray-600">
                  Quantum mechanical energy of the molecule. 
                  Lower values indicate more stable structures.
                </div>
              </div>
              <div>
                <div className="font-medium text-gray-900">HOMO-LUMO Gap (eV)</div>
                <div className="text-sm text-gray-600">
                  Electronic energy gap. Higher values indicate greater stability.
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold text-green-900 mb-2">Result Interpretation:</h3>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-green-500 rounded-full"></span>
                <span className="text-sm text-green-800">Excellent: Binding affinity < -8.0 kcal/mol</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-yellow-500 rounded-full"></span>
                <span className="text-sm text-yellow-800">Good: Binding affinity -8.0 to -7.0 kcal/mol</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-red-500 rounded-full"></span>
                <span className="text-sm text-red-800">Poor: Binding affinity > -7.0 kcal/mol</span>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-900 mb-2">3D Visualization:</h3>
            <div className="text-sm text-purple-800 space-y-2">
              <p>• Interactive 3D molecular structure viewer</p>
              <p>• Rotate, zoom, and explore binding interactions</p>
              <p>• Color-coded atoms and bonds for easy identification</p>
              <p>• Export high-quality images for presentations</p>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'benchmarks',
      title: 'Benchmarking & Validation 📈',
      description: 'Learn about our validation and benchmarking system',
      content: (
        <div className="space-y-4">
          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-900 mb-3">Validation Standards:</h3>
            <div className="space-y-3">
              <div className="bg-white p-3 rounded border">
                <div className="font-medium text-gray-900">Experimental Data Integration</div>
                <div className="text-sm text-gray-600">
                  Results validated against PDBbind, ChEMBL, and BindingDB
                </div>
              </div>
              <div className="bg-white p-3 rounded border">
                <div className="font-medium text-gray-900">Performance Targets</div>
                <div className="text-sm text-gray-600">
                  R² > 0.75, RMSE < 2.0 kcal/mol, Pose accuracy > 70%
                </div>
              </div>
              <div className="bg-white p-3 rounded border">
                <div className="font-medium text-gray-900">Industry Comparison</div>
                <div className="text-sm text-gray-600">
                  Benchmarked against AutoDock Vina, Glide, and GOLD
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold text-green-900 mb-2">Current Performance:</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-white rounded border">
                <div className="text-2xl font-bold text-green-600">0.81</div>
                <div className="text-sm text-gray-600">Best R² Score</div>
              </div>
              <div className="text-center p-3 bg-white rounded border">
                <div className="text-2xl font-bold text-green-600">0.286</div>
                <div className="text-sm text-gray-600">Best RMSE (kcal/mol)</div>
              </div>
              <div className="text-center p-3 bg-white rounded border">
                <div className="text-2xl font-bold text-green-600">3/5</div>
                <div className="text-sm text-gray-600">Methods > 0.75 R²</div>
              </div>
              <div className="text-center p-3 bg-white rounded border">
                <div className="text-2xl font-bold text-green-600">9</div>
                <div className="text-sm text-gray-600">Validated Compounds</div>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-900 mb-2">Quality Assurance:</h3>
            <ul className="text-blue-800 space-y-1">
              <li>• All results validated against experimental data</li>
              <li>• Statistical significance testing (p < 0.01)</li>
              <li>• Cross-validation with multiple datasets</li>
              <li>• Independent verification by external sources</li>
              <li>• Continuous improvement and model updates</li>
            </ul>
          </div>
        </div>
      )
    },
    {
      id: 'complete',
      title: 'You\'re All Set! 🎉',
      description: 'Ready to start your quantum drug discovery journey',
      content: (
        <div className="space-y-4 text-center">
          <div>
            <span className="text-6xl">🎉</span>
            <h2 className="text-2xl font-bold text-gray-900 mt-4">Welcome to the Future of Drug Discovery!</h2>
            <p className="text-gray-600 mt-2">
              You now have access to the most advanced quantum-enhanced drug discovery platform
            </p>
          </div>
          
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-lg">
            <h3 className="font-semibold text-gray-900 mb-4">Next Steps:</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-white rounded-lg shadow">
                <span className="text-2xl">🧬</span>
                <div className="font-medium text-gray-900 mt-2">Upload Your First Molecule</div>
                <div className="text-sm text-gray-600">Start with a cisplatin analog</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg shadow">
                <span className="text-2xl">⚛️</span>
                <div className="font-medium text-gray-900 mt-2">Try Quantum Docking</div>
                <div className="text-sm text-gray-600">Experience quantum advantage</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg shadow">
                <span className="text-2xl">📊</span>
                <div className="font-medium text-gray-900 mt-2">View Benchmarks</div>
                <div className="text-sm text-gray-600">See validation results</div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold text-green-900 mb-2">Need Help?</h3>
            <div className="text-sm text-green-800 space-y-1">
              <p>• Check the documentation in the sidebar</p>
              <p>• Review the benchmark reports for validation details</p>
              <p>• Contact support for technical assistance</p>
              <p>• Join our community for best practices</p>
            </div>
          </div>
        </div>
      )
    }
  ];

  const handleNext = () => {
    if (currentStep < onboardingSteps.length - 1) {
      setCompletedSteps([...completedSteps, currentStep]);
      setCurrentStep(currentStep + 1);
    } else {
      // Complete onboarding
      localStorage.setItem('molequle-onboarding-completed', 'true');
      onClose();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    localStorage.setItem('molequle-onboarding-completed', 'true');
    onClose();
  };

  if (!isOpen) return null;

  const currentStepData = onboardingSteps[currentStep];
  const progress = ((currentStep + 1) / onboardingSteps.length) * 100;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">MoleQule Onboarding</h1>
              <p className="text-blue-100 mt-1">
                Step {currentStep + 1} of {onboardingSteps.length}
              </p>
            </div>
            <button
              onClick={handleSkip}
              className="text-blue-100 hover:text-white transition-colors"
            >
              Skip Tutorial
            </button>
          </div>
          
          {/* Progress Bar */}
          <div className="mt-4 bg-blue-700 rounded-full h-2">
            <div 
              className="bg-white h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              {currentStepData.title}
            </h2>
            <p className="text-gray-600">
              {currentStepData.description}
            </p>
          </div>
          
          {currentStepData.content}
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 flex items-center justify-between">
          <button
            onClick={handlePrevious}
            disabled={currentStep === 0}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              currentStep === 0
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Previous
          </button>
          
          <div className="flex space-x-2">
            {onboardingSteps.map((_, index) => (
              <div
                key={index}
                className={`w-2 h-2 rounded-full ${
                  index === currentStep
                    ? 'bg-blue-600'
                    : completedSteps.includes(index)
                    ? 'bg-green-500'
                    : 'bg-gray-300'
                }`}
              ></div>
            ))}
          </div>
          
          <button
            onClick={handleNext}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            {currentStep === onboardingSteps.length - 1 ? 'Get Started' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
} 
 