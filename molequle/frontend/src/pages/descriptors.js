import React, { useState } from 'react';
import Head from 'next/head';

export default function DescriptorViewer() {
  const [selectedCompound, setSelectedCompound] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  const compounds = [
    {
      id: 'cis_001',
      name: 'Cisplatin',
      smiles: 'N[Pt](N)(Cl)Cl',
      energy: -26185.2,
      homoLumo: 2.68,
      dipole: 4.12,
      status: 'analyzed'
    },
    {
      id: 'carb_023',
      name: 'Carboplatin Analog',
      smiles: 'N[Pt](N)(Br)Br',
      energy: -26152.8,
      homoLumo: 2.61,
      dipole: 3.87,
      status: 'analyzed'
    },
    {
      id: 'oxa_045',
      name: 'Oxaliplatin Derivative',
      smiles: 'N[Pt](NCCN)(O)O',
      energy: -26178.9,
      homoLumo: 2.55,
      dipole: 5.23,
      status: 'processing'
    },
    {
      id: 'satr_067',
      name: 'Satraplatin',
      smiles: 'N[Pt](N)(OC(=O)C)(OC(=O)C)',
      energy: -26203.5,
      homoLumo: 2.74,
      dipole: 3.65,
      status: 'analyzed'
    }
  ];

  const descriptorCategories = [
    {
      name: 'Quantum Descriptors',
      descriptors: [
        { name: 'Ground State Energy', value: selectedCompound?.energy, unit: 'Hartree', description: 'Total electronic energy from VQE calculation' },
        { name: 'HOMO-LUMO Gap', value: selectedCompound?.homoLumo, unit: 'eV', description: 'Energy difference between frontier orbitals' },
        { name: 'Dipole Moment', value: selectedCompound?.dipole, unit: 'Debye', description: 'Molecular charge distribution' }
      ]
    },
    {
      name: 'Molecular Properties',
      descriptors: [
        { name: 'Molecular Weight', value: 298.04, unit: 'g/mol', description: 'Total molecular mass' },
        { name: 'LogP', value: -2.19, unit: '', description: 'Lipophilicity coefficient' },
        { name: 'TPSA', value: 36.42, unit: 'Ų', description: 'Topological polar surface area' }
      ]
    },
    {
      name: 'Binding Properties',
      descriptors: [
        { name: 'Binding Affinity', value: -8.2, unit: 'kcal/mol', description: 'Predicted binding strength to DNA' },
        { name: 'Selectivity Index', value: 2.3, unit: '', description: 'Target selectivity ratio' },
        { name: 'Drug-likeness', value: 0.72, unit: '', description: 'Pharmaceutical relevance score' }
      ]
    }
  ];

  const filteredCompounds = compounds.filter(compound =>
    compound.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    compound.smiles.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <>
      <Head>
        <title>Descriptor Viewer - MoleQule</title>
        <meta name="description" content="Analyze quantum molecular descriptors and properties" />
      </Head>

      <div className="bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900">Quantum Descriptor Viewer</h1>
            <p className="mt-2 text-gray-600">
              Analyze quantum-computed molecular descriptors and properties
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Compound List */}
            <div className="lg:col-span-1">
              <div className="bg-white shadow rounded-lg">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h3 className="text-lg font-medium text-gray-900">Compounds</h3>
                  
                  {/* Search */}
                  <div className="mt-4 relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <span className="text-gray-400">🔍</span>
                    </div>
                    <input
                      type="text"
                      className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                      placeholder="Search compounds..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                    />
                  </div>
                </div>

                <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
                  {filteredCompounds.map((compound) => (
                    <div
                      key={compound.id}
                      className={`px-6 py-4 cursor-pointer hover:bg-gray-50 ${
                        selectedCompound?.id === compound.id ? 'bg-blue-50 border-r-2 border-blue-500' : ''
                      }`}
                      onClick={() => setSelectedCompound(compound)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <h4 className="text-sm font-medium text-gray-900 truncate">
                            {compound.name}
                          </h4>
                          <p className="text-xs text-gray-500 truncate font-mono">
                            {compound.smiles}
                          </p>
                        </div>
                        <div className={`flex-shrink-0 px-2 py-1 text-xs font-medium rounded-full ${
                          compound.status === 'analyzed' ? 'bg-green-100 text-green-800' :
                          compound.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {compound.status}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Descriptor Details */}
            <div className="lg:col-span-2">
              {selectedCompound ? (
                <div className="space-y-6">
                  {/* Compound Info */}
                  <div className="bg-white shadow rounded-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h2 className="text-xl font-semibold text-gray-900">{selectedCompound.name}</h2>
                        <p className="text-sm text-gray-500 font-mono">{selectedCompound.smiles}</p>
                      </div>
                      <div className="flex space-x-2">
                        <button className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                          <span className="mr-2">⬇️</span>
                          Export
                        </button>
                        <button className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                          <span className="mr-2">📊</span>
                          Visualize
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Descriptor Categories */}
                  {descriptorCategories.map((category, categoryIndex) => (
                    <div key={categoryIndex} className="bg-white shadow rounded-lg">
                      <div className="px-6 py-4 border-b border-gray-200">
                        <h3 className="text-lg font-medium text-gray-900">{category.name}</h3>
                      </div>
                      <div className="p-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          {category.descriptors.map((descriptor, index) => (
                            <div key={index} className="border border-gray-200 rounded-lg p-4">
                              <div className="flex items-center justify-between mb-2">
                                <h4 className="text-sm font-medium text-gray-900">
                                  {descriptor.name}
                                </h4>
                                <span className="text-lg font-semibold text-blue-600">
                                  {typeof descriptor.value === 'number' ? descriptor.value.toFixed(2) : 'N/A'}
                                  {descriptor.unit && <span className="text-sm text-gray-500 ml-1">{descriptor.unit}</span>}
                                </span>
                              </div>
                              <p className="text-xs text-gray-500">{descriptor.description}</p>
                              
                              {/* Visual indicator */}
                              {typeof descriptor.value === 'number' && (
                                <div className="mt-3">
                                  <div className="bg-gray-200 rounded-full h-1">
                                    <div
                                      className="bg-blue-600 h-1 rounded-full"
                                      style={{ 
                                        width: `${Math.min(100, Math.abs(descriptor.value) * 10)}%` 
                                      }}
                                    />
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Descriptor Comparison */}
                  <div className="bg-white shadow rounded-lg">
                    <div className="px-6 py-4 border-b border-gray-200">
                      <h3 className="text-lg font-medium text-gray-900">Descriptor Analysis</h3>
                    </div>
                    <div className="p-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="text-sm font-medium text-gray-900 mb-3">Key Insights</h4>
                          <ul className="space-y-2 text-sm text-gray-600">
                            <li className="flex items-start">
                              <span className="w-2 h-2 bg-green-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                              Strong quantum correlation between HOMO-LUMO gap and binding affinity
                            </li>
                            <li className="flex items-start">
                              <span className="w-2 h-2 bg-blue-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                              Optimal dipole moment range for DNA binding selectivity
                            </li>
                            <li className="flex items-start">
                              <span className="w-2 h-2 bg-yellow-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                              Ground state energy indicates favorable coordination geometry
                            </li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-gray-900 mb-3">Recommendations</h4>
                          <ul className="space-y-2 text-sm text-gray-600">
                            <li className="flex items-start">
                              <span className="w-2 h-2 bg-purple-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                              Consider for lead optimization based on quantum descriptors
                            </li>
                            <li className="flex items-start">
                              <span className="w-2 h-2 bg-red-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                              Monitor ADMET properties during structural modifications
                            </li>
                            <li className="flex items-start">
                              <span className="w-2 h-2 bg-indigo-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                              Validate predictions with experimental binding assays
                            </li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white shadow rounded-lg p-12 text-center">
                  <span className="text-6xl">📄</span>
                  <h3 className="mt-4 text-lg font-medium text-gray-900">Select a Compound</h3>
                  <p className="mt-2 text-sm text-gray-500">
                    Choose a compound from the list to view its quantum descriptors and molecular properties.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
} 