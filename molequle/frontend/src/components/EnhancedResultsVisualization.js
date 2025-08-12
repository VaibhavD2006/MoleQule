import React, { useState, useEffect, useRef } from 'react';

export default function EnhancedResultsVisualization({ results, analogs }) {
  const [selectedView, setSelectedView] = useState('overview');
  const [selectedAnalog, setSelectedAnalog] = useState(null);
  const [chartData, setChartData] = useState(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (analogs && analogs.length > 0) {
      prepareChartData();
    }
  }, [analogs]);

  useEffect(() => {
    if (selectedAnalog && canvasRef.current) {
      render3DMolecule(selectedAnalog);
    }
  }, [selectedAnalog]);

  const prepareChartData = () => {
    const bindingAffinities = analogs.map(a => a.binding_affinity);
    const finalScores = analogs.map(a => a.final_score);
    const energies = analogs.map(a => a.energy);
    const labels = analogs.map((a, i) => `Analog ${i + 1}`);

    setChartData({
      bindingAffinities,
      finalScores,
      energies,
      labels
    });
  };

  const render3DMolecule = (analog) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set canvas size
    canvas.width = 400;
    canvas.height = 300;
    
    // Draw 3D molecular representation
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw molecular structure (simplified 3D representation)
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Draw atoms as circles with different colors
    const atoms = [
      { x: centerX - 60, y: centerY - 30, color: '#3b82f6', label: 'N' },
      { x: centerX, y: centerY, color: '#f59e0b', label: 'Pt' },
      { x: centerX + 60, y: centerY - 30, color: '#3b82f6', label: 'N' },
      { x: centerX - 40, y: centerY + 40, color: '#ef4444', label: 'Cl' },
      { x: centerX + 40, y: centerY + 40, color: '#ef4444', label: 'Cl' }
    ];
    
    // Draw bonds
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(atoms[0].x, atoms[0].y);
    ctx.lineTo(atoms[1].x, atoms[1].y);
    ctx.lineTo(atoms[2].x, atoms[2].y);
    ctx.moveTo(atoms[1].x, atoms[1].y);
    ctx.lineTo(atoms[3].x, atoms[3].y);
    ctx.moveTo(atoms[1].x, atoms[1].y);
    ctx.lineTo(atoms[4].x, atoms[4].y);
    ctx.stroke();
    
    // Draw atoms
    atoms.forEach(atom => {
      ctx.beginPath();
      ctx.arc(atom.x, atom.y, 15, 0, 2 * Math.PI);
      ctx.fillStyle = atom.color;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();
      
      // Draw atom labels
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(atom.label, atom.x, atom.y + 4);
    });
    
    // Add 3D effect with shadows
    ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
    ctx.shadowBlur = 10;
    ctx.shadowOffsetX = 5;
    ctx.shadowOffsetY = 5;
  };

  const renderBarChart = (data, labels, title, color) => {
    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Chart dimensions
    const margin = 50;
    const chartWidth = canvas.width - 2 * margin;
    const chartHeight = canvas.height - 2 * margin;
    
    // Find data range
    const maxValue = Math.max(...data);
    const minValue = Math.min(...data);
    const range = maxValue - minValue;
    
    // Draw bars
    const barWidth = chartWidth / data.length;
    data.forEach((value, index) => {
      const barHeight = ((value - minValue) / range) * chartHeight;
      const x = margin + index * barWidth;
      const y = canvas.height - margin - barHeight;
      
      // Draw bar
      ctx.fillStyle = color;
      ctx.fillRect(x + 5, y, barWidth - 10, barHeight);
      
      // Draw value
      ctx.fillStyle = '#374151';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(value.toFixed(2), x + barWidth / 2, y - 10);
      
      // Draw label
      ctx.fillText(labels[index], x + barWidth / 2, canvas.height - margin + 20);
    });
    
    // Draw title
    ctx.fillStyle = '#111827';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(title, canvas.width / 2, 30);
    
    return canvas.toDataURL();
  };

  const renderScatterPlot = (xData, yData, xLabel, yLabel, title) => {
    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 400;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Chart dimensions
    const margin = 60;
    const chartWidth = canvas.width - 2 * margin;
    const chartHeight = canvas.height - 2 * margin;
    
    // Find data ranges
    const xMax = Math.max(...xData);
    const xMin = Math.min(...xData);
    const yMax = Math.max(...yData);
    const yMin = Math.min(...yData);
    
    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, canvas.height - margin);
    ctx.lineTo(canvas.width - margin, canvas.height - margin);
    ctx.stroke();
    
    // Draw data points
    xData.forEach((x, index) => {
      const y = yData[index];
      const plotX = margin + ((x - xMin) / (xMax - xMin)) * chartWidth;
      const plotY = canvas.height - margin - ((y - yMin) / (yMax - yMin)) * chartHeight;
      
      ctx.beginPath();
      ctx.arc(plotX, plotY, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#3b82f6';
      ctx.fill();
    });
    
    // Draw labels
    ctx.fillStyle = '#374151';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, canvas.width / 2, canvas.height - 20);
    
    ctx.save();
    ctx.translate(30, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    
    // Draw title
    ctx.fillText(title, canvas.width / 2, 30);
    
    return canvas.toDataURL();
  };

  if (!results || !analogs) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="text-center text-gray-500">
          <span className="text-4xl">📊</span>
          <p className="mt-2">No results available for visualization</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8 px-6">
          {[
            { id: 'overview', label: 'Overview', icon: '📊' },
            { id: 'binding', label: 'Binding Affinity', icon: '⚛️' },
            { id: 'scores', label: 'Final Scores', icon: '🏆' },
            { id: 'energy', label: 'Energy Analysis', icon: '⚡' },
            { id: '3d', label: '3D Visualization', icon: '🔬' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setSelectedView(tab.id)}
              className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                selectedView === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Content Area */}
      <div className="p-6">
        {selectedView === 'overview' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-blue-900 mb-2">📊 Summary Statistics</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-blue-700">Total Analogs:</span>
                    <span className="font-semibold">{analogs.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-700">Avg Binding Affinity:</span>
                    <span className="font-semibold">
                      {(analogs.reduce((sum, a) => sum + a.binding_affinity, 0) / analogs.length).toFixed(2)} kcal/mol
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-700">Best Score:</span>
                    <span className="font-semibold">
                      {Math.max(...analogs.map(a => a.final_score)).toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-green-900 mb-2">🏆 Top Performers</h3>
                <div className="space-y-2">
                  {analogs
                    .sort((a, b) => b.final_score - a.final_score)
                    .slice(0, 3)
                    .map((analog, index) => (
                      <div key={index} className="flex justify-between items-center">
                        <span className="text-green-700">#{index + 1}</span>
                        <span className="font-semibold">{analog.final_score.toFixed(3)}</span>
                      </div>
                    ))}
                </div>
              </div>

              <div className="bg-purple-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-purple-900 mb-2">⚡ Energy Range</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-purple-700">Min Energy:</span>
                    <span className="font-semibold">
                      {Math.min(...analogs.map(a => a.energy)).toFixed(1)} Hartree
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-purple-700">Max Energy:</span>
                    <span className="font-semibold">
                      {Math.max(...analogs.map(a => a.energy)).toFixed(1)} Hartree
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-purple-700">Range:</span>
                    <span className="font-semibold">
                      {(Math.max(...analogs.map(a => a.energy)) - Math.min(...analogs.map(a => a.energy))).toFixed(1)} Hartree
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {chartData && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Binding Affinity Distribution</h3>
                  <img 
                    src={renderBarChart(chartData.bindingAffinities, chartData.labels, 'Binding Affinity (kcal/mol)', '#3b82f6')}
                    alt="Binding Affinity Chart"
                    className="w-full rounded-lg border"
                  />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Final Score vs Binding Affinity</h3>
                  <img 
                    src={renderScatterPlot(chartData.bindingAffinities, chartData.finalScores, 'Binding Affinity (kcal/mol)', 'Final Score', 'Correlation Analysis')}
                    alt="Scatter Plot"
                    className="w-full rounded-lg border"
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {selectedView === 'binding' && (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-gray-900">Binding Affinity Analysis</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Analog</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Binding Affinity</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {analogs
                    .sort((a, b) => a.binding_affinity - b.binding_affinity)
                    .map((analog, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {analog.analog_id || `Analog ${index + 1}`}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {analog.binding_affinity.toFixed(2)} kcal/mol
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          #{index + 1}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                            analog.binding_affinity < -8.0 
                              ? 'bg-green-100 text-green-800' 
                              : analog.binding_affinity < -7.0 
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {analog.binding_affinity < -8.0 ? 'Excellent' : 
                             analog.binding_affinity < -7.0 ? 'Good' : 'Poor'}
                          </span>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {selectedView === 'scores' && (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-gray-900">Final Score Analysis</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-4">Score Distribution</h4>
                <img 
                  src={renderBarChart(chartData?.finalScores || [], chartData?.labels || [], 'Final Scores', '#10b981')}
                  alt="Final Scores Chart"
                  className="w-full rounded-lg border"
                />
              </div>
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-4">Top 5 Analogs</h4>
                <div className="space-y-3">
                  {analogs
                    .sort((a, b) => b.final_score - a.final_score)
                    .slice(0, 5)
                    .map((analog, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div>
                          <span className="font-medium text-gray-900">#{index + 1}</span>
                          <span className="ml-2 text-sm text-gray-600">{analog.analog_id || `Analog ${index + 1}`}</span>
                        </div>
                        <div className="text-right">
                          <div className="font-semibold text-gray-900">{analog.final_score.toFixed(3)}</div>
                          <div className="text-sm text-gray-500">{analog.binding_affinity.toFixed(2)} kcal/mol</div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedView === 'energy' && (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-gray-900">Energy Analysis</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-4">Energy Distribution</h4>
                <img 
                  src={renderBarChart(chartData?.energies || [], chartData?.labels || [], 'Energy (Hartree)', '#8b5cf6')}
                  alt="Energy Chart"
                  className="w-full rounded-lg border"
                />
              </div>
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-4">Energy Statistics</h4>
                <div className="bg-gray-50 p-4 rounded-lg space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-700">Minimum Energy:</span>
                    <span className="font-semibold">
                      {Math.min(...analogs.map(a => a.energy)).toFixed(3)} Hartree
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Maximum Energy:</span>
                    <span className="font-semibold">
                      {Math.max(...analogs.map(a => a.energy)).toFixed(3)} Hartree
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Average Energy:</span>
                    <span className="font-semibold">
                      {(analogs.reduce((sum, a) => sum + a.energy, 0) / analogs.length).toFixed(3)} Hartree
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Energy Range:</span>
                    <span className="font-semibold">
                      {(Math.max(...analogs.map(a => a.energy)) - Math.min(...analogs.map(a => a.energy))).toFixed(3)} Hartree
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedView === '3d' && (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-gray-900">3D Molecular Visualization</h3>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <div className="bg-gray-100 rounded-lg p-4">
                  <canvas
                    ref={canvasRef}
                    className="w-full h-64 border border-gray-300 rounded"
                  />
                </div>
                <div className="mt-4 text-center">
                  <p className="text-sm text-gray-600">
                    Interactive 3D molecular structure visualization
                  </p>
                </div>
              </div>
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-4">Select Analog</h4>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {analogs.map((analog, index) => (
                    <button
                      key={index}
                      onClick={() => setSelectedAnalog(analog)}
                      className={`w-full p-3 text-left rounded-lg border transition-colors ${
                        selectedAnalog === analog
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="font-medium text-gray-900">
                        {analog.analog_id || `Analog ${index + 1}`}
                      </div>
                      <div className="text-sm text-gray-600">
                        Score: {analog.final_score.toFixed(3)}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 
 