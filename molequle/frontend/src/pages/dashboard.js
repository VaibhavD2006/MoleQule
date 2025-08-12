import React, { useState, useEffect } from 'react';
import Link from 'next/link';

export default function Dashboard() {
  const [stats, setStats] = useState({
    compoundsAnalyzed: 0,
    dockingRuns: 0,
    leadsFound: 0,
    activeProjects: 0,
    recentActivity: []
  });

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading real data
    const loadDashboardData = async () => {
      setLoading(true);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data - in production this would come from API
      setStats({
        compoundsAnalyzed: 156,
        dockingRuns: 892,
        leadsFound: 23,
        activeProjects: 8,
        recentActivity: [
          {
            id: 1,
            type: 'job_completed',
            title: 'Cisplatin analog analysis completed',
            description: 'Generated 15 analogs with binding affinity predictions',
            timestamp: '2 hours ago',
            status: 'success'
          },
          {
            id: 2,
            type: 'docking_run',
            title: 'DNA target docking completed',
            description: 'QAOA quantum docking achieved R² = 0.752',
            timestamp: '4 hours ago',
            status: 'success'
          },
          {
            id: 3,
            type: 'benchmark_report',
            title: 'Benchmark validation report generated',
            description: '3 methods achieved R² > 0.75 target',
            timestamp: '1 day ago',
            status: 'success'
          },
          {
            id: 4,
            type: 'project_created',
            title: 'New project: Pancreatic cancer drug discovery',
            description: 'Project Alpha initiated with 25 compounds',
            timestamp: '2 days ago',
            status: 'info'
          },
          {
            id: 5,
            type: 'lead_identified',
            title: 'High-potential lead compound identified',
            description: 'Compound X-247 shows -8.2 kcal/mol binding affinity',
            timestamp: '3 days ago',
            status: 'success'
          }
        ]
      });
      
      setLoading(false);
    };

    loadDashboardData();
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'success': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'error': return 'text-red-600 bg-red-100';
      case 'info': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getActivityIcon = (type) => {
    switch (type) {
      case 'job_completed': return '🧬';
      case 'docking_run': return '⚛️';
      case 'benchmark_report': return '📊';
      case 'project_created': return '📁';
      case 'lead_identified': return '💎';
      default: return '📋';
    }
  };

  if (loading) {
    return (
      <div className="p-8">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="bg-gray-200 h-32 rounded-lg"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-200 h-64 rounded-lg"></div>
            <div className="bg-gray-200 h-64 rounded-lg"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-2">Welcome back! Here's your MoleQule overview.</p>
        </div>
        <div className="flex space-x-3">
          <Link href="/upload">
            <button className="px-6 py-3 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 transition-colors flex items-center">
              <span className="mr-2">🧬</span>
              Upload Molecule
            </button>
          </Link>
          <Link href="/benchmarks">
            <button className="px-6 py-3 bg-green-600 text-white rounded-lg shadow hover:bg-green-700 transition-colors flex items-center">
              <span className="mr-2">📊</span>
              View Benchmarks
            </button>
          </Link>
        </div>
      </div>

      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow border-l-4 border-blue-500">
          <div className="flex items-center">
            <div className="p-3 bg-blue-100 rounded-full">
              <span className="text-2xl">🧬</span>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Compounds Analyzed</p>
              <p className="text-2xl font-bold text-gray-900">{stats.compoundsAnalyzed.toLocaleString()}</p>
            </div>
          </div>
          <div className="mt-4">
            <span className="text-green-600 text-sm font-medium">+12% from last week</span>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow border-l-4 border-purple-500">
          <div className="flex items-center">
            <div className="p-3 bg-purple-100 rounded-full">
              <span className="text-2xl">⚛️</span>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Docking Runs</p>
              <p className="text-2xl font-bold text-gray-900">{stats.dockingRuns.toLocaleString()}</p>
            </div>
          </div>
          <div className="mt-4">
            <span className="text-green-600 text-sm font-medium">+8% from last week</span>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow border-l-4 border-green-500">
          <div className="flex items-center">
            <div className="p-3 bg-green-100 rounded-full">
              <span className="text-2xl">💎</span>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Leads Found</p>
              <p className="text-2xl font-bold text-gray-900">{stats.leadsFound}</p>
            </div>
          </div>
          <div className="mt-4">
            <span className="text-green-600 text-sm font-medium">+3 this week</span>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow border-l-4 border-orange-500">
          <div className="flex items-center">
            <div className="p-3 bg-orange-100 rounded-full">
              <span className="text-2xl">📁</span>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Active Projects</p>
              <p className="text-2xl font-bold text-gray-900">{stats.activeProjects}</p>
            </div>
          </div>
          <div className="mt-4">
            <span className="text-blue-600 text-sm font-medium">2 new this month</span>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Performance Metrics</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Average Binding Affinity</span>
              <span className="font-semibold text-green-600">-7.8 kcal/mol</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-green-600 h-2 rounded-full" style={{ width: '78%' }}></div>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Docking Accuracy (R²)</span>
              <span className="font-semibold text-blue-600">0.81</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-blue-600 h-2 rounded-full" style={{ width: '81%' }}></div>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Processing Speed</span>
              <span className="font-semibold text-purple-600">2.3 min/compound</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-purple-600 h-2 rounded-full" style={{ width: '85%' }}></div>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Success Rate</span>
              <span className="font-semibold text-orange-600">94.2%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-orange-600 h-2 rounded-full" style={{ width: '94%' }}></div>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Recent Activity</h2>
          <div className="space-y-4">
            {stats.recentActivity.slice(0, 5).map((activity) => (
              <div key={activity.id} className="flex items-start space-x-3">
                <div className="flex-shrink-0">
                  <span className="text-xl">{getActivityIcon(activity.type)}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">{activity.title}</p>
                  <p className="text-sm text-gray-500">{activity.description}</p>
                  <p className="text-xs text-gray-400 mt-1">{activity.timestamp}</p>
                </div>
                <div className="flex-shrink-0">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(activity.status)}`}>
                    {activity.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-gray-200">
            <Link href="/activity">
              <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                View all activity →
              </button>
            </Link>
          </div>
        </div>
      </div>

      {/* Quick Actions and Projects */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-2 gap-3">
            <Link href="/upload">
              <button className="w-full p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-center">
                <span className="text-2xl block mb-2">🧬</span>
                <span className="text-sm font-medium text-gray-700">Upload New Molecule</span>
              </button>
            </Link>
            <Link href="/descriptors">
              <button className="w-full p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-green-400 hover:bg-green-50 transition-colors text-center">
                <span className="text-2xl block mb-2">📊</span>
                <span className="text-sm font-medium text-gray-700">View Descriptors</span>
              </button>
            </Link>
            <Link href="/benchmarks">
              <button className="w-full p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-colors text-center">
                <span className="text-2xl block mb-2">📈</span>
                <span className="text-sm font-medium text-gray-700">Run Benchmarks</span>
              </button>
            </Link>
            <Link href="/comprehensive-analysis">
              <button className="w-full p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-orange-400 hover:bg-orange-50 transition-colors text-center">
                <span className="text-2xl block mb-2">🔬</span>
                <span className="text-sm font-medium text-gray-700">Comprehensive Analysis</span>
              </button>
            </Link>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Active Projects</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <div>
                <h3 className="font-medium text-gray-900">Project Alpha</h3>
                <p className="text-sm text-gray-600">Pancreatic cancer drug discovery</p>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-gray-900">75%</div>
                <div className="w-16 bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full" style={{ width: '75%' }}></div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <div>
                <h3 className="font-medium text-gray-900">Project Beta</h3>
                <p className="text-sm text-gray-600">Cisplatin analog optimization</p>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-gray-900">30%</div>
                <div className="w-16 bg-gray-200 rounded-full h-2">
                  <div className="bg-green-600 h-2 rounded-full" style={{ width: '30%' }}></div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
              <div>
                <h3 className="font-medium text-gray-900">Project Gamma</h3>
                <p className="text-sm text-gray-600">Quantum docking validation</p>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-gray-900">90%</div>
                <div className="w-16 bg-gray-200 rounded-full h-2">
                  <div className="bg-purple-600 h-2 rounded-full" style={{ width: '90%' }}></div>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-200">
            <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
              View all projects →
            </button>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="mt-8 bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">System Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-700">Backend API</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-700">ML Service</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-700">Docking Service</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-700">Database</span>
          </div>
        </div>
      </div>
    </div>
  );
}