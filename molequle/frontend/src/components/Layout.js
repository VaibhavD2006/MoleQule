import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

const Layout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const router = useRouter();

  const navigation = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: '🏠',
      description: 'Overview of your drug discovery projects'
    },
    {
      name: 'Molecular Docking',
      href: '/upload',
      icon: '🧪',
      description: 'Upload structures and run docking analysis'
    },
    {
      name: 'Descriptor Viewer',
      href: '/descriptors',
      icon: '🔍',
      description: 'Analyze quantum molecular descriptors'
    },
    {
      name: 'Binding Score Benchmarks',
      href: '/benchmarks',
      icon: '📊',
      description: 'Compare performance against industry standards'
    },
    {
      name: 'Comprehensive Analysis',
      href: '/comprehensive-analysis',
      icon: '🔬',
      description: 'Complete ADMET, synthetic, stability, and clinical analysis'
    }
  ];

  const isActivePage = (href) => {
    if (href === '/dashboard' && router.pathname === '/') return true;
    return router.pathname.startsWith(href);
  };

  return (
    <div className="h-screen flex overflow-hidden bg-gray-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setSidebarOpen(false)} />
        </div>
      )}

      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-80 bg-white shadow-xl transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="flex items-center justify-between h-16 px-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
                <span className="text-sm font-bold">🧬</span>
              </div>
              <div>
                <h1 className="text-lg font-semibold">MoleQule</h1>
                <p className="text-xs text-blue-100">Quantum Drug Discovery</p>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-1 rounded-md hover:bg-white hover:bg-opacity-20"
            >
              ✕
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
            {navigation.map((item) => {
              const isActive = isActivePage(item.href);
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`group flex items-start p-4 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-blue-50 text-blue-700 border border-blue-200'
                      : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                  }`}
                  onClick={() => setSidebarOpen(false)}
                >
                  <span
                    className={`flex-shrink-0 text-lg mt-0.5 mr-4 ${
                      isActive ? 'text-blue-600' : 'text-gray-400 group-hover:text-gray-600'
                    }`}
                  >
                    {item.icon}
                  </span>
                  <div className="flex-1">
                    <div className={`font-medium ${isActive ? 'text-blue-900' : 'text-gray-900'}`}>
                      {item.name}
                    </div>
                    <div className={`text-xs mt-1 ${
                      isActive ? 'text-blue-600' : 'text-gray-500 group-hover:text-gray-600'
                    }`}>
                      {item.description}
                    </div>
                  </div>
                </Link>
              );
            })}
          </nav>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-gray-200">
            <div className="text-xs text-gray-500 text-center">
              <p>MoleQule v2.0</p>
              <p>Enhanced Drug Discovery Platform</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <div className="bg-white shadow-sm border-b border-gray-200">
          <div className="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
            >
              <span className="text-xl">☰</span>
            </button>
            
            <div className="flex-1 flex items-center justify-center lg:justify-end">
              <div className="text-sm text-gray-500">
                Quantum-Enhanced Drug Discovery
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout; 