import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';

export default function MobileResponsiveLayout({ children }) {
  const [isMobile, setIsMobile] = useState(false);
  const [isTablet, setIsTablet] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const checkScreenSize = () => {
      const width = window.innerWidth;
      setIsMobile(width < 768);
      setIsTablet(width >= 768 && width < 1024);
      if (width >= 1024) {
        setSidebarOpen(true);
      } else {
        setSidebarOpen(false);
      }
    };

    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: '🏠' },
    { name: 'Molecular Docking', href: '/docking', icon: '⚛️' },
    { name: 'Descriptor Viewer', href: '/descriptors', icon: '📊' },
    { name: 'Binding Score Benchmarks', href: '/benchmarks', icon: '📈' }
  ];

  const currentPath = router.pathname;

  return (
    <div className="h-screen flex overflow-hidden bg-gray-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && isMobile && (
        <div 
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:relative lg:translate-x-0
      `}>
        {/* Sidebar Header */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
          <div className="flex items-center">
            <span className="text-2xl mr-2">🧬</span>
            <h1 className="text-xl font-bold text-gray-900">MoleQule</h1>
          </div>
          {isMobile && (
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-600"
            >
              <span className="text-xl">✕</span>
            </button>
          )}
        </div>

        {/* Navigation */}
        <nav className="mt-6 px-4">
          <div className="space-y-2">
            {navigation.map((item) => {
              const isActive = currentPath === item.href;
              return (
                <a
                  key={item.name}
                  href={item.href}
                  className={`
                    group flex items-center px-3 py-3 text-sm font-medium rounded-lg transition-colors
                    ${isActive 
                      ? 'bg-blue-100 text-blue-700 border-r-2 border-blue-500' 
                      : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                    }
                  `}
                  onClick={() => isMobile && setSidebarOpen(false)}
                >
                  <span className="text-lg mr-3">{item.icon}</span>
                  <span className="truncate">{item.name}</span>
                </a>
              );
            })}
          </div>
        </nav>

        {/* Sidebar Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
              <span className="text-blue-600 font-medium">U</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">User</p>
              <p className="text-xs text-gray-500 truncate">user@example.com</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Navigation Bar (Mobile) */}
        <div className="lg:hidden bg-white shadow-sm border-b border-gray-200">
          <div className="flex items-center justify-between h-16 px-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 rounded-md text-gray-400 hover:text-gray-600"
            >
              <span className="text-xl">☰</span>
            </button>
            
            <div className="flex items-center">
              <span className="text-xl mr-2">🧬</span>
              <h1 className="text-lg font-bold text-gray-900">MoleQule</h1>
            </div>
            
            <div className="flex items-center space-x-2">
              <button className="p-2 rounded-md text-gray-400 hover:text-gray-600">
                <span className="text-lg">🔔</span>
              </button>
              <button className="p-2 rounded-md text-gray-400 hover:text-gray-600">
                <span className="text-lg">⚙️</span>
              </button>
            </div>
          </div>
        </div>

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto">
          <div className={`
            ${isMobile ? 'p-4' : isTablet ? 'p-6' : 'p-8'}
          `}>
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

// Mobile-specific components
export const MobileCard = ({ children, className = '' }) => (
  <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
    {children}
  </div>
);

export const MobileButton = ({ children, onClick, variant = 'primary', className = '' }) => {
  const baseClasses = 'w-full py-3 px-4 rounded-lg font-medium transition-colors text-center';
  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
    success: 'bg-green-600 text-white hover:bg-green-700',
    danger: 'bg-red-600 text-white hover:bg-red-700'
  };

  return (
    <button 
      onClick={onClick}
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
    >
      {children}
    </button>
  );
};

export const MobileInput = ({ label, placeholder, value, onChange, type = 'text', className = '' }) => (
  <div className={`mb-4 ${className}`}>
    {label && (
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
    )}
    <input
      type={type}
      placeholder={placeholder}
      value={value}
      onChange={onChange}
      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
    />
  </div>
);

export const MobileSelect = ({ label, options, value, onChange, className = '' }) => (
  <div className={`mb-4 ${className}`}>
    {label && (
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
    )}
    <select
      value={value}
      onChange={onChange}
      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  </div>
);

export const MobileTable = ({ headers, rows, className = '' }) => (
  <div className={`overflow-x-auto ${className}`}>
    <table className="min-w-full divide-y divide-gray-200">
      <thead className="bg-gray-50">
        <tr>
          {headers.map((header, index) => (
            <th
              key={index}
              className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
            >
              {header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="bg-white divide-y divide-gray-200">
        {rows.map((row, rowIndex) => (
          <tr key={rowIndex} className="hover:bg-gray-50">
            {row.map((cell, cellIndex) => (
              <td key={cellIndex} className="px-3 py-2 text-sm text-gray-900">
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export const MobileGrid = ({ children, cols = 1, className = '' }) => {
  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-2',
    3: 'grid-cols-3',
    4: 'grid-cols-4'
  };

  return (
    <div className={`grid ${gridCols[cols]} gap-4 ${className}`}>
      {children}
    </div>
  );
};

export const MobileModal = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" onClick={onClose} />
        
        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">{title}</h3>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600"
              >
                <span className="text-xl">✕</span>
              </button>
            </div>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

export const MobileTabs = ({ tabs, activeTab, onTabChange }) => (
  <div className="border-b border-gray-200">
    <nav className="-mb-px flex space-x-8">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`
            py-2 px-1 border-b-2 font-medium text-sm whitespace-nowrap
            ${activeTab === tab.id
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }
          `}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  </div>
);

export const MobileAlert = ({ type = 'info', title, message, onClose }) => {
  const alertStyles = {
    info: 'bg-blue-50 border-blue-200 text-blue-800',
    success: 'bg-green-50 border-green-200 text-green-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    error: 'bg-red-50 border-red-200 text-red-800'
  };

  const icons = {
    info: 'ℹ️',
    success: '✅',
    warning: '⚠️',
    error: '❌'
  };

  return (
    <div className={`border rounded-lg p-4 ${alertStyles[type]}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          <span className="text-lg">{icons[type]}</span>
        </div>
        <div className="ml-3 flex-1">
          {title && (
            <h3 className="text-sm font-medium">{title}</h3>
          )}
          {message && (
            <p className="text-sm mt-1">{message}</p>
          )}
        </div>
        {onClose && (
          <div className="ml-auto pl-3">
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <span className="text-lg">✕</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}; 
 