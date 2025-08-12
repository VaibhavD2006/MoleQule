import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null,
      showDetails: false 
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });

    // Log error to console for debugging
    console.error('MoleQule Error Boundary caught an error:', error, errorInfo);
    
    // In production, you would send this to an error reporting service
    // this.logErrorToService(error, errorInfo);
  }

  logErrorToService = (error, errorInfo) => {
    // Example error reporting service integration
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    // Send to error reporting service (e.g., Sentry, LogRocket)
    console.log('Error report:', errorReport);
  };

  handleRetry = () => {
    this.setState({ 
      hasError: false, 
      error: null, 
      errorInfo: null,
      showDetails: false 
    });
  };

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/dashboard';
  };

  toggleDetails = () => {
    this.setState(prevState => ({ showDetails: !prevState.showDetails }));
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="max-w-2xl w-full bg-white rounded-lg shadow-xl">
            {/* Header */}
            <div className="bg-red-50 border-b border-red-200 px-6 py-4">
              <div className="flex items-center">
                <span className="text-2xl mr-3">⚠️</span>
                <div>
                  <h1 className="text-xl font-semibold text-red-900">
                    Something went wrong
                  </h1>
                  <p className="text-red-700 mt-1">
                    We encountered an unexpected error in MoleQule
                  </p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-6">
              <div className="text-center mb-6">
                <span className="text-6xl mb-4 block">🧬</span>
                <h2 className="text-lg font-medium text-gray-900 mb-2">
                  MoleQule Error Recovery
                </h2>
                <p className="text-gray-600">
                  Don't worry! This is likely a temporary issue. Here are some options to get you back on track.
                </p>
              </div>

              {/* Error Details */}
              {this.state.error && (
                <div className="mb-6">
                  <button
                    onClick={this.toggleDetails}
                    className="text-blue-600 hover:text-blue-800 text-sm font-medium mb-2"
                  >
                    {this.state.showDetails ? 'Hide' : 'Show'} Error Details
                  </button>
                  
                  {this.state.showDetails && (
                    <div className="bg-gray-50 p-4 rounded-lg border">
                      <h3 className="font-medium text-gray-900 mb-2">Error Information:</h3>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="font-medium text-gray-700">Message:</span>
                          <div className="bg-white p-2 rounded border font-mono text-red-600">
                            {this.state.error.message}
                          </div>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Component Stack:</span>
                          <div className="bg-white p-2 rounded border font-mono text-xs overflow-x-auto">
                            {this.state.errorInfo.componentStack}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Recovery Options */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <button
                  onClick={this.handleRetry}
                  className="p-4 border-2 border-dashed border-blue-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-center"
                >
                  <span className="text-2xl block mb-2">🔄</span>
                  <div className="font-medium text-gray-900">Retry</div>
                  <div className="text-sm text-gray-600">Try the same action again</div>
                </button>

                <button
                  onClick={this.handleReload}
                  className="p-4 border-2 border-dashed border-green-300 rounded-lg hover:border-green-400 hover:bg-green-50 transition-colors text-center"
                >
                  <span className="text-2xl block mb-2">🔄</span>
                  <div className="font-medium text-gray-900">Reload Page</div>
                  <div className="text-sm text-gray-600">Refresh the application</div>
                </button>

                <button
                  onClick={this.handleGoHome}
                  className="p-4 border-2 border-dashed border-purple-300 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-colors text-center"
                >
                  <span className="text-2xl block mb-2">🏠</span>
                  <div className="font-medium text-gray-900">Go to Dashboard</div>
                  <div className="text-sm text-gray-600">Return to main page</div>
                </button>
              </div>

              {/* Common Solutions */}
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-blue-900 mb-3">Common Solutions:</h3>
                <div className="space-y-2 text-sm text-blue-800">
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Check your internet connection and try again</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Clear your browser cache and cookies</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Try using a different browser</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Ensure your molecular file format is supported</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Check if the file size is within limits (10MB max)</span>
                  </div>
                </div>
              </div>

              {/* Support Information */}
              <div className="mt-6 pt-6 border-t border-gray-200">
                <div className="text-center">
                  <p className="text-gray-600 mb-3">
                    Still having issues? Our support team is here to help.
                  </p>
                  <div className="flex justify-center space-x-4">
                    <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                      Contact Support
                    </button>
                    <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                      View Documentation
                    </button>
                    <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                      Report Bug
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Higher-order component for wrapping specific components
export const withErrorBoundary = (WrappedComponent, fallbackUI = null) => {
  return class extends React.Component {
    render() {
      return (
        <ErrorBoundary fallbackUI={fallbackUI}>
          <WrappedComponent {...this.props} />
        </ErrorBoundary>
      );
    }
  };
};

// Hook for functional components
export const useErrorHandler = () => {
  const [error, setError] = React.useState(null);

  const handleError = React.useCallback((error) => {
    console.error('Error caught by useErrorHandler:', error);
    setError(error);
  }, []);

  const clearError = React.useCallback(() => {
    setError(null);
  }, []);

  return { error, handleError, clearError };
};

// Error reporting utility
export const reportError = (error, context = {}) => {
  const errorReport = {
    message: error.message,
    stack: error.stack,
    context,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href,
    userId: localStorage.getItem('molequle-user-id') || 'anonymous'
  };

  // In production, send to error reporting service
  console.error('Error Report:', errorReport);
  
  // Example: Send to backend API
  // fetch('/api/errors', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify(errorReport)
  // }).catch(console.error);
};

export default ErrorBoundary; 
 