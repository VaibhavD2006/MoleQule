import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle, ArrowRight, Clock, Trash2, Eye, Plus, History } from 'lucide-react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function MolecularDockingPage() {
  const [uploadState, setUploadState] = useState({
    isUploading: false,
    jobId: null,
    error: null,
    success: false
  });
  const [previousEntries, setPreviousEntries] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [showNewSubmission, setShowNewSubmission] = useState(false);
  const router = useRouter();

  // Fetch previous entries on component mount
  useEffect(() => {
    fetchPreviousEntries();
  }, []);

  const fetchPreviousEntries = async () => {
    try {
      setLoadingHistory(true);
      // In a real app, this would fetch from your API
      // For now, we'll use localStorage to persist entries
      const stored = localStorage.getItem('molequle_entries');
      if (stored) {
        setPreviousEntries(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Error fetching previous entries:', error);
    } finally {
      setLoadingHistory(false);
    }
  };

  const saveEntry = (entry) => {
    const updatedEntries = [entry, ...previousEntries];
    setPreviousEntries(updatedEntries);
    localStorage.setItem('molequle_entries', JSON.stringify(updatedEntries));
  };

  const deleteEntry = (jobId) => {
    const updatedEntries = previousEntries.filter(entry => entry.job_id !== jobId);
    setPreviousEntries(updatedEntries);
    localStorage.setItem('molequle_entries', JSON.stringify(updatedEntries));
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploadState({ isUploading: true, jobId: null, error: null, success: false });

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_URL}/api/v1/upload-molecule`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 seconds timeout
      });

      if (response.data) {
        const newEntry = {
          job_id: response.data.job_id,
          filename: file.name,
          upload_time: new Date().toISOString(),
          status: 'processing',
          file_size: file.size,
          file_type: file.type
        };

        saveEntry(newEntry);

        setUploadState({
          isUploading: false,
          jobId: response.data.job_id,
          error: null,
          success: true
        });
        
        // Redirect to results page after a short delay
        setTimeout(() => {
          router.push(`/results/${response.data.job_id}`);
        }, 2000);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadState({
        isUploading: false,
        jobId: null,
        error: error.response?.data?.detail || error.message || 'Upload failed',
        success: false
      });
    }
  }, [router]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.smi', '.smiles'],
      'chemical/x-mol': ['.mol'],
      'chemical/x-xyz': ['.xyz'],
      'application/octet-stream': ['.smi', '.smiles', '.mol', '.xyz']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB max
  });

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'processing': return 'text-yellow-600 bg-yellow-100';
      case 'failed': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <>
      <Head>
        <title>Molecular Docking - MoleQule</title>
        <meta name="description" content="Upload molecular structures and view docking analysis history" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Molecular Docking
              </h1>
              <p className="text-gray-600">
                Upload molecular structures and analyze binding interactions
              </p>
            </div>

            {/* New Submission Section */}
            <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                  <Plus className="h-5 w-5 mr-2 text-blue-600" />
                  New Analysis
                </h2>
                <button
                  onClick={() => setShowNewSubmission(!showNewSubmission)}
                  className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                >
                  {showNewSubmission ? 'Hide' : 'Show'} Upload Area
                </button>
              </div>

              {showNewSubmission && (
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 ${
                    isDragActive
                      ? 'border-blue-400 bg-blue-50 scale-105'
                      : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                  }`}
                >
                  <input {...getInputProps()} />
                  
                  {uploadState.isUploading ? (
                    <div className="space-y-4">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                      <p className="text-gray-600">Uploading and processing...</p>
                      <p className="text-sm text-gray-500">This may take a few minutes</p>
                    </div>
                  ) : uploadState.success ? (
                    <div className="space-y-4">
                      <CheckCircle className="mx-auto h-12 w-12 text-green-500" />
                      <div>
                        <p className="text-lg font-medium text-green-900">
                          Upload Successful!
                        </p>
                        <p className="text-sm text-green-600 mt-2">
                          Redirecting to results...
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Upload className="mx-auto h-12 w-12 text-gray-400" />
                      <div>
                        <p className="text-lg font-medium text-gray-900">
                          {isDragActive ? 'Drop your file here' : 'Upload molecular structure'}
                        </p>
                        <p className="text-sm text-gray-500 mt-2">
                          Supports SMILES (.smi, .smiles), MOL (.mol), and XYZ (.xyz) formats
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          Maximum file size: 10MB
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Error Message */}
              {uploadState.error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center">
                    <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
                    <p className="text-red-800">{uploadState.error}</p>
                  </div>
                </div>
              )}

              {/* Success Message */}
              {uploadState.success && uploadState.jobId && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <CheckCircle className="h-5 w-5 text-green-400 mr-2" />
                      <p className="text-green-800">
                        Job ID: {uploadState.jobId}
                      </p>
                    </div>
                    <ArrowRight className="h-5 w-5 text-green-400 animate-pulse" />
                  </div>
                </div>
              )}
            </div>

            {/* Previous Entries Section */}
            <div className="bg-white rounded-lg shadow-sm">
              <div className="px-6 py-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                    <History className="h-5 w-5 mr-2 text-purple-600" />
                    Analysis History
                  </h2>
                  <span className="text-sm text-gray-500">
                    {previousEntries.length} entries
                  </span>
                </div>
              </div>

              {loadingHistory ? (
                <div className="p-8 text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Loading history...</p>
                </div>
              ) : previousEntries.length === 0 ? (
                <div className="p-8 text-center">
                  <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No previous analyses</h3>
                  <p className="text-gray-500 mb-4">
                    Upload your first molecular structure to get started
                  </p>
                  <button
                    onClick={() => setShowNewSubmission(true)}
                    className="btn-primary flex items-center mx-auto"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Start First Analysis
                  </button>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          File
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Job ID
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Upload Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {previousEntries.map((entry) => (
                        <tr key={entry.job_id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <FileText className="h-5 w-5 text-gray-400 mr-3" />
                              <div>
                                <div className="text-sm font-medium text-gray-900">
                                  {entry.filename}
                                </div>
                                <div className="text-sm text-gray-500">
                                  {formatFileSize(entry.file_size)}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-mono">
                            {entry.job_id}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {formatDate(entry.upload_time)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(entry.status)}`}>
                              {entry.status === 'processing' && <Clock className="h-3 w-3 mr-1 animate-pulse" />}
                              {entry.status === 'completed' && <CheckCircle className="h-3 w-3 mr-1" />}
                              {entry.status === 'failed' && <AlertCircle className="h-3 w-3 mr-1" />}
                              {entry.status}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                            <div className="flex items-center space-x-2">
                              <button
                                onClick={() => router.push(`/results/${entry.job_id}`)}
                                className="text-blue-600 hover:text-blue-900 flex items-center"
                              >
                                <Eye className="h-4 w-4 mr-1" />
                                View
                              </button>
                              <button
                                onClick={() => deleteEntry(entry.job_id)}
                                className="text-red-600 hover:text-red-900 flex items-center"
                              >
                                <Trash2 className="h-4 w-4 mr-1" />
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Instructions */}
            <div className="mt-8 bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                How Molecular Docking Works
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                    <span className="text-blue-600 font-bold">1</span>
                  </div>
                  <h4 className="font-medium text-gray-900 mb-2">Upload Structure</h4>
                  <p className="text-sm text-gray-600">
                    Upload your molecular structure in SMILES, MOL, or XYZ format
                  </p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                    <span className="text-purple-600 font-bold">2</span>
                  </div>
                  <h4 className="font-medium text-gray-900 mb-2">Quantum Analysis</h4>
                  <p className="text-sm text-gray-600">
                    Our quantum-enhanced pipeline generates optimized analogs
                  </p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                    <span className="text-green-600 font-bold">3</span>
                  </div>
                  <h4 className="font-medium text-gray-900 mb-2">View Results</h4>
                  <p className="text-sm text-gray-600">
                    Analyze binding affinities and download results
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
} 