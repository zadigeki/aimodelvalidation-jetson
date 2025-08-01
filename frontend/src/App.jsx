import React, { useState, useRef } from 'react'
import { Upload, Play, Download, FileText, BarChart3, Eye, Clock, AlertTriangle, CheckCircle } from 'lucide-react'
import VideoUpload from './components/VideoUpload'
import AnalysisResults from './components/AnalysisResults'
import AnnotatedVideo from './components/AnnotatedVideo'
import ReportExport from './components/ReportExport'
import ProgressTracker from './components/ProgressTracker'
import { analyzeVideo, getAnalysisResults } from './utils/api'

function App() {
  const [currentStep, setCurrentStep] = useState(0)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [analysisResults, setAnalysisResults] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [error, setError] = useState(null)

  const steps = [
    { id: 0, title: 'Upload Video', icon: Upload, description: 'Select driver monitoring footage' },
    { id: 1, title: 'AI Analysis', icon: Eye, description: 'Processing with computer vision' },
    { id: 2, title: 'Results', icon: BarChart3, description: 'View detailed analysis' },
    { id: 3, title: 'Annotated Video', icon: Play, description: 'Watch annotated footage' },
    { id: 4, title: 'Export', icon: Download, description: 'Download reports' }
  ]

  const handleVideoUpload = async (file, config) => {
    setUploadedFile(file)
    setIsAnalyzing(true)
    setError(null)
    setCurrentStep(1)
    
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setAnalysisProgress(prev => {
          const newProgress = prev + Math.random() * 15
          return newProgress > 90 ? 90 : newProgress
        })
      }, 1000)

      const response = await analyzeVideo(file, config)
      
      clearInterval(progressInterval)
      setAnalysisProgress(100)
      
      setSessionId(response.session_id)
      
      // Get detailed results
      const results = await getAnalysisResults(response.session_id)
      setAnalysisResults(results)
      
      setCurrentStep(2)
    } catch (err) {
      setError(err.message || 'Analysis failed')
      setCurrentStep(0)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleNextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handlePrevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleReset = () => {
    setCurrentStep(0)
    setUploadedFile(null)
    setAnalysisResults(null)
    setSessionId(null)
    setAnalysisProgress(0)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Eye className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Driver Monitoring System</h1>
                <p className="text-sm text-gray-600">AI-Powered Video Analysis</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>AI Ready</span>
              </div>
              
              {sessionId && (
                <div className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                  Session: {sessionId}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Progress Tracker */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <ProgressTracker 
          steps={steps} 
          currentStep={currentStep} 
          isAnalyzing={isAnalyzing}
          progress={analysisProgress}
        />
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-medium text-red-800">Analysis Error</h3>
              <p className="text-sm text-red-700">{error}</p>
            </div>
            <button 
              onClick={handleReset}
              className="ml-auto btn-secondary text-xs"
            >
              Try Again
            </button>
          </div>
        )}

        <div className="animate-fadeIn">
          {/* Step 0: Video Upload */}
          {currentStep === 0 && (
            <VideoUpload 
              onUpload={handleVideoUpload}
              isUploading={isAnalyzing}
            />
          )}

          {/* Step 1: Analysis in Progress */}
          {currentStep === 1 && (
            <div className="card">
              <div className="card-body text-center py-12">
                <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Eye className="w-10 h-10 text-white animate-pulse" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Analyzing Your Video
                </h2>
                <p className="text-gray-600 mb-8 max-w-md mx-auto">
                  Our AI is processing your driver monitoring footage using advanced computer vision algorithms.
                </p>
                
                <div className="max-w-md mx-auto">
                  <div className="progress-bar mb-4">
                    <div 
                      className="progress-fill bg-gradient-to-r from-blue-500 to-purple-600"
                      style={{ width: `${analysisProgress}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-600">
                    Progress: {Math.round(analysisProgress)}%
                  </p>
                </div>

                <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
                  <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
                    <Eye className="w-5 h-5 text-blue-500" />
                    <span className="text-sm text-blue-700">Face Detection</span>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-lg">
                    <BarChart3 className="w-5 h-5 text-purple-500" />
                    <span className="text-sm text-purple-700">Behavior Analysis</span>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span className="text-sm text-green-700">Event Detection</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Analysis Results */}
          {currentStep === 2 && analysisResults && (
            <AnalysisResults 
              results={analysisResults}
              onNext={handleNextStep}
              onPrev={handlePrevStep}
            />
          )}

          {/* Step 3: Annotated Video */}
          {currentStep === 3 && analysisResults && uploadedFile && (
            <AnnotatedVideo 
              originalVideo={uploadedFile}
              analysisResults={analysisResults}
              onNext={handleNextStep}
              onPrev={handlePrevStep}
            />
          )}

          {/* Step 4: Export Reports */}
          {currentStep === 4 && analysisResults && (
            <ReportExport 
              results={analysisResults}
              sessionId={sessionId}
              onPrev={handlePrevStep}
              onReset={handleReset}
            />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center text-sm text-gray-600">
            <p>Driver Monitoring Validation System v2.0.0</p>
            <p className="mt-1">
              Powered by MediaPipe, YOLO, and Roboflow Supervision
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App