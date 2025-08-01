import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  ArrowPathIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline'
import { useAppStore } from '@/store/appStore'
import { ValidationFile, ValidationProgress } from '@/types'
import ValidationProgressComponent from './ValidationProgress'
import FilePreview from './FilePreview'
import ValidationControls from './ValidationControls'
import wsService from '@/services/websocket'
import toast from 'react-hot-toast'

const ValidationPage: React.FC = () => {
  const { fileId } = useParams<{ fileId: string }>()
  const navigate = useNavigate()
  const { 
    files, 
    selectedFile, 
    selectFile, 
    validationProgress, 
    setValidationProgress, 
    clearValidationProgress,
    updateFile 
  } = useAppStore()

  const [currentFile, setCurrentFile] = useState<ValidationFile | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [canStart, setCanStart] = useState(false)

  useEffect(() => {
    // Find file by ID or use selected file
    let file: ValidationFile | null = null
    
    if (fileId) {
      file = files.find(f => f.id === fileId) || null
    } else if (selectedFile) {
      file = selectedFile
    } else if (files.length > 0) {
      // Use first available file
      file = files.find(f => f.status !== 'uploading') || null
    }

    if (file) {
      setCurrentFile(file)
      selectFile(file)
      setCanStart(file.status === 'processing' || file.status === 'completed')
      setIsValidating(file.status === 'processing' && validationProgress[file.id] !== undefined)
    } else if (fileId) {
      toast.error('File not found')
      navigate('/upload')
    }
  }, [fileId, files, selectedFile, selectFile, navigate, validationProgress])

  useEffect(() => {
    // Listen for validation progress updates
    const handleValidationProgress = (event: CustomEvent) => {
      const progress: ValidationProgress = event.detail
      setValidationProgress(progress.fileId, progress)
      
      if (progress.stage === 'complete') {
        setIsValidating(false)
        clearValidationProgress(progress.fileId)
        
        // Update file status
        updateFile(progress.fileId, { 
          status: 'completed',
          processingProgress: 100
        })
        
        toast.success('Validation completed successfully!')
        
        // Navigate to results
        setTimeout(() => {
          navigate(`/results/${progress.fileId}`)
        }, 1000)
      }
    }

    const handleValidationError = (event: CustomEvent) => {
      const { fileId, message } = event.detail
      setIsValidating(false)
      clearValidationProgress(fileId)
      
      updateFile(fileId, { status: 'error' })
      toast.error(`Validation failed: ${message}`)
    }

    window.addEventListener('validation_progress', handleValidationProgress as EventListener)
    window.addEventListener('validation_error', handleValidationError as EventListener)

    return () => {
      window.removeEventListener('validation_progress', handleValidationProgress as EventListener)
      window.removeEventListener('validation_error', handleValidationError as EventListener)
    }
  }, [setValidationProgress, clearValidationProgress, updateFile, navigate])

  const startValidation = async () => {
    if (!currentFile) return

    try {
      setIsValidating(true)
      
      // Update file status
      updateFile(currentFile.id, { status: 'processing' })
      
      // Subscribe to WebSocket updates for this file
      wsService.subscribeToFile(currentFile.id)
      
      // Start validation (simulated)
      await simulateValidation(currentFile)
      
    } catch (error) {
      setIsValidating(false)
      updateFile(currentFile.id, { status: 'error' })
      toast.error('Failed to start validation')
      console.error('Validation error:', error)
    }
  }

  const stopValidation = () => {
    if (!currentFile) return

    setIsValidating(false)
    clearValidationProgress(currentFile.id)
    updateFile(currentFile.id, { status: 'processing' })
    wsService.unsubscribeFromFile(currentFile.id)
    toast.success('Validation stopped')
  }

  const simulateValidation = async (file: ValidationFile) => {
    const stages = [
      { stage: 'preprocessing', message: 'Preprocessing file...', duration: 2000 },
      { stage: 'detection', message: 'Running object detection...', duration: 5000 },
      { stage: 'annotation', message: 'Generating annotations...', duration: 3000 },
      { stage: 'validation', message: 'Validating results...', duration: 2000 },
      { stage: 'complete', message: 'Validation completed', duration: 500 },
    ]

    for (let i = 0; i < stages.length; i++) {
      const stage = stages[i]
      const progress = Math.round(((i + 1) / stages.length) * 100)

      const progressData: ValidationProgress = {
        fileId: file.id,
        stage: stage.stage as any,
        progress,
        message: stage.message,
        timestamp: new Date(),
        estimatedTimeRemaining: i < stages.length - 1 ? 
          stages.slice(i + 1).reduce((acc, s) => acc + s.duration, 0) / 1000 : 0
      }

      setValidationProgress(file.id, progressData)
      
      if (stage.stage !== 'complete') {
        await new Promise(resolve => setTimeout(resolve, stage.duration))
      }
    }

    // Generate mock results
    const mockResults = {
      confidence: 0.85,
      objects: [
        {
          id: '1',
          class: 'person',
          confidence: 0.92,
          bbox: { x: 100, y: 50, width: 150, height: 300 },
        },
        {
          id: '2', 
          class: 'car',
          confidence: 0.78,
          bbox: { x: 300, y: 200, width: 200, height: 100 },
        },
      ],
      annotations: [],
      summary: {
        totalObjects: 2,
        classDistribution: { person: 1, car: 1 },
        averageConfidence: 0.85,
        processingTime: 12.5,
        qualityScore: 0.89,
      },
      processedAt: new Date(),
    }

    updateFile(file.id, { 
      results: mockResults,
      status: 'completed'
    })

    // Dispatch completion event
    const event = new CustomEvent('validation_complete', {
      detail: { fileId: file.id, fileName: file.name }
    })
    window.dispatchEvent(event)
  }

  if (!currentFile) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center py-12">
          <ChartBarIcon className="h-16 w-16 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-2">
            No File Selected
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Please select a file to start validation
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="btn-primary"
          >
            Upload Files
          </button>
        </div>
      </div>
    )
  }

  const progress = validationProgress[currentFile.id]

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Validation
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Run AI model validation on your files
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* File preview */}
          <div className="lg:col-span-2">
            <FilePreview 
              file={currentFile}
              showAnnotations={isValidating || currentFile.status === 'completed'}
            />
          </div>

          {/* Controls and progress */}
          <div className="lg:col-span-1 space-y-6">
            {/* Validation controls */}
            <div className="card">
              <div className="card-header">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Validation Controls
                </h2>
              </div>
              <div className="card-content">
                <ValidationControls
                  file={currentFile}
                  isValidating={isValidating}
                  canStart={canStart}
                  onStart={startValidation}
                  onStop={stopValidation}
                  onRestart={() => {
                    stopValidation()
                    setTimeout(startValidation, 1000)
                  }}
                />
              </div>
            </div>

            {/* Progress */}
            {progress && (
              <div className="card">
                <div className="card-header">
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                    Progress
                  </h2>
                </div>
                <div className="card-content">
                  <ValidationProgressComponent progress={progress} />
                </div>
              </div>
            )}

            {/* File info */}
            <div className="card">
              <div className="card-header">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  File Information
                </h2>
              </div>
              <div className="card-content space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Name:</span>
                  <span className="font-medium text-gray-900 dark:text-white truncate max-w-[200px]">
                    {currentFile.name}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Type:</span>
                  <span className="font-medium text-gray-900 dark:text-white capitalize">
                    {currentFile.type}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Size:</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {(currentFile.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Status:</span>
                  <span className={`font-medium capitalize ${
                    currentFile.status === 'completed' ? 'text-green-600 dark:text-green-400' :
                    currentFile.status === 'processing' ? 'text-yellow-600 dark:text-yellow-400' :
                    currentFile.status === 'error' ? 'text-red-600 dark:text-red-400' :
                    'text-gray-600 dark:text-gray-400'
                  }`}>
                    {currentFile.status}
                  </span>
                </div>
                {currentFile.metadata && (
                  <>
                    {currentFile.metadata.width && currentFile.metadata.height && (
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Resolution:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {currentFile.metadata.width} × {currentFile.metadata.height}
                        </span>
                      </div>
                    )}
                    {currentFile.metadata.duration && (
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Duration:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {Math.round(currentFile.metadata.duration)}s
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default ValidationPage