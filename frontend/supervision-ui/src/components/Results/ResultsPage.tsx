import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ChartBarIcon,
  EyeIcon,
  DocumentArrowDownIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline'
import { useAppStore } from '@/store/appStore'
import { ValidationFile } from '@/types'
import ResultsViewer from './ResultsViewer'
import ResultsStats from './ResultsStats'
import ConfidenceFilter from './ConfidenceFilter'
import ClassFilter from './ClassFilter'
import ExportOptions from './ExportOptions'
import { formatConfidence, getConfidenceColor } from '@/utils/formatters'

const ResultsPage: React.FC = () => {
  const { fileId } = useParams<{ fileId: string }>()
  const navigate = useNavigate()
  const { files, selectedFile, selectFile } = useAppStore()

  const [currentFile, setCurrentFile] = useState<ValidationFile | null>(null)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5)
  const [selectedClasses, setSelectedClasses] = useState<string[]>([])
  const [showAnnotations, setShowAnnotations] = useState(true)
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'grid'>('overview')

  useEffect(() => {
    // Find file by ID or use selected file
    let file: ValidationFile | null = null
    
    if (fileId) {
      file = files.find(f => f.id === fileId) || null
    } else if (selectedFile?.results) {
      file = selectedFile
    } else {
      // Use first file with results
      file = files.find(f => f.results) || null
    }

    if (file) {
      setCurrentFile(file)
      selectFile(file)
      
      // Initialize class filter with all available classes
      if (file.results) {
        const classes = Object.keys(file.results.summary.classDistribution)
        setSelectedClasses(classes)
      }
    } else if (fileId) {
      navigate('/validation')
    }
  }, [fileId, files, selectedFile, selectFile, navigate])

  const filteredObjects = currentFile?.results?.objects.filter(obj => {
    const meetsConfidence = obj.confidence >= confidenceThreshold
    const meetsClass = selectedClasses.length === 0 || selectedClasses.includes(obj.class)
    return meetsConfidence && meetsClass
  }) || []

  const availableClasses = currentFile?.results ? 
    Object.keys(currentFile.results.summary.classDistribution) : []

  if (!currentFile) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center py-12">
          <ChartBarIcon className="h-16 w-16 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-2">
            No Results Available
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Please complete validation first to view results
          </p>
          <button
            onClick={() => navigate('/validation')}
            className="btn-primary"
          >
            Start Validation
          </button>
        </div>
      </div>
    )
  }

  if (!currentFile.results) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center py-12">
          <EyeIcon className="h-16 w-16 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-2">
            Validation In Progress
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Results will appear here once validation is complete
          </p>
          <button
            onClick={() => navigate(`/validation/${currentFile.id}`)}
            className="btn-primary"
          >
            View Validation Progress
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Validation Results
              </h1>
              <p className="mt-2 text-gray-600 dark:text-gray-400">
                Analysis results for {currentFile.name}
              </p>
            </div>
            
            {/* View mode toggle */}
            <div className="flex items-center space-x-2">
              <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
                {[
                  { key: 'overview', label: 'Overview', icon: ChartBarIcon },
                  { key: 'detailed', label: 'Detailed', icon: EyeIcon },
                  { key: 'grid', label: 'Grid', icon: AdjustmentsHorizontalIcon },
                ].map((mode) => (
                  <button
                    key={mode.key}
                    onClick={() => setViewMode(mode.key as any)}
                    className={`flex items-center space-x-1 px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                      viewMode === mode.key
                        ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                        : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    }`}
                  >
                    <mode.icon className="h-4 w-4" />
                    <span>{mode.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Quick stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {currentFile.results.summary.totalObjects}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Objects Detected</div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatConfidence(currentFile.results.summary.averageConfidence)}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Avg Confidence</div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {availableClasses.length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Classes Found</div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatConfidence(currentFile.results.summary.qualityScore)}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Quality Score</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Main content */}
          <div className="lg:col-span-3">
            <ResultsViewer
              file={currentFile}
              filteredObjects={filteredObjects}
              showAnnotations={showAnnotations}
              viewMode={viewMode}
            />
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* Filters */}
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Filters
                </h2>
                <div className="flex items-center space-x-2">
                  <label className="flex items-center space-x-1 text-sm">
                    <input
                      type="checkbox"
                      checked={showAnnotations}
                      onChange={(e) => setShowAnnotations(e.target.checked)}
                      className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="text-gray-700 dark:text-gray-300">Show Annotations</span>
                  </label>
                </div>
              </div>
              <div className="card-content space-y-4">
                {/* Confidence filter */}
                <ConfidenceFilter
                  threshold={confidenceThreshold}
                  onChange={setConfidenceThreshold}
                  totalObjects={currentFile.results.summary.totalObjects}
                  filteredCount={filteredObjects.length}
                />

                {/* Class filter */}
                <ClassFilter
                  availableClasses={availableClasses}
                  selectedClasses={selectedClasses}
                  onChange={setSelectedClasses}
                  classDistribution={currentFile.results.summary.classDistribution}
                />
              </div>
            </div>

            {/* Statistics */}
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Statistics
                </h2>
              </div>
              <div className="card-content">
                <ResultsStats
                  results={currentFile.results}
                  filteredObjects={filteredObjects}
                />
              </div>
            </div>

            {/* Export options */}
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center space-x-2">
                  <DocumentArrowDownIcon className="h-5 w-5" />
                  <span>Export</span>
                </h2>
              </div>
              <div className="card-content">
                <ExportOptions
                  file={currentFile}
                  filteredObjects={filteredObjects}
                />
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default ResultsPage