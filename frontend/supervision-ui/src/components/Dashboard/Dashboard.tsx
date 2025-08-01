import React from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  CloudArrowUpIcon,
  EyeIcon,
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import { useAppStore } from '@/store/appStore'
import { formatFileSize, formatNumber } from '@/utils/formatters'

const Dashboard: React.FC = () => {
  const { files, validationProgress } = useAppStore()
  const navigate = useNavigate()

  // Calculate statistics
  const stats = {
    totalFiles: files.length,
    completedFiles: files.filter(f => f.status === 'completed').length,
    processingFiles: files.filter(f => f.status === 'processing').length,
    errorFiles: files.filter(f => f.status === 'error').length,
    totalObjects: files.reduce((acc, f) => acc + (f.results?.summary.totalObjects || 0), 0),
    totalSize: files.reduce((acc, f) => acc + f.size, 0),
    averageConfidence: files.length > 0 
      ? files.reduce((acc, f) => acc + (f.results?.summary.averageConfidence || 0), 0) / files.length
      : 0,
  }

  const recentFiles = files
    .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
    .slice(0, 5)

  const activeValidations = Object.keys(validationProgress).length

  const statCards = [
    {
      title: 'Total Files',
      value: formatNumber(stats.totalFiles),
      icon: ChartBarIcon,
      color: 'blue',
      change: '+12% from last week',
    },
    {
      title: 'Completed',
      value: formatNumber(stats.completedFiles),
      icon: CheckCircleIcon,
      color: 'green',
      change: `${stats.totalFiles > 0 ? Math.round((stats.completedFiles / stats.totalFiles) * 100) : 0}% completion rate`,
    },
    {
      title: 'Processing',
      value: formatNumber(stats.processingFiles + activeValidations),
      icon: ClockIcon,
      color: 'yellow',
      change: `${activeValidations} active validations`,
    },
    {
      title: 'Objects Detected',
      value: formatNumber(stats.totalObjects),
      icon: EyeIcon,
      color: 'purple',
      change: `Avg confidence: ${(stats.averageConfidence * 100).toFixed(1)}%`,
    },
  ]

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
            Dashboard
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Overview of your AI model validation activities
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {statCards.map((stat, index) => (
            <motion.div
              key={stat.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card"
            >
              <div className="card-content">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                      {stat.title}
                    </p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {stat.value}
                    </p>
                  </div>
                  <div className={`p-3 rounded-full bg-${stat.color}-100 dark:bg-${stat.color}-900/20`}>
                    <stat.icon className={`h-6 w-6 text-${stat.color}-600 dark:text-${stat.color}-400`} />
                  </div>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                  {stat.change}
                </p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="card cursor-pointer hover:shadow-lg transition-all duration-200"
            onClick={() => navigate('/upload')}
          >
            <div className="card-content">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-primary-100 dark:bg-primary-900/20 rounded-lg">
                  <CloudArrowUpIcon className="h-8 w-8 text-primary-600 dark:text-primary-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Upload Files
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Start by uploading images or videos
                  </p>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="card cursor-pointer hover:shadow-lg transition-all duration-200"
            onClick={() => navigate('/validation')}
          >
            <div className="card-content">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-lg">
                  <EyeIcon className="h-8 w-8 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Start Validation
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Run AI model validation
                  </p>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="card cursor-pointer hover:shadow-lg transition-all duration-200"
            onClick={() => navigate('/results')}
          >
            <div className="card-content">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
                  <ChartBarIcon className="h-8 w-8 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    View Results
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Analyze validation results
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Recent Files */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="card"
          >
            <div className="card-header">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Recent Files
              </h2>
            </div>
            <div className="card-content">
              {recentFiles.length > 0 ? (
                <div className="space-y-4">
                  {recentFiles.map((file) => (
                    <div
                      key={file.id}
                      className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                      onClick={() => {
                        if (file.results) {
                          navigate(`/results/${file.id}`)
                        } else {
                          navigate(`/validation/${file.id}`)
                        }
                      }}
                    >
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-primary-100 dark:bg-primary-900/20 rounded">
                          {file.type === 'video' ? (
                            <svg className="h-4 w-4 text-primary-600 dark:text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          ) : (
                            <svg className="h-4 w-4 text-primary-600 dark:text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                          )}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            {file.name}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {formatFileSize(file.size)} • {file.status}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {file.status === 'completed' && (
                          <CheckCircleIcon className="h-4 w-4 text-green-500" />
                        )}
                        {file.status === 'processing' && (
                          <ClockIcon className="h-4 w-4 text-yellow-500" />
                        )}
                        {file.status === 'error' && (
                          <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <CloudArrowUpIcon className="h-12 w-12 mx-auto mb-4 text-gray-300 dark:text-gray-600" />
                  <p>No files uploaded yet</p>
                  <button
                    onClick={() => navigate('/upload')}
                    className="btn-primary mt-4"
                  >
                    Upload Your First File
                  </button>
                </div>
              )}
            </div>
          </motion.div>

          {/* System Status */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7 }}
            className="card"
          >
            <div className="card-header">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                System Status
              </h2>
            </div>
            <div className="card-content space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">API Service</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-sm font-medium text-green-600 dark:text-green-400">Online</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">WebSocket</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse-slow"></div>
                  <span className="text-sm font-medium text-green-600 dark:text-green-400">Connected</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Storage</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                  <span className="text-sm font-medium text-yellow-600 dark:text-yellow-400">78% Used</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Processing Queue</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  <span className="text-sm font-medium text-blue-600 dark:text-blue-400">{activeValidations} Active</span>
                </div>
              </div>

              {stats.totalSize > 0 && (
                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Total Storage Used</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {formatFileSize(stats.totalSize)}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

export default Dashboard