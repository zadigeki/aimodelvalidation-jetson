import React from 'react'
import { motion } from 'framer-motion'
import { ValidationFile } from '@/types'
import { formatFileSize } from '@/utils/formatters'

interface UploadProgressProps {
  files: ValidationFile[]
  progress: { [fileId: string]: number }
}

const UploadProgress: React.FC<UploadProgressProps> = ({ files, progress }) => {
  return (
    <div className="space-y-4">
      {files.map((file) => {
        const fileProgress = progress[file.id] || 0
        
        return (
          <motion.div
            key={file.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-primary-100 dark:bg-primary-900/20 rounded-lg">
                  {file.type === 'video' ? (
                    <svg className="h-5 w-5 text-primary-600 dark:text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  ) : (
                    <svg className="h-5 w-5 text-primary-600 dark:text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  )}
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                    {file.name}
                  </h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {formatFileSize(file.size)}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {fileProgress}%
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {file.status === 'uploading' ? 'Uploading...' : 'Processing...'}
                </p>
              </div>
            </div>

            {/* Progress bar */}
            <div className="relative">
              <div className="progress">
                <motion.div
                  className="progress-indicator"
                  initial={{ width: 0 }}
                  animate={{ width: `${fileProgress}%` }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />
              </div>
              
              {/* Animated shimmer effect */}
              {fileProgress < 100 && (
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                  animate={{ x: ['-100%', '100%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                />
              )}
            </div>

            {/* Status indicator */}
            <div className="flex items-center justify-between mt-3 text-xs">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  file.status === 'uploading' ? 'bg-blue-500 animate-pulse' :
                  file.status === 'processing' ? 'bg-yellow-500 animate-pulse' :
                  file.status === 'completed' ? 'bg-green-500' :
                  'bg-red-500'
                }`} />
                <span className="text-gray-600 dark:text-gray-400 capitalize">
                  {file.status === 'uploading' ? 'Uploading file' :
                   file.status === 'processing' ? 'Processing file' :
                   file.status === 'completed' ? 'Upload complete' :
                   'Upload failed'}
                </span>
              </div>
              
              {fileProgress > 0 && fileProgress < 100 && (
                <span className="text-gray-500 dark:text-gray-400">
                  {Math.round((file.size * fileProgress) / 100 / 1024 / 1024 * 100) / 100} MB of {Math.round(file.size / 1024 / 1024 * 100) / 100} MB
                </span>
              )}
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

export default UploadProgress