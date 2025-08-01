import React from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  EyeIcon,
  TrashIcon,
  PlayIcon,
  DocumentTextIcon,
} from '@heroicons/react/24/outline'
import { ValidationFile } from '@/types'
import { useAppStore } from '@/store/appStore'
import { formatFileSize, formatDate } from '@/utils/formatters'

interface FileListProps {
  files: ValidationFile[]
}

const FileList: React.FC<FileListProps> = ({ files }) => {
  const { removeFile, selectFile } = useAppStore()
  const navigate = useNavigate()

  const handleViewFile = (file: ValidationFile) => {
    selectFile(file)
    if (file.results) {
      navigate(`/results/${file.id}`)
    } else {
      navigate(`/validation/${file.id}`)
    }
  }

  const handleDeleteFile = (fileId: string) => {
    if (window.confirm('Are you sure you want to delete this file?')) {
      removeFile(fileId)
    }
  }

  const getStatusBadge = (status: ValidationFile['status']) => {
    const badges = {
      uploading: 'badge-primary',
      processing: 'badge-warning',
      completed: 'badge-success',
      error: 'badge-error',
    }
    
    const labels = {
      uploading: 'Uploading',
      processing: 'Processing',
      completed: 'Completed',
      error: 'Error',
    }

    return (
      <span className={`badge ${badges[status]}`}>
        {labels[status]}
      </span>
    )
  }

  if (files.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500 dark:text-gray-400">
        No files uploaded yet
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {files.map((file, index) => (
        <motion.div
          key={file.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="card overflow-hidden hover:shadow-lg transition-shadow duration-200"
        >
          {/* File preview */}
          <div className="relative aspect-video bg-gray-100 dark:bg-gray-700 overflow-hidden">
            {file.type === 'image' && file.url ? (
              <img
                src={file.url}
                alt={file.name}
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.currentTarget.src = '/placeholder-image.png'
                }}
              />
            ) : file.type === 'video' && file.metadata?.thumbnailUrl ? (
              <div className="relative">
                <img
                  src={file.metadata.thumbnailUrl}
                  alt={file.name}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="bg-black/50 rounded-full p-3">
                    <PlayIcon className="h-8 w-8 text-white" />
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full">
                <DocumentTextIcon className="h-16 w-16 text-gray-400 dark:text-gray-600" />
              </div>
            )}

            {/* Status overlay */}
            <div className="absolute top-2 right-2">
              {getStatusBadge(file.status)}
            </div>

            {/* Processing overlay */}
            {file.status === 'processing' && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <div className="text-center text-white">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
                  <p className="text-sm">Processing...</p>
                  {file.processingProgress && (
                    <p className="text-xs mt-1">{file.processingProgress}%</p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* File info */}
          <div className="card-content">
            <div className="mb-3">
              <h3 className="font-medium text-gray-900 dark:text-white truncate">
                {file.name}
              </h3>
              <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400 mt-1">
                <span>{formatFileSize(file.size)}</span>
                <span>{formatDate(file.createdAt)}</span>
              </div>
            </div>

            {/* Metadata */}
            {file.metadata && (
              <div className="mb-3 text-xs text-gray-600 dark:text-gray-400 space-y-1">
                {file.metadata.width && file.metadata.height && (
                  <div>Resolution: {file.metadata.width} × {file.metadata.height}</div>
                )}
                {file.metadata.duration && (
                  <div>Duration: {Math.round(file.metadata.duration)}s</div>
                )}
                {file.metadata.frameRate && (
                  <div>Frame Rate: {file.metadata.frameRate} fps</div>
                )}
              </div>
            )}

            {/* Results summary */}
            {file.results && (
              <div className="mb-3 p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <div>Objects: {file.results.summary.totalObjects}</div>
                  <div>Confidence: {(file.results.summary.averageConfidence * 100).toFixed(1)}%</div>
                  <div>Quality: {(file.results.summary.qualityScore * 100).toFixed(1)}%</div>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center justify-between">
              <button
                onClick={() => handleViewFile(file)}
                className="btn-outline btn-sm flex items-center space-x-1"
                disabled={file.status === 'uploading'}
              >
                <EyeIcon className="h-4 w-4" />
                <span>
                  {file.results ? 'View Results' : 'View Details'}
                </span>
              </button>

              <button
                onClick={() => handleDeleteFile(file.id)}
                className="p-1 text-gray-400 hover:text-red-500 transition-colors"
              >
                <TrashIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  )
}

export default FileList