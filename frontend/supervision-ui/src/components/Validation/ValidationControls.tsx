import React from 'react'
import { motion } from 'framer-motion'
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline'
import { ValidationFile } from '@/types'

interface ValidationControlsProps {
  file: ValidationFile
  isValidating: boolean
  canStart: boolean
  onStart: () => void
  onStop: () => void
  onRestart: () => void
}

const ValidationControls: React.FC<ValidationControlsProps> = ({
  file,
  isValidating,
  canStart,
  onStart,
  onStop,
  onRestart,
}) => {
  const getStatusColor = (status: ValidationFile['status']) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 dark:text-green-400'
      case 'processing':
        return 'text-yellow-600 dark:text-yellow-400'
      case 'error':
        return 'text-red-600 dark:text-red-400'
      default:
        return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusMessage = () => {
    if (isValidating) {
      return 'Validation is currently running...'
    }
    
    switch (file.status) {
      case 'uploading':
        return 'File is still uploading. Please wait before starting validation.'
      case 'processing':
        return 'File is ready for validation.'
      case 'completed':
        return 'Validation has been completed for this file.'
      case 'error':
        return 'An error occurred. You can retry validation.'
      default:
        return 'Unknown status'
    }
  }

  return (
    <div className="space-y-6">
      {/* Status indicator */}
      <div className="text-center">
        <div className={`text-lg font-semibold mb-2 ${getStatusColor(file.status)}`}>
          <motion.div
            key={file.status}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-center space-x-2"
          >
            <div className={`w-3 h-3 rounded-full ${
              file.status === 'completed' ? 'bg-green-500' :
              file.status === 'processing' || isValidating ? 'bg-yellow-500 animate-pulse' :
              file.status === 'error' ? 'bg-red-500' :
              'bg-gray-400'
            }`} />
            <span className="capitalize">{file.status}</span>
          </motion.div>
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {getStatusMessage()}
        </p>
      </div>

      {/* Control buttons */}
      <div className="flex flex-col space-y-3">
        {!isValidating ? (
          <>
            {/* Start validation */}
            <motion.button
              whileHover={{ scale: canStart ? 1.02 : 1 }}
              whileTap={{ scale: canStart ? 0.98 : 1 }}
              onClick={onStart}
              disabled={!canStart}
              className={`btn w-full flex items-center justify-center space-x-2 ${
                canStart 
                  ? 'btn-primary' 
                  : 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
              }`}
            >
              <PlayIcon className="h-5 w-5" />
              <span>
                {file.status === 'completed' ? 'Re-run Validation' : 'Start Validation'}
              </span>
            </motion.button>

            {/* Restart validation (only if completed or error) */}
            {(file.status === 'completed' || file.status === 'error') && (
              <button
                onClick={onRestart}
                className="btn-outline w-full flex items-center justify-center space-x-2"
              >
                <ArrowPathIcon className="h-5 w-5" />
                <span>Restart Validation</span>
              </button>
            )}
          </>
        ) : (
          /* Stop validation */
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onStop}
            className="btn bg-red-600 hover:bg-red-700 text-white w-full flex items-center justify-center space-x-2"
          >
            <StopIcon className="h-5 w-5" />
            <span>Stop Validation</span>
          </motion.button>
        )}
      </div>

      {/* Validation options */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
          Validation Options
        </h3>
        <div className="space-y-3 text-sm">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              defaultChecked
              className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-gray-700 dark:text-gray-300">Object Detection</span>
          </label>
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              defaultChecked
              className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-gray-700 dark:text-gray-300">Classification</span>
          </label>
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-gray-700 dark:text-gray-300">Segmentation</span>
          </label>
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              defaultChecked
              className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-gray-700 dark:text-gray-300">Quality Analysis</span>
          </label>
        </div>
      </div>

      {/* Confidence threshold */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
        <label className="block text-sm font-medium text-gray-900 dark:text-white mb-2">
          Confidence Threshold
        </label>
        <div className="space-y-2">
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            defaultValue="0.5"
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>0.0</span>
            <span>0.5</span>
            <span>1.0</span>
          </div>
        </div>
      </div>

      {/* Progress info */}
      {file.processingProgress !== undefined && file.processingProgress < 100 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
          <div className="flex items-center space-x-2 text-sm text-blue-800 dark:text-blue-200">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            <span>Processing: {file.processingProgress}%</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default ValidationControls