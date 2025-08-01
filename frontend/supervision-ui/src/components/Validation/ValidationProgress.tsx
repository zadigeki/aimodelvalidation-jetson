import React from 'react'
import { motion } from 'framer-motion'
import { ValidationProgress as ValidationProgressType } from '@/types'
import { formatDuration } from '@/utils/formatters'

interface ValidationProgressProps {
  progress: ValidationProgressType
}

const ValidationProgress: React.FC<ValidationProgressProps> = ({ progress }) => {
  const getStageIcon = (stage: string, isActive: boolean, isCompleted: boolean) => {
    const iconClass = `h-4 w-4 ${
      isCompleted ? 'text-green-500' : 
      isActive ? 'text-primary-500' : 
      'text-gray-400 dark:text-gray-600'
    }`

    switch (stage) {
      case 'upload':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        )
      case 'preprocessing':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        )
      case 'detection':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        )
      case 'annotation':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
          </svg>
        )
      case 'validation':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      case 'complete':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )
      default:
        return (
          <div className={`w-4 h-4 rounded-full border-2 ${
            isCompleted ? 'bg-green-500 border-green-500' :
            isActive ? 'bg-primary-500 border-primary-500' :
            'border-gray-300 dark:border-gray-600'
          }`} />
        )
    }
  }

  const stages = [
    { key: 'upload', label: 'Upload' },
    { key: 'preprocessing', label: 'Preprocessing' },
    { key: 'detection', label: 'Object Detection' },
    { key: 'annotation', label: 'Annotation' },
    { key: 'validation', label: 'Validation' },
    { key: 'complete', label: 'Complete' },
  ]

  const currentStageIndex = stages.findIndex(s => s.key === progress.stage)

  return (
    <div className="space-y-6">
      {/* Progress bar */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Progress
          </span>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {progress.progress}%
          </span>
        </div>
        <div className="progress">
          <motion.div
            className="progress-indicator"
            initial={{ width: 0 }}
            animate={{ width: `${progress.progress}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Current status */}
      <div className="text-center">
        <motion.div
          key={progress.stage}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-lg font-medium text-gray-900 dark:text-white mb-2"
        >
          {progress.message}
        </motion.div>
        
        {progress.estimatedTimeRemaining && progress.estimatedTimeRemaining > 0 && (
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Estimated time remaining: {formatDuration(progress.estimatedTimeRemaining)}
          </div>
        )}
      </div>

      {/* Stage indicators */}
      <div className="space-y-3">
        {stages.map((stage, index) => {
          const isActive = stage.key === progress.stage
          const isCompleted = index < currentStageIndex || progress.stage === 'complete'
          
          return (
            <motion.div
              key={stage.key}
              initial={{ opacity: 0.5 }}
              animate={{ 
                opacity: isActive || isCompleted ? 1 : 0.5,
                scale: isActive ? 1.02 : 1 
              }}
              className={`flex items-center space-x-3 p-2 rounded-lg transition-colors ${
                isActive ? 'bg-primary-50 dark:bg-primary-900/20' : ''
              }`}
            >
              <div className="flex-shrink-0">
                {getStageIcon(stage.key, isActive, isCompleted)}
              </div>
              
              <div className="flex-grow">
                <div className={`text-sm font-medium ${
                  isCompleted ? 'text-green-600 dark:text-green-400' :
                  isActive ? 'text-primary-600 dark:text-primary-400' :
                  'text-gray-500 dark:text-gray-400'
                }`}>
                  {stage.label}
                </div>
              </div>
              
              <div className="flex-shrink-0">
                {isCompleted && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="w-2 h-2 bg-green-500 rounded-full"
                  />
                )}
                {isActive && !isCompleted && (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                    className="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full"
                  />
                )}
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Processing details */}
      {progress.stage !== 'complete' && (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
            <motion.div
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="w-2 h-2 bg-primary-500 rounded-full"
            />
            <span>Processing in progress...</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default ValidationProgress