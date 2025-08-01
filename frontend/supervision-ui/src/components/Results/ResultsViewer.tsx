import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { ValidationFile, DetectedObject } from '@/types'
import VideoPlayer from '../Video/VideoPlayer'
import { PhotoIcon, VideoCameraIcon } from '@heroicons/react/24/outline'
import { formatConfidence, getConfidenceColor } from '@/utils/formatters'

interface ResultsViewerProps {
  file: ValidationFile
  filteredObjects: DetectedObject[]
  showAnnotations: boolean
  viewMode: 'overview' | 'detailed' | 'grid'
}

const ResultsViewer: React.FC<ResultsViewerProps> = ({
  file,
  filteredObjects,
  showAnnotations,
  viewMode,
}) => {
  const [selectedObject, setSelectedObject] = useState<DetectedObject | null>(null)
  const [currentFrame, setCurrentFrame] = useState(0)

  const handleFrameChange = (frameNumber: number, timestamp: number) => {
    setCurrentFrame(frameNumber)
    
    // Find objects in current frame for videos
    if (file.type === 'video' && file.results?.frames) {
      const frameResult = file.results.frames.find(f => 
        Math.abs(f.frameNumber - frameNumber) <= 1
      )
      if (frameResult && frameResult.objects.length > 0) {
        setSelectedObject(frameResult.objects[0])
      }
    }
  }

  const handleObjectClick = (object: DetectedObject) => {
    setSelectedObject(object)
    
    // For videos, seek to the frame containing this object
    if (file.type === 'video' && object.frame !== undefined) {
      setCurrentFrame(object.frame)
    }
  }

  if (viewMode === 'grid') {
    return <GridView file={file} filteredObjects={filteredObjects} onObjectClick={handleObjectClick} />
  }

  return (
    <div className="space-y-6">
      {/* Main viewer */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              {viewMode === 'overview' ? 'Results Overview' : 'Detailed Analysis'}
            </h2>
            <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
              <span>{filteredObjects.length} objects visible</span>
              {file.type === 'video' && (
                <span>• Frame {currentFrame}</span>
              )}
            </div>
          </div>
        </div>
        
        <div className="card-content">
          {file.type === 'video' ? (
            <VideoPlayer
              file={file}
              frames={file.results?.frames}
              showAnnotations={showAnnotations}
              onFrameChange={handleFrameChange}
              className="w-full rounded-lg overflow-hidden"
            />
          ) : (
            <ImageViewer
              file={file}
              objects={filteredObjects}
              showAnnotations={showAnnotations}
              selectedObject={selectedObject}
              onObjectClick={handleObjectClick}
            />
          )}
        </div>
      </div>

      {/* Object details */}
      {viewMode === 'detailed' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Object list */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Detected Objects ({filteredObjects.length})
              </h3>
            </div>
            <div className="card-content max-h-96 overflow-y-auto custom-scrollbar">
              <div className="space-y-2">
                {filteredObjects.map((object, index) => (
                  <motion.div
                    key={object.id || index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      selectedObject?.id === object.id
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                    onClick={() => handleObjectClick(object)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${
                          object.confidence >= 0.8 ? 'bg-green-500' :
                          object.confidence >= 0.6 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`} />
                        <div>
                          <div className="font-medium text-gray-900 dark:text-white capitalize">
                            {object.class}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            {formatConfidence(object.confidence)}
                            {object.frame !== undefined && (
                              <span> • Frame {object.frame}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className={`badge ${
                        object.confidence >= 0.8 ? 'badge-success' :
                        object.confidence >= 0.6 ? 'badge-warning' :
                        'badge-error'
                      }`}>
                        {getConfidenceColor(object.confidence)}
                      </div>
                    </div>
                    
                    {/* Bounding box info */}
                    <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                      Position: ({object.bbox.x}, {object.bbox.y}) 
                      Size: {object.bbox.width} × {object.bbox.height}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>

          {/* Selected object details */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Object Details
              </h3>
            </div>
            <div className="card-content">
              {selectedObject ? (
                <ObjectDetails object={selectedObject} file={file} />
              ) : (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <div className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </div>
                  <p>Select an object to view details</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Image viewer component
const ImageViewer: React.FC<{
  file: ValidationFile
  objects: DetectedObject[]
  showAnnotations: boolean
  selectedObject: DetectedObject | null
  onObjectClick: (object: DetectedObject) => void
}> = ({ file, objects, showAnnotations, selectedObject, onObjectClick }) => {
  return (
    <div className="relative aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg overflow-hidden">
      <img
        src={file.url}
        alt={file.name}
        className="w-full h-full object-contain"
      />
      
      {/* Annotations */}
      {showAnnotations && (
        <div className="absolute inset-0">
          {objects.map((object, index) => {
            const isSelected = selectedObject?.id === object.id
            const confidenceColor = object.confidence >= 0.8 ? 'border-green-500' :
                                   object.confidence >= 0.6 ? 'border-yellow-500' :
                                   'border-red-500'
            
            return (
              <motion.div
                key={object.id || index}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className={`absolute border-2 cursor-pointer transition-all ${confidenceColor} ${
                  isSelected ? 'bg-primary-500/20 border-4' : 'bg-black/10 hover:bg-black/20'
                }`}
                style={{
                  left: `${(object.bbox.x / (file.metadata?.width || 1)) * 100}%`,
                  top: `${(object.bbox.y / (file.metadata?.height || 1)) * 100}%`,
                  width: `${(object.bbox.width / (file.metadata?.width || 1)) * 100}%`,
                  height: `${(object.bbox.height / (file.metadata?.height || 1)) * 100}%`,
                }}
                onClick={() => onObjectClick(object)}
              >
                <div className="absolute -top-6 left-0 bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                  {object.class} {formatConfidence(object.confidence)}
                </div>
              </motion.div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// Grid view component
const GridView: React.FC<{
  file: ValidationFile
  filteredObjects: DetectedObject[]
  onObjectClick: (object: DetectedObject) => void
}> = ({ file, filteredObjects, onObjectClick }) => {
  return (
    <div className="card">
      <div className="card-header">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Objects Grid View
        </h2>
      </div>
      <div className="card-content">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {filteredObjects.map((object, index) => (
            <motion.div
              key={object.id || index}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              onClick={() => onObjectClick(object)}
            >
              <div className="text-center">
                <div className={`w-12 h-12 mx-auto mb-3 rounded-full flex items-center justify-center ${
                  object.confidence >= 0.8 ? 'bg-green-100 dark:bg-green-900/20' :
                  object.confidence >= 0.6 ? 'bg-yellow-100 dark:bg-yellow-900/20' :
                  'bg-red-100 dark:bg-red-900/20'
                }`}>
                  <span className={`text-lg ${
                    object.confidence >= 0.8 ? 'text-green-600 dark:text-green-400' :
                    object.confidence >= 0.6 ? 'text-yellow-600 dark:text-yellow-400' :
                    'text-red-600 dark:text-red-400'
                  }`}>
                    {object.class.charAt(0).toUpperCase()}
                  </span>
                </div>
                <div className="font-medium text-gray-900 dark:text-white capitalize mb-1">
                  {object.class}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  {formatConfidence(object.confidence)}
                </div>
                {object.frame !== undefined && (
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Frame {object.frame}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Object details component
const ObjectDetails: React.FC<{
  object: DetectedObject
  file: ValidationFile
}> = ({ object, file }) => {
  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-3">
        <div className={`w-4 h-4 rounded-full ${
          object.confidence >= 0.8 ? 'bg-green-500' :
          object.confidence >= 0.6 ? 'bg-yellow-500' :
          'bg-red-500'
        }`} />
        <h4 className="text-lg font-semibold text-gray-900 dark:text-white capitalize">
          {object.class}
        </h4>
      </div>

      <div className="space-y-3 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-600 dark:text-gray-400">Confidence</span>
          <span className="font-medium text-gray-900 dark:text-white">
            {formatConfidence(object.confidence)}
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-600 dark:text-gray-400">Position</span>
          <span className="font-medium text-gray-900 dark:text-white">
            ({object.bbox.x}, {object.bbox.y})
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-600 dark:text-gray-400">Size</span>
          <span className="font-medium text-gray-900 dark:text-white">
            {object.bbox.width} × {object.bbox.height}
          </span>
        </div>

        {object.frame !== undefined && (
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Frame</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {object.frame}
            </span>
          </div>
        )}

        {object.timestamp !== undefined && (
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Timestamp</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {object.timestamp.toFixed(2)}s
            </span>
          </div>
        )}
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
          <span>Confidence Level</span>
          <span>{formatConfidence(object.confidence)}</span>
        </div>
        <div className="confidence-bar">
          <div
            className={`confidence-fill ${
              object.confidence >= 0.8 ? 'bg-green-500' :
              object.confidence >= 0.6 ? 'bg-yellow-500' :
              'bg-red-500'
            }`}
            style={{ width: `${object.confidence * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}

export default ResultsViewer