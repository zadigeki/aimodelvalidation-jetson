import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { ValidationFile } from '@/types'
import VideoPlayer from '../Video/VideoPlayer'
import { PhotoIcon, VideoCameraIcon } from '@heroicons/react/24/outline'

interface FilePreviewProps {
  file: ValidationFile
  showAnnotations?: boolean
  className?: string
}

const FilePreview: React.FC<FilePreviewProps> = ({
  file,
  showAnnotations = false,
  className = '',
}) => {
  const [currentFrame, setCurrentFrame] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)

  const handleFrameChange = (frameNumber: number, timestamp: number) => {
    setCurrentFrame(frameNumber)
    setCurrentTime(timestamp)
  }

  const handleTimeUpdate = (time: number) => {
    setCurrentTime(time)
  }

  if (!file.url) {
    return (
      <div className={`card ${className}`}>
        <div className="card-content">
          <div className="aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center">
            <div className="text-center text-gray-500 dark:text-gray-400">
              <div className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600">
                {file.type === 'video' ? (
                  <VideoCameraIcon className="w-full h-full" />
                ) : (
                  <PhotoIcon className="w-full h-full" />
                )}
              </div>
              <p>File not available for preview</p>
              <p className="text-sm mt-1">Status: {file.status}</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`card ${className}`}>
      <div className="card-header">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            File Preview
          </h2>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {file.name}
            </span>
            {showAnnotations && (
              <span className="badge badge-primary text-xs">
                Annotations ON
              </span>
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
            onTimeUpdate={handleTimeUpdate}
            className="w-full rounded-lg overflow-hidden"
          />
        ) : (
          <ImagePreview
            file={file}
            showAnnotations={showAnnotations}
            className="w-full rounded-lg overflow-hidden"
          />
        )}

        {/* File metadata */}
        {file.metadata && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {file.metadata.width && file.metadata.height && (
              <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <div className="font-medium text-gray-900 dark:text-white">
                  {file.metadata.width} × {file.metadata.height}
                </div>
                <div className="text-gray-500 dark:text-gray-400">Resolution</div>
              </div>
            )}
            
            {file.metadata.duration && (
              <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <div className="font-medium text-gray-900 dark:text-white">
                  {Math.round(file.metadata.duration)}s
                </div>
                <div className="text-gray-500 dark:text-gray-400">Duration</div>
              </div>
            )}
            
            {file.metadata.frameRate && (
              <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <div className="font-medium text-gray-900 dark:text-white">
                  {file.metadata.frameRate} fps
                </div>
                <div className="text-gray-500 dark:text-gray-400">Frame Rate</div>
              </div>
            )}
            
            <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
              <div className="font-medium text-gray-900 dark:text-white">
                {(file.size / 1024 / 1024).toFixed(1)} MB
              </div>
              <div className="text-gray-500 dark:text-gray-400">File Size</div>
            </div>
          </div>
        )}

        {/* Current frame info for videos */}
        {file.type === 'video' && (
          <div className="mt-4 flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
            <div>
              Frame: {currentFrame}
            </div>
            <div>
              Time: {Math.round(currentTime * 100) / 100}s
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Image preview component with annotation support
const ImagePreview: React.FC<{
  file: ValidationFile
  showAnnotations: boolean
  className?: string
}> = ({ file, showAnnotations, className = '' }) => {
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageError, setImageError] = useState(false)

  return (
    <div className={`relative ${className}`}>
      <div className="aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg overflow-hidden relative">
        {!imageError ? (
          <>
            <img
              src={file.url}
              alt={file.name}
              className="w-full h-full object-contain"
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageError(true)}
            />
            
            {/* Loading overlay */}
            {!imageLoaded && (
              <div className="absolute inset-0 bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
              </div>
            )}

            {/* Annotations overlay */}
            {showAnnotations && imageLoaded && file.results?.objects && (
              <AnnotationOverlay
                objects={file.results.objects}
                imageWidth={file.metadata?.width || 0}
                imageHeight={file.metadata?.height || 0}
              />
            )}
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
            <div className="text-center">
              <PhotoIcon className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600" />
              <p>Failed to load image</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Annotation overlay component
const AnnotationOverlay: React.FC<{
  objects: any[]
  imageWidth: number
  imageHeight: number
}> = ({ objects, imageWidth, imageHeight }) => {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {objects.map((obj, index) => {
        const { bbox, confidence, class: className } = obj
        
        // Calculate relative positions
        const left = (bbox.x / imageWidth) * 100
        const top = (bbox.y / imageHeight) * 100
        const width = (bbox.width / imageWidth) * 100
        const height = (bbox.height / imageHeight) * 100
        
        const confidenceColor = confidence >= 0.8 ? 'border-green-500 bg-green-500/10' :
                               confidence >= 0.6 ? 'border-yellow-500 bg-yellow-500/10' :
                               'border-red-500 bg-red-500/10'

        return (
          <motion.div
            key={index}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            className={`absolute border-2 ${confidenceColor}`}
            style={{
              left: `${left}%`,
              top: `${top}%`,
              width: `${width}%`,
              height: `${height}%`,
            }}
          >
            {/* Label */}
            <div className="absolute -top-6 left-0 bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
              {className} {(confidence * 100).toFixed(1)}%
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

export default FilePreview