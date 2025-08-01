import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CloudArrowUpIcon,
  DocumentIcon,
  VideoCameraIcon,
  PhotoIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'
import { formatFileSize } from '@/utils/formatters'

interface UploadDropzoneProps {
  onFilesSelected: (files: File[]) => void
  maxFiles?: number
  maxFileSize?: number
  acceptedTypes?: string[]
  disabled?: boolean
}

const UploadDropzone: React.FC<UploadDropzoneProps> = ({
  onFilesSelected,
  maxFiles = 10,
  maxFileSize = 100 * 1024 * 1024, // 100MB
  acceptedTypes = ['image/*', 'video/*'],
  disabled = false,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [errors, setErrors] = useState<string[]>([])

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setErrors([])
    
    // Handle rejected files
    const newErrors: string[] = []
    rejectedFiles.forEach(({ file, errors }) => {
      errors.forEach((error: any) => {
        switch (error.code) {
          case 'file-too-large':
            newErrors.push(`${file.name} is too large (max ${formatFileSize(maxFileSize)})`)
            break
          case 'file-invalid-type':
            newErrors.push(`${file.name} is not a supported file type`)
            break
          case 'too-many-files':
            newErrors.push(`Too many files selected (max ${maxFiles})`)
            break
          default:
            newErrors.push(`Error with ${file.name}: ${error.message}`)
        }
      })
    })

    if (newErrors.length > 0) {
      setErrors(newErrors)
      return
    }

    // Add accepted files to selected files
    const newFiles = [...selectedFiles, ...acceptedFiles]
    if (newFiles.length > maxFiles) {
      setErrors([`Maximum ${maxFiles} files allowed`])
      return
    }

    setSelectedFiles(newFiles)
  }, [selectedFiles, maxFiles, maxFileSize])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: acceptedTypes.reduce((acc, type) => {
      acc[type] = []
      return acc
    }, {} as Record<string, string[]>),
    maxSize: maxFileSize,
    maxFiles,
    disabled,
    multiple: true,
  })

  const removeFile = (fileToRemove: File) => {
    setSelectedFiles(files => files.filter(file => file !== fileToRemove))
  }

  const uploadFiles = () => {
    if (selectedFiles.length > 0) {
      onFilesSelected(selectedFiles)
      setSelectedFiles([])
    }
  }

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('video/')) {
      return <VideoCameraIcon className="h-8 w-8 text-purple-500" />
    } else if (file.type.startsWith('image/')) {
      return <PhotoIcon className="h-8 w-8 text-blue-500" />
    }
    return <DocumentIcon className="h-8 w-8 text-gray-500" />
  }

  return (
    <div className="space-y-6">
      {/* Dropzone */}
      <motion.div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-8 transition-all duration-200 cursor-pointer
          ${isDragActive && !isDragReject ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 drag-active' : ''}
          ${isDragReject ? 'border-red-500 bg-red-50 dark:bg-red-900/20 drag-reject' : ''}
          ${!isDragActive ? 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500' : ''}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        whileHover={!disabled ? { scale: 1.01 } : {}}
        whileTap={!disabled ? { scale: 0.99 } : {}}
      >
        <input {...getInputProps()} />
        
        <div className="text-center">
          <motion.div
            initial={{ scale: 1 }}
            animate={{ scale: isDragActive ? 1.1 : 1 }}
            transition={{ duration: 0.2 }}
          >
            <CloudArrowUpIcon className="mx-auto h-16 w-16 text-gray-400 dark:text-gray-600 mb-4" />
          </motion.div>
          
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            {isDragActive ? 'Drop files here' : 'Upload files'}
          </h3>
          
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            {isDragActive 
              ? 'Release to upload files'
              : 'Drag and drop files here, or click to browse'
            }
          </p>
          
          <div className="text-sm text-gray-500 dark:text-gray-500 space-y-1">
            <p>Supported formats: Images (JPG, PNG, GIF) and Videos (MP4, AVI, MOV)</p>
            <p>Maximum file size: {formatFileSize(maxFileSize)}</p>
            <p>Maximum files: {maxFiles}</p>
          </div>
        </div>

        {disabled && (
          <div className="absolute inset-0 bg-white/50 dark:bg-gray-900/50 rounded-lg flex items-center justify-center">
            <div className="text-gray-500 dark:text-gray-400 font-medium">
              Upload in progress...
            </div>
          </div>
        )}
      </motion.div>

      {/* Error messages */}
      <AnimatePresence>
        {errors.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4"
          >
            <div className="flex">
              <XMarkIcon className="h-5 w-5 text-red-400 mt-0.5 mr-3 flex-shrink-0" />
              <div>
                <h3 className="text-sm font-medium text-red-800 dark:text-red-200 mb-2">
                  Upload Errors
                </h3>
                <ul className="text-sm text-red-700 dark:text-red-300 space-y-1">
                  {errors.map((error, index) => (
                    <li key={index}>• {error}</li>
                  ))}
                </ul>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Selected files */}
      <AnimatePresence>
        {selectedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                Selected Files ({selectedFiles.length})
              </h3>
              <button
                onClick={uploadFiles}
                className="btn-primary"
              >
                Upload Files
              </button>
            </div>

            <div className="space-y-3 max-h-60 overflow-y-auto custom-scrollbar">
              {selectedFiles.map((file, index) => (
                <motion.div
                  key={`${file.name}-${index}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    {getFileIcon(file)}
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {file.name}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {formatFileSize(file.size)} • {file.type}
                      </p>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => removeFile(file)}
                    className="p-1 rounded-full text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default UploadDropzone