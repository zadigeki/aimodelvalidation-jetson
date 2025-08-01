import React, { useState } from 'react'
import { motion } from 'framer-motion'
import UploadDropzone from './UploadDropzone'
import UploadProgress from './UploadProgress'
import FileList from './FileList'
import { useAppStore } from '@/store/appStore'
import { ValidationFile } from '@/types'
import toast from 'react-hot-toast'

const UploadPage: React.FC = () => {
  const { files, addFile, updateFile } = useAppStore()
  const [uploadingFiles, setUploadingFiles] = useState<{ [key: string]: number }>({})

  const handleFilesSelected = async (selectedFiles: File[]) => {
    for (const file of selectedFiles) {
      // Create initial validation file object
      const validationFile: ValidationFile = {
        id: `upload-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: file.name,
        type: file.type.startsWith('video/') ? 'video' : 'image',
        size: file.size,
        url: '',
        status: 'uploading',
        uploadProgress: 0,
        metadata: {
          fileFormat: file.type,
        },
        createdAt: new Date(),
        updatedAt: new Date(),
      }

      addFile(validationFile)

      try {
        // Start upload with progress tracking
        setUploadingFiles(prev => ({ ...prev, [validationFile.id]: 0 }))

        // Simulate upload progress (replace with actual API call)
        await simulateUpload(validationFile.id, file, (progress) => {
          setUploadingFiles(prev => ({ ...prev, [validationFile.id]: progress }))
          updateFile(validationFile.id, { uploadProgress: progress })
        })

        // Update file with completed status
        updateFile(validationFile.id, {
          status: 'processing',
          url: URL.createObjectURL(file), // Temporary URL for preview
          uploadProgress: 100,
        })

        setUploadingFiles(prev => {
          const newState = { ...prev }
          delete newState[validationFile.id]
          return newState
        })

        toast.success(`${file.name} uploaded successfully`)
      } catch (error) {
        updateFile(validationFile.id, { status: 'error' })
        toast.error(`Failed to upload ${file.name}`)
        console.error('Upload error:', error)
      }
    }
  }

  const simulateUpload = (fileId: string, file: File, onProgress: (progress: number) => void): Promise<void> => {
    return new Promise((resolve) => {
      let progress = 0
      const interval = setInterval(() => {
        progress += Math.random() * 15
        if (progress >= 100) {
          progress = 100
          clearInterval(interval)
          onProgress(progress)
          setTimeout(resolve, 500)
        } else {
          onProgress(Math.floor(progress))
        }
      }, 200)
    })
  }

  const uploadingFilesList = files.filter(f => f.status === 'uploading')
  const completedFiles = files.filter(f => f.status !== 'uploading')

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
            Upload Files
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Upload images and videos for AI model validation and analysis
          </p>
        </div>

        {/* Upload area */}
        <div className="mb-8">
          <UploadDropzone
            onFilesSelected={handleFilesSelected}
            maxFiles={10}
            maxFileSize={100 * 1024 * 1024} // 100MB
            acceptedTypes={['image/*', 'video/*']}
            disabled={Object.keys(uploadingFiles).length > 0}
          />
        </div>

        {/* Upload progress */}
        {uploadingFilesList.length > 0 && (
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Uploading Files
            </h2>
            <UploadProgress
              files={uploadingFilesList}
              progress={uploadingFiles}
            />
          </div>
        )}

        {/* File list */}
        {completedFiles.length > 0 && (
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Uploaded Files ({completedFiles.length})
            </h2>
            <FileList files={completedFiles} />
          </div>
        )}

        {/* Empty state */}
        {files.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-center py-12"
          >
            <div className="text-gray-500 dark:text-gray-400">
              <svg
                className="mx-auto h-16 w-16 text-gray-400 dark:text-gray-600 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="text-lg font-medium">No files uploaded yet</p>
              <p className="text-sm">Drag and drop files above or click to browse</p>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  )
}

export default UploadPage