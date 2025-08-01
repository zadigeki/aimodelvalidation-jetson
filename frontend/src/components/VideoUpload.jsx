import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Film, Settings, AlertCircle, CheckCircle, X, Eye } from 'lucide-react'

const VideoUpload = ({ onUpload, isUploading }) => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [config, setConfig] = useState({
    driver_id: '',
    vehicle_id: '',
    fatigue_sensitivity: 0.7,
    distraction_sensitivity: 0.8,
    check_seatbelt: true,
    check_phone_usage: true
  })
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [dragActive, setDragActive] = useState(false)

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      alert('Please select a valid video file (MP4, AVI, MOV, MKV)')
      return
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      setSelectedFile(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024, // 500MB limit
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    disabled: isUploading
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    if (selectedFile && !isUploading) {
      onUpload(selectedFile, config)
    }
  }

  const removeFile = () => {
    setSelectedFile(null)
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDuration = (file) => {
    return new Promise((resolve) => {
      const video = document.createElement('video')
      video.preload = 'metadata'
      video.onloadedmetadata = () => {
        const duration = video.duration
        const minutes = Math.floor(duration / 60)
        const seconds = Math.floor(duration % 60)
        resolve(`${minutes}:${seconds.toString().padStart(2, '0')}`)
      }
      video.src = URL.createObjectURL(file)
    })
  }

  const [videoDuration, setVideoDuration] = useState(null)

  React.useEffect(() => {
    if (selectedFile) {
      formatDuration(selectedFile).then(setVideoDuration)
    }
  }, [selectedFile])

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Upload Area */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-xl font-semibold flex items-center space-x-2">
            <Upload className="w-5 h-5" />
            <span>Upload Driver Monitoring Video</span>
          </h2>
          <p className="text-gray-600 mt-1">
            Select in-cab driver footage for AI-powered behavior analysis
          </p>
        </div>
        
        <div className="card-body">
          {!selectedFile ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200 cursor-pointer
                ${isDragActive || dragActive
                  ? 'border-blue-400 bg-blue-50' 
                  : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                }
                ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              <input {...getInputProps()} />
              
              <div className="space-y-4">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto">
                  <Film className="w-8 h-8 text-white" />
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {isDragActive ? 'Drop video here' : 'Choose driver monitoring video'}
                  </h3>
                  <p className="text-gray-600 mb-4">
                    Drag and drop your video file here, or click to browse
                  </p>
                  
                  <div className="text-sm text-gray-500 space-y-1">
                    <p>Supported formats: MP4, AVI, MOV, MKV, WebM</p>
                    <p>Maximum file size: 500MB</p>
                    <p>Recommended: 30+ seconds of driver footage</p>
                  </div>
                </div>
                
                <button 
                  type="button"
                  className="btn-primary"
                  disabled={isUploading}
                >
                  Select Video File
                </button>
              </div>
            </div>
          ) : (
            /* Selected File Preview */
            <div className="bg-gray-50 rounded-lg p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                    <CheckCircle className="w-6 h-6 text-green-600" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-gray-900 truncate">
                      {selectedFile.name}
                    </h3>
                    <div className="mt-2 space-y-1 text-sm text-gray-600">
                      <p>Size: {formatFileSize(selectedFile.size)}</p>
                      <p>Type: {selectedFile.type}</p>
                      {videoDuration && <p>Duration: {videoDuration}</p>}
                      <p>Modified: {new Date(selectedFile.lastModified).toLocaleDateString()}</p>
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={removeFile}
                  className="text-gray-400 hover:text-gray-600 p-1"
                  disabled={isUploading}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Configuration */}
      {selectedFile && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Settings className="w-5 h-5" />
                <span className="text-lg font-semibold">Analysis Configuration</span>
              </div>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-sm text-blue-600 hover:text-blue-700"
              >
                {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
              </button>
            </div>
          </div>
          
          <div className="card-body">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Basic Settings */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Driver ID (Optional)
                  </label>
                  <input
                    type="text"
                    value={config.driver_id}
                    onChange={(e) => setConfig({ ...config, driver_id: e.target.value })}
                    placeholder="e.g., DRIVER_123"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Vehicle ID (Optional)
                  </label>
                  <input
                    type="text"
                    value={config.vehicle_id}
                    onChange={(e) => setConfig({ ...config, vehicle_id: e.target.value })}
                    placeholder="e.g., VEHICLE_ABC"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Advanced Settings */}
              {showAdvanced && (
                <div className="space-y-6 border-t border-gray-200 pt-6">
                  <h3 className="text-lg font-medium text-gray-900">Detection Sensitivity</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Fatigue Sensitivity: {config.fatigue_sensitivity}
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.1"
                        value={config.fatigue_sensitivity}
                        onChange={(e) => setConfig({ ...config, fatigue_sensitivity: parseFloat(e.target.value) })}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Less Sensitive</span>
                        <span>More Sensitive</span>
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Distraction Sensitivity: {config.distraction_sensitivity}
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.1"
                        value={config.distraction_sensitivity}
                        onChange={(e) => setConfig({ ...config, distraction_sensitivity: parseFloat(e.target.value) })}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Less Sensitive</span>
                        <span>More Sensitive</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h3 className="text-lg font-medium text-gray-900">Detection Features</h3>
                    
                    <div className="space-y-3">
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          checked={config.check_seatbelt}
                          onChange={(e) => setConfig({ ...config, check_seatbelt: e.target.checked })}
                          className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-700">Seatbelt Compliance Detection</span>
                      </label>
                      
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          checked={config.check_phone_usage}
                          onChange={(e) => setConfig({ ...config, check_phone_usage: e.target.checked })}
                          className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-700">Phone Usage Detection</span>
                      </label>
                    </div>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <div className="flex justify-center pt-6">
                <button
                  type="submit"
                  disabled={isUploading}
                  className="btn-primary text-lg px-8 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isUploading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Analyzing Video...
                    </>
                  ) : (
                    <>
                      <Eye className="w-5 h-5 mr-2" />
                      Start AI Analysis
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Help Section */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-blue-900 mb-2">Tips for Best Results</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Ensure good lighting and clear view of the driver's face</li>
              <li>• Include at least 30 seconds of footage for comprehensive analysis</li>
              <li>• Video should show the driver's eyes and mouth clearly</li>
              <li>• Minimize camera shake for better face detection accuracy</li>
              <li>• Higher resolution videos (720p+) provide better analysis results</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default VideoUpload