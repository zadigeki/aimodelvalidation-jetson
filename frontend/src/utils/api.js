import axios from 'axios'

// API Configuration
const API_BASE_URL = 'http://localhost:8002'
const API_TIMEOUT = 300000 // 5 minutes for video processing

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})

// API Functions
export const analyzeVideo = async (videoFile, config) => {
  try {
    const formData = new FormData()
    formData.append('video', videoFile)
    formData.append('driver_id', config.driver_id || 'UNKNOWN')
    formData.append('vehicle_id', config.vehicle_id || 'UNKNOWN')
    formData.append('fatigue_sensitivity', config.fatigue_sensitivity.toString())
    formData.append('distraction_sensitivity', config.distraction_sensitivity.toString())
    formData.append('check_seatbelt', config.check_seatbelt.toString())
    formData.append('check_phone_usage', config.check_phone_usage.toString())

    const response = await api.post('/api/driver-monitoring/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        )
        console.log(`Upload Progress: ${percentCompleted}%`)
      },
    })

    return response.data
  } catch (error) {
    console.error('Video analysis error:', error)
    throw new Error(
      error.response?.data?.detail || 
      error.message || 
      'Failed to analyze video'
    )
  }
}

export const getAnalysisResults = async (sessionId) => {
  try {
    const response = await api.get(`/api/driver-monitoring/results/${sessionId}`, {
      headers: {
        'Content-Type': 'application/json',
      },
    })
    return response.data
  } catch (error) {
    console.error('Get results error:', error)
    throw new Error(
      error.response?.data?.detail || 
      error.message || 
      'Failed to get analysis results'
    )
  }
}

export const getAnalysisStatus = async (sessionId) => {
  try {
    const response = await api.get(`/api/driver-monitoring/status/${sessionId}`, {
      headers: {
        'Content-Type': 'application/json',
      },
    })
    return response.data
  } catch (error) {
    console.error('Get status error:', error)
    throw new Error(
      error.response?.data?.detail || 
      error.message || 
      'Failed to get analysis status'
    )
  }
}

export const checkHealthStatus = async () => {
  try {
    const response = await api.get('/health', {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 5000, // 5 second timeout for health check
    })
    return response.data
  } catch (error) {
    console.error('Health check error:', error)
    throw new Error(
      error.response?.data?.detail || 
      error.message || 
      'Failed to check system health'
    )
  }
}

// Export utility for generating annotated video
export const generateAnnotatedVideoBlobUrl = (originalVideoFile, analysisResults) => {
  // For now, return the original video
  // In a full implementation, this would process the video with annotations
  return URL.createObjectURL(originalVideoFile)
}

// Utility functions for data processing
export const formatDuration = (seconds) => {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`
}

export const formatTimestamp = (timestamp) => {
  if (typeof timestamp === 'string' && timestamp.includes(':')) {
    return timestamp
  }
  return formatDuration(timestamp)
}

export const getSeverityColor = (severity) => {
  switch (severity?.toLowerCase()) {
    case 'critical':
      return 'text-red-600 bg-red-50 border-red-200'
    case 'high':
      return 'text-orange-600 bg-orange-50 border-orange-200'
    case 'medium':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    case 'low':
      return 'text-blue-600 bg-blue-50 border-blue-200'
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200'
  }
}

export const getEventTypeIcon = (type, description) => {
  const desc = description?.toLowerCase() || ''
  
  if (type === 'fatigue' || desc.includes('yawn') || desc.includes('eye') || desc.includes('drowsy')) {
    return '😴'
  }
  if (type === 'distraction' || desc.includes('phone') || desc.includes('looking away')) {
    return '📱'
  }
  if (desc.includes('seatbelt')) {
    return '🔒'
  }
  return '⚠️'
}

export const calculateOverallRiskLevel = (results) => {
  const safetyScore = results.results?.safety_scores?.overall_safety_score || 0
  
  if (safetyScore >= 85) return { level: 'Low', color: 'text-green-600 bg-green-50' }
  if (safetyScore >= 70) return { level: 'Medium', color: 'text-yellow-600 bg-yellow-50' }
  if (safetyScore >= 50) return { level: 'High', color: 'text-orange-600 bg-orange-50' }
  return { level: 'Critical', color: 'text-red-600 bg-red-50' }
}

// Data export utilities
export const exportToCSV = (data, filename) => {
  const headers = Object.keys(data[0])
  const csvContent = [
    headers.join(','),
    ...data.map(row => headers.map(header => 
      JSON.stringify(row[header] || '')
    ).join(','))
  ].join('\n')
  
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)
  link.setAttribute('href', url)
  link.setAttribute('download', `${filename}.csv`)
  link.style.visibility = 'hidden'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

export default api