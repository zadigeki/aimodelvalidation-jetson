import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { ValidationFile, ValidationResults, ValidationReport, APIResponse, ExportOptions } from '@/types'
import toast from 'react-hot-toast'

class APIService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token')
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => Promise.reject(error)
    )

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          localStorage.removeItem('auth_token')
          window.location.href = '/login'
        } else if (error.response?.status >= 500) {
          toast.error('Server error occurred. Please try again.')
        }
        return Promise.reject(error)
      }
    )
  }

  // File upload methods
  async uploadFile(file: File, onProgress?: (progress: number) => void): Promise<ValidationFile> {
    const formData = new FormData()
    formData.append('file', file)

    const config: AxiosRequestConfig = {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    }

    const response = await this.client.post<APIResponse<ValidationFile>>('/files/upload', formData, config)
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Upload failed')
    }

    return response.data.data
  }

  async uploadMultipleFiles(
    files: File[], 
    onProgress?: (fileIndex: number, progress: number) => void
  ): Promise<ValidationFile[]> {
    const uploadPromises = files.map((file, index) => 
      this.uploadFile(file, (progress) => onProgress?.(index, progress))
    )
    
    return Promise.all(uploadPromises)
  }

  // File management
  async getFiles(): Promise<ValidationFile[]> {
    const response = await this.client.get<APIResponse<ValidationFile[]>>('/files')
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch files')
    }

    return response.data.data
  }

  async getFile(fileId: string): Promise<ValidationFile> {
    const response = await this.client.get<APIResponse<ValidationFile>>(`/files/${fileId}`)
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch file')
    }

    return response.data.data
  }

  async deleteFile(fileId: string): Promise<void> {
    const response = await this.client.delete<APIResponse<void>>(`/files/${fileId}`)
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to delete file')
    }
  }

  // Validation methods
  async startValidation(fileId: string): Promise<void> {
    const response = await this.client.post<APIResponse<void>>(`/files/${fileId}/validate`)
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to start validation')
    }
  }

  async getValidationResults(fileId: string): Promise<ValidationResults> {
    const response = await this.client.get<APIResponse<ValidationResults>>(`/files/${fileId}/results`)
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch validation results')
    }

    return response.data.data
  }

  async cancelValidation(fileId: string): Promise<void> {
    const response = await this.client.post<APIResponse<void>>(`/files/${fileId}/cancel`)
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to cancel validation')
    }
  }

  // Annotation methods
  async updateAnnotation(fileId: string, annotationId: string, updates: any): Promise<void> {
    const response = await this.client.patch<APIResponse<void>>(
      `/files/${fileId}/annotations/${annotationId}`,
      updates
    )
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to update annotation')
    }
  }

  async deleteAnnotation(fileId: string, annotationId: string): Promise<void> {
    const response = await this.client.delete<APIResponse<void>>(
      `/files/${fileId}/annotations/${annotationId}`
    )
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to delete annotation')
    }
  }

  // Report methods
  async getReports(): Promise<ValidationReport[]> {
    const response = await this.client.get<APIResponse<ValidationReport[]>>('/reports')
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch reports')
    }

    return response.data.data
  }

  async generateReport(fileIds: string[], options: ExportOptions): Promise<ValidationReport> {
    const response = await this.client.post<APIResponse<ValidationReport>>('/reports/generate', {
      fileIds,
      options,
    })
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to generate report')
    }

    return response.data.data
  }

  async downloadReport(reportId: string, format: string): Promise<Blob> {
    const response = await this.client.get(`/reports/${reportId}/download`, {
      params: { format },
      responseType: 'blob',
    })

    return response.data
  }

  async deleteReport(reportId: string): Promise<void> {
    const response = await this.client.delete<APIResponse<void>>(`/reports/${reportId}`)
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to delete report')
    }
  }

  // System methods
  async getSystemStatus(): Promise<any> {
    const response = await this.client.get<APIResponse<any>>('/system/status')
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch system status')
    }

    return response.data.data
  }

  async getSystemMetrics(): Promise<any> {
    const response = await this.client.get<APIResponse<any>>('/system/metrics')
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch system metrics')
    }

    return response.data.data
  }
}

export const apiService = new APIService()
export default apiService