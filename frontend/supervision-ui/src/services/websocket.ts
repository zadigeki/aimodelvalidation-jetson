import { io, Socket } from 'socket.io-client'
import { WebSocketMessage, ValidationProgress } from '@/types'
import toast from 'react-hot-toast'

class WebSocketService {
  private socket: Socket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000

  connect(): void {
    if (this.socket?.connected) {
      return
    }

    const token = localStorage.getItem('auth_token')
    
    this.socket = io('/ws', {
      auth: {
        token,
      },
      transports: ['websocket'],
      timeout: 20000,
    })

    this.setupEventListeners()
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
  }

  private setupEventListeners(): void {
    if (!this.socket) return

    this.socket.on('connect', () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
      toast.success('Connected to validation service')
    })

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason)
      
      if (reason === 'io server disconnect') {
        // Server-initiated disconnect, don't reconnect
        return
      }

      this.handleReconnect()
    })

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error)
      this.handleReconnect()
    })

    this.socket.on('validation_progress', (data: ValidationProgress) => {
      this.handleValidationProgress(data)
    })

    this.socket.on('validation_complete', (data: any) => {
      this.handleValidationComplete(data)
    })

    this.socket.on('validation_error', (data: any) => {
      this.handleValidationError(data)
    })

    this.socket.on('system_notification', (data: any) => {
      this.handleSystemNotification(data)
    })
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      toast.error('Failed to connect to validation service')
      return
    }

    this.reconnectAttempts++
    
    setTimeout(() => {
      console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`)
      this.connect()
    }, this.reconnectDelay * this.reconnectAttempts)
  }

  private handleValidationProgress(data: ValidationProgress): void {
    // Emit custom event for components to listen to
    const event = new CustomEvent('validation_progress', { detail: data })
    window.dispatchEvent(event)
  }

  private handleValidationComplete(data: any): void {
    const event = new CustomEvent('validation_complete', { detail: data })
    window.dispatchEvent(event)
    
    toast.success(`Validation completed for ${data.fileName}`)
  }

  private handleValidationError(data: any): void {
    const event = new CustomEvent('validation_error', { detail: data })
    window.dispatchEvent(event)
    
    toast.error(`Validation failed: ${data.message}`)
  }

  private handleSystemNotification(data: any): void {
    switch (data.type) {
      case 'info':
        toast.success(data.message)
        break
      case 'warning':
        toast(data.message, { icon: '⚠️' })
        break
      case 'error':
        toast.error(data.message)
        break
      default:
        console.log('System notification:', data)
    }
  }

  // Public methods for components to use
  subscribeToFile(fileId: string): void {
    if (this.socket?.connected) {
      this.socket.emit('subscribe_file', { fileId })
    }
  }

  unsubscribeFromFile(fileId: string): void {
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe_file', { fileId })
    }
  }

  sendMessage(type: string, data: any): void {
    if (this.socket?.connected) {
      this.socket.emit(type, data)
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false
  }
}

export const wsService = new WebSocketService()
export default wsService