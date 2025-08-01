import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import Sidebar from './Sidebar'
import Header from './Header'
import { useAppStore } from '@/store/appStore'
import wsService from '@/services/websocket'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { sidebarOpen, isDarkMode } = useAppStore()

  useEffect(() => {
    // Connect WebSocket
    wsService.connect()
    
    return () => {
      wsService.disconnect()
    }
  }, [])

  useEffect(() => {
    // Apply dark mode class to document
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDarkMode])

  return (
    <div className={`flex h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300`}>
      {/* Sidebar */}
      <motion.div
        initial={false}
        animate={{
          width: sidebarOpen ? 256 : 0,
          opacity: sidebarOpen ? 1 : 0,
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="relative flex-shrink-0 overflow-hidden"
      >
        <Sidebar />
      </motion.div>

      {/* Main content area */}
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-auto bg-gray-50 dark:bg-gray-900">
          <div className="min-h-full">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}

export default Layout