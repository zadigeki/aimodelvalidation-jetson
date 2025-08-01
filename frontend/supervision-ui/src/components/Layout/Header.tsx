import React from 'react'
import { motion } from 'framer-motion'
import {
  Bars3Icon,
  BellIcon,
  MoonIcon,
  SunIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline'
import { useAppStore } from '@/store/appStore'
import wsService from '@/services/websocket'

const Header: React.FC = () => {
  const { 
    sidebarOpen, 
    setSidebarOpen, 
    isDarkMode, 
    toggleDarkMode,
    user,
    files,
    validationProgress 
  } = useAppStore()

  const activeValidations = Object.keys(validationProgress).length
  const isConnected = wsService.isConnected()

  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side */}
        <div className="flex items-center space-x-4">
          {/* Sidebar toggle */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-md text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <Bars3Icon className="h-5 w-5" />
          </button>

          {/* Connection status */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-400 animate-pulse-slow' : 'bg-red-400'
            }`} />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Active validations indicator */}
          {activeValidations > 0 && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="flex items-center space-x-2 px-3 py-1 bg-primary-100 dark:bg-primary-900/20 rounded-full"
            >
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" />
              <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
                {activeValidations} validation{activeValidations !== 1 ? 's' : ''} active
              </span>
            </motion.div>
          )}
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* Stats */}
          <div className="hidden md:flex items-center space-x-6 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center space-x-1">
              <span className="font-medium text-gray-900 dark:text-white">
                {files.length}
              </span>
              <span>files</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="font-medium text-gray-900 dark:text-white">
                {files.filter(f => f.status === 'completed').length}
              </span>
              <span>processed</span>
            </div>
          </div>

          {/* Dark mode toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-md text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            {isDarkMode ? (
              <SunIcon className="h-5 w-5" />
            ) : (
              <MoonIcon className="h-5 w-5" />
            )}
          </button>

          {/* Notifications */}
          <button className="p-2 rounded-md text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors relative">
            <BellIcon className="h-5 w-5" />
            {activeValidations > 0 && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-primary-500 rounded-full" />
            )}
          </button>

          {/* User menu */}
          <div className="flex items-center space-x-3">
            <div className="hidden md:block text-right">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {user?.name || 'User'}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                {user?.role || 'operator'}
              </p>
            </div>
            <button className="p-1 rounded-full text-gray-400 hover:text-gray-500 dark:hover:text-gray-300">
              <UserCircleIcon className="h-8 w-8" />
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header