import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  HomeIcon,
  CloudArrowUpIcon,
  EyeIcon,
  ChartBarIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Upload', href: '/upload', icon: CloudArrowUpIcon },
  { name: 'Validation', href: '/validation', icon: EyeIcon },
  { name: 'Results', href: '/results', icon: ChartBarIcon },
  { name: 'Reports', href: '/reports', icon: DocumentTextIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
]

const Sidebar: React.FC = () => {
  const location = useLocation()

  return (
    <div className="flex flex-col w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 h-full">
      {/* Logo */}
      <div className="flex items-center px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <CpuChipIcon className="h-8 w-8 text-primary-600 dark:text-primary-400" />
        <div className="ml-3">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">
            AI Validation
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Supervision UI
          </p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href || 
            (item.href !== '/dashboard' && location.pathname.startsWith(item.href))
          
          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive: linkActive }) => `
                group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-all duration-200
                ${isActive || linkActive 
                  ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300' 
                  : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white'
                }
              `}
            >
              {({ isActive: linkActive }) => (
                <>
                  <item.icon
                    className={`mr-3 h-5 w-5 flex-shrink-0 transition-colors ${
                      isActive || linkActive
                        ? 'text-primary-600 dark:text-primary-400'
                        : 'text-gray-400 dark:text-gray-500 group-hover:text-gray-500 dark:group-hover:text-gray-300'
                    }`}
                  />
                  {item.name}
                  {(isActive || linkActive) && (
                    <motion.div
                      layoutId="sidebar-indicator"
                      className="absolute right-2 w-1 h-6 bg-primary-600 dark:bg-primary-400 rounded-full"
                      initial={false}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  )}
                </>
              )}
            </NavLink>
          )
        })}
      </nav>

      {/* Status indicator */}
      <div className="px-4 py-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse-slow"></div>
            <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
              System Online
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Sidebar