import React from 'react'
import { motion } from 'framer-motion'
import { Cog6ToothIcon } from '@heroicons/react/24/outline'
import { useAppStore } from '@/store/appStore'

const SettingsPage: React.FC = () => {
  const { settings, updateSettings, isDarkMode, toggleDarkMode } = useAppStore()

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Settings
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Customize your validation preferences
          </p>
        </div>

        <div className="space-y-6">
          {/* Appearance */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Appearance
              </h2>
            </div>
            <div className="card-content space-y-4">
              <label className="flex items-center justify-between">
                <span className="text-gray-700 dark:text-gray-300">Dark mode</span>
                <input
                  type="checkbox"
                  checked={isDarkMode}
                  onChange={toggleDarkMode}
                  className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                />
              </label>
            </div>
          </div>

          {/* Validation */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Validation
              </h2>
            </div>
            <div className="card-content space-y-4">
              <label className="flex items-center justify-between">
                <span className="text-gray-700 dark:text-gray-300">Show confidence scores</span>
                <input
                  type="checkbox"
                  checked={settings.showConfidenceScores}
                  onChange={(e) => updateSettings({ showConfidenceScores: e.target.checked })}
                  className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                />
              </label>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Default confidence threshold: {(settings.confidenceThreshold * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={settings.confidenceThreshold}
                  onChange={(e) => updateSettings({ confidenceThreshold: parseFloat(e.target.value) })}
                  className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Export */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Export
              </h2>
            </div>
            <div className="card-content space-y-4">
              <label className="flex items-center justify-between">
                <span className="text-gray-700 dark:text-gray-300">Auto-export results</span>
                <input
                  type="checkbox"
                  checked={settings.autoExport}
                  onChange={(e) => updateSettings({ autoExport: e.target.checked })}
                  className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                />
              </label>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Default export format
                </label>
                <select
                  value={settings.exportFormat}
                  onChange={(e) => updateSettings({ exportFormat: e.target.value as any })}
                  className="input"
                >
                  <option value="json">JSON</option>
                  <option value="csv">CSV</option>
                  <option value="pdf">PDF</option>
                </select>
              </div>
            </div>
          </div>

          {/* Video */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Video Player
              </h2>
            </div>
            <div className="card-content space-y-4">
              <label className="flex items-center justify-between">
                <span className="text-gray-700 dark:text-gray-300">Auto-play videos</span>
                <input
                  type="checkbox"
                  checked={settings.autoPlay}
                  onChange={(e) => updateSettings({ autoPlay: e.target.checked })}
                  className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                />
              </label>
            </div>
          </div>

          {/* Notifications */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Notifications
              </h2>
            </div>
            <div className="card-content space-y-4">
              <label className="flex items-center justify-between">
                <span className="text-gray-700 dark:text-gray-300">Enable notifications</span>
                <input
                  type="checkbox"
                  checked={settings.notifications}
                  onChange={(e) => updateSettings({ notifications: e.target.checked })}
                  className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                />
              </label>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default SettingsPage