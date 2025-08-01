import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { motion } from 'framer-motion'

import Layout from './components/Layout/Layout'
import Dashboard from './components/Dashboard/Dashboard'
import UploadPage from './components/Upload/UploadPage'
import ValidationPage from './components/Validation/ValidationPage'
import ResultsPage from './components/Results/ResultsPage'
import ReportsPage from './components/Reports/ReportsPage'
import SettingsPage from './components/Settings/SettingsPage'

import { useAppStore } from './store/appStore'

const App: React.FC = () => {
  const { isDarkMode } = useAppStore()

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      isDarkMode ? 'dark bg-gray-900' : 'bg-gray-50'
    }`}>
      <Layout>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex-1"
        >
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/validation" element={<ValidationPage />} />
            <Route path="/validation/:fileId" element={<ValidationPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/results/:fileId" element={<ResultsPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </motion.div>
      </Layout>
    </div>
  )
}

export default App