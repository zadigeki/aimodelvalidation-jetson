import React from 'react'
import { motion } from 'framer-motion'
import { DocumentTextIcon } from '@heroicons/react/24/outline'

const ReportsPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center py-12"
      >
        <DocumentTextIcon className="h-16 w-16 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Reports
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Advanced reporting features coming soon
        </p>
      </motion.div>
    </div>
  )
}

export default ReportsPage