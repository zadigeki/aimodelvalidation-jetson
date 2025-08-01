import React from 'react'
import { motion } from 'framer-motion'

interface ClassFilterProps {
  availableClasses: string[]
  selectedClasses: string[]
  onChange: (classes: string[]) => void
  classDistribution: { [className: string]: number }
}

const ClassFilter: React.FC<ClassFilterProps> = ({
  availableClasses,
  selectedClasses,
  onChange,
  classDistribution,
}) => {
  const toggleClass = (className: string) => {
    if (selectedClasses.includes(className)) {
      onChange(selectedClasses.filter(c => c !== className))
    } else {
      onChange([...selectedClasses, className])
    }
  }

  const toggleAll = () => {
    if (selectedClasses.length === availableClasses.length) {
      onChange([])
    } else {
      onChange(availableClasses)
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Object Classes
        </label>
        <button
          onClick={toggleAll}
          className="text-xs text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
        >
          {selectedClasses.length === availableClasses.length ? 'None' : 'All'}
        </button>
      </div>
      
      <div className="space-y-2 max-h-48 overflow-y-auto custom-scrollbar">
        {availableClasses.map((className, index) => {
          const isSelected = selectedClasses.includes(className)
          const count = classDistribution[className] || 0
          
          return (
            <motion.label
              key={className}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                isSelected 
                  ? 'bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800'
                  : 'hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleClass(className)}
                  className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                  {className}
                </span>
              </div>
              <span className="text-xs bg-gray-100 dark:bg-gray-600 text-gray-600 dark:text-gray-300 px-2 py-1 rounded-full">
                {count}
              </span>
            </motion.label>
          )
        })}
      </div>
    </div>
  )
}

export default ClassFilter