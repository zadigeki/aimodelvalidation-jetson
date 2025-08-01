import React from 'react'
import { formatPercentage } from '@/utils/formatters'

interface ConfidenceFilterProps {
  threshold: number
  onChange: (threshold: number) => void
  totalObjects: number
  filteredCount: number
}

const ConfidenceFilter: React.FC<ConfidenceFilterProps> = ({
  threshold,
  onChange,
  totalObjects,
  filteredCount,
}) => {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Confidence Threshold
        </label>
        <span className="text-sm font-medium text-gray-900 dark:text-white">
          {formatPercentage(threshold)}
        </span>
      </div>
      
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={threshold}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
      />
      
      <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
      
      <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
        Showing {filteredCount} of {totalObjects} objects
      </div>
    </div>
  )
}

export default ConfidenceFilter