import React from 'react'
import { ValidationResults, DetectedObject } from '@/types'
import { formatPercentage, formatNumber } from '@/utils/formatters'

interface ResultsStatsProps {
  results: ValidationResults
  filteredObjects: DetectedObject[]
}

const ResultsStats: React.FC<ResultsStatsProps> = ({ results, filteredObjects }) => {
  const classStats = Object.entries(results.summary.classDistribution).map(([className, count]) => ({
    name: className,
    total: count,
    filtered: filteredObjects.filter(obj => obj.class === className).length,
    percentage: (count / results.summary.totalObjects) * 100,
  })).sort((a, b) => b.total - a.total)

  const confidenceStats = {
    high: filteredObjects.filter(obj => obj.confidence >= 0.8).length,
    medium: filteredObjects.filter(obj => obj.confidence >= 0.6 && obj.confidence < 0.8).length,
    low: filteredObjects.filter(obj => obj.confidence < 0.6).length,
  }

  return (
    <div className="space-y-6">
      {/* Class distribution */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
          Class Distribution
        </h4>
        <div className="space-y-2">
          {classStats.map((stat) => (
            <div key={stat.name} className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-primary-500 rounded-full" />
                <span className="text-sm text-gray-700 dark:text-gray-300 capitalize">
                  {stat.name}
                </span>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  {stat.filtered}/{stat.total}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {stat.percentage.toFixed(1)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Confidence distribution */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
          Confidence Distribution
        </h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full" />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                High (≥80%)
              </span>
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {confidenceStats.high}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-yellow-500 rounded-full" />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Medium (60-79%)
              </span>
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {confidenceStats.medium}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full" />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Low (&lt;60%)
              </span>
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {confidenceStats.low}
            </span>
          </div>
        </div>
      </div>

      {/* Summary stats */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
        <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
          Summary
        </h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Processing Time</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {results.summary.processingTime?.toFixed(1)}s
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Quality Score</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {formatPercentage(results.summary.qualityScore)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Avg Confidence</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {formatPercentage(results.summary.averageConfidence)}
            </span>
          </div>
          {results.summary.frameCount && (
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Frame Count</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {formatNumber(results.summary.frameCount)}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ResultsStats