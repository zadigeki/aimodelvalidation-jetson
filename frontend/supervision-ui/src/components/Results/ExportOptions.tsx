import React, { useState } from 'react'
import { motion } from 'framer-motion'
import {
  DocumentArrowDownIcon,
  CheckIcon,
} from '@heroicons/react/24/outline'
import { ValidationFile, DetectedObject, ExportOptions as ExportOptionsType } from '@/types'
import toast from 'react-hot-toast'

interface ExportOptionsProps {
  file: ValidationFile
  filteredObjects: DetectedObject[]
}

const ExportOptions: React.FC<ExportOptionsProps> = ({ file, filteredObjects }) => {
  const [isExporting, setIsExporting] = useState(false)
  const [exportOptions, setExportOptions] = useState<ExportOptionsType>({
    format: 'json',
    includeImages: true,
    includeAnnotations: true,
    includeMetadata: true,
    confidenceThreshold: 0.5,
  })

  const handleExport = async (format: ExportOptionsType['format']) => {
    setIsExporting(true)
    
    try {
      const exportData = generateExportData(format)
      await downloadFile(exportData, `${file.name.split('.')[0]}_results.${format}`, format)
      toast.success(`Results exported as ${format.toUpperCase()}`)
    } catch (error) {
      toast.error('Failed to export results')
      console.error('Export error:', error)
    } finally {
      setIsExporting(false)
    }
  }

  const generateExportData = (format: ExportOptionsType['format']) => {
    const baseData = {
      file: {
        name: file.name,
        type: file.type,
        size: file.size,
        ...(exportOptions.includeMetadata && { metadata: file.metadata }),
      },
      results: {
        summary: file.results?.summary,
        objects: filteredObjects.map(obj => ({
          id: obj.id,
          class: obj.class,
          confidence: obj.confidence,
          bbox: obj.bbox,
          ...(obj.frame !== undefined && { frame: obj.frame }),
          ...(obj.timestamp !== undefined && { timestamp: obj.timestamp }),
        })),
        ...(exportOptions.includeAnnotations && { annotations: file.results?.annotations }),
      },
      exportOptions,
      exportedAt: new Date().toISOString(),
    }

    switch (format) {
      case 'json':
        return JSON.stringify(baseData, null, 2)
      
      case 'csv':
        return generateCSV(baseData)
      
      case 'xml':
        return generateXML(baseData)
      
      default:
        return JSON.stringify(baseData, null, 2)
    }
  }

  const generateCSV = (data: any) => {
    const headers = ['ID', 'Class', 'Confidence', 'X', 'Y', 'Width', 'Height', 'Frame', 'Timestamp']
    const rows = data.results.objects.map((obj: any) => [
      obj.id || '',
      obj.class,
      obj.confidence.toFixed(3),
      obj.bbox.x,
      obj.bbox.y,
      obj.bbox.width,
      obj.bbox.height,
      obj.frame || '',
      obj.timestamp || '',
    ])

    return [headers, ...rows].map(row => row.join(',')).join('\n')
  }

  const generateXML = (data: any) => {
    const objects = data.results.objects.map((obj: any) => `
      <object>
        <id>${obj.id || ''}</id>
        <class>${obj.class}</class>
        <confidence>${obj.confidence}</confidence>
        <bbox>
          <x>${obj.bbox.x}</x>
          <y>${obj.bbox.y}</y>
          <width>${obj.bbox.width}</width>
          <height>${obj.bbox.height}</height>
        </bbox>
        ${obj.frame !== undefined ? `<frame>${obj.frame}</frame>` : ''}
        ${obj.timestamp !== undefined ? `<timestamp>${obj.timestamp}</timestamp>` : ''}
      </object>`
    ).join('')

    return `<?xml version="1.0" encoding="UTF-8"?>
<validation_results>
  <file>
    <name>${data.file.name}</name>
    <type>${data.file.type}</type>
    <size>${data.file.size}</size>
  </file>
  <results>
    <summary>
      <total_objects>${data.results.summary.totalObjects}</total_objects>
      <average_confidence>${data.results.summary.averageConfidence}</average_confidence>
      <quality_score>${data.results.summary.qualityScore}</quality_score>
    </summary>
    <objects>${objects}
    </objects>
  </results>
  <exported_at>${data.exportedAt}</exported_at>
</validation_results>`
  }

  const downloadFile = async (content: string, filename: string, format: string) => {
    const mimeTypes = {
      json: 'application/json',
      csv: 'text/csv',
      xml: 'application/xml',
      pdf: 'application/pdf',
    }

    const blob = new Blob([content], { type: mimeTypes[format as keyof typeof mimeTypes] })
    const url = URL.createObjectURL(blob)
    
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const formats = [
    { key: 'json', label: 'JSON', description: 'Complete data with metadata' },
    { key: 'csv', label: 'CSV', description: 'Object data in spreadsheet format' },
    { key: 'xml', label: 'XML', description: 'Structured markup format' },
    { key: 'pdf', label: 'PDF', description: 'Report with visualizations (Coming Soon)', disabled: true },
  ]

  return (
    <div className="space-y-4">
      {/* Export formats */}
      <div className="space-y-2">
        {formats.map((format) => (
          <motion.button
            key={format.key}
            whileHover={!format.disabled ? { scale: 1.02 } : {}}
            whileTap={!format.disabled ? { scale: 0.98 } : {}}
            onClick={() => !format.disabled && handleExport(format.key as any)}
            disabled={format.disabled || isExporting}
            className={`w-full p-3 text-left rounded-lg border transition-all ${
              format.disabled
                ? 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 cursor-not-allowed opacity-50'
                : 'border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20'
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium text-gray-900 dark:text-white">
                  {format.label}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  {format.description}
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {isExporting ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500" />
                ) : (
                  <DocumentArrowDownIcon className="h-5 w-5 text-gray-400" />
                )}
              </div>
            </div>
          </motion.button>
        ))}
      </div>

      {/* Export options */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
        <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
          Export Options
        </h4>
        <div className="space-y-2">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={exportOptions.includeMetadata}
              onChange={(e) => setExportOptions(prev => ({
                ...prev,
                includeMetadata: e.target.checked
              }))}
              className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Include file metadata</span>
          </label>
          
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={exportOptions.includeAnnotations}
              onChange={(e) => setExportOptions(prev => ({
                ...prev,
                includeAnnotations: e.target.checked
              }))}
              className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Include annotations</span>
          </label>
          
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={exportOptions.includeImages}
              onChange={(e) => setExportOptions(prev => ({
                ...prev,
                includeImages: e.target.checked
              }))}
              className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Include image references</span>
          </label>
        </div>
      </div>

      {/* Quick stats */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
        <div className="text-sm text-gray-600 dark:text-gray-400">
          <div className="flex justify-between">
            <span>Objects to export:</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {filteredObjects.length}
            </span>
          </div>
          <div className="flex justify-between mt-1">
            <span>File size:</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {(file.size / 1024 / 1024).toFixed(1)} MB
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ExportOptions