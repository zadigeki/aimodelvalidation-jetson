import React, { useState } from 'react'
import { Download, FileText, BarChart, ChevronLeft, RotateCcw, CheckCircle, AlertTriangle } from 'lucide-react'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'
import { exportToCSV, formatTimestamp, getSeverityColor, calculateOverallRiskLevel } from '../utils/api'

const ReportExport = ({ results, sessionId, onPrev, onReset }) => {
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false)
  const [isGeneratingCSV, setIsGeneratingCSV] = useState(false)
  const [exportStatus, setExportStatus] = useState({ pdf: null, csv: null })

  const {
    session_id,
    driver_id,
    vehicle_id,
    created_at,
    results: analysisData
  } = results

  const {
    safety_scores,
    behavior_summary,
    events_detected,
    recommendations,
    processing_time,
    total_frames,
    duration_seconds,
    ai_analysis
  } = analysisData

  const riskLevel = calculateOverallRiskLevel(results)

  const generatePDFReport = async () => {
    setIsGeneratingPDF(true)
    setExportStatus(prev => ({ ...prev, pdf: null }))

    try {
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageWidth = pdf.internal.pageSize.getWidth()
      const pageHeight = pdf.internal.pageSize.getHeight()
      let yPosition = 20

      // Header
      pdf.setFontSize(20)
      pdf.setTextColor(59, 130, 246) // Blue
      pdf.text('Driver Monitoring Analysis Report', 20, yPosition)
      
      yPosition += 15
      pdf.setFontSize(12)
      pdf.setTextColor(75, 85, 99) // Gray
      pdf.text(`Generated on ${new Date().toLocaleString()}`, 20, yPosition)

      yPosition += 20

      // Session Information
      pdf.setFontSize(16)
      pdf.setTextColor(17, 24, 39) // Dark gray
      pdf.text('Session Information', 20, yPosition)
      yPosition += 10

      pdf.setFontSize(10)
      pdf.setTextColor(75, 85, 99)
      const sessionInfo = [
        `Session ID: ${session_id}`,
        `Driver ID: ${driver_id || 'Not specified'}`,
        `Vehicle ID: ${vehicle_id || 'Not specified'}`,
        `Analysis Date: ${new Date(created_at).toLocaleString()}`,
        `Video Duration: ${Math.round(duration_seconds)}s (${total_frames} frames)`,
        `Processing Time: ${processing_time.toFixed(1)}s`
      ]

      sessionInfo.forEach(info => {
        pdf.text(info, 20, yPosition)
        yPosition += 6
      })

      yPosition += 15

      // Overall Risk Assessment
      pdf.setFontSize(16)
      pdf.setTextColor(17, 24, 39)
      pdf.text('Risk Assessment', 20, yPosition)
      yPosition += 10

      pdf.setFontSize(14)
      pdf.setTextColor(
        riskLevel.level === 'Low' ? [34, 197, 94] :
        riskLevel.level === 'Medium' ? [245, 158, 11] :
        [239, 68, 68]
      )
      pdf.text(`Overall Risk Level: ${riskLevel.level.toUpperCase()}`, 20, yPosition)
      yPosition += 15

      // Safety Scores
      pdf.setFontSize(16)
      pdf.setTextColor(17, 24, 39)
      pdf.text('Safety Scores', 20, yPosition)
      yPosition += 10

      pdf.setFontSize(10)
      pdf.setTextColor(75, 85, 99)
      const scores = [
        `Overall Safety Score: ${safety_scores.overall_safety_score.toFixed(1)}%`,
        `Fatigue Score: ${safety_scores.fatigue_score.toFixed(1)}%`,
        `Attention Score: ${safety_scores.attention_score.toFixed(1)}%`,
        `Compliance Score: ${safety_scores.compliance_score.toFixed(1)}%`
      ]

      scores.forEach(score => {
        pdf.text(score, 20, yPosition)
        yPosition += 6
      })

      yPosition += 15

      // Behavior Summary
      pdf.setFontSize(16)
      pdf.setTextColor(17, 24, 39)
      pdf.text('Behavior Summary', 20, yPosition)
      yPosition += 10

      pdf.setFontSize(10)
      pdf.setTextColor(75, 85, 99)
      const behaviors = [
        `Alert Time: ${behavior_summary.alert_percentage.toFixed(1)}%`,
        `Drowsy Time: ${behavior_summary.drowsy_percentage.toFixed(1)}%`,
        `Distracted Time: ${behavior_summary.distracted_percentage.toFixed(1)}%`,
        `Yawning Time: ${(behavior_summary.yawning_percentage || 0).toFixed(1)}%`
      ]

      behaviors.forEach(behavior => {
        pdf.text(behavior, 20, yPosition)
        yPosition += 6
      })

      yPosition += 15

      // Events
      pdf.setFontSize(16)
      pdf.setTextColor(17, 24, 39)
      pdf.text(`Detected Events (${events_detected.length} total)`, 20, yPosition)
      yPosition += 10

      if (events_detected.length > 0) {
        pdf.setFontSize(9)
        events_detected.slice(0, 15).forEach((event, index) => { // Limit to first 15 events
          if (yPosition > pageHeight - 30) {
            pdf.addPage()
            yPosition = 20
          }

          pdf.setTextColor(
            event.severity === 'critical' ? [239, 68, 68] :
            event.severity === 'high' ? [245, 158, 11] :
            [75, 85, 99]
          )
          
          pdf.text(`${index + 1}. [${formatTimestamp(event.timestamp)}] ${event.description}`, 20, yPosition)
          yPosition += 5
          pdf.setTextColor(107, 114, 128)
          pdf.text(`   Severity: ${event.severity} | Confidence: ${(event.confidence * 100).toFixed(0)}% | Frame: ${event.frame_number}`, 20, yPosition)
          yPosition += 8
        })

        if (events_detected.length > 15) {
          pdf.setTextColor(107, 114, 128)
          pdf.text(`... and ${events_detected.length - 15} more events`, 20, yPosition)
          yPosition += 10
        }
      } else {
        pdf.setTextColor(34, 197, 94)
        pdf.text('No concerning events detected during analysis.', 20, yPosition)
        yPosition += 10
      }

      yPosition += 10

      // Recommendations
      pdf.setFontSize(16)
      pdf.setTextColor(17, 24, 39)
      pdf.text('Safety Recommendations', 20, yPosition)
      yPosition += 10

      pdf.setFontSize(10)
      pdf.setTextColor(75, 85, 99)
      recommendations.forEach((recommendation, index) => {
        if (yPosition > pageHeight - 20) {
          pdf.addPage()
          yPosition = 20
        }
        
        const lines = pdf.splitTextToSize(`${index + 1}. ${recommendation}`, pageWidth - 40)
        lines.forEach(line => {
          pdf.text(line, 20, yPosition)
          yPosition += 6
        })
        yPosition += 3
      })

      // AI Analysis Details
      if (yPosition > pageHeight - 40) {
        pdf.addPage()
        yPosition = 20
      }

      yPosition += 10
      pdf.setFontSize(16)
      pdf.setTextColor(17, 24, 39)
      pdf.text('AI Analysis Details', 20, yPosition)
      yPosition += 10

      pdf.setFontSize(10)
      pdf.setTextColor(75, 85, 99)
      const aiDetails = [
        `MediaPipe Face Detection: ${ai_analysis.mediapipe_face_detection ? 'Enabled' : 'Disabled'}`,
        `YOLO Object Detection: ${ai_analysis.yolo_object_detection ? 'Enabled' : 'Disabled'}`,
        `Eye Aspect Ratio Analysis: ${ai_analysis.eye_aspect_ratio_analysis ? 'Enabled' : 'Disabled'}`,
        `Mouth Aspect Ratio Analysis: ${ai_analysis.mouth_aspect_ratio_analysis ? 'Enabled' : 'Disabled'}`,
        `Yawn Detection: ${ai_analysis.yawn_detection ? 'Enabled' : 'Disabled'}`,
        `Head Pose Estimation: ${ai_analysis.head_pose_estimation ? 'Enabled' : 'Disabled'}`,
        `Real-time Processing: ${ai_analysis.real_time_processing ? 'Enabled' : 'Disabled'}`
      ]

      aiDetails.forEach(detail => {
        pdf.text(detail, 20, yPosition)
        yPosition += 6
      })

      // Footer
      const timestamp = new Date().toISOString()
      pdf.setFontSize(8)
      pdf.setTextColor(156, 163, 175)
      pdf.text(`Driver Monitoring System v2.0.0 - Report generated at ${timestamp}`, 20, pageHeight - 10)

      // Save PDF
      const filename = `driver_monitoring_report_${session_id}_${new Date().toISOString().split('T')[0]}.pdf`
      pdf.save(filename)

      setExportStatus(prev => ({ ...prev, pdf: 'success' }))
    } catch (error) {
      console.error('PDF generation error:', error)
      setExportStatus(prev => ({ ...prev, pdf: 'error' }))
    } finally {
      setIsGeneratingPDF(false)
    }
  }

  const generateCSVReport = async () => {
    setIsGeneratingCSV(true)
    setExportStatus(prev => ({ ...prev, csv: null }))

    try {
      // Main session data
      const sessionData = [{
        session_id,
        driver_id: driver_id || 'Not specified',
        vehicle_id: vehicle_id || 'Not specified',
        analysis_date: new Date(created_at).toISOString(),
        video_duration_seconds: duration_seconds,
        total_frames,
        processing_time_seconds: processing_time,
        overall_safety_score: safety_scores.overall_safety_score,
        fatigue_score: safety_scores.fatigue_score,
        attention_score: safety_scores.attention_score,
        compliance_score: safety_scores.compliance_score,
        alert_percentage: behavior_summary.alert_percentage,
        drowsy_percentage: behavior_summary.drowsy_percentage,
        distracted_percentage: behavior_summary.distracted_percentage,
        yawning_percentage: behavior_summary.yawning_percentage || 0,
        risk_level: riskLevel.level,
        total_events: events_detected.length,
        critical_events: events_detected.filter(e => e.severity === 'critical').length,
        high_severity_events: events_detected.filter(e => e.severity === 'high').length,
        medium_severity_events: events_detected.filter(e => e.severity === 'medium').length,
        low_severity_events: events_detected.filter(e => e.severity === 'low').length
      }]

      // Events data
      const eventsData = events_detected.map((event, index) => ({
        event_number: index + 1,
        session_id,
        timestamp: formatTimestamp(event.timestamp),
        frame_number: event.frame_number,
        event_type: event.type,
        description: event.description,
        severity: event.severity,
        confidence: (event.confidence * 100).toFixed(2) + '%'
      }))

      // Export session summary
      exportToCSV(sessionData, `driver_monitoring_summary_${session_id}_${new Date().toISOString().split('T')[0]}`)
      
      // Export events if any
      if (eventsData.length > 0) {
        setTimeout(() => {
          exportToCSV(eventsData, `driver_monitoring_events_${session_id}_${new Date().toISOString().split('T')[0]}`)
        }, 1000)
      }

      setExportStatus(prev => ({ ...prev, csv: 'success' }))
    } catch (error) {
      console.error('CSV generation error:', error)
      setExportStatus(prev => ({ ...prev, csv: 'error' }))
    } finally {
      setIsGeneratingCSV(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-xl font-semibold">Export Analysis Reports</h2>
          <p className="text-gray-600">Download comprehensive reports in multiple formats</p>
        </div>
      </div>

      {/* Export Options */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* PDF Report */}
        <div className="card">
          <div className="card-body">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                <FileText className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">PDF Report</h3>
                <p className="text-sm text-gray-600">Comprehensive analysis report</p>
              </div>
            </div>
            
            <div className="space-y-3 mb-6">
              <div className="text-sm text-gray-600">
                <h4 className="font-medium text-gray-900 mb-2">Includes:</h4>
                <ul className="space-y-1">
                  <li>• Session and driver information</li>
                  <li>• Safety scores and risk assessment</li>
                  <li>• Detailed event timeline</li>
                  <li>• AI analysis specifications</li>
                  <li>• Safety recommendations</li>
                </ul>
              </div>
            </div>

            {exportStatus.pdf && (
              <div className={`mb-4 p-3 rounded-lg flex items-center space-x-2 ${
                exportStatus.pdf === 'success' 
                  ? 'bg-green-50 text-green-700 border border-green-200' 
                  : 'bg-red-50 text-red-700 border border-red-200'
              }`}>
                {exportStatus.pdf === 'success' ? (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-sm">PDF report downloaded successfully!</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4" />
                    <span className="text-sm">Failed to generate PDF report.</span>
                  </>
                )}
              </div>
            )}

            <button
              onClick={generatePDFReport}
              disabled={isGeneratingPDF}
              className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGeneratingPDF ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  Generating PDF...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Download PDF Report
                </>
              )}
            </button>
          </div>
        </div>

        {/* CSV Data Export */}
        <div className="card">
          <div className="card-body">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <BarChart className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">CSV Data Export</h3>
                <p className="text-sm text-gray-600">Raw data for analysis tools</p>
              </div>
            </div>
            
            <div className="space-y-3 mb-6">
              <div className="text-sm text-gray-600">
                <h4 className="font-medium text-gray-900 mb-2">Includes:</h4>
                <ul className="space-y-1">
                  <li>• Session summary data</li>
                  <li>• Individual event records</li>
                  <li>• Behavioral metrics</li>
                  <li>• Timestamps and confidence scores</li>
                  <li>• Compatible with Excel/Google Sheets</li>
                </ul>
              </div>
            </div>

            {exportStatus.csv && (
              <div className={`mb-4 p-3 rounded-lg flex items-center space-x-2 ${
                exportStatus.csv === 'success' 
                  ? 'bg-green-50 text-green-700 border border-green-200' 
                  : 'bg-red-50 text-red-700 border border-red-200'
              }`}>
                {exportStatus.csv === 'success' ? (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-sm">CSV files downloaded successfully!</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4" />
                    <span className="text-sm">Failed to generate CSV files.</span>
                  </>
                )}
              </div>
            )}

            <button
              onClick={generateCSVReport}
              disabled={isGeneratingCSV}
              className="w-full btn-success disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGeneratingCSV ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  Generating CSV...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Download CSV Data
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Summary Information */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold">Export Summary</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{Math.round(duration_seconds)}s</div>
              <div className="text-sm text-blue-800">Video Duration</div>
            </div>
            <div className="bg-red-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-red-600">{events_detected.length}</div>
              <div className="text-sm text-red-800">Events Analyzed</div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{safety_scores.overall_safety_score.toFixed(0)}%</div>
              <div className="text-sm text-green-800">Safety Score</div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between items-center">
        <button
          onClick={onPrev}
          className="btn-secondary flex items-center space-x-2"
        >
          <ChevronLeft className="w-4 h-4" />
          <span>Back to Video</span>
        </button>
        
        <button
          onClick={onReset}
          className="btn-primary flex items-center space-x-2"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Analyze New Video</span>
        </button>
      </div>
    </div>
  )
}

export default ReportExport