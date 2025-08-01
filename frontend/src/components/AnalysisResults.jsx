import React from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Eye, 
  Phone, 
  UserCheck,
  ChevronRight,
  ChevronLeft
} from 'lucide-react'
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js'
import { Doughnut, Bar } from 'react-chartjs-2'
import { formatTimestamp, getSeverityColor, getEventTypeIcon, calculateOverallRiskLevel } from '../utils/api'

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement)

const AnalysisResults = ({ results, onNext, onPrev }) => {
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
    duration_seconds
  } = analysisData

  const riskLevel = calculateOverallRiskLevel(results)

  // Chart configurations
  const behaviorChartData = {
    labels: ['Alert', 'Drowsy', 'Distracted', 'Yawning'],
    datasets: [{
      data: [
        behavior_summary.alert_percentage,
        behavior_summary.drowsy_percentage,
        behavior_summary.distracted_percentage,
        behavior_summary.yawning_percentage || 0
      ],
      backgroundColor: [
        '#10b981', // green for alert
        '#f59e0b', // yellow for drowsy
        '#ef4444', // red for distracted
        '#8b5cf6'  // purple for yawning
      ],
      borderWidth: 2,
      borderColor: '#ffffff'
    }]
  }

  const safetyScoresData = {
    labels: ['Overall', 'Fatigue', 'Attention', 'Compliance'],
    datasets: [{
      label: 'Safety Scores',
      data: [
        safety_scores.overall_safety_score,
        safety_scores.fatigue_score,
        safety_scores.attention_score,
        safety_scores.compliance_score
      ],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(139, 92, 246, 0.8)'
      ],
      borderColor: [
        'rgb(59, 130, 246)',
        'rgb(16, 185, 129)',
        'rgb(245, 158, 11)',
        'rgb(139, 92, 246)'
      ],
      borderWidth: 1
    }]
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 20,
          font: {
            size: 12
          }
        }
      }
    }
  }

  const barChartOptions = {
    ...chartOptions,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%'
          }
        }
      }
    }
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header Summary */}
      <div className="card">
        <div className="card-body">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Session Info */}
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Analysis Complete</h2>
              <div className="space-y-2 text-sm text-gray-600">
                <p><span className="font-medium">Session:</span> {session_id}</p>
                <p><span className="font-medium">Driver:</span> {driver_id}</p>
                <p><span className="font-medium">Vehicle:</span> {vehicle_id}</p>
                <p><span className="font-medium">Duration:</span> {Math.round(duration_seconds)}s ({total_frames} frames)</p>
                <p><span className="font-medium">Processing Time:</span> {processing_time.toFixed(1)}s</p>
              </div>
            </div>

            {/* Overall Risk Level */}
            <div className="text-center">
              <div className={`inline-flex items-center px-6 py-3 rounded-full text-lg font-semibold ${riskLevel.color}`}>
                {riskLevel.level === 'Low' && <CheckCircle className="w-5 h-5 mr-2" />}
                {riskLevel.level === 'Medium' && <AlertTriangle className="w-5 h-5 mr-2" />}
                {riskLevel.level === 'High' && <AlertTriangle className="w-5 h-5 mr-2" />}
                {riskLevel.level === 'Critical' && <AlertTriangle className="w-5 h-5 mr-2" />}
                {riskLevel.level} Risk
              </div>
              <p className="text-sm text-gray-600 mt-2">Overall Safety Assessment</p>
            </div>

            {/* Key Metrics */}
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-700">Overall Safety</span>
                <span className="text-lg font-bold text-blue-600">{safety_scores.overall_safety_score.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-700">Events Detected</span>
                <span className="text-lg font-bold text-red-600">{events_detected.length}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-700">Alert Time</span>
                <span className="text-lg font-bold text-green-600">{behavior_summary.alert_percentage.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Behavior Distribution */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold">Behavior Distribution</h3>
            <p className="text-sm text-gray-600">Time spent in different states</p>
          </div>
          <div className="card-body">
            <div className="h-64">
              <Doughnut data={behaviorChartData} options={chartOptions} />
            </div>
          </div>
        </div>

        {/* Safety Scores */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold">Safety Scores</h3>
            <p className="text-sm text-gray-600">Performance across different categories</p>
          </div>
          <div className="card-body">
            <div className="h-64">
              <Bar data={safetyScoresData} options={barChartOptions} />
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="report-grid">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="metric-value text-green-600">{safety_scores.fatigue_score.toFixed(1)}%</div>
              <div className="metric-label">Fatigue Score</div>
            </div>
            <Eye className="w-8 h-8 text-green-500" />
          </div>
          <div className="progress-bar mt-3">
            <div 
              className="progress-fill bg-green-500"
              style={{ width: `${safety_scores.fatigue_score}%` }}
            />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="metric-value text-blue-600">{safety_scores.attention_score.toFixed(1)}%</div>
              <div className="metric-label">Attention Score</div>
            </div>
            <UserCheck className="w-8 h-8 text-blue-500" />
          </div>
          <div className="progress-bar mt-3">
            <div 
              className="progress-fill bg-blue-500"
              style={{ width: `${safety_scores.attention_score}%` }}
            />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="metric-value text-purple-600">{safety_scores.compliance_score.toFixed(1)}%</div>
              <div className="metric-label">Compliance Score</div>
            </div>
            <Phone className="w-8 h-8 text-purple-500" />
          </div>
          <div className="progress-bar mt-3">
            <div 
              className="progress-fill bg-purple-500"
              style={{ width: `${safety_scores.compliance_score}%` }}
            />
          </div>
        </div>
      </div>

      {/* Events Timeline */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold">Detected Events</h3>
          <p className="text-sm text-gray-600">{events_detected.length} events detected during analysis</p>
        </div>
        <div className="card-body">
          {events_detected.length > 0 ? (
            <div className="timeline">
              {events_detected.map((event, index) => (
                <div key={index} className="timeline-item">
                  <div className={`timeline-marker ${
                    event.severity === 'critical' ? 'danger' : 
                    event.severity === 'high' ? 'warning' : 'info'
                  }`} />
                  
                  <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3">
                        <span className="text-2xl">{getEventTypeIcon(event.type, event.description)}</span>
                        <div>
                          <h4 className="font-semibold text-gray-900">{event.description}</h4>
                          <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                            <div className="flex items-center space-x-1">
                              <Clock className="w-4 h-4" />
                              <span>{formatTimestamp(event.timestamp)}</span>
                            </div>
                            <div>Frame: {event.frame_number}</div>
                            <div>Confidence: {(event.confidence * 100).toFixed(0)}%</div>
                          </div>
                        </div>
                      </div>
                      
                      <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getSeverityColor(event.severity)}`}>
                        {event.severity.toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <CheckCircle className="w-12 h-12 mx-auto mb-4 text-green-500" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Concerning Events</h3>
              <p>No significant safety events were detected during the analysis.</p>
            </div>
          )}
        </div>
      </div>

      {/* Recommendations */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold">Safety Recommendations</h3>
          <p className="text-sm text-gray-600">AI-generated recommendations based on analysis</p>
        </div>
        <div className="card-body">
          <div className="space-y-3">
            {recommendations.map((recommendation, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <AlertTriangle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <p className="text-blue-800">{recommendation}</p>
              </div>
            ))}
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
          <span>Back to Upload</span>
        </button>
        
        <button
          onClick={onNext}
          className="btn-primary flex items-center space-x-2"
        >
          <span>View Annotated Video</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

export default AnalysisResults