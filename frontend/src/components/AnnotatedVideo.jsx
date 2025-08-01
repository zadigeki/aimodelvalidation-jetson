import React, { useState, useRef, useEffect } from 'react'
import { Play, Pause, SkipBack, SkipForward, Volume2, ChevronLeft, ChevronRight, Download } from 'lucide-react'
import { formatTimestamp, getEventTypeIcon, getSeverityColor } from '../utils/api'

const AnnotatedVideo = ({ originalVideo, analysisResults, onNext, onPrev }) => {
  const videoRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [showControls, setShowControls] = useState(true)
  const [activeEvents, setActiveEvents] = useState([])
  
  const { events_detected, duration_seconds } = analysisResults.results
  
  // Create video URL from uploaded file
  const videoUrl = React.useMemo(() => {
    return URL.createObjectURL(originalVideo)
  }, [originalVideo])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadedMetadata = () => {
      setDuration(video.duration)
    }

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime)
      
      // Find active events at current time
      const currentEvents = events_detected.filter(event => {
        // Convert timestamp to seconds if it's in MM:SS format
        let eventTime = 0
        if (typeof event.timestamp === 'string' && event.timestamp.includes(':')) {
          const parts = event.timestamp.split(':')
          eventTime = parseInt(parts[0]) * 60 + parseInt(parts[1])
        } else {
          eventTime = parseFloat(event.timestamp) || 0
        }
        
        // Show event for 3 seconds around the timestamp
        return Math.abs(video.currentTime - eventTime) <= 3
      })
      
      setActiveEvents(currentEvents)
    }

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      URL.revokeObjectURL(videoUrl)
    }
  }, [videoUrl, events_detected])

  const togglePlay = () => {
    const video = videoRef.current
    if (video.paused) {
      video.play()
    } else {
      video.pause()
    }
  }

  const handleSeek = (e) => {
    const video = videoRef.current
    const rect = e.currentTarget.getBoundingClientRect()
    const percentage = (e.clientX - rect.left) / rect.width
    video.currentTime = percentage * duration
  }

  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value)
    setVolume(newVolume)
    videoRef.current.volume = newVolume
  }

  const skipToEvent = (eventIndex) => {
    const event = events_detected[eventIndex]
    if (!event || !videoRef.current) return
    
    let eventTime = 0
    if (typeof event.timestamp === 'string' && event.timestamp.includes(':')) {
      const parts = event.timestamp.split(':')
      eventTime = parseInt(parts[0]) * 60 + parseInt(parts[1])
    } else {
      eventTime = parseFloat(event.timestamp) || 0
    }
    
    videoRef.current.currentTime = eventTime
  }

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  const downloadVideo = () => {
    const link = document.createElement('a')
    link.href = videoUrl
    link.download = `annotated_${originalVideo.name}`
    link.click()
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-xl font-semibold">Annotated Video Playback</h2>
          <p className="text-gray-600">Watch your video with real-time event annotations</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Video Player */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="card-body p-0">
              <div 
                className="relative bg-black rounded-t-xl overflow-hidden group"
                onMouseEnter={() => setShowControls(true)}
                onMouseLeave={() => setShowControls(false)}
              >
                <video
                  ref={videoRef}
                  src={videoUrl}
                  className="w-full h-auto max-h-96 object-contain"
                  poster=""
                />
                
                {/* Active Event Overlays */}
                {activeEvents.map((event, index) => (
                  <div
                    key={index}
                    className="absolute top-4 right-4 bg-black bg-opacity-75 text-white px-4 py-2 rounded-lg text-sm animate-fadeIn"
                  >
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getEventTypeIcon(event.type, event.description)}</span>
                      <div>
                        <div className="font-semibold">{event.type.toUpperCase()}</div>
                        <div className="text-xs opacity-90">{event.description}</div>
                      </div>
                    </div>
                  </div>
                ))}
                
                {/* Play/Pause Overlay */}
                {!isPlaying && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <button
                      onClick={togglePlay}
                      className="w-20 h-20 bg-black bg-opacity-50 hover:bg-opacity-75 rounded-full flex items-center justify-center text-white transition-all"
                    >
                      <Play className="w-8 h-8 ml-1" />
                    </button>
                  </div>
                )}
                
                {/* Video Controls */}
                {showControls && (
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4">
                    {/* Progress Bar */}
                    <div className="mb-4">
                      <div 
                        className="w-full h-2 bg-gray-600 rounded cursor-pointer"
                        onClick={handleSeek}
                      >
                        <div 
                          className="h-full bg-blue-500 rounded relative"
                          style={{ width: `${(currentTime / duration) * 100}%` }}
                        >
                          <div className="absolute right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-blue-500 rounded-full border-2 border-white"></div>
                        </div>
                      </div>
                      
                      {/* Event Markers */}
                      <div className="relative h-1 mt-1">
                        {events_detected.map((event, index) => {
                          let eventTime = 0
                          if (typeof event.timestamp === 'string' && event.timestamp.includes(':')) {
                            const parts = event.timestamp.split(':')
                            eventTime = parseInt(parts[0]) * 60 + parseInt(parts[1])
                          } else {
                            eventTime = parseFloat(event.timestamp) || 0
                          }
                          
                          const position = (eventTime / duration) * 100
                          return (
                            <div
                              key={index}
                              className={`absolute w-2 h-2 rounded-full cursor-pointer transform -translate-x-1/2 ${
                                event.severity === 'critical' ? 'bg-red-500' :
                                event.severity === 'high' ? 'bg-orange-500' :
                                'bg-yellow-500'
                              }`}
                              style={{ left: `${position}%` }}
                              onClick={() => skipToEvent(index)}
                              title={`${event.description} at ${formatTimestamp(event.timestamp)}`}
                            />
                          )
                        })}
                      </div>
                    </div>
                    
                    {/* Control Buttons */}
                    <div className="flex items-center justify-between text-white">
                      <div className="flex items-center space-x-4">
                        <button onClick={togglePlay} className="hover:text-blue-400">
                          {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
                        </button>
                        
                        <div className="flex items-center space-x-2">
                          <Volume2 className="w-5 h-5" />
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={volume}
                            onChange={handleVolumeChange}
                            className="w-16"
                          />
                        </div>
                        
                        <div className="text-sm">
                          {formatTime(currentTime)} / {formatTime(duration)}
                        </div>
                      </div>
                      
                      <button
                        onClick={downloadVideo}
                        className="hover:text-blue-400"
                        title="Download Video"
                      >
                        <Download className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Event Timeline Sidebar */}
        <div className="lg:col-span-1">
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-semibold">Event Timeline</h3>
              <p className="text-sm text-gray-600">{events_detected.length} events detected</p>
            </div>
            <div className="card-body max-h-96 overflow-y-auto">
              {events_detected.length > 0 ? (
                <div className="space-y-3">
                  {events_detected.map((event, index) => (
                    <div
                      key={index}
                      className="p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                      onClick={() => skipToEvent(index)}
                    >
                      <div className="flex items-start space-x-3">
                        <span className="text-lg">{getEventTypeIcon(event.type, event.description)}</span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium text-gray-900 truncate">
                              {formatTimestamp(event.timestamp)}
                            </span>
                            <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${getSeverityColor(event.severity)}`}>
                              {event.severity}
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 leading-tight">{event.description}</p>
                          <div className="flex items-center space-x-2 mt-2 text-xs text-gray-500">
                            <span>Frame {event.frame_number}</span>
                            <span>•</span>
                            <span>{(event.confidence * 100).toFixed(0)}% confidence</span>
                          </div>
                          
                          {/* Event Thumbnail in Sidebar */}
                          {event.event_id && (
                            <div className="mt-2">
                              <img
                                src={`http://localhost:8002/api/driver-monitoring/thumbnail/${event.event_id}`}
                                alt={`Event preview`}
                                className="w-20 h-15 object-cover rounded border border-gray-300"
                                onError={(e) => {
                                  e.target.style.display = 'none'
                                }}
                              />
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p>No events detected</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Video Information */}
      <div className="card">
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{events_detected.length}</div>
              <div className="text-sm text-blue-800">Total Events</div>
            </div>
            <div className="bg-red-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-red-600">
                {events_detected.filter(e => e.severity === 'critical').length}
              </div>
              <div className="text-sm text-red-800">Critical Events</div>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {events_detected.filter(e => e.severity === 'high').length}
              </div>
              <div className="text-sm text-orange-800">High Severity</div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {formatTime(duration)}
              </div>
              <div className="text-sm text-green-800">Total Duration</div>
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
          <span>Back to Results</span>
        </button>
        
        <button
          onClick={onNext}
          className="btn-primary flex items-center space-x-2"
        >
          <span>Export Reports</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

export default AnnotatedVideo