import React, { useRef, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  PlayIcon,
  PauseIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  ArrowsPointingOutIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline'
import { ValidationFile, FrameResult } from '@/types'
import { formatDuration } from '@/utils/formatters'

interface VideoPlayerProps {
  file: ValidationFile
  frames?: FrameResult[]
  showAnnotations?: boolean
  onFrameChange?: (frameNumber: number, timestamp: number) => void
  onTimeUpdate?: (currentTime: number) => void
  className?: string
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
  file,
  frames = [],
  showAnnotations = true,
  onFrameChange,
  onTimeUpdate,
  className = '',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const progressRef = useRef<HTMLDivElement>(null)
  
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showControls, setShowControls] = useState(true)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isLoading, setIsLoading] = useState(true)

  const frameRate = file.metadata?.frameRate || 30

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadedMetadata = () => {
      setDuration(video.duration)
      setIsLoading(false)
    }

    const handleTimeUpdate = () => {
      const time = video.currentTime
      setCurrentTime(time)
      
      // Calculate current frame
      const frame = Math.floor(time * frameRate)
      setCurrentFrame(frame)
      
      onTimeUpdate?.(time)
      onFrameChange?.(frame, time)
    }

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleVolumeChange = () => {
      setVolume(video.volume)
      setIsMuted(video.muted)
    }

    const handleLoadStart = () => setIsLoading(true)
    const handleCanPlay = () => setIsLoading(false)

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('volumechange', handleVolumeChange)
    video.addEventListener('loadstart', handleLoadStart)
    video.addEventListener('canplay', handleCanPlay)

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('volumechange', handleVolumeChange)
      video.removeEventListener('loadstart', handleLoadStart)
      video.removeEventListener('canplay', handleCanPlay)
    }
  }, [frameRate, onFrameChange, onTimeUpdate])

  useEffect(() => {
    // Draw annotations on canvas
    if (showAnnotations && canvasRef.current && videoRef.current) {
      drawAnnotations()
    }
  }, [currentFrame, showAnnotations, frames])

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return

    if (isPlaying) {
      video.pause()
    } else {
      video.play()
    }
  }

  const toggleMute = () => {
    const video = videoRef.current
    if (!video) return

    video.muted = !video.muted
  }

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current
    const progressBar = progressRef.current
    if (!video || !progressBar) return

    const rect = progressBar.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const percentage = clickX / rect.width
    const newTime = percentage * duration

    video.currentTime = newTime
  }

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current
    if (!video) return

    const newVolume = parseFloat(e.target.value)
    video.volume = newVolume
    video.muted = newVolume === 0
  }

  const seekToFrame = (frameNumber: number) => {
    const video = videoRef.current
    if (!video) return

    const time = frameNumber / frameRate
    video.currentTime = Math.min(time, duration)
  }

  const nextFrame = () => {
    seekToFrame(currentFrame + 1)
  }

  const prevFrame = () => {
    seekToFrame(Math.max(currentFrame - 1, 0))
  }

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      videoRef.current?.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  const drawAnnotations = () => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Find frame results for current frame
    const frameResult = frames.find(f => Math.abs(f.frameNumber - currentFrame) <= 1)
    if (!frameResult) return

    // Set canvas size to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw bounding boxes
    frameResult.objects.forEach((obj, index) => {
      const { bbox, confidence, class: className } = obj
      
      // Calculate colors based on confidence
      const confidenceColor = confidence >= 0.8 ? '#22c55e' : 
                             confidence >= 0.6 ? '#f59e0b' : '#ef4444'

      // Draw bounding box
      ctx.strokeStyle = confidenceColor
      ctx.lineWidth = 2
      ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height)

      // Draw background for label
      const label = `${className} ${(confidence * 100).toFixed(1)}%`
      ctx.font = '14px Arial'
      const textMetrics = ctx.measureText(label)
      const labelHeight = 20
      
      ctx.fillStyle = confidenceColor
      ctx.fillRect(bbox.x, bbox.y - labelHeight, textMetrics.width + 8, labelHeight)

      // Draw label text
      ctx.fillStyle = 'white'
      ctx.fillText(label, bbox.x + 4, bbox.y - 6)
    })
  }

  const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div 
      className={`relative bg-black rounded-lg overflow-hidden ${className}`}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      {/* Video element */}
      <video
        ref={videoRef}
        src={file.url}
        className="w-full h-full object-contain"
        preload="metadata"
        crossOrigin="anonymous"
      />

      {/* Annotation canvas */}
      {showAnnotations && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
      )}

      {/* Loading overlay */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/50 flex items-center justify-center"
          >
            <div className="text-white text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
              <p>Loading video...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Controls overlay */}
      <AnimatePresence>
        {showControls && !isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent"
          >
            {/* Top controls */}
            <div className="absolute top-4 right-4 flex items-center space-x-2">
              <button
                onClick={toggleFullscreen}
                className="p-2 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors"
              >
                <ArrowsPointingOutIcon className="h-5 w-5" />
              </button>
            </div>

            {/* Bottom controls */}
            <div className="absolute bottom-0 left-0 right-0 p-4">
              {/* Progress bar */}
              <div className="mb-4">
                <div
                  ref={progressRef}
                  className="h-2 bg-white/30 rounded-full cursor-pointer group"
                  onClick={handleProgressClick}
                >
                  <div
                    className="h-full bg-primary-500 rounded-full transition-all duration-150 group-hover:h-3"
                    style={{ width: `${progressPercentage}%` }}
                  />
                </div>
              </div>

              {/* Controls bar */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {/* Play/Pause */}
                  <button
                    onClick={togglePlay}
                    className="p-2 bg-white/20 text-white rounded-full hover:bg-white/30 transition-colors"
                  >
                    {isPlaying ? (
                      <PauseIcon className="h-6 w-6" />
                    ) : (
                      <PlayIcon className="h-6 w-6" />
                    )}
                  </button>

                  {/* Frame controls */}
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={prevFrame}
                      className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                    >
                      <ChevronLeftIcon className="h-4 w-4" />
                    </button>
                    <span className="text-white text-sm px-2">
                      Frame {currentFrame}
                    </span>
                    <button
                      onClick={nextFrame}
                      className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                    >
                      <ChevronRightIcon className="h-4 w-4" />
                    </button>
                  </div>

                  {/* Time display */}
                  <span className="text-white text-sm">
                    {formatDuration(currentTime)} / {formatDuration(duration)}
                  </span>
                </div>

                <div className="flex items-center space-x-3">
                  {/* Volume control */}
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={toggleMute}
                      className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                    >
                      {isMuted || volume === 0 ? (
                        <SpeakerXMarkIcon className="h-5 w-5" />
                      ) : (
                        <SpeakerWaveIcon className="h-5 w-5" />
                      )}
                    </button>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={isMuted ? 0 : volume}
                      onChange={handleVolumeChange}
                      className="w-20 h-1 bg-white/30 rounded-full outline-none"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Frame annotations info */}
            {showAnnotations && frames.length > 0 && (
              <div className="absolute top-4 left-4 bg-black/70 text-white p-3 rounded-lg">
                <p className="text-sm mb-1">Frame {currentFrame}</p>
                {frames.find(f => Math.abs(f.frameNumber - currentFrame) <= 1) && (
                  <p className="text-xs text-gray-300">
                    {frames.find(f => Math.abs(f.frameNumber - currentFrame) <= 1)?.objects.length || 0} objects detected
                  </p>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default VideoPlayer