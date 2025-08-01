import React from 'react'
import { Check, Clock } from 'lucide-react'

const ProgressTracker = ({ steps, currentStep, isAnalyzing, progress }) => {
  return (
    <div className="card">
      <div className="card-body">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">Analysis Pipeline</h2>
          {isAnalyzing && (
            <div className="flex items-center space-x-2 text-sm text-blue-600">
              <Clock className="w-4 h-4" />
              <span>Processing...</span>
            </div>
          )}
        </div>
        
        <div className="relative">
          {/* Progress Line */}
          <div className="absolute top-5 left-0 w-full h-0.5 bg-gray-200">
            <div 
              className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-500 ease-out"
              style={{ 
                width: `${(currentStep / (steps.length - 1)) * 100}%` 
              }}
            />
          </div>
          
          {/* Steps */}
          <div className="relative flex justify-between">
            {steps.map((step, index) => {
              const isCompleted = index < currentStep
              const isCurrent = index === currentStep
              const isUpcoming = index > currentStep
              
              return (
                <div key={step.id} className="flex flex-col items-center">
                  {/* Step Circle */}
                  <div className={`
                    relative w-10 h-10 rounded-full border-2 flex items-center justify-center transition-all duration-300
                    ${isCompleted 
                      ? 'bg-green-500 border-green-500 text-white' 
                      : isCurrent 
                        ? isAnalyzing
                          ? 'bg-blue-500 border-blue-500 text-white animate-pulse'
                          : 'bg-blue-500 border-blue-500 text-white'
                        : 'bg-white border-gray-300 text-gray-400'
                    }
                  `}>
                    {isCompleted ? (
                      <Check className="w-5 h-5" />
                    ) : (
                      React.createElement(step.icon, { className: "w-5 h-5" })
                    )}
                    
                    {/* Loading Ring for Current Step */}
                    {isCurrent && isAnalyzing && (
                      <div className="absolute inset-0 rounded-full border-2 border-blue-200 border-t-blue-500 animate-spin" />
                    )}
                  </div>
                  
                  {/* Step Info */}
                  <div className="mt-3 text-center">
                    <div className={`text-sm font-medium ${
                      isCompleted || isCurrent ? 'text-gray-900' : 'text-gray-500'
                    }`}>
                      {step.title}
                    </div>
                    <div className={`text-xs mt-1 max-w-24 ${
                      isCompleted || isCurrent ? 'text-gray-600' : 'text-gray-400'
                    }`}>
                      {step.description}
                    </div>
                    
                    {/* Progress for current step */}
                    {isCurrent && isAnalyzing && progress > 0 && (
                      <div className="mt-2 w-20 mx-auto">
                        <div className="progress-bar h-1">
                          <div 
                            className="progress-fill bg-blue-500"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                        <div className="text-xs text-blue-600 mt-1">
                          {Math.round(progress)}%
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
        
        {/* Current Step Details */}
        {currentStep < steps.length && (
          <div className="mt-8 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                {React.createElement(steps[currentStep].icon, { className: "w-4 h-4 text-blue-600" })}
              </div>
              <div>
                <h3 className="font-medium text-gray-900">
                  {steps[currentStep].title}
                </h3>
                <p className="text-sm text-gray-600">
                  {steps[currentStep].description}
                </p>
              </div>
            </div>
            
            {isAnalyzing && currentStep === 1 && (
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">AI Processing Progress</span>
                  <span className="text-blue-600 font-medium">{Math.round(progress)}%</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill bg-gradient-to-r from-blue-500 to-purple-600"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500">
                  Analyzing video with MediaPipe and YOLO models...
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ProgressTracker