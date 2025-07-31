/**
 * Mock Factory for TDD London School
 * 
 * Provides mock creation utilities for dependency injection and contract testing.
 * Focuses on behavior verification and interaction testing.
 */

/**
 * Creates a comprehensive mock object with Jest spies
 * @param {Object} contractDefinition - Interface definition with method signatures
 * @param {Object} defaultBehaviors - Default return values or implementations
 * @returns {Object} Mock object with spies and utilities
 */
export const createContractMock = (contractDefinition, defaultBehaviors = {}) => {
  const mock = {};
  
  Object.entries(contractDefinition).forEach(([methodName, signature]) => {
    const mockFn = jest.fn();
    
    // Set default behavior if provided
    if (defaultBehaviors[methodName]) {
      if (typeof defaultBehaviors[methodName] === 'function') {
        mockFn.mockImplementation(defaultBehaviors[methodName]);
      } else {
        mockFn.mockResolvedValue(defaultBehaviors[methodName]);
      }
    }
    
    // Add metadata about the method signature
    mockFn.signature = signature;
    mockFn.methodName = methodName;
    
    mock[methodName] = mockFn;
  });
  
  // Add contract verification utilities
  mock._verifyContract = (expectedInteractions) => {
    expectedInteractions.forEach(({ method, args, times = 1 }) => {
      if (args) {
        expect(mock[method]).toHaveBeenCalledWith(...args);
      }
      expect(mock[method]).toHaveBeenCalledTimes(times);
    });
  };
  
  mock._getInteractionHistory = () => {
    const history = [];
    Object.keys(contractDefinition).forEach(method => {
      mock[method].mock.calls.forEach((call, index) => {
        history.push({
          method,
          args: call,
          callIndex: index,
          timestamp: mock[method].mock.invocationCallOrder[index]
        });
      });
    });
    return history.sort((a, b) => a.timestamp - b.timestamp);
  };
  
  mock._reset = () => {
    Object.keys(contractDefinition).forEach(method => {
      mock[method].mockReset();
    });
  };
  
  return mock;
};

/**
 * AI Model Service Mock Contracts
 */
export const AIModelServiceContract = {
  validate: { 
    params: ['modelData', 'validationRules'], 
    returns: 'Promise<ValidationResult>' 
  },
  predict: { 
    params: ['inputData'], 
    returns: 'Promise<PredictionResult>' 
  },
  train: { 
    params: ['trainingData', 'config'], 
    returns: 'Promise<TrainingResult>' 
  },
  evaluate: { 
    params: ['testData', 'metrics'], 
    returns: 'Promise<EvaluationResult>' 
  }
};

export const ValidationRepositoryContract = {
  save: { 
    params: ['validationResult'], 
    returns: 'Promise<string>' 
  },
  findById: { 
    params: ['id'], 
    returns: 'Promise<ValidationResult>' 
  },
  findByModel: { 
    params: ['modelId'], 
    returns: 'Promise<ValidationResult[]>' 
  },
  delete: { 
    params: ['id'], 
    returns: 'Promise<boolean>' 
  }
};

export const NotificationServiceContract = {
  sendValidationComplete: { 
    params: ['validationResult'], 
    returns: 'Promise<boolean>' 
  },
  sendValidationFailed: { 
    params: ['error', 'context'], 
    returns: 'Promise<boolean>' 
  },
  sendAlert: { 
    params: ['message', 'severity'], 
    returns: 'Promise<boolean>' 
  }
};

export const MetricsCollectorContract = {
  recordValidation: { 
    params: ['modelId', 'duration', 'result'], 
    returns: 'void' 
  },
  recordPrediction: { 
    params: ['modelId', 'inputSize', 'latency'], 
    returns: 'void' 
  },
  getMetrics: { 
    params: ['modelId', 'timeRange'], 
    returns: 'Promise<MetricsData>' 
  }
};

/**
 * Creates service mocks with realistic behaviors for AI model validation
 */
export const createAIModelServiceMock = (customBehaviors = {}) => {
  const defaultBehaviors = {
    validate: async (modelData, validationRules) => ({
      isValid: true,
      score: 0.85,
      errors: [],
      warnings: [],
      metadata: { modelId: modelData.id, timestamp: Date.now() }
    }),
    predict: async (inputData) => ({
      prediction: [0.7, 0.3],
      confidence: 0.85,
      modelVersion: '1.0.0',
      processingTime: 150
    }),
    train: async (trainingData, config) => ({
      modelId: 'model-123',
      accuracy: 0.92,
      loss: 0.08,
      epochs: config.epochs || 100,
      duration: 3600000
    }),
    evaluate: async (testData, metrics) => ({
      accuracy: 0.89,
      precision: 0.91,
      recall: 0.87,
      f1Score: 0.89,
      confusionMatrix: [[45, 5], [3, 47]]
    })
  };
  
  return createContractMock(AIModelServiceContract, { ...defaultBehaviors, ...customBehaviors });
};

export const createValidationRepositoryMock = (customBehaviors = {}) => {
  const defaultBehaviors = {
    save: async (validationResult) => `validation-${Date.now()}`,
    findById: async (id) => ({ id, status: 'completed', result: { isValid: true } }),
    findByModel: async (modelId) => [{ id: '1', modelId, status: 'completed' }],
    delete: async (id) => true
  };
  
  return createContractMock(ValidationRepositoryContract, { ...defaultBehaviors, ...customBehaviors });
};

export const createNotificationServiceMock = (customBehaviors = {}) => {
  const defaultBehaviors = {
    sendValidationComplete: async (validationResult) => true,
    sendValidationFailed: async (error, context) => true,
    sendAlert: async (message, severity) => true
  };
  
  return createContractMock(NotificationServiceContract, { ...defaultBehaviors, ...customBehaviors });
};

export const createMetricsCollectorMock = (customBehaviors = {}) => {
  const defaultBehaviors = {
    recordValidation: jest.fn(),
    recordPrediction: jest.fn(),
    getMetrics: async (modelId, timeRange) => ({
      validations: 150,
      predictions: 1250,
      averageLatency: 200,
      errorRate: 0.02
    })
  };
  
  return createContractMock(MetricsCollectorContract, { ...defaultBehaviors, ...customBehaviors });
};

/**
 * Swarm mock coordination for distributed testing
 */
export const createSwarmMock = (serviceName, contractDef, customBehaviors = {}) => {
  const mock = createContractMock(contractDef, customBehaviors);
  
  // Add swarm coordination metadata
  mock._swarmId = `swarm-${serviceName}-${Date.now()}`;
  mock._serviceName = serviceName;
  mock._coordinationHistory = [];
  
  // Override methods to track coordination
  Object.keys(contractDef).forEach(method => {
    const originalMethod = mock[method];
    mock[method] = jest.fn(async (...args) => {
      // Record coordination event
      mock._coordinationHistory.push({
        method,
        args,
        timestamp: Date.now(),
        swarmId: mock._swarmId
      });
      
      // Call original mock behavior
      return await originalMethod(...args);
    });
  });
  
  return mock;
};

/**
 * Mock evolution for contract changes
 */
export const extendSwarmMock = (baseMock, extensionContract, newBehaviors = {}) => {
  const extended = { ...baseMock };
  
  Object.entries(extensionContract).forEach(([methodName, signature]) => {
    const mockFn = jest.fn();
    
    if (newBehaviors[methodName]) {
      if (typeof newBehaviors[methodName] === 'function') {
        mockFn.mockImplementation(newBehaviors[methodName]);
      } else {
        mockFn.mockResolvedValue(newBehaviors[methodName]);
      }
    }
    
    mockFn.signature = signature;
    mockFn.methodName = methodName;
    extended[methodName] = mockFn;
  });
  
  return extended;
};