/**
 * Domain Contracts for AI Model Validation
 * London School TDD - Interface definitions driven by mock expectations
 */

// Data Capture Contracts
export const WebcamCaptureContract = {
  initialize: () => Promise.resolve(Boolean),
  captureFrame: (options = {}) => Promise.resolve(Object),
  startContinuousCapture: (config) => Promise.resolve(void 0),
  stopContinuousCapture: () => Promise.resolve(void 0),
  isActive: () => Boolean
};

export const DataStorageContract = {
  save: (data) => Promise.resolve(Object),
  load: (id) => Promise.resolve(Object),
  exists: (id) => Boolean,
  delete: (id) => Promise.resolve(Boolean),
  list: (pattern) => Promise.resolve(Array)
};

// Annotation Contracts
export const AnnotationServiceContract = {
  createProject: (config) => Promise.resolve(Object),
  uploadData: (data) => Promise.resolve(Object),
  getAnnotations: (taskId) => Promise.resolve(Object),
  exportAnnotations: (taskId, format) => Promise.resolve(Object),
  validateAnnotations: (annotations) => Promise.resolve(Object)
};

export const CvatClientContract = {
  connect: () => Promise.resolve(Boolean),
  authenticate: (credentials) => Promise.resolve(Object),
  createTask: (taskConfig) => Promise.resolve(Object),
  uploadImages: (images) => Promise.resolve(Object),
  getTaskStatus: (taskId) => Promise.resolve(Object)
};

// Validation Contracts
export const ValidationServiceContract = {
  validateDataset: (dataset) => Promise.resolve(Object),
  validateModel: (model, testData) => Promise.resolve(Object),
  generateReport: (validationResults) => Promise.resolve(Object)
};

export const DeepchecksClientContract = {
  runSuite: (suite, data) => Promise.resolve(Object),
  runCheck: (check, data) => Promise.resolve(Object),
  generateHtmlReport: (results) => Promise.resolve(String),
  getCheckResults: (runId) => Promise.resolve(Array)
};

// Model Training Contracts
export const ModelTrainerContract = {
  initialize: (config) => Promise.resolve(Boolean),
  train: (trainingConfig) => Promise.resolve(Object),
  validateModel: (validationData) => Promise.resolve(Object),
  saveModel: (path) => Promise.resolve(Object),
  loadModel: (path) => Promise.resolve(Object)
};

export const UltralyticsClientContract = {
  createModel: (modelType) => Promise.resolve(Object),
  trainModel: (trainingData, config) => Promise.resolve(Object),
  predict: (model, input) => Promise.resolve(Object),
  exportModel: (model, format) => Promise.resolve(Object)
};

// Infrastructure Contracts
export const LoggerContract = {
  info: (message, metadata) => void 0,
  warn: (message, metadata) => void 0,
  error: (message, metadata) => void 0,
  debug: (message, metadata) => void 0,
  createChild: (name) => Object
};

export const ConfigManagerContract = {
  get: (key) => Object,
  set: (key, value) => Boolean,
  load: (source) => Promise.resolve(Boolean),
  validate: (config) => Boolean
};

export const FileSystemContract = {
  readFile: (path) => Promise.resolve(String),
  writeFile: (path, content) => Promise.resolve(Boolean),
  createDirectory: (path) => Promise.resolve(Boolean),
  exists: (path) => Boolean,
  getStats: (path) => Promise.resolve(Object)
};

// Workflow Orchestration Contracts
export const WorkflowOrchestratorContract = {
  executeWorkflow: (workflowConfig) => Promise.resolve(Object),
  resumeWorkflow: (workflowState) => Promise.resolve(Object),
  getWorkflowStatus: (workflowId) => Promise.resolve(Object),
  cancelWorkflow: (workflowId) => Promise.resolve(Boolean)
};

// Contract validation utility
export const validateContract = (implementation, contract) => {
  const contractMethods = Object.keys(contract);
  const implementationMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(implementation))
    .filter(name => typeof implementation[name] === 'function' && name !== 'constructor');
  
  const missingMethods = contractMethods.filter(method => 
    !implementationMethods.includes(method)
  );
  
  if (missingMethods.length > 0) {
    throw new Error(`Implementation missing contract methods: ${missingMethods.join(', ')}`);
  }
  
  return true;
};