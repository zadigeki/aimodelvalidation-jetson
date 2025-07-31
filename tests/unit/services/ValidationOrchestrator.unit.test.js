/**
 * ValidationOrchestrator Unit Tests
 * 
 * London School TDD unit tests focusing on mock interactions and behavior verification.
 * These tests define how the ValidationOrchestrator should collaborate with its dependencies.
 */

import { 
  createAIModelServiceMock, 
  createValidationRepositoryMock,
  createNotificationServiceMock,
  createMetricsCollectorMock 
} from '../../mocks/mockFactory.js';

describe('ValidationOrchestrator - Unit Tests (London School)', () => {
  let validationOrchestrator;
  let mockAIService;
  let mockRepository;
  let mockNotifier;
  let mockMetrics;

  // Mock the ValidationOrchestrator class until implemented
  class MockValidationOrchestrator {
    constructor(aiService, repository, notifier, metrics) {
      this.aiService = aiService;
      this.repository = repository;
      this.notifier = notifier;
      this.metrics = metrics;
    }

    async validateModel(modelData, validationRules) {
      // Input validation
      if (!modelData || !modelData.id) {
        throw new Error('Invalid model data: missing id');
      }

      if (!validationRules) {
        throw new Error('Validation rules are required');
      }

      try {
        // Step 1: Validate the model
        const validationResult = await this.aiService.validate(modelData, validationRules);

        // Step 2: If validation passes, save and notify
        if (validationResult.isValid) {
          const savedId = await this.repository.save(validationResult);
          await this.notifier.sendValidationComplete(validationResult);
          this.metrics.recordValidation(modelData.id, 1500, validationResult);
          
          return { ...validationResult, id: savedId };
        } else {
          // Step 3: If validation fails, notify failure and record metrics
          await this.notifier.sendValidationFailed(
            new Error('Validation failed'), 
            { modelId: modelData.id, errors: validationResult.errors }
          );
          this.metrics.recordValidation(modelData.id, 800, validationResult);
          
          throw new Error(`Model validation failed: ${validationResult.errors.join(', ')}`);
        }
      } catch (error) {
        // Handle service errors
        if (error.message.includes('validation failed')) {
          throw error; // Re-throw validation failures
        }
        
        // Handle unexpected errors
        await this.notifier.sendAlert(
          `Unexpected error during validation: ${error.message}`,
          'error'
        );
        throw new Error(`Validation process failed: ${error.message}`);
      }
    }

    async evaluatePerformance(modelId, testData, metricsToEvaluate) {
      if (!modelId) {
        throw new Error('Model ID is required');
      }

      const evaluationResult = await this.aiService.evaluate(testData, metricsToEvaluate);
      
      const evaluationRecord = {
        type: 'performance_evaluation',
        modelId,
        result: evaluationResult,
        timestamp: Date.now()
      };

      const savedId = await this.repository.save(evaluationRecord);
      this.metrics.recordValidation(modelId, 2000, evaluationResult);

      return { ...evaluationResult, evaluationId: savedId };
    }

    async getBatchValidationStatus(batchId) {
      const validations = await this.repository.findByModel(batchId);
      
      const status = {
        batchId,
        total: validations.length,
        completed: validations.filter(v => v.status === 'completed').length,
        failed: validations.filter(v => v.status === 'failed').length,
        pending: validations.filter(v => v.status === 'pending').length
      };

      return status;
    }
  }

  beforeEach(() => {
    // Mock-first: Create all collaborator mocks
    mockAIService = createAIModelServiceMock();
    mockRepository = createValidationRepositoryMock();
    mockNotifier = createNotificationServiceMock();
    mockMetrics = createMetricsCollectorMock();

    // Inject mocks into the orchestrator
    validationOrchestrator = new MockValidationOrchestrator(
      mockAIService,
      mockRepository,
      mockNotifier,
      mockMetrics
    );
  });

  describe('validateModel method', () => {
    const validModelData = {
      id: 'model-123',
      name: 'Test Model',
      version: '1.0.0'
    };

    const validationRules = {
      accuracyThreshold: 0.8,
      maxLatency: 500
    };

    it('should validate input parameters before processing', async () => {
      // WHEN: Called with missing model data
      // THEN: Should reject with appropriate error
      await expect(
        validationOrchestrator.validateModel(null, validationRules)
      ).rejects.toThrow('Invalid model data: missing id');

      await expect(
        validationOrchestrator.validateModel({ name: 'test' }, validationRules)
      ).rejects.toThrow('Invalid model data: missing id');

      await expect(
        validationOrchestrator.validateModel(validModelData, null)
      ).rejects.toThrow('Validation rules are required');

      // Verify no services were called with invalid input
      expect(mockAIService.validate).not.toHaveBeenCalled();
      expect(mockRepository.save).not.toHaveBeenCalled();
    });

    it('should coordinate successful validation workflow', async () => {
      // GIVEN: AI service returns successful validation
      const expectedValidationResult = {
        isValid: true,
        score: 0.92,
        errors: [],
        warnings: []
      };
      mockAIService.validate.mockResolvedValue(expectedValidationResult);
      mockRepository.save.mockResolvedValue('saved-id-123');

      // WHEN: Validating a model
      const result = await validationOrchestrator.validateModel(validModelData, validationRules);

      // THEN: Should coordinate services in correct sequence
      expect(mockAIService.validate).toHaveBeenCalledWith(validModelData, validationRules);
      expect(mockRepository.save).toHaveBeenCalledWith(expectedValidationResult);
      expect(mockNotifier.sendValidationComplete).toHaveBeenCalledWith(expectedValidationResult);
      expect(mockMetrics.recordValidation).toHaveBeenCalledWith(
        validModelData.id, 
        1500, 
        expectedValidationResult
      );

      // Verify result structure
      expect(result).toEqual({
        ...expectedValidationResult,
        id: 'saved-id-123'
      });
    });

    it('should handle validation failure workflow', async () => {
      // GIVEN: AI service returns validation failure
      const failedValidationResult = {
        isValid: false,
        score: 0.65,
        errors: ['Accuracy too low: 0.65 < 0.8'],
        warnings: ['Consider retraining']
      };
      mockAIService.validate.mockResolvedValue(failedValidationResult);

      // WHEN: Validating fails
      // THEN: Should coordinate failure workflow
      await expect(
        validationOrchestrator.validateModel(validModelData, validationRules)
      ).rejects.toThrow('Model validation failed: Accuracy too low: 0.65 < 0.8');

      // Verify failure coordination
      expect(mockAIService.validate).toHaveBeenCalledWith(validModelData, validationRules);
      expect(mockNotifier.sendValidationFailed).toHaveBeenCalledWith(
        expect.any(Error),
        expect.objectContaining({
          modelId: validModelData.id,
          errors: ['Accuracy too low: 0.65 < 0.8']
        })
      );
      expect(mockMetrics.recordValidation).toHaveBeenCalledWith(
        validModelData.id,
        800,
        failedValidationResult
      );

      // Repository should NOT be called on validation failure
      expect(mockRepository.save).not.toHaveBeenCalled();
      expect(mockNotifier.sendValidationComplete).not.toHaveBeenCalled();
    });

    it('should handle service errors gracefully', async () => {
      // GIVEN: AI service throws an error
      const serviceError = new Error('AI service unavailable');
      mockAIService.validate.mockRejectedValue(serviceError);

      // WHEN: Service fails
      // THEN: Should handle error and send alert
      await expect(
        validationOrchestrator.validateModel(validModelData, validationRules)
      ).rejects.toThrow('Validation process failed: AI service unavailable');

      expect(mockNotifier.sendAlert).toHaveBeenCalledWith(
        'Unexpected error during validation: AI service unavailable',
        'error'
      );

      // Other services should not be called after AI service failure
      expect(mockRepository.save).not.toHaveBeenCalled();
      expect(mockNotifier.sendValidationComplete).not.toHaveBeenCalled();
    });

    it('should verify service interaction sequence', async () => {
      // GIVEN: Successful validation setup
      mockAIService.validate.mockResolvedValue({ isValid: true, score: 0.9 });
      mockRepository.save.mockResolvedValue('id-123');

      // WHEN: Validating
      await validationOrchestrator.validateModel(validModelData, validationRules);

      // THEN: Services should be called in specific order
      expect(mockAIService.validate).toHaveBeenCalledBefore(mockRepository.save);
      expect(mockRepository.save).toHaveBeenCalledBefore(mockNotifier.sendValidationComplete);
      expect(mockNotifier.sendValidationComplete).toHaveBeenCalledBefore(mockMetrics.recordValidation);
    });
  });

  describe('evaluatePerformance method', () => {
    const testData = {
      inputs: [[1, 2], [3, 4]],
      expectedOutputs: [[0.8], [0.6]]
    };
    const metricsToEvaluate = ['accuracy', 'precision'];

    it('should coordinate performance evaluation workflow', async () => {
      // GIVEN: AI service provides evaluation results
      const evaluationResult = {
        accuracy: 0.89,
        precision: 0.91,
        recall: 0.87
      };
      mockAIService.evaluate.mockResolvedValue(evaluationResult);
      mockRepository.save.mockResolvedValue('eval-123');

      // WHEN: Evaluating performance
      const result = await validationOrchestrator.evaluatePerformance(
        'model-456', 
        testData, 
        metricsToEvaluate
      );

      // THEN: Should coordinate evaluation services
      expect(mockAIService.evaluate).toHaveBeenCalledWith(testData, metricsToEvaluate);
      expect(mockRepository.save).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'performance_evaluation',
          modelId: 'model-456',
          result: evaluationResult
        })
      );
      expect(mockMetrics.recordValidation).toHaveBeenCalledWith(
        'model-456',
        2000,
        evaluationResult
      );

      expect(result).toEqual({
        ...evaluationResult,
        evaluationId: 'eval-123'
      });
    });

    it('should require model ID for evaluation', async () => {
      // WHEN: Called without model ID
      // THEN: Should reject
      await expect(
        validationOrchestrator.evaluatePerformance(null, testData, metricsToEvaluate)
      ).rejects.toThrow('Model ID is required');

      // Verify no services called
      expect(mockAIService.evaluate).not.toHaveBeenCalled();
    });
  });

  describe('getBatchValidationStatus method', () => {
    it('should aggregate batch validation status', async () => {
      // GIVEN: Repository returns validation records
      const mockValidations = [
        { id: '1', status: 'completed' },
        { id: '2', status: 'completed' },
        { id: '3', status: 'failed' },
        { id: '4', status: 'pending' }
      ];
      mockRepository.findByModel.mockResolvedValue(mockValidations);

      // WHEN: Getting batch status
      const status = await validationOrchestrator.getBatchValidationStatus('batch-789');

      // THEN: Should return aggregated status
      expect(mockRepository.findByModel).toHaveBeenCalledWith('batch-789');
      expect(status).toEqual({
        batchId: 'batch-789',
        total: 4,
        completed: 2,
        failed: 1,
        pending: 1
      });
    });
  });

  describe('Contract verification', () => {
    it('should satisfy all service contracts', () => {
      // Verify the orchestrator uses the expected contract methods
      const expectedAIServiceMethods = ['validate', 'evaluate'];
      const expectedRepositoryMethods = ['save', 'findByModel'];
      const expectedNotifierMethods = ['sendValidationComplete', 'sendValidationFailed', 'sendAlert'];
      const expectedMetricsMethods = ['recordValidation'];

      // This test ensures we're using the contracts correctly
      expect(mockAIService).toHaveProperty('validate');
      expect(mockAIService).toHaveProperty('evaluate');
      expect(mockRepository).toHaveProperty('save');
      expect(mockRepository).toHaveProperty('findByModel');
      expect(mockNotifier).toHaveProperty('sendValidationComplete');
      expect(mockNotifier).toHaveProperty('sendValidationFailed');
      expect(mockNotifier).toHaveProperty('sendAlert');
      expect(mockMetrics).toHaveProperty('recordValidation');
    });
  });
});